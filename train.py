import os
import json
import argparse
from pathlib import Path

import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from config import Config
from audio_preprocessor import AudioPreprocessor
from data_utils import (
    parse_transcript_file,
    prepare_data_splits,
    DatasetProcessor,
    combine_transcript_with_audio
)
from training_utils import (
    WhisperTrainingCallback,
    MetricsComputer,
    create_data_collator,
    TrainingMonitor
)
from inference import WhisperSanskritTranscriber, evaluate_on_test_set


def setup_model_and_processor(config: Config):
    
    print(f"Loading Whisper model: {config.model_name}")
    
    # Load processor and model
    processor = WhisperProcessor.from_pretrained(config.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
    
    # Configure for Sanskrit
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    # Set language and task
    model.generation_config.language = config.language
    model.generation_config.task = config.task
    
    # Move to device
    model.to(config.device)
    
    print(f"Model loaded. Parameters: {model.num_parameters():,}")
    print(f"Device: {config.device}")
    
    return model, processor


def prepare_data(config: Config, processor: WhisperProcessor):
    """Prepare all data for training"""
    print("\n=== Data Preparation ===")
    
    # Create directories
    os.makedirs(config.cache_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Parse transcripts
    print("\n1. Parsing transcript file...")
    transcript_data = parse_transcript_file(config.transcript_file)
    
    # Initialize audio preprocessor
    print("\n2. Converting audio files...")
    audio_preprocessor = AudioPreprocessor(config.cache_dir, config.sample_rate)
    
    # Get list of audio files from transcript data
    audio_files = [item['audio_filename'] for item in transcript_data]
    converted_paths = audio_preprocessor.convert_audio_batch(audio_files, config.audio_dir)
    
    # Combine transcript data with converted audio paths
    print("\n3. Combining data...")
    combined_data = combine_transcript_with_audio(transcript_data, converted_paths)
    
    # Split data
    print("\n4. Creating data splits...")
    train_data, val_data, test_data = prepare_data_splits(combined_data, config)
    
    # Create datasets
    print("\n5. Processing datasets...")
    # dataset_processor = DatasetProcessor(processor, config.sample_rate)
    dataset_processor = DatasetProcessor(
        processor=processor,
        sample_rate=config.sample_rate,
        use_prosodic_features=config.use_prosodic_features,  # Pass the flag from config
        config=config
    )
    train_dataset = dataset_processor.prepare_dataset(
        train_data, audio_preprocessor, config.max_duration
    )
    val_dataset = dataset_processor.prepare_dataset(
        val_data, audio_preprocessor, config.max_duration
    )
    test_dataset = dataset_processor.prepare_dataset(
        test_data, audio_preprocessor, config.max_duration
    )
    
    print(f"\nFinal dataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset, test_data


def train_model(
    config: Config,
    model,
    processor,
    train_dataset,
    val_dataset,
    training_monitor: TrainingMonitor = None
):
    """Train the model"""
    print("\n=== Training Configuration ===")
    
    # Create training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=config.fp16,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        report_to=["tensorboard"],
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        push_to_hub=False,
        predict_with_generate=True,
        generation_max_length=config.generation_max_length,
        save_total_limit=config.save_total_limit,
        remove_unused_columns=False,
    )
    
    # Create data collator
    data_collator = create_data_collator(processor, model)
    
    # Create metrics computer
    compute_metrics = MetricsComputer(processor)
    
    # Create callbacks
    callbacks = [WhisperTrainingCallback(verbose=True)]
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=callbacks,
    )
    
    print(f"\nTraining configuration:")
    print(f"  Total steps: {config.max_steps}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Evaluation every {config.eval_steps} steps")
    
    # Start training
    print("\n=== Starting Training ===")
    train_result = trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)
    
    return trainer, train_result


def evaluate_model(config: Config, trainer, test_dataset, test_data):
    """Evaluate the model on test set"""
    print("\n=== Evaluation ===")
    
    # Evaluate with trainer
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    
    print("\nTest Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save evaluation results
    results_path = os.path.join(config.output_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Additional evaluation with transcriber
    print("\nPerforming detailed evaluation...")
    transcriber = WhisperSanskritTranscriber(config.output_dir)
    detailed_results = evaluate_on_test_set(
        transcriber,
        test_data,
        os.path.join(config.output_dir, "detailed_test_results.json")
    )
    
    print("\nDetailed Metrics:")
    for key, value in detailed_results.items():
        print(f"  {key}: {value:.4f}")
    
    return test_results, detailed_results


def save_training_metadata(config: Config, train_result, test_results):
    """Save training metadata and configuration"""
    metadata = {
        "model_name": config.model_name,
        "language": config.language,
        "task": config.task,
        "sample_rate": config.sample_rate,
        "max_duration": config.max_duration,
        "training_steps": train_result.global_step,
        "training_loss": train_result.training_loss,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "test_wer": test_results.get("eval_wer", None),
        "device": str(config.device),
    }
    
    metadata_path = os.path.join(config.output_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save config
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2)
    
    print(f"\nMetadata saved to {metadata_path}")
    print(f"Config saved to {config_path}")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for Sanskrit")
    parser.add_argument("--config", type=str, help="Path to custom config file")
    parser.add_argument("--model_name", type=str, help="Whisper model to use")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--max_steps", type=int, help="Maximum training steps")
    parser.add_argument("--audio_dir", type=str, help="Directory containing audio files")
    parser.add_argument("--transcript_file", type=str, help="Path to transcript file")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Override config with command line arguments
    if args.model_name:
        config.model_name = args.model_name
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.audio_dir:
        config.audio_dir = args.audio_dir
    if args.transcript_file:
        config.transcript_file = args.transcript_file
    if args.output_dir:
        config.output_dir = args.output_dir
    
    print("=== Whisper Sanskrit Fine-tuning ===")
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    
    # Setup model and processor
    model, processor = setup_model_and_processor(config)
    
    # Prepare data
    train_dataset, val_dataset, test_dataset, test_data = prepare_data(config, processor)
    # if config.use_prosodic_features:
    #     print("\n[INFO] Performing a quick check on the first training sample...")
    #     first_sample = train_dataset[0]
    #     input_features = torch.tensor(first_sample['input_features'])

    #     print(f"Shape of input_features: {input_features.shape}")

    #     if input_features.shape[0] == 80:
    #         mel_part = input_features[:40, :]
    #         prosodic_part = input_features[40:, :]
    #         print(f"  - Mel part shape: {mel_part.shape}")
    #         print(f"  - Prosodic part shape: {prosodic_part.shape}")
            
    #         # Check if the first 10 rows of the prosodic part are identical (as they should be)
    #         is_repeated_correctly = torch.all(prosodic_part[0] == prosodic_part[9])
            
    #         print(f"  - Mean of mel part: {mel_part.mean():.4f}")
    #         print(f"  - Mean of prosodic part: {prosodic_part.mean():.4f}")
    #         print(f"  - Prosodic part appears correctly repeated (checked 1st block): {is_repeated_correctly}")
    #         print("[INFO] Check complete. The feature fusion seems to be working as expected.")
    #     else:
    #         print(f"[WARNING] Expected 80 feature channels, but got {input_features.shape[0]}.")
        
    #     print("\n[INFO] Exiting after check. Remove this block to proceed with training.")
    #     exit()
    # Train model
    training_monitor = TrainingMonitor()
    trainer, train_result = train_model(
        config, model, processor, train_dataset, val_dataset, training_monitor
    )
    
    # Evaluate model
    test_results, detailed_results = evaluate_model(
        config, trainer, test_dataset, test_data
    )
    
    # Save metadata
    save_training_metadata(config, train_result, test_results)
    
    print("\n=== Training Complete! ===")
    print(f"Model saved to: {config.output_dir}")
    print(f"Final WER: {test_results.get('eval_wer', 'N/A'):.4f}")
    
    # Plot training history if matplotlib is available
    try:
        training_monitor.plot_metrics(
            os.path.join(config.output_dir, "training_curves.png")
        )
    except:
        pass


if __name__ == "__main__":
    main()