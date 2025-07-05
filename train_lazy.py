import os
import json
import argparse
from pathlib import Path
import gc
import torch

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments
)

from lazy_config import LazyConfig
from audio_preprocessor import AudioPreprocessor
from lazy_data_utils import (
    parse_transcript_file,
    prepare_data_splits,
    combine_transcript_with_audio,
    LazyAudioDataset,
    MemoryEfficientDataLoader
)
from lazy_training_utils import (
    WhisperTrainingCallback,
    MemoryMonitorCallback,
    MetricsComputer,
    create_data_collator,
    TrainingMonitor,
    MemoryEfficientTrainer
)
from inference import WhisperSanskritTranscriber, evaluate_on_test_set


def setup_model_and_processor(config: LazyConfig):
    """Setup model with memory optimization"""
    print(f"Loading Whisper model: {config.model_name}")
    
    # Load processor and model
    processor = WhisperProcessor.from_pretrained(config.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
    
    # Configure for Sanskrit
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # Disable KV cache for training
    
    # Set language and task
    model.generation_config.language = config.language
    model.generation_config.task = config.task
    
    # Enable gradient checkpointing if requested
    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Move to device
    model.to(config.device)
    
    print(f"Model loaded. Parameters: {model.num_parameters():,}")
    print(f"Device: {config.device}")
    
    # Clear cache after model loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model, processor


def prepare_lazy_data(config: LazyConfig, processor: WhisperProcessor):
    """Prepare data with lazy loading - no audio files loaded yet!"""
    print("\n=== Lazy Data Preparation ===")
    
    # Parse transcripts (just text, no audio loading)
    print("\n1. Parsing transcript file...")
    transcript_data = parse_transcript_file(config.transcript_file)
    
    # Initialize audio preprocessor
    print("\n2. Setting up audio preprocessing...")
    audio_preprocessor = AudioPreprocessor(config.cache_dir, config.sample_rate)
    
    # Convert audio files if needed
    audio_files = [item['audio_filename'] for item in transcript_data]
    converted_paths = audio_preprocessor.convert_audio_batch(audio_files, config.audio_dir)
    
    # Combine transcript with paths (still no audio loading)
    print("\n3. Combining metadata...")
    combined_data = combine_transcript_with_audio(transcript_data, converted_paths)
    
    # Split data
    print("\n4. Creating data splits...")
    train_data, val_data, test_data = prepare_data_splits(combined_data, config)
    
    # Create lazy datasets - AUDIO NOT LOADED YET!
    print("\n5. Creating lazy datasets...")
    
    train_dataset = LazyAudioDataset(
        train_data, 
        processor,
        config.sample_rate,
        config.max_duration,
        config.lazy_cache_dir
    )
    
    val_dataset = LazyAudioDataset(
        val_data,
        processor,
        config.sample_rate,
        config.max_duration,
        config.lazy_cache_dir
    )
    
    test_dataset = LazyAudioDataset(
        test_data,
        processor,
        config.sample_rate,
        config.max_duration,
        config.lazy_cache_dir
    )
    
    print(f"\nLazy datasets created (no audio loaded yet):")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples") 
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset, test_data


def train_model_lazy(
    config: LazyConfig,
    model,
    processor,
    train_dataset,
    val_dataset,
    training_monitor: TrainingMonitor = None
):
    """Train with lazy loading and memory optimization"""
    print("\n=== Lazy Training Configuration ===")
    
    # Create training arguments with memory optimization
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
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        push_to_hub=False,
        predict_with_generate=True,
        generation_max_length=config.generation_max_length,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        dataloader_drop_last=False,
        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=config.pin_memory,
        dataloader_persistent_workers=config.persistent_workers,
        # Optimization settings
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        label_smoothing_factor=config.label_smoothing_factor,
    )
    
    # Create data collator for lazy loading
    data_collator = create_data_collator(processor, model)
    
    # Create metrics computer
    compute_metrics = MetricsComputer(processor)
    
    # Create callbacks
    callbacks = [
        WhisperTrainingCallback(verbose=True),
        MemoryMonitorCallback(log_every_n_steps=config.logging_steps)
    ]
    
    # Create memory-efficient trainer
    trainer = MemoryEfficientTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=callbacks,
        clear_cache_every_n_steps=config.clear_cache_every_n_steps,
    )
    
    print(f"\nTraining configuration:")
    print(f"  Total steps: {config.max_steps}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Effective batch size: {config.effective_batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Workers: {config.num_workers}")
    print(f"  Lazy loading: {config.use_lazy_loading}")
    print(f"  Memory efficient mode: {config.memory_efficient_mode}")
    
    # Start training
    print("\n=== Starting Lazy Training ===")
    print("Audio files will be loaded on-demand during training...")
    
    # Clear memory before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    train_result = trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)
    
    return trainer, train_result


def evaluate_model(config: LazyConfig, trainer, test_dataset, test_data):
    """Evaluate the model on test set with lazy loading"""
    print("\n=== Lazy Evaluation ===")
    
    # Clear memory before evaluation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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


def save_training_metadata(config: LazyConfig, train_result, test_results):
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
        "effective_batch_size": config.effective_batch_size,
        "test_wer": test_results.get("eval_wer", None),
        "device": str(config.device),
        "lazy_loading": config.use_lazy_loading,
        "num_workers": config.num_workers,
    }
    
    metadata_path = os.path.join(config.output_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save config
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\nMetadata saved to {metadata_path}")
    print(f"Config saved to {config_path}")


def main():
    """Main training pipeline with lazy loading"""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for Sanskrit with Lazy Loading")
    parser.add_argument("--config", type=str, help="Path to custom config file")
    parser.add_argument("--model_name", type=str, help="Whisper model to use")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--max_steps", type=int, help="Maximum training steps")
    parser.add_argument("--audio_dir", type=str, help="Directory containing audio files")
    parser.add_argument("--transcript_file", type=str, help="Path to transcript file")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--num_workers", type=int, help="Number of data loading workers")
    parser.add_argument("--no_lazy_loading", action="store_true", help="Disable lazy loading")
    
    args = parser.parse_args()
    
    # Load config
    config = LazyConfig()
    
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
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.no_lazy_loading:
        config.use_lazy_loading = False
    
    print("=== Whisper Sanskrit Fine-tuning with Lazy Loading ===")
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Lazy Loading: {config.use_lazy_loading}")
    
    # Setup model and processor
    model, processor = setup_model_and_processor(config)
    
    # Prepare data with lazy loading
    train_dataset, val_dataset, test_dataset, test_data = prepare_lazy_data(config, processor)
    
    # Train model with memory optimization
    training_monitor = TrainingMonitor()
    trainer, train_result = train_model_lazy(
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
    
    # Final memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()