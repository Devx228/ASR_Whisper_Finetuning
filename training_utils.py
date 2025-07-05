"""
Training utilities including metrics, callbacks, and data collator
"""
import numpy as np
import torch
import evaluate
from transformers import TrainerCallback, WhisperProcessor
from dataclasses import dataclass
from typing import Any, Dict, List, Union


class WhisperTrainingCallback(TrainerCallback):
    """Custom callback for monitoring training progress"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.verbose:
            # Log training progress
            log_items = []
            if "loss" in logs:
                log_items.append(f"Loss: {logs['loss']:.4f}")
            if "eval_wer" in logs:
                log_items.append(f"WER: {logs['eval_wer']:.4f}")
            if "learning_rate" in logs:
                log_items.append(f"LR: {logs['learning_rate']:.2e}")
            
            if log_items:
                print(f"Step {state.global_step}: {' | '.join(log_items)}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.verbose:
            print(f"\nCompleted epoch {state.epoch}")
    
    def on_evaluation(self, args, state, control, metrics=None, **kwargs):
        if metrics and self.verbose:
            print("\nEvaluation Results:")
            for key, value in metrics.items():
                if key != "epoch":
                    print(f"  {key}: {value:.4f}")


class MetricsComputer:
    """Handles metric computation for evaluation"""
    
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")
    
    def __call__(self, eval_pred):
        """Compute Word Error Rate (WER) for evaluation"""
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        
        # Decode predictions
        decoded_preds = self.processor.batch_decode(
            predictions, 
            skip_special_tokens=True
        )
        
        # Replace -100 in labels
        labels = np.where(
            labels != -100, 
            labels, 
            self.processor.tokenizer.pad_token_id
        )
        decoded_labels = self.processor.batch_decode(
            labels, 
            skip_special_tokens=True
        )
        
        # Normalize text (optional - add more normalization as needed)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # Compute WER
        wer = self.wer_metric.compute(
            predictions=decoded_preds, 
            references=decoded_labels
        )
        cer = self.cer_metric.compute(
            predictions=decoded_preds, 
            references=decoded_labels
        )
        
        return {"wer": wer, "cer": cer}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Compatible with Whisper fine-tuning.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def create_data_collator(processor: WhisperProcessor, model):
    """Create data collator for Whisper Seq2Seq training"""
    return DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )


class TrainingMonitor:
    """Monitor and log training statistics"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_wer': [],
            'learning_rate': []
        }
    
    def update(self, logs):
        """Update training history"""
        for key in self.history:
            if key in logs:
                self.history[key].append(logs[key])
    
    def plot_metrics(self, save_path: str = None):
        """Plot training metrics (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Training Metrics', fontsize=16)
            
            # Plot train loss
            if self.history['train_loss']:
                axes[0, 0].plot(self.history['train_loss'])
                axes[0, 0].set_title('Training Loss')
                axes[0, 0].set_xlabel('Steps')
                axes[0, 0].set_ylabel('Loss')
            
            # Plot eval loss
            if self.history['eval_loss']:
                axes[0, 1].plot(self.history['eval_loss'], 'r')
                axes[0, 1].set_title('Validation Loss')
                axes[0, 1].set_xlabel('Eval Steps')
                axes[0, 1].set_ylabel('Loss')
            
            # Plot WER
            if self.history['eval_wer']:
                axes[1, 0].plot(self.history['eval_wer'], 'g')
                axes[1, 0].set_title('Word Error Rate')
                axes[1, 0].set_xlabel('Eval Steps')
                axes[1, 0].set_ylabel('WER')
            
            # Plot learning rate
            if self.history['learning_rate']:
                axes[1, 1].plot(self.history['learning_rate'], 'orange')
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Steps')
                axes[1, 1].set_ylabel('LR')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            plt.show()
            
        except ImportError:
            print("Matplotlib not installed. Cannot plot metrics.")