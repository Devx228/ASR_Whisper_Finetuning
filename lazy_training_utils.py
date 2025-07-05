"""
Memory-efficient training utilities with lazy loading support
"""
import numpy as np
import torch
import evaluate
from transformers import TrainerCallback, WhisperProcessor, Trainer
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import gc
import psutil
import GPUtil


class MemoryMonitorCallback(TrainerCallback):
    """Monitor memory usage during training"""
    
    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_n_steps == 0:
            # CPU Memory
            cpu_percent = psutil.virtual_memory().percent
            cpu_used_gb = psutil.virtual_memory().used / (1024**3)
            
            log_str = f"Step {state.global_step} - Memory: CPU {cpu_percent:.1f}% ({cpu_used_gb:.1f}GB)"
            
            # GPU Memory
            if self.gpu_available and self.gpu:
                gpu_memory_mb = self.gpu.memoryUsed
                gpu_percent = self.gpu.memoryUtil * 100
                log_str += f", GPU {gpu_memory_mb}MB ({gpu_percent:.1f}%)"
                
                # Clear cache if memory usage is high
                if gpu_percent > 85:
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"  ⚠️ High GPU memory usage - cleared cache")
            
            print(log_str)


class WhisperTrainingCallback(TrainerCallback):
    """Custom callback for monitoring training progress"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.verbose:
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
        
        # Normalize text
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # Compute WER
        wer = self.wer_metric.compute(
            predictions=decoded_preds, 
            references=decoded_labels
        )
        
        return {"wer": wer}


@dataclass
class LazyDataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for lazy loading that handles variable-length sequences
    """
    processor: Any
    decoder_start_token_id: int
    padding: bool = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Filter out error samples
        valid_features = [f for f in features if f.get("duration", 0) > 0]
        
        if not valid_features:
            # Return dummy batch if all samples failed
            return {
                "input_features": torch.zeros(1, 80, 3000),
                "labels": torch.tensor([[-100]]),
                "attention_mask": torch.zeros(1, 1)
            }
        
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in valid_features]
        label_features = [{"input_ids": feature["labels"]} for feature in valid_features]
        
        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt"
        )
        
        # Replace padding with -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        # Remove decoder start token if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        return batch


def create_data_collator(processor: WhisperProcessor, model):
    """Create lazy-loading compatible data collator"""
    return LazyDataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )


class MemoryEfficientTrainer(Trainer):
    """Custom trainer with aggressive memory management for lazy loading"""
    
    def __init__(self, *args, clear_cache_every_n_steps=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_cache_every_n_steps = clear_cache_every_n_steps
    
    def training_step(self, model, inputs):
        """Override to add memory cleanup"""
        loss = super().training_step(model, inputs)
        
        # Periodic memory cleanup
        if self.state.global_step % self.clear_cache_every_n_steps == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return loss
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                       ignore_keys=None, metric_key_prefix="eval"):
        """Override to handle lazy loading in evaluation"""
        # Force garbage collection before evaluation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return super().evaluation_loop(
            dataloader, description, prediction_loss_only,
            ignore_keys, metric_key_prefix
        )


class TrainingMonitor:
    """Monitor and log training statistics"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_wer': [],
            'learning_rate': [],
            'memory_cpu': [],
            'memory_gpu': []
        }
        self.gpu_available = torch.cuda.is_available()
    
    def update(self, logs):
        """Update training history"""
        for key in ['train_loss', 'eval_loss', 'eval_wer', 'learning_rate']:
            if key in logs:
                self.history[key].append(logs[key])
        
        # Track memory usage
        self.history['memory_cpu'].append(psutil.virtual_memory().percent)
        if self.gpu_available and GPUtil.getGPUs():
            self.history['memory_gpu'].append(GPUtil.getGPUs()[0].memoryUtil * 100)
    
    def plot_metrics(self, save_path: str = None):
        """Plot training metrics including memory usage"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Training Metrics with Memory Usage', fontsize=16)
            
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
                axes[0, 2].plot(self.history['eval_wer'], 'g')
                axes[0, 2].set_title('Word Error Rate')
                axes[0, 2].set_xlabel('Eval Steps')
                axes[0, 2].set_ylabel('WER')
            
            # Plot learning rate
            if self.history['learning_rate']:
                axes[1, 0].plot(self.history['learning_rate'], 'orange')
                axes[1, 0].set_title('Learning Rate')
                axes[1, 0].set_xlabel('Steps')
                axes[1, 0].set_ylabel('LR')
            
            # Plot CPU memory
            if self.history['memory_cpu']:
                axes[1, 1].plot(self.history['memory_cpu'], 'purple')
                axes[1, 1].set_title('CPU Memory Usage')
                axes[1, 1].set_xlabel('Steps')
                axes[1, 1].set_ylabel('Percentage')
                axes[1, 1].set_ylim(0, 100)
            
            # Plot GPU memory
            if self.history['memory_gpu']:
                axes[1, 2].plot(self.history['memory_gpu'], 'cyan')
                axes[1, 2].set_title('GPU Memory Usage')
                axes[1, 2].set_xlabel('Steps')
                axes[1, 2].set_ylabel('Percentage')
                axes[1, 2].set_ylim(0, 100)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            plt.show()
            
        except ImportError:
            print("Matplotlib not installed. Cannot plot metrics.")