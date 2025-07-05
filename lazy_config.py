"""
Configuration settings for Whisper Sanskrit fine-tuning with lazy loading
"""
from dataclasses import dataclass
from typing import Optional
import torch
import os


@dataclass
class LazyConfig:
    """Configuration for memory-efficient fine-tuning with lazy loading"""
    
    # Paths
    audio_dir: str = "./sanskrit_audio"
    transcript_file: str = "./transcript.txt"
    cache_dir: str = "./audio_cache"
    output_dir: str = "./whisper_sanskrit_finetuned"
    
    # Lazy loading specific
    lazy_cache_dir: str = "./lazy_cache"  # Cache processed audio arrays
    use_lazy_loading: bool = True
    prefetch_factor: int = 2
    num_workers: int = 4  # Parallel data loading workers
    persistent_workers: bool = True
    pin_memory: bool = True
    
    # Model settings
    model_name: str = "openai/whisper-small"  # Upgraded from tiny
    language: str = "sa"
    task: str = "transcribe"
    
    # Audio settings
    sample_rate: int = 16000
    max_duration: float = 30.0
    
    # Training settings - Optimized for better performance
    batch_size: int = 4  # Small batch for memory efficiency
    gradient_accumulation_steps: int = 16  # Effective batch size = 64
    learning_rate: float = 1e-4  # Higher learning rate
    warmup_steps: int = 2000
    max_steps: int = 15000  # More training steps
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 25
    
    # Memory optimization
    gradient_checkpointing: bool = True  # Trade compute for memory
    fp16: bool = torch.cuda.is_available()
    optim: str = "adamw_torch"  # Use "adamw_8bit" if bitsandbytes installed
    
    # Data split
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Advanced settings
    generation_max_length: int = 448
    save_total_limit: int = 3  # Save fewer checkpoints
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    label_smoothing_factor: float = 0.1
    
    # Remote training settings
    clear_cache_every_n_steps: int = 100
    memory_efficient_mode: bool = True
    
    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @property
    def effective_batch_size(self):
        return self.batch_size * self.gradient_accumulation_steps
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.lazy_cache_dir, exist_ok=True)
        
        # Adjust workers based on system
        if self.num_workers == -1:
            self.num_workers = min(4, os.cpu_count() or 1)
        
        # Disable workers if using Windows
        if os.name == 'nt':  # Windows
            self.num_workers = 0
            self.persistent_workers = False
            
        print(f"Lazy loading config initialized:")
        print(f"  - Model: {self.model_name}")
        print(f"  - Device: {self.device}")
        print(f"  - Effective batch size: {self.effective_batch_size}")
        print(f"  - Workers: {self.num_workers}")
        print(f"  - Memory efficient mode: {self.memory_efficient_mode}")