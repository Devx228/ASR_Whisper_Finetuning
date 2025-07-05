"""
Configuration settings for Whisper Sanskrit fine-tuning
"""
from dataclasses import dataclass,field
from typing import Optional
import torch
from datetime import datetime
import os

@dataclass
class Config:
    """Configuration for the fine-tuning process"""
    
    audio_dir: str = "./sanskrit_audio"  
    transcript_file: str = "./transcript.txt"
    cache_dir: str = "./audio_cache" 
    # output_dir: str = "./whisper_sanskrit_finetuned"
    output_dir: str = field(init=False)
    
    # Model settings
    model_name: str = "openai/whisper-base"  # tiny, base, small, medium, large-v2
    language: str = "sa" 
    task: str = "transcribe"
    
    # Audio settings
    sample_rate: int = 16000  # Whisper uses 16kHz
    max_duration: float = 30.0  # Maximum audio duration in seconds
    
    # Training settings
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    max_steps: int = 7000
    eval_steps: int = 250
    save_steps: int = 250
    logging_steps: int = 50
    
    # Data split
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Hardware settings
    fp16: bool = torch.cuda.is_available()
    gradient_checkpointing: bool = False
    
    generation_max_length: int = 448
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    
    use_prosodic_features: bool = True 
    prosodic_mu: float = 255.0 
    prosodic_dim_per_feature: int = 10
    prosodic_frame_shift: float = 0.01  # 10ms frame shift for prosodic extraction
    prosodic_frame_length: float = 0.025  # 25ms frame length for prosodic extraction

    def __post_init__(self):
        """Set dynamic output directory after initialization."""
        model_name_safe = self.model_name.split('/')[-1]
        prosody_tag = "with_prosody" if self.use_prosodic_features else "no_prosody"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = f"./experiments/{model_name_safe}_{prosody_tag}_{timestamp}"

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")