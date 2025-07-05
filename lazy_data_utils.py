"""
Lazy loading data utilities for memory-efficient training
"""
import os
from typing import List, Dict, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import WhisperProcessor
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
import librosa
from pathlib import Path
import gc


class LazyAudioDataset(TorchDataset):
    """Lazy loading dataset that loads audio files on-demand"""
    
    def __init__(self, data_items: List[Dict], processor: WhisperProcessor, 
                 sample_rate: int = 16000, max_duration: float = 30.0,
                 cache_dir: str = None):
        self.data_items = data_items
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Pre-validate files exist
        self.valid_indices = self._validate_files()
        print(f"Lazy dataset initialized with {len(self.valid_indices)}/{len(data_items)} valid files")
    
    def _validate_files(self):
        """Check which files exist and are accessible"""
        valid_indices = []
        for idx, item in enumerate(self.data_items):
            if os.path.exists(item['audio']):
                valid_indices.append(idx)
            else:
                print(f"Warning: File not found: {item['audio']}")
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Load and process a single item on-demand"""
        actual_idx = self.valid_indices[idx]
        item = self.data_items[actual_idx]
        
        try:
            # Load audio file - THIS IS WHERE LAZY LOADING HAPPENS!
            audio_path = item['audio']
            
            # Use cache if available
            if self.cache_dir:
                cache_path = self.cache_dir / f"{Path(audio_path).stem}.npy"
                if cache_path.exists():
                    audio_array = np.load(cache_path)
                else:
                    audio_array, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                    np.save(cache_path, audio_array)
            else:
                audio_array, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Check duration and truncate if needed
            duration = len(audio_array) / self.sample_rate
            if duration > self.max_duration:
                max_samples = int(self.max_duration * self.sample_rate)
                audio_array = audio_array[:max_samples]
                duration = self.max_duration
            
            # Normalize
            audio_array = librosa.util.normalize(audio_array)
            
            # Process audio to log-mel spectrogram
            inputs = self.processor.feature_extractor(
                audio_array,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            
            # Tokenize transcript
            labels = self.processor.tokenizer(
                item['transcript'],
                padding=False,
                truncation=True,
                max_length=448,
                return_tensors="pt"
            )
            
            return {
                "input_features": inputs.input_features.squeeze(0),
                "labels": labels.input_ids.squeeze(0),
                "duration": duration,
                "original_filename": item.get('original_filename', audio_path)
            }
            
        except Exception as e:
            print(f"Error loading {item.get('audio', 'unknown')}: {e}")
            # Return a dummy sample on error
            return {
                "input_features": torch.zeros(80, 3000),
                "labels": torch.tensor([self.processor.tokenizer.pad_token_id]),
                "duration": 0.0,
                "original_filename": "error"
            }


# Keep your original functions but update them to work with lazy loading
def parse_transcript_file(transcript_path: str) -> List[Dict[str, str]]:
    """Parse tab-separated transcript file - same as before"""
    data = []
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                audio_filename = parts[0]
                transcript = '\t'.join(parts[1:])
                
                data.append({
                    'audio_filename': audio_filename,
                    'transcript': transcript
                })
            else:
                print(f"Warning: Skipping malformed line {line_num}: {line[:50]}...")
    
    print(f"Loaded {len(data)} transcript entries")
    return data


def prepare_data_splits(converted_data: List[Dict], config) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train, validation, and test sets - same as before"""
    # First split: train+val vs test
    train_val_data, test_data = train_test_split(
        converted_data, 
        test_size=config.test_split, 
        random_state=42
    )
    
    # Second split: train vs val
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=config.val_split / (1 - config.test_split),
        random_state=42
    )
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    return train_data, val_data, test_data


def combine_transcript_with_audio(transcript_data: List[Dict], 
                                 converted_paths: List[str]) -> List[Dict]:
    """Combine transcript data with converted audio paths - same as before"""
    # Create a mapping of original filename to wav path
    path_mapping = {}
    for wav_path in converted_paths:
        wav_name = os.path.basename(wav_path)
        original_name = wav_name.replace('.wav', '.m4a')
        path_mapping[original_name] = wav_path
    
    # Combine data
    combined_data = []
    for item in transcript_data:
        audio_filename = item['audio_filename']
        if audio_filename in path_mapping:
            combined_data.append({
                'audio': path_mapping[audio_filename],
                'transcript': item['transcript'],
                'original_filename': audio_filename
            })
    
    print(f"Successfully matched {len(combined_data)} audio files with transcripts")
    return combined_data


class MemoryEfficientDataLoader:
    """Wrapper for memory-efficient data loading"""
    
    @staticmethod
    def create_lazy_dataloader(
        data: List[Dict],
        processor: WhisperProcessor,
        config,
        shuffle: bool = True,
        cache_dir: str = None
    ) -> DataLoader:
        """Create a memory-efficient dataloader"""
        
        dataset = LazyAudioDataset(
            data_items=data,
            processor=processor,
            sample_rate=config.sample_rate,
            max_duration=config.max_duration,
            cache_dir=cache_dir
        )
        
        # Determine number of workers based on system
        num_workers = min(4, os.cpu_count() or 1)
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=False
        )
        
        return dataloader