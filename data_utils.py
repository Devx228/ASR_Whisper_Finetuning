# """
# Data processing utilities for transcript parsing and dataset creation
# """
# import os
# from typing import List, Dict, Tuple
# import numpy as np
# from sklearn.model_selection import train_test_split
# from datasets import Dataset
# from transformers import WhisperProcessor
# import torch


# def parse_transcript_file(transcript_path: str) -> List[Dict[str, str]]:
#     """
#     Parse tab-separated transcript file.
#     Expected format: audio_filename.m4a\\ttranscript_text
    
#     Args:
#         transcript_path: Path to transcript file
        
#     Returns:
#         List of dictionaries with 'audio_filename' and 'transcript' keys
#     """
#     data = []
    
#     with open(transcript_path, 'r', encoding='utf-8') as f:
#         for line_num, line in enumerate(f, 1):
#             line = line.strip()
#             if not line:
#                 continue
            
#             parts = line.split('\t')
#             if len(parts) >= 2:
#                 audio_filename = parts[0]
#                 transcript = '\t'.join(parts[1:])  # Handle tabs in transcript
                
#                 data.append({
#                     'audio_filename': audio_filename,
#                     'transcript': transcript
#                 })
#             else:
#                 print(f"Warning: Skipping malformed line {line_num}: {line[:50]}...")
    
#     print(f"Loaded {len(data)} transcript entries")
#     return data


# def prepare_data_splits(converted_data: List[Dict], config) -> Tuple[List[Dict], List[Dict], List[Dict]]:
#     """
#     Split data into train, validation, and test sets.
    
#     Args:
#         converted_data: List of data dictionaries
#         config: Configuration object with split ratios
        
#     Returns:
#         Tuple of (train_data, val_data, test_data)
#     """
#     # First split: train+val vs test
#     train_val_data, test_data = train_test_split(
#         converted_data, 
#         test_size=config.test_split, 
#         random_state=42
#     )
    
#     # Second split: train vs val
#     train_data, val_data = train_test_split(
#         train_val_data,
#         test_size=config.val_split / (1 - config.test_split),
#         random_state=42
#     )
    
#     print(f"Dataset splits:")
#     print(f"  Train: {len(train_data)} samples")
#     print(f"  Validation: {len(val_data)} samples")
#     print(f"  Test: {len(test_data)} samples")
    
#     return train_data, val_data, test_data


# class DatasetProcessor:
#     """Handles dataset creation and processing for Whisper"""
    
#     def __init__(self, processor: WhisperProcessor, sample_rate: int = 16000):
#         self.processor = processor
#         self.sample_rate = sample_rate
    
#     def prepare_dataset(self, data: List[Dict], audio_preprocessor, 
#                        max_duration: float = 30.0) -> Dataset:
#         """
#         Prepare dataset for Whisper fine-tuning.
        
#         Args:
#             data: List of data dictionaries with 'audio' and 'transcript' keys
#             audio_preprocessor: AudioPreprocessor instance
#             max_duration: Maximum audio duration in seconds
            
#         Returns:
#             Processed HuggingFace Dataset
#         """
#         def preprocess_function(examples):
#             # Load audio
#             audio_arrays = []
#             transcripts = []
            
#             for audio_path, transcript in zip(examples['audio'], examples['transcript']):
#                 try:
#                     # Load audio
#                     y, sr = audio_preprocessor.load_and_preprocess_audio(audio_path)
                    
#                     # Check duration
#                     duration = len(y) / sr
#                     if duration <= max_duration:
#                         audio_arrays.append(y)
#                         transcripts.append(transcript)
#                 except Exception as e:
#                     print(f"Error processing {audio_path}: {e}")
#                     continue
            
#             if not audio_arrays:
#                 return {
#                     "input_features": [],
#                     "labels": []
#                 }
            
#             # Process audio to log-mel spectrograms
#             inputs = self.processor.feature_extractor(
#                 audio_arrays, 
#                 sampling_rate=self.sample_rate,
#                 return_tensors="pt"
#             )
            
#             # Tokenize transcripts
#             labels = self.processor.tokenizer(
#                 transcripts,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt"
#             )
            
#             # Replace padding token id with -100 (ignored by loss)
#             labels["input_ids"] = labels["input_ids"].masked_fill(
#                 labels["attention_mask"].eq(0), -100
#             )
            
#             return {
#                 "input_features": inputs.input_features,
#                 "labels": labels.input_ids
#             }
        
#         # Create dataset
#         dataset = Dataset.from_list(data)
        
#         # Process in batches
#         dataset = dataset.map(
#             preprocess_function,
#             batched=True,
#             batch_size=8,
#             remove_columns=dataset.column_names,
#             desc="Processing audio files"
#         )
        
#         # Filter out empty examples
#         dataset = dataset.filter(lambda x: len(x['input_features']) > 0)
        
#         return dataset


# def combine_transcript_with_audio(transcript_data: List[Dict], 
#                                  converted_paths: List[str]) -> List[Dict]:
#     """
#     Combine transcript data with converted audio paths.
    
#     Args:
#         transcript_data: Original transcript data
#         converted_paths: List of converted WAV file paths
        
#     Returns:
#         Combined data list
#     """
#     # Create a mapping of original filename to wav path
#     path_mapping = {}
#     for wav_path in converted_paths:
#         # Extract original filename from wav path
#         wav_name = os.path.basename(wav_path)
#         original_name = wav_name.replace('.wav', '.m4a')
#         path_mapping[original_name] = wav_path
    
#     # Combine data
#     combined_data = []
#     for item in transcript_data:
#         audio_filename = item['audio_filename']
#         if audio_filename in path_mapping:
#             combined_data.append({
#                 'audio': path_mapping[audio_filename],
#                 'transcript': item['transcript'],
#                 'original_filename': audio_filename
#             })
    
#     print(f"Successfully matched {len(combined_data)} audio files with transcripts")
#     return combined_data



"""
Data processing utilities for transcript parsing and dataset creation with prosodic features
"""
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import WhisperProcessor
import torch
import torch.nn as nn
import librosa


def parse_transcript_file(transcript_path: str) -> List[Dict[str, str]]:
    """
    Parse tab-separated transcript file.
    Expected format: audio_filename.m4a\\ttranscript_text
    
    Args:
        transcript_path: Path to transcript file
        
    Returns:
        List of dictionaries with 'audio_filename' and 'transcript' keys
    """
    data = []
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                audio_filename = parts[0]
                transcript = '\t'.join(parts[1:])  # Handle tabs in transcript
                
                data.append({
                    'audio_filename': audio_filename,
                    'transcript': transcript
                })
            else:
                print(f"Warning: Skipping malformed line {line_num}: {line[:50]}...")
    
    print(f"Loaded {len(data)} transcript entries")
    return data


def prepare_data_splits(converted_data: List[Dict], config) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        converted_data: List of data dictionaries
        config: Configuration object with split ratios
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
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


class ProsodicFeatureExtractor:
    """Extract prosodic features (pitch, energy, delta, delta-delta) from audio."""
    
    def __init__(self, sample_rate: int = 16000, frame_shift: float = 0.01, frame_length: float = 0.025):
        self.sample_rate = sample_rate
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.hop_length = int(frame_shift * sample_rate)
        self.n_fft = int(frame_length * sample_rate)
        
    def extract_pitch(self, audio: np.ndarray) -> np.ndarray:
        """Extract pitch (F0) using librosa's piptrack."""
        pitches, magnitudes = librosa.piptrack(
            y=audio, 
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            threshold=0.1
        )
        # Get the pitch with maximum magnitude at each frame
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_values.append(pitch if pitch > 0 else 0)
        return np.array(pitch_values)
    
    def extract_energy(self, audio: np.ndarray) -> np.ndarray:
        """Extract energy (RMS) from audio."""
        energy = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )[0]
        return energy
    
    def compute_deltas(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute delta and delta-delta features."""
        delta = librosa.feature.delta(features)
        delta_delta = librosa.feature.delta(features, order=2)
        return delta, delta_delta
    
    def extract_prosodic_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract all prosodic features from audio."""
        # Extract base features
        pitch = self.extract_pitch(audio)
        energy = self.extract_energy(audio)
        
        # Compute deltas
        pitch_delta, pitch_delta_delta = self.compute_deltas(pitch)
        energy_delta, energy_delta_delta = self.compute_deltas(energy)
        
        # Stack features: [pitch, energy, delta, delta-delta]
        # Using pitch_delta and energy_delta as representative deltas
        features = np.stack([pitch, energy, pitch_delta, energy_delta_delta], axis=0)
        
        return features


class ProsodicFeatureFusion:
    """Fuse prosodic features with Whisper's mel spectrogram."""
    
    def __init__(self, mu: float = 255.0, prosodic_dim_per_feature: int = 10, use_prosodic_features: bool = True):
        self.mu = mu
        self.prosodic_dim_per_feature = prosodic_dim_per_feature
        self.use_prosodic_features = use_prosodic_features
        self.prosodic_extractor = ProsodicFeatureExtractor()
        
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [-1, 1] range."""
        # Handle zero variance case
        if features.std() == 0:
            return np.zeros_like(features)
        normalized = (features - features.mean()) / (features.std() + 1e-8)
        # Clip to [-3, 3] sigma and rescale to [-1, 1]
        normalized = np.clip(normalized, -3, 3) / 3
        return normalized
    
    def mu_law_companding(self, x: np.ndarray) -> np.ndarray:
        """Apply μ-law companding to features."""
        return np.sign(x) * (np.log(1 + self.mu * np.abs(x)) / np.log(1 + self.mu))
    
    def expand_prosodic_features(self, features: np.ndarray, target_frames: int) -> np.ndarray:
        """Expand prosodic features to match mel spectrogram frames and dimensions."""
        # features shape: (4, time)
        # Interpolate to match target frames if needed
        if features.shape[1] != target_frames:
            features_expanded = np.zeros((4, target_frames))
            for i in range(4):
                features_expanded[i] = np.interp(
                    np.linspace(0, features.shape[1]-1, target_frames),
                    np.arange(features.shape[1]),
                    features[i]
                )
            features = features_expanded
        
        # Expand each feature to prosodic_dim_per_feature dimensions
        # This creates a 4 x 10 = 40 dimensional representation
        expanded = np.zeros((4 * self.prosodic_dim_per_feature, target_frames))
        for i in range(4):
            # Repeat each prosodic feature value across 10 bins
            start_idx = i * self.prosodic_dim_per_feature
            end_idx = start_idx + self.prosodic_dim_per_feature
            expanded[start_idx:end_idx, :] = np.repeat(features[i:i+1, :], self.prosodic_dim_per_feature, axis=0)
        
        return expanded
    
    def fuse_with_mel_spectrogram(self, audio: np.ndarray, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Fuse prosodic features with Whisper's mel spectrogram.
        
        Args:
            audio: Raw audio waveform (numpy array)
            mel_spectrogram: Whisper's mel spectrogram (80, time)
            
        Returns:
            Fused features: 
            - If use_prosodic_features=True: (80, time) where first 40 are truncated mel, last 40 are prosodic
            - If use_prosodic_features=False: (80, time) original mel spectrogram unchanged
        """
        # If not using prosodic features, return original mel spectrogram
        if not self.use_prosodic_features:
            return mel_spectrogram
        
        # Extract prosodic features
        prosodic_features = self.prosodic_extractor.extract_prosodic_features(audio)
        
        # Normalize each prosodic feature independently
        for i in range(4):
            prosodic_features[i] = self.normalize_features(prosodic_features[i])
        
        # Apply μ-law companding
        prosodic_features = self.mu_law_companding(prosodic_features)
        
        # Get target number of frames from mel spectrogram
        target_frames = mel_spectrogram.shape[-1]
        
        # Expand prosodic features to 40 dimensions (4 features × 10 dims each)
        prosodic_expanded = self.expand_prosodic_features(prosodic_features, target_frames)
        
        # Convert to torch tensor
        prosodic_tensor = torch.from_numpy(prosodic_expanded).float()
        
        # Ensure same device as mel spectrogram
        prosodic_tensor = prosodic_tensor.to(mel_spectrogram.device)
        
        # Truncate mel spectrogram to first 40 bins (lower frequencies)
        mel_truncated = mel_spectrogram[:40, :]
        
        # Stack: [mel_truncated (40); prosodic (40)] = 80 dims
        fused_features = torch.cat([mel_truncated, prosodic_tensor], dim=0)
        
        return fused_features


class DatasetProcessor:
    """Handles dataset creation and processing for Whisper with optional prosodic features"""
    
    def __init__(self, processor: WhisperProcessor, sample_rate: int = 16000, 
                 use_prosodic_features: bool = False, config=None):
        self.processor = processor
        self.sample_rate = sample_rate
        self.use_prosodic_features = use_prosodic_features
        self.config = config
        
        # Initialize prosodic fusion with config parameters if available
        if config and hasattr(config, 'prosodic_mu'):
            self.prosodic_fusion = ProsodicFeatureFusion(
                use_prosodic_features=use_prosodic_features,
                mu=config.prosodic_mu,
                prosodic_dim_per_feature=config.prosodic_dim_per_feature
            )
        else:
            self.prosodic_fusion = ProsodicFeatureFusion(use_prosodic_features=use_prosodic_features)
    
    def prepare_dataset(self, data: List[Dict], audio_preprocessor, 
                       max_duration: float = 30.0) -> Dataset:
        """
        Prepare dataset for Whisper fine-tuning with optional prosodic features.
        
        Args:
            data: List of data dictionaries with 'audio' and 'transcript' keys
            audio_preprocessor: AudioPreprocessor instance
            max_duration: Maximum audio duration in seconds
            
        Returns:
            Processed HuggingFace Dataset
        """
        def preprocess_function(examples):
            # Load audio
            audio_arrays = []
            transcripts = []
            
            for audio_path, transcript in zip(examples['audio'], examples['transcript']):
                try:
                    # Load audio
                    y, sr = audio_preprocessor.load_and_preprocess_audio(audio_path)
                    
                    # Check duration
                    duration = len(y) / sr
                    if duration <= max_duration:
                        audio_arrays.append(y)
                        transcripts.append(transcript)
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
                    continue
            
            if not audio_arrays:
                return {
                    "input_features": [],
                    "labels": []
                }
            
            # Process audio to log-mel spectrograms using WhisperProcessor
            if self.use_prosodic_features:
                # First get standard mel spectrograms
                inputs = self.processor.feature_extractor(
                    audio_arrays, 
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                )
                
                # Apply prosodic feature fusion
                fused_features_list = []
                for i, audio in enumerate(audio_arrays):
                    mel_spectrogram = inputs.input_features[i]
                    fused_features = self.prosodic_fusion.fuse_with_mel_spectrogram(audio, mel_spectrogram)
                    fused_features_list.append(fused_features)
                
                # Stack all fused features
                input_features = torch.stack(fused_features_list)
            else:
                # Standard processing without prosodic features
                inputs = self.processor.feature_extractor(
                    audio_arrays, 
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                )
                input_features = inputs.input_features
            
            # Tokenize transcripts
            labels = self.processor.tokenizer(
                transcripts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Replace padding token id with -100 (ignored by loss)
            labels["input_ids"] = labels["input_ids"].masked_fill(
                labels["attention_mask"].eq(0), -100
            )
            
            return {
                "input_features": input_features,
                "labels": labels.input_ids
            }
        
        # Create dataset
        dataset = Dataset.from_list(data)
        
        # Process in batches
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=8,
            remove_columns=dataset.column_names,
            desc=f"Processing audio files {'with' if self.use_prosodic_features else 'without'} prosodic features"
        )
        
        # Filter out empty examples
        dataset = dataset.filter(lambda x: len(x['input_features']) > 0)
        
        return dataset


def combine_transcript_with_audio(transcript_data: List[Dict], 
                                 converted_paths: List[str]) -> List[Dict]:
    """
    Combine transcript data with converted audio paths.
    
    Args:
        transcript_data: Original transcript data
        converted_paths: List of converted WAV file paths
        
    Returns:
        Combined data list
    """
    # Create a mapping of original filename to wav path
    path_mapping = {}
    for wav_path in converted_paths:
        # Extract original filename from wav path
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