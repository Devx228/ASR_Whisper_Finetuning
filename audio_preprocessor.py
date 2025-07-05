"""
Audio preprocessing utilities for converting and processing audio files
"""
import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from pydub import AudioSegment
import librosa
import soundfile as sf
from tqdm import tqdm


class AudioPreprocessor:
    """Handles audio format conversion and preprocessing"""
    
    def __init__(self, cache_dir: str, sample_rate: int = 16000):
        self.cache_dir = Path(cache_dir)
        self.sample_rate = sample_rate
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def m4a_to_wav(self, m4a_path: str, force_convert: bool = False) -> Optional[str]:
        """
        Convert M4A file to WAV format with proper sample rate.
        
        Args:
            m4a_path: Path to M4A file
            force_convert: Force conversion even if cached file exists
            
        Returns:
            Path to converted WAV file or None if conversion fails
        """
        m4a_path = Path(m4a_path)
        wav_filename = m4a_path.stem + ".wav"
        wav_path = self.cache_dir / wav_filename
        
        # Check if already converted
        if wav_path.exists() and not force_convert:
            return str(wav_path)
        
        try:
            # Load M4A file using pydub
            audio = AudioSegment.from_file(str(m4a_path), format="m4a")
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Set sample rate
            audio = audio.set_frame_rate(self.sample_rate)
            
            # Export as WAV
            audio.export(str(wav_path), format="wav")
            
            # Verify the conversion
            y, sr = librosa.load(str(wav_path), sr=self.sample_rate)
            duration = len(y) / sr
            
            print(f"Converted: {m4a_path.name} -> {wav_filename} (Duration: {duration:.2f}s)")
            
            return str(wav_path)
            
        except Exception as e:
            print(f"Error converting {m4a_path}: {e}")
            return None
    
    def load_and_preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and preprocess for Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Handle M4A files
        if audio_path.endswith('.m4a'):
            wav_path = self.m4a_to_wav(audio_path)
            if wav_path:
                audio_path = wav_path
            else:
                raise ValueError(f"Failed to convert M4A file: {audio_path}")
        
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        return y, sr
    
    def convert_audio_batch(self, audio_files: list, audio_dir: str) -> list:
        """
        Convert a batch of audio files from M4A to WAV.
        
        Args:
            audio_files: List of audio filenames
            audio_dir: Directory containing the audio files
            
        Returns:
            List of successfully converted file paths
        """
        converted_files = []
        
        for audio_file in tqdm(audio_files, desc="Converting audio files"):
            audio_path = os.path.join(audio_dir, audio_file)
            
            if os.path.exists(audio_path):
                wav_path = self.m4a_to_wav(audio_path)
                if wav_path:
                    converted_files.append(wav_path)
            else:
                print(f"Warning: Audio file not found: {audio_path}")
        
        return converted_files
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        return len(y) / sr