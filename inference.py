"""
Inference utilities for using the fine-tuned Whisper model
"""
import torch
import librosa
from typing import List, Optional, Dict
from pathlib import Path
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class WhisperSanskritTranscriber:
    """Easy-to-use interface for transcribing Sanskrit audio"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the transcriber with a fine-tuned model.
        
        Args:
            model_path: Path to fine-tuned model directory
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load processor and model
        print(f"Loading model from {model_path}...")
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def transcribe(
        self, 
        audio_path: str, 
        language: str = "sa",
        task: str = "transcribe",
        return_timestamps: bool = False,
        **generate_kwargs
    ) -> Dict[str, any]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (WAV, M4A, MP3, etc.)
            language: Language code (default: 'sa' for Sanskrit)
            task: Task to perform ('transcribe' or 'translate')
            return_timestamps: Whether to return word timestamps
            **generate_kwargs: Additional arguments for model.generate()
            
        Returns:
            Dictionary with transcription and optional metadata
        """
        # Load audio
        audio_array, sampling_rate = self._load_audio(audio_path)
        
        # Process audio
        inputs = self.processor.feature_extractor(
            audio_array, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        ).to(self.device)
        
        # Set generation parameters
        gen_kwargs = {
            "language": language,
            "task": task,
            "return_timestamps": return_timestamps,
            **generate_kwargs
        }
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_features,
                **gen_kwargs
            )
        
        # Decode
        transcription = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        result = {
            "text": transcription.strip(),
            "language": language,
            "duration": len(audio_array) / sampling_rate
        }
        
        # Add timestamps if requested
        if return_timestamps:
            # Decode with timestamps
            output_with_timestamps = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
                output_offsets=True
            )[0]
            result["timestamps"] = output_with_timestamps
        
        return result
    
    def transcribe_batch(
        self, 
        audio_paths: List[str],
        batch_size: int = 8,
        **transcribe_kwargs
    ) -> List[Dict[str, any]]:
        """
        Transcribe multiple audio files in batches.
        
        Args:
            audio_paths: List of audio file paths
            batch_size: Batch size for processing
            **transcribe_kwargs: Arguments passed to transcribe()
            
        Returns:
            List of transcription results
        """
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(audio_paths), batch_size), desc="Transcribing"):
            batch_paths = audio_paths[i:i + batch_size]
            batch_audio = []
            
            # Load audio files
            for audio_path in batch_paths:
                try:
                    audio_array, sr = self._load_audio(audio_path)
                    batch_audio.append(audio_array)
                except Exception as e:
                    print(f"Error loading {audio_path}: {e}")
                    results.append({"text": "", "error": str(e)})
                    continue
            
            if not batch_audio:
                continue
            
            # Process batch
            inputs = self.processor.feature_extractor(
                batch_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Generate for batch
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_features,
                    language=transcribe_kwargs.get("language", "sa"),
                    task=transcribe_kwargs.get("task", "transcribe")
                )
            
            # Decode batch
            transcriptions = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            # Add to results
            for transcription, audio_array in zip(transcriptions, batch_audio):
                results.append({
                    "text": transcription.strip(),
                    "duration": len(audio_array) / 16000
                })
        
        return results
    
    def _load_audio(self, audio_path: str) -> tuple:
        """Load and preprocess audio file"""
        # Convert path to string if Path object
        audio_path = str(audio_path)
        
        # Load with librosa (handles multiple formats)
        audio_array, sampling_rate = librosa.load(
            audio_path, 
            sr=16000,  # Whisper uses 16kHz
            mono=True
        )
        
        # Normalize
        audio_array = librosa.util.normalize(audio_array)
        
        return audio_array, sampling_rate
    
    def benchmark(self, test_audio_path: str) -> Dict[str, float]:
        """
        Benchmark inference speed on a test audio file.
        
        Args:
            test_audio_path: Path to test audio file
            
        Returns:
            Dictionary with benchmark metrics
        """
        import time
        
        # Load audio
        audio_array, sr = self._load_audio(test_audio_path)
        audio_duration = len(audio_array) / sr
        
        # Warm up
        _ = self.transcribe(test_audio_path)
        
        # Time multiple runs
        n_runs = 5
        times = []
        
        for _ in range(n_runs):
            start = time.time()
            _ = self.transcribe(test_audio_path)
            times.append(time.time() - start)
        
        avg_time = sum(times) / n_runs
        rtf = avg_time / audio_duration  # Real-time factor
        
        return {
            "audio_duration": audio_duration,
            "avg_inference_time": avg_time,
            "real_time_factor": rtf,
            "speed_ratio": 1 / rtf if rtf > 0 else 0
        }


def evaluate_on_test_set(
    transcriber: WhisperSanskritTranscriber,
    test_data: List[Dict[str, str]],
    output_file: str = None
) -> Dict[str, float]:
    """
    Evaluate the model on a test set and compute metrics.
    
    Args:
        transcriber: WhisperSanskritTranscriber instance
        test_data: List of dicts with 'audio' and 'transcript' keys
        output_file: Optional file to save results
        
    Returns:
        Dictionary with evaluation metrics
    """
    from jiwer import wer, cer
    import json
    
    predictions = []
    references = []
    results = []
    
    print(f"Evaluating on {len(test_data)} test samples...")
    
    for item in tqdm(test_data, desc="Evaluating"):
        # Transcribe
        result = transcriber.transcribe(item['audio'])
        predicted_text = result['text']
        reference_text = item['transcript']
        
        predictions.append(predicted_text)
        references.append(reference_text)
        
        # Store detailed results
        results.append({
            'audio': item.get('original_filename', item['audio']),
            'reference': reference_text,
            'prediction': predicted_text,
            'duration': result['duration']
        })
    
    # Compute metrics
    word_error_rate = wer(references, predictions)
    char_error_rate = cer(references, predictions)
    
    metrics = {
        'word_error_rate': word_error_rate,
        'character_error_rate': char_error_rate,
        'num_samples': len(test_data)
    }
    
    # Save results if requested
    if output_file:
        output_data = {
            'metrics': metrics,
            'results': results
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    
    return metrics