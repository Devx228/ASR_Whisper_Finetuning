import os
import sys
import glob
import json
from datetime import datetime
from inference import WhisperSanskritTranscriber

def find_audio_files(path):
    """Find all audio files in a path (file or folder)"""
    audio_extensions = ['.m4a', '.wav', '.mp3', '.flac', '.ogg', '.aac']
    
    if os.path.isfile(path):
        # Single file
        if any(path.lower().endswith(ext) for ext in audio_extensions):
            return [path]
        else:
            print(f"❌ '{path}' is not a supported audio file")
            return []
    
    elif os.path.isdir(path):
        audio_files = []
        for ext in audio_extensions:
            pattern = os.path.join(path, f"*{ext}")
            audio_files.extend(glob.glob(pattern))
            # Also check for uppercase extensions
            pattern = os.path.join(path, f"*{ext.upper()}")
            audio_files.extend(glob.glob(pattern))
        
        if not audio_files:
            print(f"❌ No audio files found in '{path}'")
            return []
        
        audio_files.sort()
        print(f"📁 Found {len(audio_files)} audio files in folder")
        return audio_files
    
    else:
        print(f"❌ Path '{path}' does not exist!")
        return []

def transcribe_single_file(transcriber, audio_path, show_details=True):
    """Transcribe a single audio file"""
    try:
        if show_details:
            print(f"🎧 Transcribing: {os.path.basename(audio_path)}")
        
        result = transcriber.transcribe(audio_path)
        
        if show_details:
            print(f"   📝 Result: '{result['text'][:50]}{'...' if len(result['text']) > 50 else ''}'")
            print(f"   ⏱️  Duration: {result['duration']:.1f}s")
        
        # Save transcript to file
        base_name = os.path.splitext(audio_path)[0]
        output_file = f"{base_name}_transcript.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        
        return {
            'audio_file': os.path.basename(audio_path),
            'transcript': result['text'],
            'duration': result['duration'],
            'output_file': output_file,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            'audio_file': os.path.basename(audio_path),
            'error': str(e),
            'status': 'failed'
        }

def transcribe_multiple_files(audio_files, model_path="./whisper_sanskrit_finetuned"):
    """Transcribe multiple audio files"""
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model directory '{model_path}' not found!")
        print("Make sure you've completed training first.")
        return None
    
    print(f"🎵 Loading model from: {model_path}")
    try:
        transcriber = WhisperSanskritTranscriber(model_path)
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Transcribe all files
    results = []
    successful = 0
    failed = 0
    
    print(f"🚀 Starting transcription of {len(audio_files)} files...\n")
    
    for i, audio_path in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}]", end=" ")
        result = transcribe_single_file(transcriber, audio_path, show_details=True)
        results.append(result)
        
        if result['status'] == 'success':
            successful += 1
        else:
            failed += 1
        print()  # Empty line for spacing
    
    # Create summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"transcription_summary_{timestamp}.json"
    
    summary = {
        'timestamp': timestamp,
        'total_files': len(audio_files),
        'successful': successful,
        'failed': failed,
        'model_path': model_path,
        'results': results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    print("="*60)
    print("📊 TRANSCRIPTION SUMMARY")
    print("="*60)
    print(f"Total files processed: {len(audio_files)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Summary saved to: {summary_file}")
    print("\nIndividual transcripts saved as: [filename]_transcript.txt")
    print("="*60)
    
    return results

def main():
    """Main function"""
    print("=== Sanskrit Audio Transcriber ===")
    print("Supports single files and entire folders\n")
    
    # Get path from command line or user input
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        print("Enter path to:")
        print("  📄 Single audio file (e.g., audio.m4a)")
        print("  📁 Folder with audio files (e.g., ./my_audio_folder)")
        input_path = input("\nPath: ").strip().strip('"\'')
    
    # Check if path is empty
    if not input_path:
        print("❌ No path provided!")
        return
    
    # Find audio files
    audio_files = find_audio_files(input_path)
    if not audio_files:
        return
    
    # Show what we found
    if len(audio_files) == 1:
        print(f"📄 Processing single file: {os.path.basename(audio_files[0])}")
    else:
        print(f"📁 Processing {len(audio_files)} files from folder")
        print("Files found:")
        for i, f in enumerate(audio_files[:5], 1):  # Show first 5
            print(f"  {i}. {os.path.basename(f)}")
        if len(audio_files) > 5:
            print(f"  ... and {len(audio_files) - 5} more files")
    
    # Confirm if multiple files
    if len(audio_files) > 1:
        confirm = input(f"\nProceed with transcribing {len(audio_files)} files? (y/n): ")
        if confirm.lower() != 'y':
            print("❌ Cancelled by user")
            return
    
    print()  # Empty line
    
    # Transcribe
    results = transcribe_multiple_files(audio_files)
    
    if results:
        successful = sum(1 for r in results if r['status'] == 'success')
        if successful > 0:
            print(f"\n✅ Successfully transcribed {successful} files!")
            print(f"📊 Model Performance: ~{(1-0.76)*100:.0f}% word accuracy")
        else:
            print("\n❌ No files were successfully transcribed!")

if __name__ == "__main__":
    main()