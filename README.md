# Fine-Tuning Whisper for Sanskrit Automatic Speech Recognition

This project provides a complete pipeline for fine-tuning OpenAI's Whisper models for Automatic Speech Recognition (ASR) on the Sanskrit language. It includes scripts for data preprocessing, training, inference, and results visualization.

A key feature of this repository is the experimental implementation of **prosodic feature fusion**, allowing the model to leverage pitch and energy alongside standard Mel spectrograms to potentially improve recognition accuracy.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow.svg)

---

## Features

-   **Whisper Model Fine-Tuning**: Easily fine-tune any model from the Whisper family (e.g., `tiny`, `base`, `small`).
-   **Prosodic Feature Fusion**: An advanced, experimental feature to combine pitch, energy, and their deltas with Mel spectrograms.
-   **Automated Data Handling**: Scripts to automatically preprocess audio files and split data into training, validation, and test sets.
-   **Comprehensive Training**: Leverages Hugging Face `transformers` and `accelerate` for efficient, multi-GPU-ready training.
-   **Detailed Logging & Visualization**: Automatically logs metrics (Loss, WER, CER) to TensorBoard and includes a script to generate publication-ready plots.
-   **Organized Experiment Tracking**: Each training run is saved to a unique, timestamped directory to prevent overwriting results and facilitate comparison.
-   **Simple Inference**: A straightforward script to run transcription on new audio files using your fine-tuned models.

## Project Structure

```
Whisper_ASR/
│
├── experiments/              # Output directory for all training runs (ignored by Git)
├── sanskrit_audio/           # Your raw audio files (ignored by Git)
│
├── config.py                 # Main configuration file for all settings
├── train.py                  # Main script to start the training process
├── inference.py              # Script to transcribe audio with a trained model
├── plot_results.py           # Script to visualize training metrics
│
├── data_utils.py             # Utilities for data loading and preprocessing
├── training_utils.py         # Helper classes for training (metrics, callbacks)
│
├── transcript.txt            # Your master transcript file
├── requirements.txt          # Python package dependencies
└── .gitignore                # Specifies files and folders for Git to ignore
```

## Setup and Installation

### 1. Prerequisites
-   Python 3.9+
-   [FFmpeg](https://ffmpeg.org/download.html): Ensure it is installed and accessible in your system's PATH. This is required for audio file processing.

### 2. Clone the Repository
```bash
git clone https://github.com/Devx228/ASR_Whisper_Finetuning.git
cd ASR_Whisper_Finetuning
```

### 3. Set Up Virtual Environment and Install Dependencies
```bash
# Create and activate a virtual environment
python -m venv wenv
.\wenv\Scripts\activate  # On Windows

# Install the required packages
pip install -r requirements.txt
```

### 4. Data Preparation
1.  Place all your audio files (e.g., `.m4a`, `.mp3`, `.wav`) inside the `sanskrit_audio/` directory.
2.  Create a `transcript.txt` file in the root directory. Each line should contain the audio filename and its corresponding transcription, separated by a tab.

    **Example `transcript.txt`:**
    ```
    audio_001.m4a	संस्कृतवाक्यम् अत्र लिख्यताम्।
    audio_002.m4a	अन्यत् संस्कृतवाक्यम्।
    ```

## How to Use

### 1. Configure Your Training Run
Open `config.py` and adjust the settings to your needs. Key parameters include:
-   `model_name`: The Whisper model to fine-tune (e.g., `"openai/whisper-base"`).
-   `use_prosodic_features`: Set to `True` to enable prosodic feature fusion, or `False` for standard training.
-   `batch_size`, `learning_rate`, `max_steps`: Standard training hyperparameters.

### 2. Start Training
Run the main training script. It will automatically handle data processing, training, and evaluation.
```bash
python train.py
```
Each training run will be saved in a new, unique folder inside the `experiments/` directory.

### 3. Transcribe Audio with a Trained Model
Use the `inference.py` script to perform transcription. You need to provide the path to your trained model directory.
```bash
python inference.py --model_path "experiments/whisper-base_no_prosody_20250704-011828" --audio_path "path/to/your/test_audio.wav"
```

### 4. Plot Training and Evaluation Results
To visualize the metrics (Loss, WER, CER) from a training run, use the `plot_results.py` script.
```bash
python plot_results.py --dir "experiments/whisper-base_no_prosody_20250704-011828"
```
This will generate and save a `training_plots_grid.png` image inside the specified experiment
