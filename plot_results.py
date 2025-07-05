import os
import argparse
from tbparse import SummaryReader
import matplotlib.pyplot as plt
import pandas as pd

def plot_training_results(experiment_dir: str):
    """
    Reads TensorBoard logs from an experiment directory and plots the results in a 2x2 grid.

    Args:
        experiment_dir: Path to the specific experiment directory.
    """
    if not os.path.isdir(experiment_dir):
        print(f"❌ Error: Directory not found at '{experiment_dir}'")
        return

    log_dir = None
    for root, dirs, files in os.walk(experiment_dir):
        if any(file.startswith("events.out.tfevents") for file in files):
            log_dir = root
            break

    if log_dir is None:
        print(f"❌ Error: No TensorBoard event files found in '{experiment_dir}' or its subdirectories.")
        return

    print(f"📊 Reading logs from: {log_dir}")
    reader = SummaryReader(log_dir)
    df = reader.scalars

    if df.empty:
        print("❌ Error: No data found in the TensorBoard logs.")
        return

    # --- Data Extraction ---
    train_loss = df[df["tag"] == "train/loss"].dropna()
    eval_loss = df[df["tag"] == "eval/loss"].dropna()
    eval_wer = df[df["tag"] == "eval/wer"].dropna()
    eval_cer = df[df["tag"] == "eval/cer"].dropna()

    # --- Plotting ---
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Training Metrics for: {os.path.basename(experiment_dir)}', fontsize=14)

    # Plot 1: Training Loss
    axs[0, 0].plot(train_loss["step"], train_loss["value"], label="Training Loss", color='royalblue')
    axs[0, 0].set_title("Training Loss vs. Steps", fontsize=12)
    axs[0, 0].set_xlabel("Steps")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].legend()

    # Plot 2: Evaluation Loss
    axs[0, 1].plot(eval_loss["step"], eval_loss["value"], label="Evaluation Loss", color='darkorange', marker='.', linestyle='--')
    axs[0, 1].set_title("Evaluation Loss vs. Steps", fontsize=12)
    axs[0, 1].set_xlabel("Steps")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].legend()

    # Plot 3: Evaluation WER
    axs[1, 0].plot(eval_wer["step"], eval_wer["value"], label="Evaluation WER", color='forestgreen', marker='.', linestyle='--')
    axs[1, 0].set_title("Word Error Rate (WER) vs. Steps", fontsize=12)
    axs[1, 0].set_xlabel("Steps")
    axs[1, 0].set_ylabel("WER")
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    axs[1, 0].legend()

    # Plot 4: Evaluation CER
    if not eval_cer.empty:
        axs[1, 1].plot(eval_cer["step"], eval_cer["value"], label="Evaluation CER", color='crimson', marker='.', linestyle='--')
        axs[1, 1].set_title("Character Error Rate (CER) vs. Steps", fontsize=12)
        axs[1, 1].set_xlabel("Steps")
        axs[1, 1].set_ylabel("CER")
        axs[1, 1].grid(True, linestyle='--', alpha=0.6)
        axs[1, 1].legend()
    else:
        # Hide the empty subplot if there's no CER data
        axs[1, 1].axis('off')
        axs[1, 1].text(0.5, 0.5, 'No CER data found', ha='center', va='center', fontsize=12, color='gray')


    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    output_path = os.path.join(experiment_dir, "training_plots_grid.png")
    plt.savefig(output_path)
    print(f"✅ Plot saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics from a Whisper fine-tuning experiment.")
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to the experiment directory containing TensorBoard logs."
    )
    args = parser.parse_args()
    plot_training_results(args.dir)