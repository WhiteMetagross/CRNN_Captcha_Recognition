#This program visualizes training metrics from a log file.
#It extracts metrics such as training loss, validation loss, validation character error rate (CER),
#sequence accuracy, and character accuracy, and generates plots for these metrics over epochs.

import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Function to parse the log file and extract training metrics.
def parse_log_file(log_path):
    log_content = ""
    try:
        with open(log_path, 'r') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        return None

    epoch_pattern = re.compile(r"Epoch (\d+)/(\d+)")
    train_loss_pattern = re.compile(r"Train Loss: ([\d.]+)")
    val_loss_pattern = re.compile(r"Val Loss: ([\d.]+)")
    val_cer_pattern = re.compile(r"Val CER: ([\d.]+)")
    seq_acc_pattern = re.compile(r"Seq Acc: ([\d.]+)%")
    char_acc_pattern = re.compile(r"Char Acc: ([\d.]+)%")

    epochs_data = epoch_pattern.findall(log_content)
    if not epochs_data:
        print("No epoch data found in the log file.")
        return None

    num_epochs = int(epochs_data[-1][1])
    data_records = []

    for i in range(1, num_epochs + 1):
        epoch_start_str = f"Epoch {i}/{num_epochs}"
        next_epoch_start_str = f"Epoch {i+1}/{num_epochs}"
        
        start_index = log_content.find(epoch_start_str)
        if start_index == -1:
            continue
            
        end_index = log_content.find(next_epoch_start_str)
        if end_index == -1:
            end_index = len(log_content)

        epoch_block = log_content[start_index:end_index]
        
        train_loss_match = train_loss_pattern.search(epoch_block)
        val_loss_match = val_loss_pattern.search(epoch_block)
        val_cer_match = val_cer_pattern.search(epoch_block)
        seq_acc_match = seq_acc_pattern.search(epoch_block)
        char_acc_match = char_acc_pattern.search(epoch_block)

        if all([train_loss_match, val_loss_match, val_cer_match, seq_acc_match, char_acc_match]):
            record = {
                'Epoch': i,
                'Train Loss': float(train_loss_match.group(1)),
                'Val Loss': float(val_loss_match.group(1)),
                'Val CER': float(val_cer_match.group(1)),
                'Sequence Accuracy': float(seq_acc_match.group(1)),
                'Character Accuracy': float(char_acc_match.group(1))
            }
            data_records.append(record)

    if not data_records:
        print("Could not parse any complete data records from the log file.")
        return None

    return pd.DataFrame(data_records)

#Function to plot the training metrics.
def plot_metrics(df, output_dir):
    if df is None or df.empty:
        print("No data to plot.")
        return

    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Training and Validation Metrics', fontsize=20, weight='bold')

    #Plotting training and validation loss.
    sns.lineplot(ax=axes[0, 0], x='Epoch', y='value', hue='variable', data=pd.melt(df, ['Epoch'], ['Train Loss', 'Val Loss']))
    axes[0, 0].set_title('Loss vs. Epochs', fontsize=14)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(title='Metric')

    #Plotting validation metrics.S
    sns.lineplot(ax=axes[0, 1], x='Epoch', y='Sequence Accuracy', data=df, color='g', marker='o')
    axes[0, 1].set_title('Validation Sequence Accuracy', fontsize=14)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)

    #Plotting validation character accuracy and CER.
    sns.lineplot(ax=axes[1, 0], x='Epoch', y='Character Accuracy', data=df, color='b', marker='o')
    axes[1, 0].set_title('Validation Character Accuracy', fontsize=14)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy (%)', fontsize=12)

    #Plotting validation character error rate (CER).
    sns.lineplot(ax=axes[1, 1], x='Epoch', y='Val CER', data=df, color='r', marker='o')
    axes[1, 1].set_title('Validation Character Error Rate (CER)', fontsize=14)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Error Rate', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(output_dir, "training_visualizations.png")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        plt.savefig(output_path)
        print(f"Plots saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving plots: {e}")
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training metrics from a log file.")
    parser.add_argument('--log_path', type=str, default="./logs/training.log", help="Path to the training.log file.")
    parser.add_argument('--output_dir', type=str, default="./visualizations", help="Directory to save the plots.")
    args = parser.parse_args()

    df = parse_log_file(args.log_path)
    if df is not None:
        plot_metrics(df, args.output_dir)

