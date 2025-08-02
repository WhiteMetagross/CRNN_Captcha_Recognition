#This program splits a dataset of files into training, validation, and test sets,
#analyzes character distribution, and generates plots and CSV reports.

import os
import json
import random
import shutil
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

#Load configuration from a JSON file.
##The config should contain a 'data' field with a 'charset' list of characters.
def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    if 'data' not in config or 'charset' not in config['data']:
        raise KeyError("Config must contain 'data.charset' field")
    return config

#Extracts the label from a filename based on the provided charset.
#The label is formed by keeping only characters that are in the charset.
def extract_label_from_filename(filename, charset):
    name_without_ext = os.path.splitext(filename)[0]
    return ''.join(char for char in name_without_ext if char in charset)

#Randomly splits a list of files into training, validation, and test sets based on specified ratios.
#The ratios must sum to 1.
def random_split(files, train_ratio, val_ratio, test_ratio):
    random.shuffle(files)
    
    n_train = int(len(files) * train_ratio)
    n_val = int(len(files) * val_ratio)
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    
    return train_files, val_files, test_files

#Analyzes the character distribution in the dataset files.
#It counts the occurrences of each character in the labels extracted from filenames.
def analyze_character_distribution(files, charset):
    char_counter = Counter()
    total_sequences = 0
    
    for filename in files:
        label = extract_label_from_filename(filename, charset)
        if label:
            char_counter.update(label)
            total_sequences += 1
    
    return char_counter, total_sequences

#Copies files from source to destination in parallel using a thread pool.
#It handles exceptions during copying and reports errors.
def copy_files_parallel(file_pairs, desc, workers):
    if not file_pairs:
        return
    
    def copy_file(pair):
        src, dst = pair
        try:
            shutil.copyfile(src, dst)
        except Exception as e:
            print(f"Error copying {src} to {dst}: {e}")
            raise
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(tqdm(executor.map(copy_file, file_pairs),
                  total=len(file_pairs), desc=desc))

#Creates analysis plots showing character frequency distribution and dataset split distribution.
#It generates a bar plot for character frequencies and a pie chart for dataset splits.
def create_analysis_plots(train_files, val_files, test_files, charset, output_dir):
    train_chars, train_seqs = analyze_character_distribution(train_files, charset)
    val_chars, val_seqs = analyze_character_distribution(val_files, charset)
    test_chars, test_seqs = analyze_character_distribution(test_files, charset)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    all_chars = sorted(charset)
    train_counts = [train_chars.get(char, 0) for char in all_chars]
    val_counts = [val_chars.get(char, 0) for char in all_chars]
    test_counts = [test_chars.get(char, 0) for char in all_chars]
    
    x = np.arange(len(all_chars))
    width = 0.25
    
    ax1.bar(x - width, train_counts, width, label='Train', alpha=0.8, color='#2E86AB')
    ax1.bar(x, val_counts, width, label='Validation', alpha=0.8, color='#A23B72')
    ax1.bar(x + width, test_counts, width, label='Test', alpha=0.8, color='#F18F01')
    
    ax1.set_xlabel('Characters')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Character Frequency Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_chars, fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    total_files = len(train_files) + len(val_files) + len(test_files)
    if total_files > 0:
        sizes = [len(train_files), len(val_files), len(test_files)]
        labels = ['Train', 'Validation', 'Test']
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90,
                                          textprops={'fontsize': 12})
        ax2.set_title('Dataset Split Distribution')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'dataset_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

#Creates a CSV report summarizing the dataset split and character frequency analysis.
#It includes counts and percentages for each character across train, validation, and test sets.
def create_analysis_csv(train_files, val_files, test_files, charset, output_dir):
    csv_path = os.path.join(output_dir, 'dataset_analysis.csv')
    
    train_chars, train_seqs = analyze_character_distribution(train_files, charset)
    val_chars, val_seqs = analyze_character_distribution(val_files, charset)
    test_chars, test_seqs = analyze_character_distribution(test_files, charset)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(['DATASET SPLIT SUMMARY:'])
        writer.writerow(['Split', 'Files', 'Sequences', 'Percentage'])
        total_files = len(train_files) + len(val_files) + len(test_files)
        if total_files > 0:
            writer.writerow(['Train', len(train_files), train_seqs, f"{len(train_files)/total_files*100:.2f}%"])
            writer.writerow(['Validation', len(val_files), val_seqs, f"{len(val_files)/total_files*100:.2f}%"])
            writer.writerow(['Test', len(test_files), test_seqs, f"{len(test_files)/total_files*100:.2f}%"])
        
        writer.writerow([])
        writer.writerow(['CHARACTER FREQUENCY:'])
        writer.writerow(['Character', 'Train_Count', 'Val_Count', 'Test_Count', 'Total_Count', 'Train_%', 'Val_%', 'Test_%'])
        
        for char in sorted(charset):
            train_count = train_chars.get(char, 0)
            val_count = val_chars.get(char, 0)
            test_count = test_chars.get(char, 0)
            total_count = train_count + val_count + test_count
            
            if total_count > 0:
                train_pct = (train_count / total_count) * 100
                val_pct = (val_count / total_count) * 100
                test_pct = (test_count / total_count) * 100
            else:
                train_pct = val_pct = test_pct = 0
            
            writer.writerow([char, train_count, val_count, test_count, total_count,
                           f"{train_pct:.2f}", f"{val_pct:.2f}", f"{test_pct:.2f}"])
        
        total_train_chars = sum(train_chars.values())
        total_val_chars = sum(val_chars.values())
        total_test_chars = sum(test_chars.values())
        total_chars = total_train_chars + total_val_chars + total_test_chars
        
        writer.writerow([])
        if total_chars > 0:
            writer.writerow(['TOTALS', total_train_chars, total_val_chars, total_test_chars, total_chars,
                            f"{(total_train_chars/total_chars)*100:.2f}",
                            f"{(total_val_chars/total_chars)*100:.2f}",
                            f"{(total_test_chars/total_chars)*100:.2f}"])
        else:
            writer.writerow(['TOTALS', total_train_chars, total_val_chars, total_test_chars, total_chars,
                            "0.00", "0.00", "0.00"])
    
    return csv_path

#Splits the dataset into training, validation, and test sets,
#copies the files to respective directories, and generates analysis plots and CSV reports.
def split_dataset(input_dir, output_dir, train_ratio, val_ratio, test_ratio, config_path, workers=8):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError('train_ratio, val_ratio, and test_ratio must sum to 1')
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    config = load_config(config_path)
    charset = set(config['data']['charset'])
    
    entries = [entry.name for entry in os.scandir(input_dir) if entry.is_file()]
    
    if not entries:
        raise ValueError(f"No files found in input directory: {input_dir}")
    
    print(f"Found {len(entries)} files in dataset")
    
    train_files, val_files, test_files = random_split(
        entries, train_ratio, val_ratio, test_ratio
    )
    
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    train_tasks = [(os.path.join(input_dir, f), os.path.join(train_dir, f)) for f in train_files]
    val_tasks = [(os.path.join(input_dir, f), os.path.join(val_dir, f)) for f in val_files]
    test_tasks = [(os.path.join(input_dir, f), os.path.join(test_dir, f)) for f in test_files]
    
    copy_files_parallel(train_tasks, 'Training Split', workers)
    copy_files_parallel(val_tasks, 'Validation Split', workers)
    copy_files_parallel(test_tasks, 'Test Split', workers)
    
    print(f"\nDataset split complete:")
    print(f"Train: {len(train_files)} files ({len(train_files)/len(entries)*100:.1f}%)")
    print(f"Val: {len(val_files)} files ({len(val_files)/len(entries)*100:.1f}%)")
    print(f"Test: {len(test_files)} files ({len(test_files)/len(entries)*100:.1f}%)")
    
    plot_path = create_analysis_plots(train_files, val_files, test_files, charset, output_dir)
    print(f"Analysis plot saved to: {plot_path}")
    
    csv_path = create_analysis_csv(train_files, val_files, test_files, charset, output_dir)
    print(f"Analysis CSV saved to: {csv_path}")

if __name__ == '__main__':
    #The input directory should contain the files to be split.
    input_dir = r"C:\Users\Xeron\OneDrive\Desktop\CaptchaNew"
    #The output directory will contain the split datasets and analysis reports.
    #It will create 'train', 'val', and 'test' subdirectories.
    output_dir = r"C:\Users\Xeron\OneDrive\Documents\Programs\ProjectAA\dataset"
    config_path = "config.json"
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    workers = 8
    
    split_dataset(input_dir, output_dir, train_ratio, val_ratio, test_ratio, config_path, workers)