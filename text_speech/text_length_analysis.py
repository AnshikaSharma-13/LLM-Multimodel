"""
GoEmotions Sequence Length Analysis (Phase 3, Step 1)

This script analyzes the tokenized length distribution of all training examples
in GoEmotions using the distilbert-base-uncased tokenizer (including [CLS] and [SEP]).
This helps determine an optimal max_length hyperparameter for model training.
"""

import statistics
from datasets import load_dataset
from transformers import AutoTokenizer


def analyze_text_lengths():
    # Step 1: Load the GoEmotions dataset
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("google-research-datasets/go_emotions")
    train_data = dataset["train"]

    # Step 2: Load distilbert-base-uncased tokenizer
    print("Loading distilbert-base-uncased tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Step 3: Compute tokenized length (including special tokens) for every training example
    print("Tokenizing training examples and calculating sequence lengths...")
    token_lengths = []

    for example in train_data:
        # tokenizer.encode returns token IDs including special tokens [CLS] and [SEP]
        length = len(tokenizer.encode(example["text"]))
        token_lengths.append(length)

    total_examples = len(token_lengths)

    # Step 4: Calculate length statistics (min, max, mean, median)
    min_length = min(token_lengths)
    max_length = max(token_lengths)
    mean_length = statistics.mean(token_lengths)
    median_length = statistics.median(token_lengths)

    # Step 5: Count and calculate percentages for lengths > 64, > 128, and > 256
    count_gt_64 = sum(1 for length in token_lengths if length > 64)
    count_gt_128 = sum(1 for length in token_lengths if length > 128)
    count_gt_256 = sum(1 for length in token_lengths if length > 256)

    pct_gt_64 = (count_gt_64 / total_examples) * 100
    pct_gt_128 = (count_gt_128 / total_examples) * 100
    pct_gt_256 = (count_gt_256 / total_examples) * 100

    # Step 6: Display analysis summary
    print("\n--- Sequence Length Analysis Summary ---")
    print(f"Total Training Examples : {total_examples}")
    print(f"Minimum Token Length    : {min_length}")
    print(f"Maximum Token Length    : {max_length}")
    print(f"Mean Token Length       : {mean_length:.2f}")
    print(f"Median Token Length     : {median_length:.1f}")

    print("\n--- Length Threshold Analysis ---")
    print(f"Examples with length > 64  : {count_gt_64:<6} ({pct_gt_64:.2f}%)")
    print(f"Examples with length > 128 : {count_gt_128:<6} ({pct_gt_128:.2f}%)")
    print(f"Examples with length > 256 : {count_gt_256:<6} ({pct_gt_256:.2f}%)")


if __name__ == "__main__":
    analyze_text_lengths()
