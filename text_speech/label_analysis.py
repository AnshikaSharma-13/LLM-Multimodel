"""
GoEmotions Label Analysis Script (Phase 2, Step 2)

This script analyzes the training split of the GoEmotions dataset to understand:
1. Multi-label distribution (how many labels each example has).
2. Frequency of each of the 28 emotion categories across the dataset.
"""

from collections import Counter
from datasets import load_dataset


def analyze_labels():
    # Step 1: Load the GoEmotions dataset
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("google-research-datasets/go_emotions")
    train_data = dataset["train"]

    # Step 2: Retrieve label names from dataset metadata
    label_names = train_data.features["labels"].feature.names

    # Step 3: Track distribution of label counts per example and frequency of each emotion
    label_count_distribution = Counter()
    emotion_frequency = Counter()

    for example in train_data:
        labels = example["labels"]
        # Count how many labels this example has
        num_labels = len(labels)
        label_count_distribution[num_labels] += 1

        # Count occurrences of individual emotion label IDs
        for label_id in labels:
            emotion_frequency[label_id] += 1

    # Step 4: Print multi-label distribution summary
    print("\n--- Multi-Label Distribution per Example ---")
    print(f"Total training examples: {len(train_data)}\n")

    print(f"Examples with exactly 1 label  : {label_count_distribution[1]}")
    print(f"Examples with exactly 2 labels : {label_count_distribution[2]}")
    print(f"Examples with exactly 3 labels : {label_count_distribution[3]}")

    # Calculate examples with more than 3 labels
    more_than_3 = sum(count for num_labels, count in label_count_distribution.items() if num_labels > 3)
    print(f"Examples with > 3 labels       : {more_than_3}")

    # Step 5: Print frequency count for each of the 28 emotion labels
    print("\n--- Individual Emotion Label Frequencies ---")
    print(f"{'ID':<5} {'Emotion Name':<20} {'Count':<10}")
    print("-" * 38)

    for label_id, name in enumerate(label_names):
        count = emotion_frequency[label_id]
        print(f"{label_id:<5} {name:<20} {count:<10}")


if __name__ == "__main__":
    analyze_labels()
