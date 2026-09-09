"""
Multi-Hot Label Encoding Demo (Phase 2, Step 4)

This script demonstrates how multi-label raw integer IDs (e.g., [8, 20])
are converted into a fixed 28-dimensional binary "multi-hot" vector suitable
for multi-label classification models.
"""

from datasets import load_dataset


def demo_multi_hot_encoding():
    # Step 1: Load GoEmotions dataset
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("google-research-datasets/go_emotions")
    train_data = dataset["train"]

    # Step 2: Retrieve label names and count from dataset metadata
    label_names = train_data.features["labels"].feature.names
    num_classes = len(label_names)  # 28 classes

    # Step 3: Find the first example in training split with exactly 2 labels
    target_example = None
    for example in train_data:
        if len(example["labels"]) == 2:
            target_example = example
            break

    # Extract text, raw label IDs, and mapped emotion names
    text = target_example["text"]
    raw_label_ids = target_example["labels"]
    emotion_names = [label_names[idx] for idx in raw_label_ids]

    # Step 4: Print details of the chosen example
    print("\n--- Selected Training Example ---")
    print(f"Text          : {text}")
    print(f"Raw Label IDs : {raw_label_ids}")
    print(f"Emotion Names : {emotion_names}")

    # Step 5: Create a 28-element list initialized with 0
    multi_hot_vector = [0] * num_classes

    # Step 6: Set positions corresponding to label IDs to 1
    for label_id in raw_label_ids:
        multi_hot_vector[label_id] = 1

    # Step 7: Print the resulting multi-hot vector and stats
    print("\n--- Multi-Hot Vector Encoding ---")
    print(f"Multi-Hot Vector : {multi_hot_vector}")
    print(f"Vector Length    : {len(multi_hot_vector)}")

    # Identify and display active positions (where value == 1)
    active_positions = [i for i, val in enumerate(multi_hot_vector) if val == 1]
    print(f"Positions with 1 : {active_positions}")
    for pos in active_positions:
        print(f"  Index {pos} ({label_names[pos]}) -> 1")


if __name__ == "__main__":
    demo_multi_hot_encoding()
