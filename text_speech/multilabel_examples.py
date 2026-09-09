"""
GoEmotions Multi-Label Inspection Script (Phase 2, Step 3)

This script inspects real training examples from GoEmotions that contain:
- Exactly 2 emotion labels
- Exactly 3 emotion labels
- More than 3 emotion labels

It converts the raw integer label IDs to their corresponding human-readable emotion names.
"""

from datasets import load_dataset


def inspect_multilabel_examples():
    # Step 1: Load the GoEmotions training dataset
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("google-research-datasets/go_emotions")
    train_data = dataset["train"]

    # Step 2: Retrieve label names mapping from dataset metadata
    label_names = train_data.features["labels"].feature.names

    # Storage buckets for each multi-label category
    examples_2_labels = []
    examples_3_labels = []
    examples_more_than_3 = []

    # Step 3: Collect up to 3 examples for each multi-label category
    for example in train_data:
        labels = example["labels"]
        num_labels = len(labels)

        if num_labels == 2 and len(examples_2_labels) < 3:
            examples_2_labels.append(example)
        elif num_labels == 3 and len(examples_3_labels) < 3:
            examples_3_labels.append(example)
        elif num_labels > 3 and len(examples_more_than_3) < 3:
            examples_more_than_3.append(example)

        # Stop iteration early if all 3 buckets are full
        if (
            len(examples_2_labels) == 3
            and len(examples_3_labels) == 3
            and len(examples_more_than_3) == 3
        ):
            break

    # Helper function to format and print sample records
    def display_examples(title, examples_list):
        print(f"\n==========================================")
        print(f" {title}")
        print(f"==========================================")

        if not examples_list:
            print("No examples found for this category.")
            return

        for idx, ex in enumerate(examples_list, 1):
            raw_labels = ex["labels"]
            # Convert label IDs to emotion names using dataset metadata
            mapped_names = [label_names[label_id] for label_id in raw_labels]

            print(f"\nExample {idx}:")
            print(f"  Text          : {ex['text']}")
            print(f"  Raw Label IDs : {raw_labels}")
            print(f"  Emotion Names : {mapped_names}")

    # Step 4: Display examples for all categories
    display_examples("Category 1: Exactly 2 Emotion Labels", examples_2_labels)
    display_examples("Category 2: Exactly 3 Emotion Labels", examples_3_labels)
    display_examples("Category 3: More than 3 Emotion Labels", examples_more_than_3)


if __name__ == "__main__":
    inspect_multilabel_examples()
