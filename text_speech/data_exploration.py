"""
GoEmotions Dataset Exploration Script (Phase 2, Step 1)

This script loads the GoEmotions dataset from Hugging Face and inspects its structure,
splits, column names, raw example entries, and label metadata without performing
any preprocessing or model training.
"""

from datasets import load_dataset


def explore_go_emotions():
    # Step 1: Load the GoEmotions dataset from Hugging Face
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("google-research-datasets/go_emotions")

    # Step 2: Print the dataset object to view all available splits (train, validation, test)
    print("\n--- Dataset Overview & Splits ---")
    print(dataset)

    # Step 3: Print column names of the training split
    train_split = dataset["train"]
    print("\n--- Training Split Columns ---")
    print(train_split.column_names)

    # Step 4: Print the number of examples in each split
    print("\n--- Example Counts Per Split ---")
    print(f"train: {len(dataset['train'])}")
    print(f"validation: {len(dataset['validation'])}")
    print(f"test: {len(dataset['test'])}")

    # Step 5: Extract available label names from dataset metadata if available
    print("\n--- Dataset Metadata & Available Label Names ---")
    try:
        label_feature = train_split.features["labels"]
        # In GoEmotions, 'labels' is a List of ClassLabel elements
        if hasattr(label_feature, "feature") and hasattr(label_feature.feature, "names"):
            label_names = label_feature.feature.names
            print(f"Found {len(label_names)} label names:")
            for idx, name in enumerate(label_names):
                print(f"  [{idx}] {name}")
        else:
            print("Label feature structure:", label_feature)
    except Exception as e:
        print(f"Could not retrieve label names from metadata: {e}")

    # Step 6: Print 5 raw examples from the training split (text and raw labels)
    print("\n--- First 5 Examples from Training Split ---")
    for i in range(5):
        example = train_split[i]
        print(f"\nExample {i + 1}:")
        print(f"  Text      : {example['text']}")
        print(f"  Raw Labels: {example['labels']}")


if __name__ == "__main__":
    explore_go_emotions()
