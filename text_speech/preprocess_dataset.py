"""
Full GoEmotions Dataset Preprocessing Script (Phase 3, Step 4)

This script loads the full GoEmotions dataset (train, validation, and test splits),
tokenizes all text using distilbert-base-uncased (max_length=64, padding="max_length", truncation=True),
and converts all raw label ID lists into 28-dimensional multi-hot vectors.
"""

from datasets import load_dataset
from transformers import AutoTokenizer


def preprocess_full_dataset():
    # Step 1: Load the full GoEmotions dataset (train, validation, test)
    print("Loading full GoEmotions dataset...")
    raw_dataset = load_dataset("google-research-datasets/go_emotions")

    # Step 2: Load the pre-trained DistilBERT tokenizer
    print("Loading distilbert-base-uncased tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Retrieve total number of emotion categories (28) from dataset metadata
    label_names = raw_dataset["train"].features["labels"].feature.names
    num_classes = len(label_names)  # 28

    # Step 3: Print original dataset split sizes
    print("\n--- Original Dataset Split Sizes ---")
    for split_name, split_data in raw_dataset.items():
        print(f"  {split_name:<12}: {len(split_data)} examples")

    # Step 4: Define batch preprocessing function
    def preprocess_function(examples):
        # Tokenize a batch of text entries with max_length=64 and padding/truncation
        model_inputs = tokenizer(
            examples["text"],
            max_length=64,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )

        # Convert each raw label ID list in the batch into a 28-dimensional multi-hot float vector
        batch_multi_hot_labels = []
        for label_ids in examples["labels"]:
            multi_hot = [0.0] * num_classes
            for lid in label_ids:
                multi_hot[lid] = 1.0
            batch_multi_hot_labels.append(multi_hot)

        # Assign multi-hot vectors to the 'labels' feature
        model_inputs["labels"] = batch_multi_hot_labels
        return model_inputs

    # Step 5: Preprocess all 3 splits (train, validation, test) using Dataset.map() with batched=True
    print("\nPreprocessing all dataset splits (train, validation, test)...")
    processed_dataset = raw_dataset.map(preprocess_function, batched=True)

    # Step 6: Print verification metrics
    print("\n--- Processed Dataset Verification ---")
    print("Processed Split Sizes:")
    for split_name, split_data in processed_dataset.items():
        print(f"  {split_name:<12}: {len(split_data)} examples")

    train_processed = processed_dataset["train"]
    print(f"\nProcessed Columns: {train_processed.column_names}")

    # Inspect one sample from the processed training split
    sample_example = train_processed[0]
    input_ids_len = len(sample_example["input_ids"])
    attention_mask_len = len(sample_example["attention_mask"])
    label_vector = sample_example["labels"]
    active_labels_count = sum(1 for val in label_vector if val == 1.0)

    print("\n--- Sample Processed Entry Details (Train Example 0) ---")
    print(f"input_ids length      : {input_ids_len}")
    print(f"attention_mask length : {attention_mask_len}")
    print(f"label vector length   : {len(label_vector)}")
    print(f"Active labels count   : {active_labels_count}")
    print(f"Label vector          : {label_vector}")

    return processed_dataset


if __name__ == "__main__":
    preprocess_full_dataset()
