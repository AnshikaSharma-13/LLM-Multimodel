"""
Dataset.map() Preprocessing Demo (Phase 3, Step 3)

This script demonstrates how to preprocess a Hugging Face Dataset batch-wise using `dataset.map(..., batched=True)`.
It tokenizes texts with DistilBERT and converts multi-label integer IDs into 28-dimensional multi-hot vectors.
"""

from datasets import load_dataset
from transformers import AutoTokenizer


def demo_full_preprocessing():
    # Step 1: Load the GoEmotions dataset
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("google-research-datasets/go_emotions")
    train_data = dataset["train"]

    # Step 2: Load distilbert-base-uncased tokenizer
    print("Loading distilbert-base-uncased tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Get total number of emotion classes (28)
    label_names = train_data.features["labels"].feature.names
    num_classes = len(label_names)  # 28

    # Step 3: Define batch preprocessing function
    def preprocess_function(examples):
        # Tokenize a batch of texts with fixed length (max_length=64)
        model_inputs = tokenizer(
            examples["text"],
            max_length=64,
            padding="max_length",
            truncation=True,
        )

        # Convert each raw label ID list in the batch into a 28-dimensional multi-hot float vector
        batch_multi_hot_labels = []
        for label_ids in examples["labels"]:
            multi_hot = [0.0] * num_classes
            for lid in label_ids:
                multi_hot[lid] = 1.0
            batch_multi_hot_labels.append(multi_hot)

        # Assign multi-hot vectors to the 'labels' field
        model_inputs["labels"] = batch_multi_hot_labels
        return model_inputs

    # Step 4: Select a small subset of 5 training examples for inspection
    train_subset = train_data.select(range(5))

    # Step 5: Process subset using dataset.map() with batched=True
    # `batched=True` processes data in batches (lists of texts/labels), which is vastly faster than element-wise processing
    processed_subset = train_subset.map(preprocess_function, batched=True)

    # Step 6: Print dataset comparison metrics
    print("\n--- Dataset Preprocessing Overview ---")
    print(f"Original dataset columns             : {train_data.column_names}")
    print(f"Number of examples in selected subset: {len(train_subset)}")
    print(f"Processed dataset columns            : {processed_subset.column_names}")

    # Inspect the first processed example
    first_example = processed_subset[0]
    input_ids_len = len(first_example["input_ids"])
    attention_mask_len = len(first_example["attention_mask"])
    label_vector = first_example["labels"]
    active_labels_count = sum(1 for val in label_vector if val == 1.0)

    print("\n--- Sample Processed Entry Details ---")
    print(f"One processed example's input_ids length      : {input_ids_len}")
    print(f"One processed example's attention_mask length : {attention_mask_len}")
    print(f"One processed example's label vector          : {label_vector}")
    print(f"Label vector length                           : {len(label_vector)}")
    print(f"Number of active labels in that example        : {active_labels_count}")


if __name__ == "__main__":
    demo_full_preprocessing()
