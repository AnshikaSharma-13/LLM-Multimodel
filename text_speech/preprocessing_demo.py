"""
Educational Preprocessing Demo (Phase 3, Step 2)

This script demonstrates how a single GoEmotions training example is transformed
into model-ready numerical inputs (tokenized IDs, attention mask, and multi-hot label vector)
for training DistilBERT on multi-label emotion classification.
"""

from datasets import load_dataset
from transformers import AutoTokenizer


def demo_preprocessing():
    # Step 1: Load the GoEmotions dataset
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("google-research-datasets/go_emotions")
    train_data = dataset["train"]

    # Step 2: Load the pre-trained DistilBERT tokenizer
    print("Loading distilbert-base-uncased tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Step 3: Extract the 28 emotion label names from dataset metadata
    label_names = train_data.features["labels"].feature.names
    num_classes = len(label_names)  # 28

    # Step 4: Select one training example
    example = train_data[0]
    text = example["text"]
    raw_label_ids = example["labels"]
    emotion_names = [label_names[lid] for lid in raw_label_ids]

    # Step 5: Convert raw label IDs into a 28-dimensional multi-hot vector
    multi_hot_labels = [0] * num_classes
    for label_id in raw_label_ids:
        multi_hot_labels[label_id] = 1

    # Step 6: Tokenize text using max_length=64, padding="max_length", truncation=True
    encoded = tokenizer(
        text,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Count real tokens (1s) vs padding tokens (0s) in the attention mask
    count_ones = attention_mask.count(1)
    count_zeros = attention_mask.count(0)

    # Step 7: Display all preprocessing components
    print("\n--- 1. Raw Sample Details ---")
    print(f"Original Text        : {text}")
    print(f"Raw Label IDs        : {raw_label_ids}")
    print(f"Emotion Names        : {emotion_names}")

    print("\n--- 2. Multi-Hot Label Vector ---")
    print(f"Multi-Hot Labels     : {multi_hot_labels}")
    print(f"Label Vector Length  : {len(multi_hot_labels)}")

    print("\n--- 3. Tokenizer Outputs ---")
    print(f"Tokens               : {tokens}")
    print(f"Input IDs            : {input_ids}")
    print(f"Attention Mask       : {attention_mask}")
    print(f"Number of Input IDs  : {len(input_ids)}")
    print(f"Attention Mask 1s    : {count_ones} (real tokens)")
    print(f"Attention Mask 0s    : {count_zeros} (padding tokens)")


if __name__ == "__main__":
    demo_preprocessing()
