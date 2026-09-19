"""
GoEmotions PyTorch DataLoader Demo (Phase 4, Step 9)

This script demonstrates how tokenized GoEmotions text inputs and multi-hot label vectors
are wrapped into a PyTorch TensorDataset and batched using DataLoader.
"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizerFast


def demo_goemotions_dataloader():
    model_name = "distilbert-base-uncased"
    num_examples = 12
    batch_size = 3
    num_labels = 28

    print("--- 1. Loading Dataset & Tokenizer ---")
    print("Loading GoEmotions dataset...")
    raw_dataset = load_dataset("google-research-datasets/go_emotions")

    # Select only the first 12 training examples for this educational demo
    train_subset = raw_dataset["train"].select(range(num_examples))

    print(f"Loading DistilBertTokenizerFast from '{model_name}'...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    # Tokenize the 12 text examples
    texts = [example["text"] for example in train_subset]
    raw_labels = [example["labels"] for example in train_subset]

    encoded = tokenizer(
        texts,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Convert label ID lists into 28-dimensional multi-hot float tensors
    multi_hot_list = []
    for label_ids in raw_labels:
        multi_hot = [0.0] * num_labels
        for lid in label_ids:
            multi_hot[lid] = 1.0
        multi_hot_list.append(multi_hot)

    labels_tensor = torch.tensor(multi_hot_list, dtype=torch.float32)

    # Create PyTorch TensorDataset holding input_ids, attention_mask, and labels
    tensor_dataset = TensorDataset(input_ids, attention_mask, labels_tensor)

    # Create PyTorch DataLoader with batch_size=3 and shuffle=False
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

    print("\n--- 2. DataLoader Batch Iteration ---")
    # Iterate through DataLoader and print tensor shapes for each batch
    for batch_num, (b_input_ids, b_attention_mask, b_labels) in enumerate(dataloader, start=1):
        print(f"Batch {batch_num}:")
        print(f"  input_ids shape     : {list(b_input_ids.shape)}")
        print(f"  attention_mask shape: {list(b_attention_mask.shape)}")
        print(f"  labels shape        : {list(b_labels.shape)}")
        print(f"  Examples in batch   : {b_input_ids.shape[0]}\n")

    # Print final summary
    print("--- 3. Summary Statistics ---")
    print(f"total examples    = {len(tensor_dataset)}")
    print(f"batch size        = {batch_size}")
    print(f"number of batches = {len(dataloader)}")

    # EDUCATIONAL CONCEPT EXPLANATION:
    # - DataLoader groups individual examples into batched tensors.
    # - Each batch contains multiple examples.
    # - 3 examples x 64 token positions gives input_ids shape [3, 64].
    # - 3 examples x 28 emotion labels gives labels shape [3, 28].


if __name__ == "__main__":
    demo_goemotions_dataloader()
