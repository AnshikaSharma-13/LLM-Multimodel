"""
Mini Real Training Script (Phase 4, Step 10)

This script connects the real GoEmotions dataset (a 32-example subset) to a DistilBERT
multi-label classification model using PyTorch DataLoader, BCEWithLogitsLoss, and AdamW.
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification


def run_mini_real_training():
    model_name = "distilbert-base-uncased"
    num_examples = 32
    num_labels = 28
    batch_size = 8
    num_epochs = 2
    learning_rate = 5e-5
    problem_type = "multi_label_classification"

    print("--- 1. Loading GoEmotions Subset & DistilBERT ---")
    print("Loading GoEmotions dataset...")
    raw_dataset = load_dataset("google-research-datasets/go_emotions")

    # Select only the first 32 examples from the training split
    train_subset = raw_dataset["train"].select(range(num_examples))

    print(f"Loading tokenizer & model for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
    )
    model.train()

    # Tokenize texts into PyTorch tensors
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

    # Convert label IDs into 28-dimensional multi-hot float tensors
    multi_hot_list = []
    for label_ids in raw_labels:
        multi_hot = [0.0] * num_labels
        for lid in label_ids:
            multi_hot[lid] = 1.0
        multi_hot_list.append(multi_hot)

    labels_tensor = torch.tensor(multi_hot_list, dtype=torch.float32)

    # Create TensorDataset and DataLoader (batch_size=8, shuffle=False)
    tensor_dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

    # Setup Loss function and AdamW optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    print("\n--- 2. Executing Training Loop (2 Epochs, 32 Examples) ---")
    global_step = 0

    # Train for exactly 2 epochs
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch} ---")

        for b_input_ids, b_attention_mask, b_labels in dataloader:
            global_step += 1

            # Step sequence: reset grads -> forward pass -> loss -> backward pass -> optimizer step
            optimizer.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, b_labels)
            loss.backward()
            optimizer.step()

            # Print step progress metrics
            print(
                f"Epoch {epoch} | Step {global_step:<2} | Loss: {loss.item():.4f} | Logits Shape: {list(logits.shape)}"
            )

    # Final summary
    print("\n--- 3. Training Run Summary ---")
    print(f"number of examples   = {num_examples}")
    print(f"batch size           = {batch_size}")
    print(f"number of epochs     = {num_epochs}")
    print(f"total optimizer steps = {global_step}")


if __name__ == "__main__":
    run_mini_real_training()
