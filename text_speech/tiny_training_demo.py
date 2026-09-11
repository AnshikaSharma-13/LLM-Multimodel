"""
Tiny DistilBERT Training Experiment (Phase 4, Step 6)

This script demonstrates a complete 3-step training loop on a 4-example subset
of the GoEmotions dataset using DistilBERT, PyTorch BCEWithLogitsLoss, and AdamW.
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


def run_tiny_training_demo():
    model_name = "distilbert-base-uncased"
    num_labels = 28
    problem_type = "multi_label_classification"
    num_examples = 4

    print("--- 1. Loading Dataset & Tokenizer ---")
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("google-research-datasets/go_emotions")

    # Select only 4 training examples for this educational micro-experiment
    train_subset = dataset["train"].select(range(num_examples))

    print(f"Loading DistilBertTokenizerFast for '{model_name}'...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    print(f"Loading DistilBertForSequenceClassification ({num_labels} labels)...")
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
    )

    # Set model to training mode
    model.train()

    # --- 2. Preprocessing & Batch Preparation ---
    texts = [example["text"] for example in train_subset]
    raw_labels = [example["labels"] for example in train_subset]

    # Tokenize texts into PyTorch tensors
    encoded = tokenizer(
        texts,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Convert raw label ID lists into 28-dimensional multi-hot float tensors [4, 28]
    multi_hot_list = []
    for label_ids in raw_labels:
        multi_hot = [0.0] * num_labels
        for lid in label_ids:
            multi_hot[lid] = 1.0
        multi_hot_list.append(multi_hot)

    target_labels = torch.tensor(multi_hot_list, dtype=torch.float32)

    # Display dataset and tensor specifications
    print("\n--- 2. Batch & Data Specifications ---")
    print(f"Number of training examples : {num_examples}")
    print(f"Number of emotion labels    : {num_labels}")
    print(f"input_ids shape             : {input_ids.shape}")
    print(f"attention_mask shape        : {attention_mask.shape}")
    print(f"target_labels shape        : {target_labels.shape}")

    # --- 3. Optimizer & Loss Function Setup ---
    learning_rate = 5e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    # --- 4. Micro Training Loop (3 Steps) ---
    print("\n--- 3. Executing 3 Micro-Training Steps ---")
    for step in range(1, 4):
        # Step A: Reset old gradients
        optimizer.zero_grad()

        # Step B: Forward pass through DistilBERT
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Step C: Compute multi-label BCE loss
        loss = loss_fn(logits, target_labels)

        # Step D: Backward pass (compute gradients for all parameters)
        loss.backward()

        # Step E: Optimizer step (update model weights)
        optimizer.step()

        # Print step metrics
        print(f"Step {step}: Loss = {loss.item():.4f} | Logits Shape = {list(logits.shape)}")

    # --- 5. Educational Explanation ---
    print("\n--- Training Loop Execution Breakdown ---")
    print("Step 1: forward pass -> calculate loss -> backward pass (compute gradients) -> update parameters")
    print("Step 2: forward pass with updated parameters -> calculate new loss -> backward pass -> update parameters")
    print("Step 3: repeat (loss decreases as model parameters adapt to training targets)")


if __name__ == "__main__":
    run_tiny_training_demo()
