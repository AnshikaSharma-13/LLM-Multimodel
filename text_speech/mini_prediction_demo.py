"""
Multi-Label Prediction Demo (Phase 4, Step 11B)

This script demonstrates the complete end-to-end multi-label emotion prediction flow:
text -> tokenizer -> DistilBERT -> logits -> sigmoid -> threshold -> emotion labels
by training on 32 GoEmotions samples for 2 epochs and performing inference on a sample text.
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification


def run_prediction_demo():
    model_name = "distilbert-base-uncased"
    num_examples = 32
    num_labels = 28
    batch_size = 8
    num_epochs = 2
    learning_rate = 5e-5
    problem_type = "multi_label_classification"
    threshold = 0.5

    print("--- 1. Loading Dataset, Tokenizer & Model ---")
    raw_dataset = load_dataset("google-research-datasets/go_emotions")
    train_subset = raw_dataset["train"].select(range(num_examples))

    # Retrieve available emotion category names from dataset metadata
    label_names = raw_dataset["train"].features["labels"].feature.names

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
    )

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

    # Convert raw label ID lists into 28-dimensional multi-hot float tensors
    multi_hot_list = []
    for label_ids in raw_labels:
        multi_hot = [0.0] * num_labels
        for lid in label_ids:
            multi_hot[lid] = 1.0
        multi_hot_list.append(multi_hot)

    labels_tensor = torch.tensor(multi_hot_list, dtype=torch.float32)

    # Create TensorDataset and DataLoader
    tensor_dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

    # Train for 2 epochs using BCEWithLogitsLoss & AdamW
    print("\n--- 2. Training for 2 Epochs on 32 Examples ---")
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(1, num_epochs + 1):
        for b_input_ids, b_attention_mask, b_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
            loss = loss_fn(outputs.logits, b_labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}/{num_epochs} training complete (Loss: {loss.item():.4f})")

    # Perform Inference on one example (Index 0)
    sample_index = 0
    sample_text = texts[sample_index]
    sample_raw_label_ids = raw_labels[sample_index]
    actual_emotions = [label_names[lid] for lid in sample_raw_label_ids]

    sample_input_ids = input_ids[sample_index : sample_index + 1]  # Shape: [1, 64]
    sample_attention_mask = attention_mask[sample_index : sample_index + 1]  # Shape: [1, 64]

    # Switch to evaluation mode and disable gradient calculation
    model.eval()
    with torch.no_grad():
        # Pipeline Flow: text -> tokenizer -> DistilBERT -> logits
        outputs = model(input_ids=sample_input_ids, attention_mask=sample_attention_mask)
        logits = outputs.logits  # Shape: [1, 28]

        # Pipeline Flow: logits -> sigmoid -> probabilities
        probabilities = torch.sigmoid(logits)[0]  # Shape: [28]

        # Pipeline Flow: probabilities -> threshold 0.5 -> predicted emotion label IDs
        predicted_label_ids = [
            idx for idx, prob in enumerate(probabilities.tolist()) if prob >= threshold
        ]
        predicted_emotions = [label_names[pid] for pid in predicted_label_ids]
        predicted_probs = [round(probabilities[pid].item(), 4) for pid in predicted_label_ids]

        # Extract top 3 highest scoring emotions for learning visibility
        top_prob_values, top_indices = torch.topk(probabilities, k=3)
        top_3_emotions = [
            f"{label_names[idx]} ({prob.item():.4f})"
            for idx, prob in zip(top_indices.tolist(), top_prob_values)
        ]

    # Print clearly labeled outputs
    print("\n==========================================")
    print("      END-TO-END PREDICTION RESULTS      ")
    print("==========================================")
    print(f"Original text          : \"{sample_text}\"\n")
    print(f"Actual emotions        : {actual_emotions}")
    print(f"Predicted emotions     : {predicted_emotions}")
    print(f"Predicted probabilities: {predicted_probs}")
    print(f"\n[Educational Note] Top 3 highest probability scores:")
    print(f"  {', '.join(top_3_emotions)}")
    print("  (Note: With only 2 training epochs on 32 samples, probabilities are still developing towards threshold 0.5)")


if __name__ == "__main__":
    run_prediction_demo()
