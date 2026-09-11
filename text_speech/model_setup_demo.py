"""
DistilBERT Model Setup Demo (Phase 4, Step 1)

This script demonstrates how to instantiate a pre-trained DistilBERT model for multi-label
sequence classification with 28 emotion categories and inspect its raw output logits.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def demo_model_setup():
    model_name = "distilbert-base-uncased"
    num_labels = 28
    problem_type = "multi_label_classification"

    # Step 1: Load AutoTokenizer
    print(f"Loading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Step 2: Load AutoModelForSequenceClassification configured for multi-label classification
    print(f"Loading sequence classification model for '{model_name}'...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
    )

    # Set model to evaluation mode for inference demo
    model.eval()

    # Step 3: Define and tokenize sample text
    sample_text = "I am feeling nervous but also excited about tomorrow."

    # Produce PyTorch tensor inputs ('pt') for direct model execution
    inputs = tokenizer(
        sample_text,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Step 4: Forward pass through DistilBERT model to compute raw logits
    # torch.no_grad() disables gradient tracking since we are not training here
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Step 5: Display model & tensor details
    print("\n--- Model Configuration & Parameters ---")
    print(f"Model Name          : {model_name}")
    print(f"Number of Labels    : {num_labels}")
    print(f"Problem Type        : {problem_type}")

    print("\n--- Tensor Shapes ---")
    print(f"Sample Text         : \"{sample_text}\"")
    print(f"input_ids shape     : {input_ids.shape}")
    print(f"attention_mask shape: {attention_mask.shape}")
    print(f"logits shape        : {logits.shape}")

    # LOGITS EXPLANATION:
    # Logits are the raw, unnormalized real-valued output scores ((-inf, +inf)) from the linear classification head.
    # They represent relative confidence scores before any activation function.
    # They are NOT probabilities yet. For multi-label classification, Sigmoid σ(z) is applied to convert each logit into a [0, 1] probability.
    print(f"\nRaw Logits (Raw Unnormalized Scores for 28 Emotion Classes):")
    print(logits)


if __name__ == "__main__":
    demo_model_setup()
