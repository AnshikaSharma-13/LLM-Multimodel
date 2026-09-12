"""
Parameter Update Demonstration (Phase 4, Step 7)

This script demonstrates that calling optimizer.step() actually mutates and updates
the weights of the DistilBERT model classification head based on computed gradients.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, DistilBertForSequenceClassification


def demo_parameter_update():
    model_name = "distilbert-base-uncased"
    num_labels = 28
    problem_type = "multi_label_classification"

    print("--- 1. Loading Model & Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
    )
    model.train()

    # Step 2: Define sample text & tokenize
    sample_text = "I am feeling nervous today."
    inputs = tokenizer(
        sample_text,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Step 3: Create 28-dimensional target tensor with nervousness (label ID 19) = 1.0, others = 0.0
    target_labels = torch.zeros((1, num_labels), dtype=torch.float32)
    target_labels[0, 19] = 1.0  # nervousness label ID 19

    # Step 4: Setup AdamW optimizer & BCEWithLogitsLoss
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    # Step 5: Inspect model.classifier.weight before optimization
    classifier_weight = model.classifier.weight
    weight_before = classifier_weight.detach().clone()

    print("\n--- 2. Parameter Inspection BEFORE Update ---")
    print(f"Inspected Parameter: model.classifier.weight (Shape: {classifier_weight.shape})")
    print("First 5 weight values BEFORE step():")
    print(weight_before.flatten()[:5].tolist())

    # Step 6: Forward pass, loss computation, and backpropagation
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = loss_fn(outputs.logits, target_labels)
    loss.backward()

    # Step 7: Print gradient values before optimizer.step()
    print("\n--- 3. Gradient Inspection BEFORE optimizer.step() ---")
    grad_tensor = classifier_weight.grad
    print("First 5 gradient values of model.classifier.weight:")
    print(grad_tensor.flatten()[:5].tolist())

    # Step 8: Execute optimizer.step() to update parameters
    optimizer.step()

    # Step 9: Inspect model.classifier.weight AFTER optimization
    weight_after = classifier_weight.detach().clone()

    print("\n--- 4. Parameter Inspection AFTER optimizer.step() ---")
    print("First 5 weight values AFTER step():")
    print(weight_after.flatten()[:5].tolist())

    # Step 10: Verify parameter changes
    are_equal = torch.equal(weight_before, weight_after)
    max_abs_diff = torch.abs(weight_after - weight_before).max().item()

    print("\n--- 5. Parameter Change Verification ---")
    print(f"Are parameter weights equal before and after step()? : {are_equal}")
    print(f"Maximum absolute parameter change (Max Abs Diff)   : {max_abs_diff:.8f}")

    # Educational summary explanation
    print("\n--- Educational Concept Explanation ---")
    print("1. backward() calculates gradients for all trainable parameters based on loss error.")
    print("2. optimizer.step() uses those gradients and the learning rate (5e-5) to update parameter values.")
    print("3. Therefore, the parameter values before and after step() differ (torch.equal is False).")


if __name__ == "__main__":
    demo_parameter_update()
