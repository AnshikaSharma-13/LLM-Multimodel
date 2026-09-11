"""
BCEWithLogitsLoss Demo (Phase 4, Step 3)

This script demonstrates PyTorch's BCEWithLogitsLoss function for multi-label classification.
It compares loss values when the model's prediction aligns with vs. contradicts the true label.
"""

import torch
import torch.nn as nn


def demo_bce_loss():
    # Step 1: Create true target label (1.0 = emotion is present)
    # Label 1.0 means the emotion is present in the example text.
    target_label = torch.tensor([1.0])

    # Step 2: Create two candidate logits:
    # good_logit (+3.0): Model predicts emotion IS present (aligned with label 1.0)
    # bad_logit  (-3.0): Model predicts emotion is ABSENT (conflicts with label 1.0)
    good_logit = torch.tensor([3.0])
    bad_logit = torch.tensor([-3.0])

    # Step 3: Create the loss function
    # BCEWithLogitsLoss combines Sigmoid activation and Binary Cross-Entropy loss internally for numerical stability.
    loss_fn = nn.BCEWithLogitsLoss()

    # Step 4: Calculate loss for both predictions
    good_loss = loss_fn(good_logit, target_label)
    bad_loss = loss_fn(bad_logit, target_label)

    # Step 5: Convert logits to probabilities using torch.sigmoid() for inspection
    # Positive logits produce higher probabilities (> 0.5)
    # Negative logits produce lower probabilities (< 0.5)
    good_prob = torch.sigmoid(good_logit)
    bad_prob = torch.sigmoid(bad_logit)

    # Step 6: Display outputs and loss comparison
    print("\n--- PyTorch BCEWithLogitsLoss Demo ---")
    print(f"True Target Label : {target_label.item()} (Emotion Present)\n")

    print(f"Case 1: Good Prediction (Aligned)")
    print(f"  Logit          : {good_logit.item():>5.1f}")
    print(f"  Probability    : {good_prob.item():>6.4f}")
    print(f"  BCE Loss       : {good_loss.item():>6.4f}  (Lower loss = prediction is closer to target 1.0)\n")

    print(f"Case 2: Bad Prediction (Contradictory)")
    print(f"  Logit          : {bad_logit.item():>5.1f}")
    print(f"  Probability    : {bad_prob.item():>6.4f}")
    print(f"  BCE Loss       : {bad_loss.item():>6.4f}  (Higher loss = severe penalty for incorrect prediction)\n")

    # CONCEPT SUMMARY:
    # - Label 1.0 means the emotion is present in the text.
    # - Positive logits produce higher probabilities (> 0.5).
    # - Negative logits produce lower probabilities (< 0.5).
    # - BCEWithLogitsLoss combines sigmoid and binary cross-entropy internally for speed and numerical precision.
    # - A lower loss means the model's predicted probability is closer to the correct label target.


if __name__ == "__main__":
    demo_bce_loss()
