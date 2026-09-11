"""
Sigmoid Activation Demo (Phase 4, Step 2)

This script demonstrates how PyTorch's torch.sigmoid() function converts raw real-valued
logits (scores ranging from -infinity to +infinity) into independent probabilities in the range [0, 1].
"""

import torch


def demo_sigmoid():
    # Step 1: Create a sample tensor containing example logits [-2.0, -1.0, 0.0, 1.0, 2.0]
    logits = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Step 2: Apply torch.sigmoid() function
    # Sigmoid formula: σ(z) = 1 / (1 + e^(-z))
    probabilities = torch.sigmoid(logits)

    # Step 3: Display original logits and corresponding sigmoid probabilities
    print("\n--- Sigmoid Activation Demo ---")
    print(f"Original Logits      : {logits.tolist()}")
    print(f"Sigmoid Probabilities: {[round(p, 4) for p in probabilities.tolist()]}")

    print("\n--- Logit to Probability Mapping ---")
    for logit, prob in zip(logits.tolist(), probabilities.tolist()):
        print(f"  Logit: {logit:>5.1f}  ==>  Probability: {prob:6.4f}")

    # MULTI-LABEL CONCEPT EXPLANATION:
    # 1. Sigmoid maps any real number from (-infinity, +infinity) strictly into the range [0.0, 1.0].
    # 2. In multi-label classification (GoEmotions), Sigmoid is applied INDEPENDENTLY to each of the 28 output logits.
    # 3. Unlike Softmax (which forces all class probabilities to sum to 1.0), independent Sigmoid activation
    #    allows multiple emotions (e.g., 'joy' = 0.85 and 'excitement' = 0.78) to be active simultaneously.


if __name__ == "__main__":
    demo_sigmoid()
