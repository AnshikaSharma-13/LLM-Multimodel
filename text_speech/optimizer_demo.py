"""
AdamW Optimizer Demo (Phase 4, Step 5)

This script demonstrates how PyTorch's AdamW optimizer uses computed gradients
and a learning rate hyperparameter to update a model parameter to minimize loss.
"""

import torch
from torch.optim import AdamW


def demo_optimizer():
    # Step 1: Create a scalar parameter with requires_grad=True
    x = torch.tensor(2.0, requires_grad=True)

    # Step 2: Instantiate AdamW optimizer with learning rate = 0.1
    # The learning rate controls the step size of each parameter update.
    optimizer = AdamW([x], lr=0.1)

    # Step 3: Calculate initial loss (loss = x^2)
    initial_loss = x ** 2

    print("\n--- AdamW Optimizer Step Demo ---")
    print(f"Initial Parameter x : {x.item():.4f}")
    print(f"Initial Loss (x²)   : {initial_loss.item():.4f}\n")

    # Step 4: Clear any old gradients stored in parameters
    # `optimizer.zero_grad()` clears old accumulated gradients before computing new ones.
    optimizer.zero_grad()

    # Step 5: Compute gradient via backpropagation
    # `loss.backward()` calculates the gradient of the loss with respect to x.
    initial_loss.backward()
    print(f"Calculated x.grad   : {x.grad.item():.4f}")

    # Step 6: Update parameter x using AdamW step
    # `optimizer.step()` tells AdamW to update parameter x using its gradient and learning rate.
    optimizer.step()

    # Step 7: Calculate and print new loss with updated x
    new_loss = x ** 2
    print(f"\nUpdated Parameter x : {x.item():.4f}")
    print(f"New Loss (x²)       : {new_loss.item():.4f}")

    # CONCEPT RECAP:
    # - zero_grad() clears old gradients so they do not accumulate across iterations.
    # - backward() calculates the gradient of the loss with respect to x.
    # - step() tells AdamW to update the parameter using the computed gradient.
    # - The learning rate (lr=0.1) controls the size of updates.
    # - The goal of the optimizer step is to adjust x to reduce the loss.


if __name__ == "__main__":
    demo_optimizer()
