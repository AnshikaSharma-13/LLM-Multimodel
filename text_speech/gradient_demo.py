"""
PyTorch Autograd & Gradient Demo (Phase 4, Step 4)

This script demonstrates how PyTorch's automatic differentiation engine (autograd)
tracks tensor operations, computes derivatives using .backward(), and stores gradients in .grad.
"""

import torch


def demo_gradient():
    # Step 1: Create a scalar tensor parameter with requires_grad=True
    # `requires_grad=True` tells PyTorch to track operations involving x in a computational graph.
    x = torch.tensor(2.0, requires_grad=True)

    # Step 2: Define a simple loss function: loss = x^2
    loss = x ** 2

    # Print input parameter x and computed loss
    print("\n--- PyTorch Autograd & Gradient Demo ---")
    print(f"Parameter x Value : {x.item()}")
    print(f"Computed Loss (x²): {loss.item()}")

    # Step 3: Backpropagate to calculate gradients
    # `loss.backward()` calculates the gradient of the loss with respect to x (d(loss)/dx).
    loss.backward()

    # Step 4: Inspect the computed gradient stored in x.grad
    print(f"Calculated x.grad : {x.grad.item()}")

    # MATHEMATICAL & CONCEPTUAL EXPLANATION:
    # 1. requires_grad=True tells PyTorch to track operations involving x.
    # 2. loss.backward() calculates the gradient of the loss with respect to x.
    # 3. x.grad contains that gradient.
    # 4. For loss = x², the derivative is d/dx(x²) = 2x.
    #    When x = 2.0, the gradient is 2 * (2.0) = 4.0.


if __name__ == "__main__":
    demo_gradient()
