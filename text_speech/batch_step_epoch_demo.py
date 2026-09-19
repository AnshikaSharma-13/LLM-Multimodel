"""
Batch, Step, and Epoch Demonstration (Phase 4, Step 8)

This script demonstrates the relationship between batches, training steps, and epochs
using a simple 12-example dataset and PyTorch DataLoader.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset


def demo_batch_step_epoch():
    # Step 1: Create a simple dataset containing 12 examples (numbers 1 to 12)
    examples = torch.tensor(list(range(1, 13)))
    dataset = TensorDataset(examples)

    # Step 2: Define training hyperparameters
    batch_size = 3
    num_epochs = 2

    # Step 3: Create PyTorch DataLoader to divide 12 examples into batches of 3
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_examples = len(examples)
    batches_per_epoch = len(dataloader)  # 12 / 3 = 4 batches per epoch

    # Step 4: Track global step counter across epochs
    global_step = 0

    print("--- Batch, Step, and Epoch Demonstration ---\n")

    # Step 5: Loop over epochs and batches
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}:")

        for batch_idx, (batch_data,) in enumerate(dataloader, start=1):
            global_step += 1
            batch_list = batch_data.tolist()
            print(f"  Batch {batch_idx} -> Step {global_step:<2} | Batch Examples: {batch_list}")

        print()  # Spacer line between epochs

    # Step 6: Print summary metrics
    print("--- Training Metrics Summary ---")
    print(f"total examples       = {total_examples}")
    print(f"batch size           = {batch_size}")
    print(f"batches per epoch    = {batches_per_epoch}")
    print(f"number of epochs     = {num_epochs}")
    print(f"total training steps = {global_step}")

    # EDUCATIONAL TERMINOLOGY EXPLANATION:
    # - Batch = group of examples processed together in a single forward/backward pass
    # - Step = one training/parameter-update iteration (1 step per batch)
    # - Epoch = one complete pass through all examples in the dataset


if __name__ == "__main__":
    demo_batch_step_epoch()
