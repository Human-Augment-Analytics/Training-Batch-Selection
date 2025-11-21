"""
Training loop for vision tasks.
"""
import numpy as np
import torch
import torch.nn as nn

from config.base import DEVICE, USE_GPU, PIN_MEMORY
from config.vision import MOVING_AVG_DECAY
from tasks.vision.evaluate import evaluate


def train_model(model, train_ds, test_ds, epochs, batch_size, batch_strategy,
                loss_kwargs=None, batch_kwargs=None, seed=None, optimizer_name="Adam"):
    """
    Train a model using a specified batch strategy.

    Args:
        model: PyTorch model to train
        train_ds: Training dataset
        test_ds: Test dataset
        epochs: Number of training epochs
        batch_size: Batch size
        batch_strategy: Batch sampling strategy function
        loss_kwargs: Optional kwargs for loss function
        batch_kwargs: Optional kwargs for batch strategy
        seed: Random seed for reproducibility
        optimizer_name: Name of optimizer to use

    Returns:
        tuple: (train_accs, test_accs, train_losses, test_losses)
    """
    if loss_kwargs is None:
        loss_kwargs = {}
    if batch_kwargs is None:
        batch_kwargs = {}

    # Set random seeds for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if USE_GPU:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Move model to device (GPU if available)
    model = model.to(DEVICE)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters())

    loss_fn = nn.CrossEntropyLoss(**loss_kwargs)

    # Track per-sample loss for smart batching strategies
    per_sample_loss = np.zeros(len(train_ds))
    train_accs, test_accs, train_losses, test_losses = [], [], [], []

    for epoch in range(epochs):
        correct = 0
        n = 0
        running_loss = 0
        model.train()

        # Check if batch strategy needs loss_history parameter
        if (batch_strategy.__name__ == "batch_sampler" and
            "loss_history" in batch_strategy.__code__.co_varnames):
            batch_iter = batch_strategy(train_ds, batch_size,
                                       loss_history=per_sample_loss, **batch_kwargs)
        else:
            batch_iter = batch_strategy(train_ds, batch_size, **batch_kwargs)

        # Iterate through batches
        for idxs in batch_iter:
            # Get batch data - optimized for GPU with non_blocking transfers
            batch_data = [train_ds[i] for i in idxs]
            x = torch.stack([item[0] for item in batch_data]).view(len(idxs), -1)
            y = torch.tensor([item[1] for item in batch_data])

            # Transfer to device (GPU if available, non_blocking for efficiency)
            x = x.to(DEVICE, non_blocking=PIN_MEMORY)
            y = y.to(DEVICE, non_blocking=PIN_MEMORY)

            optimizer.zero_grad()
            y_pred = model(x)
            losses = loss_fn(y_pred, y)

            # Handle both reduction='none' and regular loss
            if len(losses.shape) > 0:
                loss = losses.mean()
            else:
                loss = losses

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (y_pred.argmax(1) == y).sum().item()
            n += x.size(0)

            # Update per-sample loss history (exponential moving average)
            if "loss_history" in batch_strategy.__code__.co_varnames:
                for k, idx in enumerate(idxs):
                    # EMA update: new = alpha * old + (1-alpha) * current
                    per_sample_loss[idx] = (
                        MOVING_AVG_DECAY * per_sample_loss[idx] +
                        (1 - MOVING_AVG_DECAY) * (
                            losses[k].item() if len(losses.shape) > 0 else losses.item()
                        )
                    )

        # Calculate metrics for this epoch
        train_accs.append(correct / n)
        train_losses.append(running_loss / n)
        test_acc, test_loss = evaluate(model, test_ds)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}: train_acc={train_accs[-1]:.4f}, test_acc={test_acc:.4f}")

    return train_accs, test_accs, train_losses, test_losses
