"""
Evaluation utilities for vision tasks.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.base import DEVICE, NUM_WORKERS, PIN_MEMORY


def evaluate(model, dataset, batch_size=256):
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model to evaluate
        dataset: Dataset to evaluate on
        batch_size: Batch size for evaluation (can be larger than training)

    Returns:
        tuple: (accuracy, loss)
    """
    # Use larger batch for eval + optimizations for GPU
    loader = DataLoader(dataset, batch_size=batch_size,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    model.eval()
    correct, n, total_loss = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            # Move to device (GPU if available)
            x = x.view(x.size(0), -1).to(DEVICE, non_blocking=PIN_MEMORY)
            y = y.to(DEVICE, non_blocking=PIN_MEMORY)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item() * x.size(0)
            correct += (y_pred.argmax(1) == y).sum().item()
            n += x.size(0)

    return correct / n, total_loss / n
