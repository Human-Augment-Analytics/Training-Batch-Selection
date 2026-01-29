"""
Random Batch Selection - Baseline

This is the standard approach: randomly sample from the batch.
Used as baseline for comparison with informed selection methods.
"""

import torch
import torch.nn as nn
from typing import Dict
from .base import BatchSelector


class RandomSelector(BatchSelector):
    """
    Random selection baseline.

    Assigns uniform random scores to all samples.
    Equivalent to standard random mini-batch SGD.

    Overhead: O(batch_size) - negligible
    """

    def __init__(self, selection_ratio: float = 1.0, device: str = "cpu"):
        super().__init__(selection_ratio, device)

    def score_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Return uniform random scores."""
        batch_size = next(iter(batch.values())).shape[0]
        return torch.rand(batch_size, device=self.device)
