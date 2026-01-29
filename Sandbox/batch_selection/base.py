"""
Base class for batch selection strategies.

All batch selectors follow a common interface:
1. score_batch() - Score samples in a batch for informativeness
2. select() - Select top-k samples based on scores
3. update() - Update internal state after training step (e.g., loss history)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn


@dataclass
class SelectionResult:
    """Result of batch selection."""
    indices: torch.Tensor  # Selected sample indices
    scores: torch.Tensor   # Scores for selected samples
    metadata: Dict[str, Any] = None  # Optional metadata (e.g., timing info)


class BatchSelector(ABC):
    """Abstract base class for batch selection strategies."""

    def __init__(
        self,
        selection_ratio: float = 0.5,
        device: str = "cpu",
    ):
        """
        Args:
            selection_ratio: Fraction of batch to select (0.0 to 1.0)
            device: Device for computations
        """
        self.selection_ratio = selection_ratio
        self.device = device
        self._step_count = 0
        self._overhead_time = 0.0  # Track selection overhead

    @abstractmethod
    def score_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """
        Score each sample in the batch for informativeness.

        Args:
            model: The model being trained
            batch: Dict with 'input_ids', 'labels', etc.

        Returns:
            Tensor of scores, shape (batch_size,)
        """
        pass

    def select(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        **kwargs,
    ) -> SelectionResult:
        """
        Select top samples from batch based on scores.

        Args:
            model: The model being trained
            batch: Input batch

        Returns:
            SelectionResult with selected indices and scores
        """
        import time
        start = time.time()

        scores = self.score_batch(model, batch, **kwargs)
        batch_size = scores.shape[0]
        k = max(1, int(batch_size * self.selection_ratio))

        # Select top-k by score
        top_scores, top_indices = torch.topk(scores, k, largest=True)

        self._overhead_time += time.time() - start
        self._step_count += 1

        return SelectionResult(
            indices=top_indices,
            scores=top_scores,
            metadata={"k": k, "batch_size": batch_size}
        )

    def update(
        self,
        indices: torch.Tensor,
        losses: torch.Tensor,
        **kwargs,
    ) -> None:
        """
        Update internal state after training on selected samples.
        Override in subclasses that maintain state (e.g., loss history).

        Args:
            indices: Indices of samples that were trained on
            losses: Per-sample losses from training
        """
        pass

    def get_overhead_stats(self) -> Dict[str, float]:
        """Return timing statistics for overhead analysis."""
        return {
            "total_overhead_seconds": self._overhead_time,
            "steps": self._step_count,
            "avg_overhead_per_step": self._overhead_time / max(1, self._step_count),
        }

    def reset_stats(self) -> None:
        """Reset timing statistics."""
        self._step_count = 0
        self._overhead_time = 0.0


class NoSelectionWrapper(BatchSelector):
    """Wrapper that returns all samples (baseline for comparison)."""

    def score_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        batch_size = next(iter(batch.values())).shape[0]
        return torch.ones(batch_size, device=self.device)

    def select(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        **kwargs,
    ) -> SelectionResult:
        batch_size = next(iter(batch.values())).shape[0]
        return SelectionResult(
            indices=torch.arange(batch_size, device=self.device),
            scores=torch.ones(batch_size, device=self.device),
            metadata={"k": batch_size, "batch_size": batch_size}
        )
