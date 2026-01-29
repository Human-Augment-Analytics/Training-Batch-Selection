"""
Loss-Based Batch Selection

Prioritizes samples with high loss (hard examples).
Variants:
1. Current loss: Compute forward pass, select high-loss samples
2. Historical loss: Use cached losses from previous epochs (lower overhead)

References:
- Online Batch Selection (Loshchilov & Hutter, 2015)
- Selective Backprop (Jiang et al., 2019)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from .base import BatchSelector


class LossBasedSelector(BatchSelector):
    """
    Loss-based selection with optional loss caching.

    Modes:
    - 'current': Compute fresh losses each step (1 forward pass overhead)
    - 'cached': Use exponential moving average of historical losses (0 overhead)
    - 'hybrid': Mix of exploration (random) and exploitation (high loss)

    Overhead:
    - current: O(forward_pass)
    - cached: O(1) lookup
    - hybrid: O(1) lookup
    """

    def __init__(
        self,
        selection_ratio: float = 0.5,
        device: str = "cpu",
        mode: str = "cached",
        ema_decay: float = 0.9,
        explore_ratio: float = 0.3,
        dataset_size: Optional[int] = None,
    ):
        """
        Args:
            selection_ratio: Fraction of batch to select
            device: Computation device
            mode: 'current', 'cached', or 'hybrid'
            ema_decay: Decay for exponential moving average of losses
            explore_ratio: Fraction of selection to be random (hybrid mode)
            dataset_size: Size of dataset for loss cache initialization
        """
        super().__init__(selection_ratio, device)
        self.mode = mode
        self.ema_decay = ema_decay
        self.explore_ratio = explore_ratio

        # Loss cache: maps sample index -> EMA loss
        self.loss_cache: Dict[int, float] = {}
        self.dataset_size = dataset_size
        self._epoch = 0

    def score_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        compute_loss_fn: Optional[callable] = None,
        sample_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Score samples by loss value.

        Args:
            model: Model for forward pass (if mode='current')
            batch: Input batch
            compute_loss_fn: Function(model, batch) -> per-sample losses
            sample_indices: Original dataset indices for cache lookup
        """
        batch_size = next(iter(batch.values())).shape[0]

        if self.mode == "current":
            # Compute fresh losses - requires forward pass
            if compute_loss_fn is None:
                raise ValueError("compute_loss_fn required for mode='current'")
            with torch.no_grad():
                losses = compute_loss_fn(model, batch)
            return losses

        elif self.mode == "cached":
            # Use cached losses - zero compute overhead
            if sample_indices is None:
                # Fallback to random if no indices provided
                return torch.rand(batch_size, device=self.device)

            scores = torch.zeros(batch_size, device=self.device)
            for i, idx in enumerate(sample_indices.tolist()):
                # Default to high score for unseen samples (explore them)
                scores[i] = self.loss_cache.get(idx, float('inf'))
            return scores

        elif self.mode == "hybrid":
            # Mix exploration and exploitation
            scores = torch.zeros(batch_size, device=self.device)

            if sample_indices is not None:
                for i, idx in enumerate(sample_indices.tolist()):
                    scores[i] = self.loss_cache.get(idx, float('inf'))

            # Add exploration noise
            noise = torch.rand(batch_size, device=self.device)
            explore_mask = noise < self.explore_ratio
            scores[explore_mask] = float('inf')  # Force selection of explore samples

            return scores

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def update(
        self,
        indices: torch.Tensor,
        losses: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Update loss cache with new loss values."""
        if sample_indices is None:
            return

        for i, (batch_idx, sample_idx) in enumerate(
            zip(indices.tolist(), sample_indices[indices].tolist())
        ):
            loss_val = losses[i].item() if i < len(losses) else losses.mean().item()

            if sample_idx in self.loss_cache:
                # Exponential moving average
                self.loss_cache[sample_idx] = (
                    self.ema_decay * self.loss_cache[sample_idx] +
                    (1 - self.ema_decay) * loss_val
                )
            else:
                self.loss_cache[sample_idx] = loss_val

    def on_epoch_end(self) -> None:
        """Called at end of epoch."""
        self._epoch += 1


class SelectiveBackpropSelector(BatchSelector):
    """
    Selective Backpropagation (Jiang et al., 2019)

    Forward pass on full batch, backward only on high-loss samples.
    This is more efficient than re-sampling because forward pass
    is typically cheaper than backward pass.

    Overhead: O(forward_pass) - but saves on backward pass
    """

    def __init__(
        self,
        selection_ratio: float = 0.5,
        device: str = "cpu",
        loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__(selection_ratio, device)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(reduction='none')

    def score_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Compute forward pass and return per-sample losses."""
        model.eval()
        with torch.no_grad():
            if "input_ids" in batch:
                # Language model
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                )
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                # Shift for causal LM
                if logits.dim() == 3:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = batch["labels"][..., 1:].contiguous()
                    losses = self.loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    ).view(shift_labels.size()).mean(dim=-1)
                else:
                    losses = self.loss_fn(logits, batch["labels"])
            else:
                # Standard classifier
                outputs = model(batch["inputs"])
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                losses = self.loss_fn(logits, batch["labels"])

        model.train()
        return losses
