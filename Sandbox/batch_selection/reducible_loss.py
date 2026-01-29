"""
Reducible Loss (RHO-Loss) Based Selection

Selects samples where the model can still improve, filtering out:
1. Already learned examples (low loss)
2. Noisy/mislabeled examples (consistently high loss)

Uses a reference model to identify "reducible" loss.

References:
- RHO-Loss: Mindermann et al., 2022
- "Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt"
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from .base import BatchSelector


class ReducibleLossSelector(BatchSelector):
    """
    RHO-Loss style selection based on reducible loss.

    Reducible loss = current_loss - reference_loss

    Where reference_loss comes from:
    - A smaller/simpler model trained on same data
    - The same model from earlier in training
    - A running EMA of per-sample losses

    This filters out both easy examples (low reducible loss)
    and noisy examples (high loss but also high reference loss).

    Overhead: Depends on reference model strategy
    - EMA baseline: O(1) lookup
    - Separate model: O(forward_pass of reference model)
    """

    def __init__(
        self,
        selection_ratio: float = 0.5,
        device: str = "cpu",
        reference_mode: str = "ema",
        ema_decay: float = 0.99,
        reference_model: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
    ):
        """
        Args:
            selection_ratio: Fraction of batch to select
            device: Computation device
            reference_mode: 'ema', 'checkpoint', or 'model'
            ema_decay: Decay for EMA baseline (if mode='ema')
            reference_model: Separate reference model (if mode='model')
            loss_fn: Loss function for computing losses
        """
        super().__init__(selection_ratio, device)
        self.reference_mode = reference_mode
        self.ema_decay = ema_decay
        self.reference_model = reference_model
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(reduction='none')

        # EMA baseline for per-sample losses
        self.loss_baseline: Dict[int, float] = {}

        # For checkpoint mode
        self.checkpoint_losses: Dict[int, float] = {}
        self._checkpoint_step = 0

    def score_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        sample_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute reducible loss scores.

        Reducible loss = current_loss - baseline_loss
        Higher reducible loss = more room for improvement
        """
        batch_size = next(iter(batch.values())).shape[0]

        # Compute current losses
        current_losses = self._compute_losses(model, batch)

        # Get baseline losses
        if self.reference_mode == "ema":
            baseline_losses = self._get_ema_baseline(sample_indices, batch_size)
        elif self.reference_mode == "checkpoint":
            baseline_losses = self._get_checkpoint_baseline(sample_indices, batch_size)
        elif self.reference_mode == "model":
            baseline_losses = self._compute_reference_losses(batch)
        else:
            raise ValueError(f"Unknown reference_mode: {self.reference_mode}")

        # Reducible loss
        reducible = current_losses - baseline_losses

        # Clamp to avoid negative scores (already better than baseline)
        reducible = torch.clamp(reducible, min=0.0)

        return reducible

    def _compute_losses(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-sample losses."""
        with torch.no_grad():
            if "input_ids" in batch:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                )
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

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
                outputs = model(batch["inputs"])
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                losses = self.loss_fn(logits, batch["labels"])

        return losses

    def _get_ema_baseline(
        self,
        sample_indices: Optional[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """Get EMA baseline losses."""
        baseline = torch.zeros(batch_size, device=self.device)

        if sample_indices is None:
            return baseline

        for i, idx in enumerate(sample_indices.tolist()):
            baseline[i] = self.loss_baseline.get(idx, 0.0)

        return baseline

    def _get_checkpoint_baseline(
        self,
        sample_indices: Optional[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """Get checkpoint baseline losses."""
        baseline = torch.zeros(batch_size, device=self.device)

        if sample_indices is None:
            return baseline

        for i, idx in enumerate(sample_indices.tolist()):
            baseline[i] = self.checkpoint_losses.get(idx, 0.0)

        return baseline

    def _compute_reference_losses(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute losses using reference model."""
        if self.reference_model is None:
            batch_size = next(iter(batch.values())).shape[0]
            return torch.zeros(batch_size, device=self.device)

        return self._compute_losses(self.reference_model, batch)

    def update(
        self,
        indices: torch.Tensor,
        losses: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Update EMA baseline with new losses."""
        if sample_indices is None or self.reference_mode != "ema":
            return

        for i, batch_idx in enumerate(indices.tolist()):
            if batch_idx < len(sample_indices):
                sample_idx = sample_indices[batch_idx].item()
                loss_val = losses[i].item() if i < len(losses) else losses.mean().item()

                if sample_idx in self.loss_baseline:
                    self.loss_baseline[sample_idx] = (
                        self.ema_decay * self.loss_baseline[sample_idx] +
                        (1 - self.ema_decay) * loss_val
                    )
                else:
                    self.loss_baseline[sample_idx] = loss_val

    def save_checkpoint(
        self,
        model: nn.Module,
        dataloader,
    ) -> None:
        """
        Save current model's per-sample losses as checkpoint baseline.
        Call this periodically during training.
        """
        self.checkpoint_losses.clear()

        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                losses = self._compute_losses(model, batch)
                sample_indices = batch.get("indices", torch.arange(len(losses)) + batch_idx * len(losses))

                for i, idx in enumerate(sample_indices.tolist()):
                    self.checkpoint_losses[idx] = losses[i].item()
        model.train()

        self._checkpoint_step += 1
