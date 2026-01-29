"""
Uncertainty-Based Batch Selection

Selects samples where model is most uncertain, measured by:
1. Prediction entropy
2. Margin (difference between top-2 predictions)
3. Least confidence (1 - max probability)

These are VERY cheap - only require forward pass logits.

References:
- Active learning literature
- Uncertainty sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .base import BatchSelector


class UncertaintySelector(BatchSelector):
    """
    Uncertainty-based selection using prediction confidence.

    Modes:
    - 'entropy': Shannon entropy of prediction distribution
    - 'margin': Difference between top-2 class probabilities
    - 'least_confidence': 1 - max(probability)
    - 'variance': For sequence models, variance across tokens

    Overhead: O(forward_pass) - reuses logits from training

    This is one of the CHEAPEST informed selection methods
    because it only requires the forward pass logits.
    """

    def __init__(
        self,
        selection_ratio: float = 0.5,
        device: str = "cpu",
        mode: str = "entropy",
        temperature: float = 1.0,
    ):
        """
        Args:
            selection_ratio: Fraction of batch to select
            device: Computation device
            mode: 'entropy', 'margin', 'least_confidence', or 'variance'
            temperature: Softmax temperature (higher = more uniform)
        """
        super().__init__(selection_ratio, device)
        self.mode = mode
        self.temperature = temperature

    def score_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        logits: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute uncertainty scores.

        Args:
            model: Model (used if logits not provided)
            batch: Input batch
            logits: Pre-computed logits (to avoid redundant forward pass)
        """
        if logits is None:
            with torch.no_grad():
                if "input_ids" in batch:
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                    )
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                else:
                    outputs = model(batch["inputs"])
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # Handle sequence models (batch, seq_len, vocab)
        if logits.dim() == 3:
            # Average uncertainty across sequence
            return self._sequence_uncertainty(logits)
        else:
            return self._classification_uncertainty(logits)

    def _classification_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty for classification logits (batch, classes)."""
        probs = F.softmax(logits / self.temperature, dim=-1)

        if self.mode == "entropy":
            # Shannon entropy: -sum(p * log(p))
            log_probs = F.log_softmax(logits / self.temperature, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            return entropy

        elif self.mode == "margin":
            # Margin: difference between top-2 probabilities
            # Lower margin = more uncertain
            top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
            if top2.shape[-1] == 2:
                margin = top2[:, 0] - top2[:, 1]
            else:
                margin = top2[:, 0]
            # Invert so higher score = more uncertain
            return 1.0 - margin

        elif self.mode == "least_confidence":
            # Least confidence: 1 - max(p)
            max_prob = probs.max(dim=-1).values
            return 1.0 - max_prob

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _sequence_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty for sequence model logits (batch, seq_len, vocab).
        Returns per-sample uncertainty by aggregating across sequence.
        """
        batch_size, seq_len, vocab_size = logits.shape

        if self.mode == "entropy":
            # Per-token entropy, then mean across sequence
            probs = F.softmax(logits / self.temperature, dim=-1)
            log_probs = F.log_softmax(logits / self.temperature, dim=-1)
            token_entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)
            return token_entropy.mean(dim=-1)  # (batch,)

        elif self.mode == "variance":
            # Variance of per-token max probabilities
            probs = F.softmax(logits / self.temperature, dim=-1)
            max_probs = probs.max(dim=-1).values  # (batch, seq_len)
            return max_probs.var(dim=-1)  # (batch,)

        elif self.mode == "margin":
            probs = F.softmax(logits / self.temperature, dim=-1)
            top2 = torch.topk(probs, k=2, dim=-1).values
            margins = top2[..., 0] - top2[..., 1]  # (batch, seq_len)
            return 1.0 - margins.mean(dim=-1)  # (batch,)

        elif self.mode == "least_confidence":
            probs = F.softmax(logits / self.temperature, dim=-1)
            max_probs = probs.max(dim=-1).values  # (batch, seq_len)
            return 1.0 - max_probs.mean(dim=-1)  # (batch,)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class CheapUncertaintySelector(UncertaintySelector):
    """
    Ultra-cheap uncertainty selection that reuses training forward pass.

    Instead of computing a separate forward pass for selection,
    this hooks into the training loop to reuse logits.

    Overhead: ~0 (reuses existing computation)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_logits = None
        self._cached_batch_id = None

    def cache_logits(self, logits: torch.Tensor, batch_id: int) -> None:
        """Cache logits from training forward pass."""
        self._cached_logits = logits.detach()
        self._cached_batch_id = batch_id

    def score_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Use cached logits if available."""
        if batch_id is not None and batch_id == self._cached_batch_id:
            logits = self._cached_logits
        else:
            logits = None  # Will compute fresh

        return super().score_batch(model, batch, logits=logits, **kwargs)
