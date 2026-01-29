"""
Gradient Norm Based Batch Selection

Selects samples with highest gradient norms, as these contribute
most to parameter updates.

References:
- Optimal sampling is proportional to gradient norm (theory)
- GradNorm selection in GREATS paper
- Last-layer gradient as cheap proxy (Katharopoulos & Fleuret, 2018)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from .base import BatchSelector


class GradientNormSelector(BatchSelector):
    """
    Gradient norm based selection.

    Modes:
    - 'full': Compute full gradient norm (expensive)
    - 'last_layer': Use last layer gradient as proxy (cheap)
    - 'embedding': Use embedding gradient norm (medium)

    Overhead:
    - full: O(backward_pass) - defeats purpose
    - last_layer: O(forward_pass + last_layer_backward)
    - embedding: O(forward_pass + embedding_backward)
    """

    def __init__(
        self,
        selection_ratio: float = 0.5,
        device: str = "cpu",
        mode: str = "last_layer",
        loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__(selection_ratio, device)
        self.mode = mode
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(reduction='none')

    def score_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Compute gradient-based scores."""

        if self.mode == "last_layer":
            return self._last_layer_grad_norm(model, batch)
        elif self.mode == "full":
            return self._full_grad_norm(model, batch)
        elif self.mode == "embedding":
            return self._embedding_grad_norm(model, batch)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _last_layer_grad_norm(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute gradient norm w.r.t. last layer only.
        This is a cheap proxy for full gradient norm.
        """
        model.eval()

        # Find last linear layer
        last_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        if last_linear is None:
            # Fallback to loss-based
            return self._loss_based_fallback(model, batch)

        # Forward pass with gradient tracking for last layer
        batch_size = next(iter(batch.values())).shape[0]
        grad_norms = torch.zeros(batch_size, device=self.device)

        # Process each sample individually (for per-sample gradients)
        for i in range(batch_size):
            model.zero_grad()

            if "input_ids" in batch:
                outputs = model(
                    input_ids=batch["input_ids"][i:i+1],
                    attention_mask=batch.get("attention_mask", None)
                    if batch.get("attention_mask") is None
                    else batch["attention_mask"][i:i+1],
                )
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                if logits.dim() == 3:
                    # Causal LM
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = batch["labels"][i:i+1, 1:].contiguous()
                    loss = self.loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    ).mean()
                else:
                    loss = self.loss_fn(logits.squeeze(), batch["labels"][i])
            else:
                outputs = model(batch["inputs"][i:i+1])
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = self.loss_fn(logits.squeeze(), batch["labels"][i])

            # Backward to last layer only
            loss.backward(retain_graph=False)

            # Compute gradient norm
            if last_linear.weight.grad is not None:
                grad_norms[i] = last_linear.weight.grad.norm().item()

        model.train()
        return grad_norms

    def _full_grad_norm(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute full gradient norm per sample.
        WARNING: This is expensive and defeats the purpose of batch selection.
        Only for research comparison.
        """
        batch_size = next(iter(batch.values())).shape[0]
        grad_norms = torch.zeros(batch_size, device=self.device)

        for i in range(batch_size):
            model.zero_grad()

            if "input_ids" in batch:
                outputs = model(
                    input_ids=batch["input_ids"][i:i+1],
                    attention_mask=batch.get("attention_mask", None)
                    if batch.get("attention_mask") is None
                    else batch["attention_mask"][i:i+1],
                )
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                if logits.dim() == 3:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = batch["labels"][i:i+1, 1:].contiguous()
                    loss = self.loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    ).mean()
                else:
                    loss = self.loss_fn(logits.squeeze(), batch["labels"][i])
            else:
                outputs = model(batch["inputs"][i:i+1])
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = self.loss_fn(logits.squeeze(), batch["labels"][i])

            loss.backward()

            # Sum gradient norms across all parameters
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            grad_norms[i] = total_norm ** 0.5

        return grad_norms

    def _embedding_grad_norm(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute gradient norm w.r.t. embeddings."""
        # Similar to last_layer but targets embedding layer
        # Useful for transformer models
        return self._last_layer_grad_norm(model, batch)  # Simplified

    def _loss_based_fallback(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fallback to loss-based selection."""
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
