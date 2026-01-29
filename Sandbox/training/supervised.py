"""
Supervised Fine-Tuning (SFT) Training Loop

Standard language model training with batch selection support.
Used as baseline for comparison with RL alignment methods.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List
import time
from dataclasses import dataclass, field
from tqdm import tqdm

import sys
sys.path.append("..")
from batch_selection.base import BatchSelector, NoSelectionWrapper


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    weight_decay: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 10


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    train_loss: List[float] = field(default_factory=list)
    eval_loss: List[float] = field(default_factory=list)
    samples_per_step: List[int] = field(default_factory=list)
    wall_time_per_step: List[float] = field(default_factory=list)
    selection_overhead: List[float] = field(default_factory=list)
    total_samples_seen: int = 0
    total_wall_time: float = 0.0


class SupervisedTrainer:
    """
    Supervised fine-tuning trainer with batch selection support.

    Supports plugging in different batch selection strategies to study
    their impact on training efficiency and final performance.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: TrainingConfig,
        batch_selector: Optional[BatchSelector] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.batch_selector = batch_selector or NoSelectionWrapper()
        self.device = config.device

        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

        # Metrics
        self.metrics = TrainingMetrics()

    def compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-sample loss for a batch."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Per-token loss
        loss_per_token = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.size())

        # Mask padding tokens
        mask = (shift_labels != -100).float()
        loss_per_sample = (loss_per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

        return loss_per_sample

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
    ) -> Dict[str, float]:
        """Single training step with batch selection."""
        step_start = time.time()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Batch selection
        selection_start = time.time()
        selection_result = self.batch_selector.select(
            self.model,
            batch,
            compute_loss_fn=self.compute_loss,
            sample_indices=batch.get("index"),
        )
        selection_time = time.time() - selection_start

        # Select samples
        selected_indices = selection_result.indices
        selected_batch = {
            k: v[selected_indices] if isinstance(v, torch.Tensor) and v.dim() > 0 else v
            for k, v in batch.items()
        }

        # Forward pass
        self.model.train()
        losses = self.compute_loss(self.model, selected_batch)
        loss = losses.mean()

        # Backward pass
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        # Gradient clipping and optimizer step
        if (step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Update batch selector state
        self.batch_selector.update(
            selected_indices,
            losses.detach(),
            sample_indices=batch.get("index"),
        )

        step_time = time.time() - step_start

        # Record metrics
        self.metrics.train_loss.append(loss.item() * self.config.gradient_accumulation_steps)
        self.metrics.samples_per_step.append(len(selected_indices))
        self.metrics.wall_time_per_step.append(step_time)
        self.metrics.selection_overhead.append(selection_time)
        self.metrics.total_samples_seen += len(selected_indices)

        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "samples": len(selected_indices),
            "selection_time": selection_time,
            "step_time": step_time,
        }

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> TrainingMetrics:
        """Full training loop."""
        total_start = time.time()
        global_step = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_samples = 0

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch in pbar:
                step_metrics = self.train_step(batch, global_step)
                global_step += 1

                epoch_loss += step_metrics["loss"]
                epoch_samples += step_metrics["samples"]

                if global_step % self.config.log_interval == 0:
                    pbar.set_postfix({
                        "loss": f"{step_metrics['loss']:.4f}",
                        "samples": step_metrics["samples"],
                        "sel_time": f"{step_metrics['selection_time']*1000:.1f}ms",
                    })

            # Epoch end
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}, samples={epoch_samples}")

            # Evaluation
            if eval_dataloader is not None:
                eval_loss = self.evaluate(eval_dataloader)
                self.metrics.eval_loss.append(eval_loss)
                print(f"  Eval loss: {eval_loss:.4f}")

            # Notify batch selector of epoch end
            if hasattr(self.batch_selector, 'on_epoch_end'):
                self.batch_selector.on_epoch_end()

        self.metrics.total_wall_time = time.time() - total_start
        return self.metrics

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> float:
        """Evaluate model on dataloader."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                losses = self.compute_loss(self.model, batch)
                total_loss += losses.sum().item()
                total_samples += len(losses)

        self.model.train()
        return total_loss / total_samples

    def get_efficiency_stats(self) -> Dict[str, Any]:
        """Get training efficiency statistics."""
        selector_stats = self.batch_selector.get_overhead_stats()

        return {
            "total_wall_time": self.metrics.total_wall_time,
            "total_samples_seen": self.metrics.total_samples_seen,
            "avg_loss": sum(self.metrics.train_loss) / len(self.metrics.train_loss) if self.metrics.train_loss else 0,
            "avg_samples_per_step": sum(self.metrics.samples_per_step) / len(self.metrics.samples_per_step) if self.metrics.samples_per_step else 0,
            "avg_selection_overhead": sum(self.metrics.selection_overhead) / len(self.metrics.selection_overhead) if self.metrics.selection_overhead else 0,
            "total_selection_overhead": sum(self.metrics.selection_overhead),
            "selection_overhead_fraction": sum(self.metrics.selection_overhead) / self.metrics.total_wall_time if self.metrics.total_wall_time > 0 else 0,
            **selector_stats,
        }
