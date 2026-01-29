"""
Direct Preference Optimization (DPO) Training Loop

DPO is a simpler alternative to RLHF that directly optimizes the policy
using preference data, without needing a separate reward model.

Key insight for batch selection:
- DPO loss depends on log probability ratios
- Samples with high preference margin may be more informative
- Samples where policy disagrees with reference may indicate learning opportunity

References:
- Rafailov et al., 2023: "Direct Preference Optimization"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List, Tuple
import time
from dataclasses import dataclass, field
from tqdm import tqdm

import sys
sys.path.append("..")
from batch_selection.base import BatchSelector, NoSelectionWrapper


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    learning_rate: float = 1e-6
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    beta: float = 0.1  # KL penalty coefficient
    label_smoothing: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 10


@dataclass
class DPOMetrics:
    """Metrics for DPO training."""
    train_loss: List[float] = field(default_factory=list)
    chosen_rewards: List[float] = field(default_factory=list)
    rejected_rewards: List[float] = field(default_factory=list)
    reward_margins: List[float] = field(default_factory=list)
    reward_accuracies: List[float] = field(default_factory=list)
    samples_per_step: List[int] = field(default_factory=list)
    selection_overhead: List[float] = field(default_factory=list)
    total_samples_seen: int = 0
    total_wall_time: float = 0.0


class DPOTrainer:
    """
    DPO trainer with batch selection support.

    The key opportunity for batch selection in DPO:
    1. Select samples with high implicit reward margin (chosen - rejected)
    2. Select samples where policy deviates most from reference
    3. Select samples with high uncertainty in preference prediction
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        tokenizer,
        config: DPOConfig,
        batch_selector: Optional[BatchSelector] = None,
    ):
        self.policy = policy_model
        self.reference = reference_model
        self.tokenizer = tokenizer
        self.config = config
        self.batch_selector = batch_selector or NoSelectionWrapper()
        self.device = config.device

        self.policy.to(self.device)
        self.reference.to(self.device)
        self.reference.eval()  # Reference is frozen

        # Optimizer (only policy)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
        )

        self.metrics = DPOMetrics()

    def get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int,
    ) -> torch.Tensor:
        """
        Compute log probabilities of response tokens.

        Only computes log probs for tokens after the prompt.
        """
        with torch.set_grad_enabled(model.training):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Shift for causal LM
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Log softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask prompt and padding
        mask = attention_mask[:, 1:].float()
        # Zero out prompt tokens
        mask[:, :prompt_length-1] = 0.0

        # Sum log probs (only response tokens)
        response_log_probs = (token_log_probs * mask).sum(dim=-1)

        return response_log_probs

    def compute_dpo_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute DPO loss for a batch.

        DPO Loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

        Where log_ratio = log(pi(y|x)) - log(pi_ref(y|x))

        Returns:
            loss: Scalar loss
            info: Dict with per-sample rewards and margins
        """
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_mask = batch["chosen_attention_mask"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.device)
        prompt_length = batch["prompt_length"]

        # Handle variable prompt lengths
        if isinstance(prompt_length, torch.Tensor):
            prompt_length = prompt_length[0].item()

        # Policy log probs
        policy_chosen_logps = self.get_log_probs(
            self.policy, chosen_ids, chosen_mask, prompt_length
        )
        policy_rejected_logps = self.get_log_probs(
            self.policy, rejected_ids, rejected_mask, prompt_length
        )

        # Reference log probs (no gradient)
        with torch.no_grad():
            ref_chosen_logps = self.get_log_probs(
                self.reference, chosen_ids, chosen_mask, prompt_length
            )
            ref_rejected_logps = self.get_log_probs(
                self.reference, rejected_ids, rejected_mask, prompt_length
            )

        # Log ratios
        chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

        # Implicit rewards (for monitoring)
        chosen_rewards = self.config.beta * chosen_log_ratio
        rejected_rewards = self.config.beta * rejected_log_ratio
        reward_margin = chosen_rewards - rejected_rewards

        # DPO loss
        losses = -F.logsigmoid(self.config.beta * (chosen_log_ratio - rejected_log_ratio))

        # Label smoothing
        if self.config.label_smoothing > 0:
            losses = (1 - self.config.label_smoothing) * losses + \
                     self.config.label_smoothing * (-F.logsigmoid(-self.config.beta * (chosen_log_ratio - rejected_log_ratio)))

        info = {
            "chosen_rewards": chosen_rewards.detach(),
            "rejected_rewards": rejected_rewards.detach(),
            "reward_margin": reward_margin.detach(),
            "losses": losses.detach(),
        }

        return losses, info

    def score_batch_for_selection(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Score samples for batch selection.

        Options:
        1. Reward margin (higher margin = clearer preference signal)
        2. Loss value (higher loss = more to learn)
        3. KL divergence from reference (higher = policy has changed more)
        """
        with torch.no_grad():
            losses, info = self.compute_dpo_loss(batch)

        # Use loss as score (higher loss = more informative)
        # Could also use: info["reward_margin"].abs() or -info["reward_margin"]
        return losses

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
    ) -> Dict[str, float]:
        """Single DPO training step with batch selection."""
        step_start = time.time()

        # Batch selection
        selection_start = time.time()

        # Create a compatible batch for the selector
        selector_batch = {
            "input_ids": batch["chosen_input_ids"].to(self.device),
            "attention_mask": batch["chosen_attention_mask"].to(self.device),
            "labels": batch["chosen_input_ids"].to(self.device),
        }

        # For DPO-specific selection, override score_batch
        original_score = self.batch_selector.score_batch
        self.batch_selector.score_batch = lambda m, b, **kw: self.score_batch_for_selection(batch)

        selection_result = self.batch_selector.select(
            self.policy,
            selector_batch,
            sample_indices=batch.get("index"),
        )

        # Restore original score function
        self.batch_selector.score_batch = original_score

        selection_time = time.time() - selection_start

        # Select samples
        selected_indices = selection_result.indices
        selected_batch = {
            k: v[selected_indices] if isinstance(v, torch.Tensor) and v.dim() > 0 else v
            for k, v in batch.items()
        }

        # Forward pass
        self.policy.train()
        losses, info = self.compute_dpo_loss(selected_batch)
        loss = losses.mean()

        # Backward pass
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        # Gradient clipping and optimizer step
        if (step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        step_time = time.time() - step_start

        # Update selector
        self.batch_selector.update(
            selected_indices,
            info["losses"],
            sample_indices=batch.get("index"),
        )

        # Record metrics
        reward_accuracy = (info["reward_margin"] > 0).float().mean().item()

        self.metrics.train_loss.append(loss.item() * self.config.gradient_accumulation_steps)
        self.metrics.chosen_rewards.append(info["chosen_rewards"].mean().item())
        self.metrics.rejected_rewards.append(info["rejected_rewards"].mean().item())
        self.metrics.reward_margins.append(info["reward_margin"].mean().item())
        self.metrics.reward_accuracies.append(reward_accuracy)
        self.metrics.samples_per_step.append(len(selected_indices))
        self.metrics.selection_overhead.append(selection_time)
        self.metrics.total_samples_seen += len(selected_indices)

        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "reward_margin": info["reward_margin"].mean().item(),
            "reward_accuracy": reward_accuracy,
            "samples": len(selected_indices),
            "selection_time": selection_time,
        }

    def train(
        self,
        train_dataloader: DataLoader,
    ) -> DPOMetrics:
        """Full DPO training loop."""
        total_start = time.time()
        global_step = 0

        for epoch in range(self.config.num_epochs):
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch in pbar:
                step_metrics = self.train_step(batch, global_step)
                global_step += 1

                if global_step % self.config.log_interval == 0:
                    pbar.set_postfix({
                        "loss": f"{step_metrics['loss']:.4f}",
                        "margin": f"{step_metrics['reward_margin']:.3f}",
                        "acc": f"{step_metrics['reward_accuracy']:.2%}",
                    })

            if hasattr(self.batch_selector, 'on_epoch_end'):
                self.batch_selector.on_epoch_end()

        self.metrics.total_wall_time = time.time() - total_start
        return self.metrics

    def get_efficiency_stats(self) -> Dict[str, Any]:
        """Get training efficiency statistics."""
        return {
            "total_wall_time": self.metrics.total_wall_time,
            "total_samples_seen": self.metrics.total_samples_seen,
            "final_reward_accuracy": self.metrics.reward_accuracies[-1] if self.metrics.reward_accuracies else 0,
            "avg_reward_margin": sum(self.metrics.reward_margins) / len(self.metrics.reward_margins) if self.metrics.reward_margins else 0,
            "total_selection_overhead": sum(self.metrics.selection_overhead),
            "selection_overhead_fraction": sum(self.metrics.selection_overhead) / self.metrics.total_wall_time if self.metrics.total_wall_time > 0 else 0,
        }
