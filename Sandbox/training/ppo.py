"""
Proximal Policy Optimization (PPO) Training Loop for RLHF

PPO is the standard algorithm for RLHF. It optimizes a policy to maximize
rewards while staying close to a reference policy (KL penalty).

Key opportunities for batch selection in PPO:
1. Experience selection: Which rollouts/trajectories to learn from
2. Mini-batch selection within PPO epochs: Which samples to prioritize
3. Advantage-based selection: Prioritize high-advantage samples

References:
- Schulman et al., 2017: "Proximal Policy Optimization Algorithms"
- Ouyang et al., 2022: "Training language models to follow instructions with human feedback"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List, Tuple
import time
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np

import sys
sys.path.append("..")
from batch_selection.base import BatchSelector, NoSelectionWrapper


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    learning_rate: float = 1e-5
    num_epochs: int = 1
    ppo_epochs: int = 4  # PPO epochs per batch of experiences
    batch_size: int = 4
    mini_batch_size: int = 2  # For PPO updates
    max_grad_norm: float = 1.0
    gamma: float = 1.0  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clipping
    vf_coef: float = 0.1  # Value function coefficient
    kl_coef: float = 0.1  # KL penalty coefficient
    target_kl: Optional[float] = 0.01  # Early stopping KL threshold
    max_response_length: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 10


@dataclass
class PPOMetrics:
    """Metrics for PPO training."""
    policy_loss: List[float] = field(default_factory=list)
    value_loss: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    kl_divergence: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    samples_per_step: List[int] = field(default_factory=list)
    selection_overhead: List[float] = field(default_factory=list)
    total_samples_seen: int = 0
    total_wall_time: float = 0.0


@dataclass
class Experience:
    """Single experience/trajectory for PPO."""
    query_ids: torch.Tensor
    response_ids: torch.Tensor
    attention_mask: torch.Tensor
    log_probs: torch.Tensor  # Log probs under old policy
    values: torch.Tensor  # Value estimates
    rewards: torch.Tensor  # Rewards from reward model
    advantages: torch.Tensor  # Computed advantages
    returns: torch.Tensor  # Computed returns


class PPOTrainer:
    """
    PPO trainer for RLHF with batch selection support.

    The training loop:
    1. Generate responses from policy
    2. Score with reward model
    3. Compute advantages
    4. Run PPO updates with batch selection

    Batch selection opportunities:
    - Experience selection: Which generated experiences to train on
    - Mini-batch selection: Which samples to prioritize in PPO epochs
    """

    def __init__(
        self,
        policy_model: nn.Module,  # Should be PolicyWithValueHead
        reference_model: nn.Module,
        reward_model: nn.Module,
        tokenizer,
        config: PPOConfig,
        batch_selector: Optional[BatchSelector] = None,
    ):
        self.policy = policy_model
        self.reference = reference_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        self.batch_selector = batch_selector or NoSelectionWrapper()
        self.device = config.device

        self.policy.to(self.device)
        self.reference.to(self.device)
        self.reward_model.to(self.device)

        self.reference.eval()
        self.reward_model.eval()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
        )

        self.metrics = PPOMetrics()

    def generate_responses(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate responses from current policy."""
        self.policy.eval()

        # Tokenize prompts
        prompt_enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.policy.generate(
                input_ids=prompt_enc["input_ids"],
                attention_mask=prompt_enc["attention_mask"],
                max_new_tokens=self.config.max_response_length,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Separate query and response
        query_ids = prompt_enc["input_ids"]
        response_ids = outputs[:, query_ids.shape[1]:]

        # Create attention mask for full sequence
        attention_mask = torch.ones_like(outputs)

        self.policy.train()
        return query_ids, response_ids, attention_mask

    def compute_rewards(
        self,
        query_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute rewards using reward model."""
        # Concatenate query and response
        full_ids = torch.cat([query_ids, response_ids], dim=1)

        with torch.no_grad():
            rewards = self.reward_model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
            )
            if hasattr(rewards, 'logits'):
                rewards = rewards.logits.squeeze(-1)

        return rewards

    def compute_log_probs_and_values(
        self,
        query_ids: torch.Tensor,
        response_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probs and values for responses."""
        full_ids = torch.cat([query_ids, response_ids], dim=1)

        logits, values = self.policy(
            input_ids=full_ids,
            attention_mask=attention_mask,
        )

        # Log probs for response tokens only
        response_start = query_ids.shape[1]
        response_logits = logits[:, response_start-1:-1, :]  # Shifted
        response_labels = response_ids

        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=response_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask padding
        response_mask = (response_ids != self.tokenizer.pad_token_id).float()
        response_log_probs = (token_log_probs * response_mask).sum(dim=-1)

        return response_log_probs, values

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        # For simplicity, treat each response as single-step
        # In full implementation, would compute per-token advantages

        returns = rewards
        advantages = rewards - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def collect_experiences(
        self,
        prompts: List[str],
    ) -> Experience:
        """Collect experiences by generating and scoring responses."""
        # Generate responses
        query_ids, response_ids, attention_mask = self.generate_responses(prompts)

        # Get rewards
        rewards = self.compute_rewards(query_ids, response_ids)

        # Get log probs and values under current policy
        with torch.no_grad():
            log_probs, values = self.compute_log_probs_and_values(
                query_ids, response_ids, attention_mask
            )

        # Compute advantages
        advantages, returns = self.compute_advantages(rewards, values.squeeze(-1))

        return Experience(
            query_ids=query_ids,
            response_ids=response_ids,
            attention_mask=attention_mask,
            log_probs=log_probs.detach(),
            values=values.detach(),
            rewards=rewards,
            advantages=advantages,
            returns=returns,
        )

    def ppo_step(
        self,
        experience: Experience,
        indices: torch.Tensor,
    ) -> Dict[str, float]:
        """Single PPO update step on selected samples."""
        # Select samples
        query_ids = experience.query_ids[indices]
        response_ids = experience.response_ids[indices]
        attention_mask = experience.attention_mask[indices]
        old_log_probs = experience.log_probs[indices]
        old_values = experience.values[indices]
        advantages = experience.advantages[indices]
        returns = experience.returns[indices]

        # Current log probs and values
        new_log_probs, new_values = self.compute_log_probs_and_values(
            query_ids, response_ids, attention_mask
        )
        new_values = new_values.squeeze(-1)

        # Policy loss (PPO clipped objective)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(new_values, returns)

        # KL divergence (for monitoring and early stopping)
        with torch.no_grad():
            ref_log_probs, _ = self.compute_log_probs_and_values(
                query_ids, response_ids, attention_mask
            )
            ref_log_probs = ref_log_probs.detach()
        kl = (new_log_probs - ref_log_probs).mean()

        # Total loss
        loss = policy_loss + self.config.vf_coef * value_loss + self.config.kl_coef * kl

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "kl": kl.item(),
            "ratio": ratio.mean().item(),
        }

    def train_on_experiences(
        self,
        experience: Experience,
        step: int,
    ) -> Dict[str, float]:
        """Run PPO epochs on collected experiences with batch selection."""
        batch_size = experience.query_ids.shape[0]
        total_policy_loss = 0
        total_value_loss = 0
        total_kl = 0

        for ppo_epoch in range(self.config.ppo_epochs):
            # Batch selection for this PPO epoch
            selection_start = time.time()

            # Create batch for selector
            selector_batch = {
                "input_ids": experience.query_ids,
                "labels": experience.response_ids,
                "attention_mask": experience.attention_mask,
            }

            # Score by advantage magnitude (higher = more informative)
            scores = experience.advantages.abs()

            # Override selector scoring temporarily
            original_score = self.batch_selector.score_batch
            self.batch_selector.score_batch = lambda m, b, **kw: scores

            selection_result = self.batch_selector.select(
                self.policy,
                selector_batch,
            )

            self.batch_selector.score_batch = original_score
            selection_time = time.time() - selection_start

            # PPO update on selected samples
            selected_indices = selection_result.indices
            step_metrics = self.ppo_step(experience, selected_indices)

            total_policy_loss += step_metrics["policy_loss"]
            total_value_loss += step_metrics["value_loss"]
            total_kl += step_metrics["kl"]

            # Early stopping on KL
            if self.config.target_kl and step_metrics["kl"] > self.config.target_kl:
                break

        n_epochs = ppo_epoch + 1
        return {
            "policy_loss": total_policy_loss / n_epochs,
            "value_loss": total_value_loss / n_epochs,
            "kl": total_kl / n_epochs,
            "ppo_epochs": n_epochs,
            "samples": len(selected_indices),
            "selection_time": selection_time,
        }

    def train(
        self,
        prompt_dataloader: DataLoader,
    ) -> PPOMetrics:
        """Full PPO training loop."""
        total_start = time.time()
        global_step = 0

        for epoch in range(self.config.num_epochs):
            pbar = tqdm(prompt_dataloader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch in pbar:
                # Extract prompts
                if isinstance(batch, dict):
                    prompts = batch.get("prompt", batch.get("text", []))
                else:
                    prompts = batch

                if isinstance(prompts, torch.Tensor):
                    prompts = self.tokenizer.batch_decode(prompts, skip_special_tokens=True)

                # Collect experiences
                experience = self.collect_experiences(prompts)

                # PPO update with batch selection
                step_metrics = self.train_on_experiences(experience, global_step)
                global_step += 1

                # Record metrics
                self.metrics.policy_loss.append(step_metrics["policy_loss"])
                self.metrics.value_loss.append(step_metrics["value_loss"])
                self.metrics.kl_divergence.append(step_metrics["kl"])
                self.metrics.rewards.append(experience.rewards.mean().item())
                self.metrics.advantages.append(experience.advantages.mean().item())
                self.metrics.samples_per_step.append(step_metrics["samples"])
                self.metrics.selection_overhead.append(step_metrics["selection_time"])
                self.metrics.total_samples_seen += step_metrics["samples"]

                if global_step % self.config.log_interval == 0:
                    pbar.set_postfix({
                        "reward": f"{experience.rewards.mean().item():.3f}",
                        "kl": f"{step_metrics['kl']:.4f}",
                        "samples": step_metrics["samples"],
                    })

        self.metrics.total_wall_time = time.time() - total_start
        return self.metrics

    def get_efficiency_stats(self) -> Dict[str, Any]:
        """Get training efficiency statistics."""
        return {
            "total_wall_time": self.metrics.total_wall_time,
            "total_samples_seen": self.metrics.total_samples_seen,
            "avg_reward": sum(self.metrics.rewards) / len(self.metrics.rewards) if self.metrics.rewards else 0,
            "final_kl": self.metrics.kl_divergence[-1] if self.metrics.kl_divergence else 0,
            "total_selection_overhead": sum(self.metrics.selection_overhead),
            "selection_overhead_fraction": sum(self.metrics.selection_overhead) / self.metrics.total_wall_time if self.metrics.total_wall_time > 0 else 0,
        }
