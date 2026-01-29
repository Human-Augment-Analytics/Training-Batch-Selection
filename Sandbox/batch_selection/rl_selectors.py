"""
Novel Batch Selection Methods for RL Alignment

These selectors are specifically designed for the RL alignment setting
(PPO, DPO, RLHF) where the learning signal comes from preferences
or reward models rather than direct labels.

Key insights for RL-specific batch selection:
1. Preference margin matters: Samples with clearer preferences are easier to learn
2. Policy-reference divergence: Samples where policy has drifted may need correction
3. Advantage variance: High-variance advantages indicate uncertain value estimates
4. Reward informativeness: Some reward model outputs are more informative than others

These methods are designed to be LIGHTWEIGHT - the key contribution is
showing that simple, low-overhead methods can match or beat expensive methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .base import BatchSelector


class PreferenceMarginSelector(BatchSelector):
    """
    Select samples based on preference margin (DPO-specific).

    Intuition: Samples with moderate preference margins are most informative.
    - Very high margin: Already learned, not much to gain
    - Very low margin: Noisy or ambiguous, may hurt training
    - Moderate margin: Sweet spot for learning

    This is CHEAP: Uses logits from training forward pass.
    """

    def __init__(
        self,
        selection_ratio: float = 0.5,
        device: str = "cpu",
        target_margin: float = 0.5,  # Prefer samples near this margin
        margin_mode: str = "moderate",  # "moderate", "high", "curriculum"
    ):
        super().__init__(selection_ratio, device)
        self.target_margin = target_margin
        self.margin_mode = margin_mode
        self._epoch = 0

    def score_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        reward_margin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Score by preference margin informativeness.

        Args:
            reward_margin: Pre-computed reward margin (chosen - rejected)
        """
        if reward_margin is None:
            # Fallback to random
            batch_size = next(iter(batch.values())).shape[0]
            return torch.rand(batch_size, device=self.device)

        margin = reward_margin.abs()

        if self.margin_mode == "moderate":
            # Prefer samples with margin close to target
            # Score = 1 / (1 + |margin - target|)
            scores = 1.0 / (1.0 + (margin - self.target_margin).abs())

        elif self.margin_mode == "high":
            # Prefer high-margin samples (clearer signal)
            scores = margin

        elif self.margin_mode == "curriculum":
            # Start with high-margin (easy), gradually include lower-margin
            # Adjusts target based on epoch
            curriculum_target = max(0.1, self.target_margin - 0.1 * self._epoch)
            scores = 1.0 / (1.0 + (margin - curriculum_target).abs())

        else:
            scores = margin

        return scores

    def on_epoch_end(self) -> None:
        self._epoch += 1


class AdvantageVarianceSelector(BatchSelector):
    """
    Select samples based on advantage estimate variance (PPO-specific).

    Intuition: High variance in advantage estimates indicates uncertainty
    in value predictions, which suggests the model can learn from these samples.

    Implementation: Track running variance of advantages per sample/prompt.
    Select samples with high variance (uncertain) or moderate variance (learnable).

    This is VERY CHEAP: Only uses advantage values already computed.
    """

    def __init__(
        self,
        selection_ratio: float = 0.5,
        device: str = "cpu",
        ema_decay: float = 0.9,
        prefer_high_variance: bool = True,
    ):
        super().__init__(selection_ratio, device)
        self.ema_decay = ema_decay
        self.prefer_high_variance = prefer_high_variance

        # Track advantage statistics per sample
        self.advantage_mean: Dict[int, float] = {}
        self.advantage_var: Dict[int, float] = {}

    def score_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        advantages: Optional[torch.Tensor] = None,
        sample_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Score by advantage variance."""
        batch_size = next(iter(batch.values())).shape[0]

        if advantages is None or sample_indices is None:
            return torch.rand(batch_size, device=self.device)

        scores = torch.zeros(batch_size, device=self.device)

        for i, idx in enumerate(sample_indices.tolist()):
            if idx in self.advantage_var:
                # Use tracked variance
                var = self.advantage_var[idx]
                scores[i] = var if self.prefer_high_variance else 1.0 / (1.0 + var)
            else:
                # New sample - give high score to explore
                scores[i] = float('inf') if self.prefer_high_variance else 0.0

        return scores

    def update(
        self,
        indices: torch.Tensor,
        losses: torch.Tensor,
        advantages: Optional[torch.Tensor] = None,
        sample_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Update advantage statistics."""
        if advantages is None or sample_indices is None:
            return

        for i, batch_idx in enumerate(indices.tolist()):
            if batch_idx >= len(sample_indices):
                continue

            sample_idx = sample_indices[batch_idx].item()
            adv = advantages[batch_idx].item() if batch_idx < len(advantages) else 0

            if sample_idx in self.advantage_mean:
                # Update EMA of mean and variance
                old_mean = self.advantage_mean[sample_idx]
                self.advantage_mean[sample_idx] = (
                    self.ema_decay * old_mean + (1 - self.ema_decay) * adv
                )
                # Online variance update
                diff = adv - old_mean
                self.advantage_var[sample_idx] = (
                    self.ema_decay * self.advantage_var.get(sample_idx, 0) +
                    (1 - self.ema_decay) * diff ** 2
                )
            else:
                self.advantage_mean[sample_idx] = adv
                self.advantage_var[sample_idx] = 0.0


class KLDivergenceSelector(BatchSelector):
    """
    Select samples based on KL divergence from reference model.

    Intuition: Samples where policy has diverged significantly from reference
    may benefit from targeted updates. Can be used to:
    1. Prioritize high-KL samples (bring back towards reference)
    2. Prioritize low-KL samples (continue safe exploration)

    This requires forward passes on both policy and reference,
    but can reuse computation from PPO training.
    """

    def __init__(
        self,
        selection_ratio: float = 0.5,
        device: str = "cpu",
        prefer_high_kl: bool = True,
        kl_threshold: Optional[float] = None,
    ):
        super().__init__(selection_ratio, device)
        self.prefer_high_kl = prefer_high_kl
        self.kl_threshold = kl_threshold

    def score_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        policy_logps: Optional[torch.Tensor] = None,
        reference_logps: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Score by KL divergence from reference."""
        batch_size = next(iter(batch.values())).shape[0]

        if policy_logps is None or reference_logps is None:
            return torch.rand(batch_size, device=self.device)

        # Approximate KL divergence
        kl = policy_logps - reference_logps  # Per-sample KL

        if self.kl_threshold is not None:
            # Binary: select samples above/below threshold
            if self.prefer_high_kl:
                scores = (kl > self.kl_threshold).float()
            else:
                scores = (kl < self.kl_threshold).float()
        else:
            # Continuous scoring
            if self.prefer_high_kl:
                scores = kl.abs()
            else:
                scores = 1.0 / (1.0 + kl.abs())

        return scores


class RewardUncertaintySelector(BatchSelector):
    """
    Select samples based on reward model uncertainty.

    Intuition: If we have ensemble reward models or can estimate
    reward uncertainty, prioritize samples where reward is uncertain.

    For single reward model: Use logit magnitude as confidence proxy.
    Lower magnitude = more uncertain.

    This is CHEAP if reusing reward model outputs from PPO.
    """

    def __init__(
        self,
        selection_ratio: float = 0.5,
        device: str = "cpu",
        prefer_uncertain: bool = True,
    ):
        super().__init__(selection_ratio, device)
        self.prefer_uncertain = prefer_uncertain

    def score_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        reward_logits: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Score by reward uncertainty."""
        batch_size = next(iter(batch.values())).shape[0]

        if reward_logits is None:
            return torch.rand(batch_size, device=self.device)

        # Use logit magnitude as inverse confidence
        # Low magnitude = uncertain
        confidence = reward_logits.abs()

        if self.prefer_uncertain:
            # Invert: low confidence = high score
            scores = 1.0 / (1.0 + confidence)
        else:
            scores = confidence

        return scores


class CombinedRLSelector(BatchSelector):
    """
    Combine multiple RL-specific selection signals.

    Weights multiple criteria:
    - Preference margin (for DPO)
    - Advantage variance (for PPO)
    - KL divergence
    - Reward uncertainty

    This allows flexible tuning for different RL settings.
    """

    def __init__(
        self,
        selection_ratio: float = 0.5,
        device: str = "cpu",
        margin_weight: float = 0.3,
        variance_weight: float = 0.3,
        kl_weight: float = 0.2,
        uncertainty_weight: float = 0.2,
    ):
        super().__init__(selection_ratio, device)

        self.weights = {
            "margin": margin_weight,
            "variance": variance_weight,
            "kl": kl_weight,
            "uncertainty": uncertainty_weight,
        }

        # Sub-selectors
        self.margin_selector = PreferenceMarginSelector(1.0, device)
        self.variance_selector = AdvantageVarianceSelector(1.0, device)
        self.kl_selector = KLDivergenceSelector(1.0, device)
        self.uncertainty_selector = RewardUncertaintySelector(1.0, device)

    def score_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Combine multiple selection signals."""
        batch_size = next(iter(batch.values())).shape[0]
        combined_scores = torch.zeros(batch_size, device=self.device)

        # Margin score
        if self.weights["margin"] > 0 and "reward_margin" in kwargs:
            margin_scores = self.margin_selector.score_batch(model, batch, **kwargs)
            margin_scores = self._normalize(margin_scores)
            combined_scores += self.weights["margin"] * margin_scores

        # Variance score
        if self.weights["variance"] > 0 and "advantages" in kwargs:
            var_scores = self.variance_selector.score_batch(model, batch, **kwargs)
            var_scores = self._normalize(var_scores)
            combined_scores += self.weights["variance"] * var_scores

        # KL score
        if self.weights["kl"] > 0 and "policy_logps" in kwargs:
            kl_scores = self.kl_selector.score_batch(model, batch, **kwargs)
            kl_scores = self._normalize(kl_scores)
            combined_scores += self.weights["kl"] * kl_scores

        # Uncertainty score
        if self.weights["uncertainty"] > 0 and "reward_logits" in kwargs:
            unc_scores = self.uncertainty_selector.score_batch(model, batch, **kwargs)
            unc_scores = self._normalize(unc_scores)
            combined_scores += self.weights["uncertainty"] * unc_scores

        # If no signals available, random
        if combined_scores.sum() == 0:
            return torch.rand(batch_size, device=self.device)

        return combined_scores

    def _normalize(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalize scores to [0, 1] range."""
        # Handle inf values
        finite_mask = torch.isfinite(scores)
        if not finite_mask.any():
            return torch.rand_like(scores)

        min_val = scores[finite_mask].min()
        max_val = scores[finite_mask].max()

        if max_val - min_val < 1e-8:
            return torch.ones_like(scores) * 0.5

        normalized = (scores - min_val) / (max_val - min_val)
        normalized[~finite_mask] = 1.0  # Give inf high score
        return normalized

    def update(self, indices, losses, **kwargs):
        """Update sub-selectors."""
        self.variance_selector.update(indices, losses, **kwargs)
