"""
Experiment: Compare Batch Selection Methods for PPO/RLHF Training

This experiment compares batch selection strategies in the PPO
setting used for RLHF.

Key research questions:
1. Can we select more informative experiences for PPO updates?
2. Does advantage-based selection improve sample efficiency?
3. How does selection interact with PPO's multiple epochs per batch?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset
from dataclasses import asdict
import time
from typing import List

from models.language_models import load_causal_lm, create_reference_model, load_reward_model, PolicyWithValueHead
from training.ppo import PPOTrainer, PPOConfig
from batch_selection import (
    RandomSelector,
    LossBasedSelector,
    UncertaintySelector,
)
from batch_selection.base import NoSelectionWrapper
from utils.metrics import MetricsLogger
from utils.visualization import plot_rl_training


class PromptDataset(Dataset):
    """Simple dataset of prompts for PPO."""

    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}


def generate_synthetic_prompts(n: int = 100) -> List[str]:
    """Generate synthetic prompts for testing."""
    templates = [
        "Explain the concept of {}.",
        "What is the difference between {} and {}?",
        "How do I {}?",
        "Write a short story about {}.",
        "Describe {} in simple terms.",
    ]
    topics = [
        "machine learning", "neural networks", "reinforcement learning",
        "natural language processing", "computer vision", "deep learning",
        "gradient descent", "backpropagation", "attention mechanism",
        "transformers", "BERT", "GPT", "batch selection", "curriculum learning",
    ]

    prompts = []
    for i in range(n):
        template = templates[i % len(templates)]
        if "{}" in template and template.count("{}") == 2:
            prompt = template.format(
                topics[i % len(topics)],
                topics[(i + 1) % len(topics)]
            )
        else:
            prompt = template.format(topics[i % len(topics)])
        prompts.append(prompt)

    return prompts


def run_experiment(
    model_name: str = "distilgpt2",
    n_prompts: int = 100,
    batch_size: int = 4,
    selection_ratio: float = 0.5,
    num_epochs: int = 1,
    output_dir: str = "./results/ppo_comparison",
):
    """
    Run comparative PPO experiment with different batch selectors.
    """
    print("=" * 60)
    print("PPO Batch Selection Experiment")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Prompts: {n_prompts}")
    print(f"Selection ratio: {selection_ratio}")
    print()

    # Initialize logger
    logger = MetricsLogger("ppo_comparison", output_dir)

    # Generate prompts
    prompts = generate_synthetic_prompts(n_prompts)
    dataset = PromptDataset(prompts)

    def collate_fn(batch):
        return {"prompt": [b["prompt"] for b in batch]}

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Define selectors
    selectors = {
        "no_selection": NoSelectionWrapper(device=device),
        "random_50": RandomSelector(selection_ratio=selection_ratio, device=device),
        "loss_based": LossBasedSelector(
            selection_ratio=selection_ratio,
            device=device,
            mode="cached",
        ),
        "uncertainty": UncertaintySelector(
            selection_ratio=selection_ratio,
            device=device,
            mode="entropy",
        ),
    }

    results = {}

    for selector_name, selector in selectors.items():
        print(f"\n{'='*40}")
        print(f"Running PPO with selector: {selector_name}")
        print(f"{'='*40}")

        # Fresh models for each selector
        policy_model, tokenizer = load_causal_lm(model_name, device=device)
        policy_with_value = PolicyWithValueHead(policy_model)
        reference_model = create_reference_model(policy_model)
        reward_model = load_reward_model(model_name, device=device)

        # Config
        config = PPOConfig(
            num_epochs=num_epochs,
            batch_size=batch_size,
            ppo_epochs=2,  # Fewer for faster testing
            device=device,
            max_response_length=64,
        )

        # Trainer
        trainer = PPOTrainer(
            policy_model=policy_with_value,
            reference_model=reference_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            batch_selector=selector,
        )

        # Start logging
        logger.start_run(
            run_name=selector_name,
            selector_name=selector_name,
            model_name=model_name,
            dataset_name="synthetic_prompts",
            config=asdict(config),
        )

        # Train
        start_time = time.time()
        metrics = trainer.train(dataloader)
        total_time = time.time() - start_time

        # Record metrics
        for i in range(len(metrics.policy_loss)):
            logger.log_step(
                loss=metrics.policy_loss[i],
                samples=metrics.samples_per_step[i] if i < len(metrics.samples_per_step) else batch_size,
                wall_time=metrics.total_wall_time / len(metrics.policy_loss),
                selection_time=metrics.selection_overhead[i] if i < len(metrics.selection_overhead) else 0,
                reward=metrics.rewards[i] if i < len(metrics.rewards) else 0,
                kl=metrics.kl_divergence[i] if i < len(metrics.kl_divergence) else 0,
            )

        logger.end_run(total_time)

        # Store results
        results[selector_name] = {
            "metrics": metrics,
            "efficiency": trainer.get_efficiency_stats(),
            "rewards": metrics.rewards,
            "policy_loss": metrics.policy_loss,
            "kl_divergence": metrics.kl_divergence,
            "samples_per_step": metrics.samples_per_step,
        }

        print(f"\nResults for {selector_name}:")
        print(f"  Final policy loss: {metrics.policy_loss[-1]:.4f}")
        print(f"  Final reward: {metrics.rewards[-1]:.4f}")
        print(f"  Final KL: {metrics.kl_divergence[-1]:.4f}")
        print(f"  Total samples: {metrics.total_samples_seen}")
        print(f"  Wall time: {total_time:.2f}s")

    # Save and visualize
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    comparison = logger.get_comparison()
    for name, stats in comparison.items():
        print(f"\n{name}:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    output_path = logger.save()
    print(f"\nResults saved to: {output_path}")

    # Generate plots
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_rl_training(
        {name: {
            "rewards": r["rewards"],
            "policy_loss": r["policy_loss"],
            "kl_divergence": r["kl_divergence"],
            "samples_per_step": r["samples_per_step"],
        } for name, r in results.items()},
        output_path=f"{output_dir}/ppo_training.png",
        title="PPO Training with Different Batch Selectors",
    )

    print(f"Plots saved to {output_dir}/")

    return results, comparison


if __name__ == "__main__":
    run_experiment()
