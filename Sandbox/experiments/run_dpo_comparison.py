"""
Experiment: Compare Batch Selection Methods for DPO Training

This experiment compares different batch selection strategies
for Direct Preference Optimization (DPO) training.

Key research questions:
1. Can batch selection reduce training time without hurting alignment quality?
2. Which selection criteria work best for preference learning?
3. What is the overhead vs speedup tradeoff?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from dataclasses import asdict
import json
import time

from models.language_models import load_causal_lm, create_reference_model
from datasets.loaders import load_preference_dataset
from training.dpo import DPOTrainer, DPOConfig
from batch_selection import (
    RandomSelector,
    LossBasedSelector,
    UncertaintySelector,
    ReducibleLossSelector,
)
from batch_selection.base import NoSelectionWrapper
from utils.metrics import MetricsLogger, compare_selectors
from utils.visualization import plot_training_curves, plot_selection_analysis


def run_experiment(
    model_name: str = "distilgpt2",
    dataset_name: str = "synthetic",
    max_samples: int = 500,
    batch_size: int = 4,
    selection_ratio: float = 0.5,
    num_epochs: int = 1,
    output_dir: str = "./results/dpo_comparison",
):
    """
    Run comparative experiment on DPO with different batch selectors.
    """
    print("=" * 60)
    print("DPO Batch Selection Experiment")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Selection ratio: {selection_ratio}")
    print()

    # Initialize logger
    logger = MetricsLogger("dpo_comparison", output_dir)

    # Load model and tokenizer
    print("Loading model...")
    policy_model, tokenizer = load_causal_lm(model_name, device=device)

    # Load dataset
    print("Loading dataset...")
    dataset = load_preference_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_length=256,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Define selectors to compare
    selectors = {
        "no_selection": NoSelectionWrapper(device=device),
        "random_50": RandomSelector(selection_ratio=selection_ratio, device=device),
        "loss_based_cached": LossBasedSelector(
            selection_ratio=selection_ratio,
            device=device,
            mode="cached",
        ),
        "loss_based_hybrid": LossBasedSelector(
            selection_ratio=selection_ratio,
            device=device,
            mode="hybrid",
            explore_ratio=0.3,
        ),
        "uncertainty_entropy": UncertaintySelector(
            selection_ratio=selection_ratio,
            device=device,
            mode="entropy",
        ),
        "reducible_loss": ReducibleLossSelector(
            selection_ratio=selection_ratio,
            device=device,
            reference_mode="ema",
        ),
    }

    results = {}

    for selector_name, selector in selectors.items():
        print(f"\n{'='*40}")
        print(f"Running with selector: {selector_name}")
        print(f"{'='*40}")

        # Fresh model for each selector
        policy_model, tokenizer = load_causal_lm(model_name, device=device)
        reference_model = create_reference_model(policy_model)

        # Config
        config = DPOConfig(
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device,
        )

        # Trainer
        trainer = DPOTrainer(
            policy_model=policy_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            config=config,
            batch_selector=selector,
        )

        # Start logging
        logger.start_run(
            run_name=selector_name,
            selector_name=selector_name,
            model_name=model_name,
            dataset_name=dataset_name,
            config=asdict(config),
        )

        # Train
        start_time = time.time()
        metrics = trainer.train(dataloader)
        total_time = time.time() - start_time

        # Record metrics
        for i, loss in enumerate(metrics.train_loss):
            logger.log_step(
                loss=loss,
                samples=metrics.samples_per_step[i] if i < len(metrics.samples_per_step) else batch_size,
                wall_time=metrics.total_wall_time / len(metrics.train_loss),
                selection_time=metrics.selection_overhead[i] if i < len(metrics.selection_overhead) else 0,
                reward_margin=metrics.reward_margins[i] if i < len(metrics.reward_margins) else 0,
                reward_accuracy=metrics.reward_accuracies[i] if i < len(metrics.reward_accuracies) else 0,
            )

        logger.end_run(total_time)

        # Store results
        results[selector_name] = {
            "metrics": metrics,
            "efficiency": trainer.get_efficiency_stats(),
            "train_loss": metrics.train_loss,
            "reward_margins": metrics.reward_margins,
            "reward_accuracies": metrics.reward_accuracies,
            "samples_per_step": metrics.samples_per_step,
        }

        print(f"\nResults for {selector_name}:")
        print(f"  Final loss: {metrics.train_loss[-1]:.4f}")
        print(f"  Final reward accuracy: {metrics.reward_accuracies[-1]:.2%}")
        print(f"  Total samples: {metrics.total_samples_seen}")
        print(f"  Wall time: {total_time:.2f}s")
        print(f"  Selection overhead: {sum(metrics.selection_overhead):.3f}s")

    # Comparison
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

    # Save results
    output_path = logger.save()
    print(f"\nResults saved to: {output_path}")

    # Generate plots
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_training_curves(
        {name: {
            "train_loss": r["train_loss"],
            "samples_per_step": r["samples_per_step"],
        } for name, r in results.items()},
        output_path=f"{output_dir}/training_curves.png",
        title="DPO Training with Different Batch Selectors",
    )

    # Selector analysis
    selector_stats = {}
    baseline = results["no_selection"]
    for name, r in results.items():
        if name == "no_selection":
            selector_stats[name] = {
                "selection_overhead_ratio": 0,
                "sample_reduction": 0,
                "net_speedup": 1.0,
            }
        else:
            selector_stats[name] = {
                "selection_overhead_ratio": sum(r["metrics"].selection_overhead) / r["metrics"].total_wall_time if r["metrics"].total_wall_time > 0 else 0,
                "sample_reduction": 1 - r["metrics"].total_samples_seen / baseline["metrics"].total_samples_seen,
                "net_speedup": baseline["metrics"].total_wall_time / r["metrics"].total_wall_time if r["metrics"].total_wall_time > 0 else 1,
            }

    plot_selection_analysis(
        selector_stats,
        output_path=f"{output_dir}/selection_analysis.png",
    )

    print(f"\nPlots saved to {output_dir}/")

    return results, comparison


if __name__ == "__main__":
    run_experiment()
