"""
Visualization utilities for experiment analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path


def plot_training_curves(
    metrics_dict: Dict[str, Dict[str, List[float]]],
    output_path: Optional[str] = None,
    title: str = "Training Comparison",
) -> None:
    """
    Plot training curves for multiple selectors.

    Args:
        metrics_dict: {selector_name: {"train_loss": [...], "eval_loss": [...]}}
        output_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_dict)))

    # Training loss
    ax = axes[0, 0]
    for (name, metrics), color in zip(metrics_dict.items(), colors):
        if "train_loss" in metrics:
            ax.plot(metrics["train_loss"], label=name, color=color, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Eval loss
    ax = axes[0, 1]
    for (name, metrics), color in zip(metrics_dict.items(), colors):
        if "eval_loss" in metrics:
            ax.plot(metrics["eval_loss"], label=name, color=color, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Eval Loss")
    ax.set_title("Evaluation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Samples per step
    ax = axes[1, 0]
    for (name, metrics), color in zip(metrics_dict.items(), colors):
        if "samples_per_step" in metrics:
            ax.plot(metrics["samples_per_step"], label=name, color=color, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Samples")
    ax.set_title("Samples per Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative samples
    ax = axes[1, 1]
    for (name, metrics), color in zip(metrics_dict.items(), colors):
        if "samples_per_step" in metrics:
            cumsum = np.cumsum(metrics["samples_per_step"])
            ax.plot(cumsum, label=name, color=color, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Samples")
    ax.set_title("Total Samples Seen")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_selection_analysis(
    selector_stats: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
) -> None:
    """
    Plot analysis of batch selection methods.

    Shows overhead vs speedup tradeoffs.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = list(selector_stats.keys())
    x = np.arange(len(names))

    # Overhead comparison
    ax = axes[0]
    overheads = [selector_stats[n].get("selection_overhead_ratio", 0) * 100 for n in names]
    bars = ax.bar(x, overheads, color="coral", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Selection Overhead (%)")
    ax.set_title("Selection Overhead")
    ax.axhline(y=5, color='r', linestyle='--', label='5% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Sample reduction
    ax = axes[1]
    reductions = [selector_stats[n].get("sample_reduction", 0) * 100 for n in names]
    bars = ax.bar(x, reductions, color="steelblue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Sample Reduction (%)")
    ax.set_title("Samples Saved")
    ax.grid(True, alpha=0.3, axis='y')

    # Net speedup
    ax = axes[2]
    speedups = [selector_stats[n].get("net_speedup", 1.0) for n in names]
    colors = ['green' if s > 1 else 'red' for s in speedups]
    bars = ax.bar(x, speedups, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Net Speedup (x)")
    ax.set_title("Net Speedup (Overhead Adjusted)")
    ax.axhline(y=1.0, color='k', linestyle='--', label='Break-even')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Batch Selection Analysis", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_rl_training(
    metrics_dict: Dict[str, Dict[str, List[float]]],
    output_path: Optional[str] = None,
    title: str = "RL Alignment Training",
) -> None:
    """
    Plot RL-specific training metrics (rewards, KL, etc.)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_dict)))

    metric_configs = [
        ("rewards", "Mean Reward", axes[0, 0]),
        ("kl_divergence", "KL Divergence", axes[0, 1]),
        ("policy_loss", "Policy Loss", axes[0, 2]),
        ("reward_margins", "Reward Margin", axes[1, 0]),
        ("reward_accuracies", "Reward Accuracy", axes[1, 1]),
        ("samples_per_step", "Samples/Step", axes[1, 2]),
    ]

    for metric_name, ylabel, ax in metric_configs:
        for (name, metrics), color in zip(metrics_dict.items(), colors):
            if metric_name in metrics and metrics[metric_name]:
                ax.plot(metrics[metric_name], label=name, color=color, alpha=0.7)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_comparison_table(
    comparison: Dict[str, Dict[str, Any]],
) -> str:
    """
    Create a markdown table comparing selectors.
    """
    headers = [
        "Selector",
        "Final Loss",
        "Total Samples",
        "Wall Time (s)",
        "Overhead (%)",
        "Net Speedup",
    ]

    rows = []
    for name, stats in comparison.items():
        row = [
            name,
            f"{stats.get('final_loss', 0):.4f}",
            f"{stats.get('total_samples', 0):,}",
            f"{stats.get('total_wall_time', 0):.2f}",
            f"{stats.get('selection_overhead_ratio', 0)*100:.1f}",
            f"{stats.get('net_speedup', 1.0):.2f}x",
        ]
        rows.append(row)

    # Create table
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    lines = []
    # Header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-|-".join("-" * w for w in col_widths))

    # Rows
    for row in rows:
        row_line = " | ".join(c.ljust(w) for c, w in zip(row, col_widths))
        lines.append(row_line)

    return "\n".join(lines)
