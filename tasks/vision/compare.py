#!/usr/bin/env python3
"""
Compare two or more batch strategies with overlay plots.

Usage:
    python -m tasks.vision.compare
    python -m tasks.vision.compare --dataset cifar10_csv
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config.base import OUTPUTS_ROOT
from config.vision import EPOCHS, ACTIVE_DATASET
from config.batch_strategies import VISION_STRATEGY_COMPARISON_PAIRS


def load_summary(run_dir):
    """Load summary statistics from a completed run."""
    summary_file = Path(run_dir) / "summary.txt"
    means, cis = {}, {}

    for metric in ["train_acc", "test_acc", "train_loss", "test_loss"]:
        means[metric] = []
        cis[metric] = []

    with open(summary_file) as f:
        for line in f:
            if 'Epoch' in line:
                for metric in ["train_acc", "test_acc", "train_loss"]:
                    try:
                        parts = line.split(f"{metric}=")[1].split("±")
                        val = float(parts[0])
                        ci_str = parts[1].split(",")[0].strip()
                        ci = float(ci_str)
                        means[metric].append(val)
                        cis[metric].append(ci)
                    except:
                        continue

    return means, cis


def find_latest_run_path(strategy_name, dataset_name):
    """Find the most recent completed run for a strategy."""
    strat_dir = OUTPUTS_ROOT / 'vision' / dataset_name / f"batching_{strategy_name.lower()}"

    if not strat_dir.exists():
        raise RuntimeError(f"No output directory found for {strategy_name} on {dataset_name}")

    runs = sorted([d for d in strat_dir.iterdir() if d.is_dir() and d.name.startswith('run-')])

    if not runs:
        raise RuntimeError(f"No run directories found for {strategy_name}")

    # Find latest run with summary.txt
    for run in reversed(runs):
        summary_file = run / "summary.txt"
        if summary_file.exists():
            return run

    raise RuntimeError(f"No completed runs found for {strategy_name}")


def plot_comparison(pair, dataset_name, epoch_count=EPOCHS):
    """Create comparison plots for two strategies."""
    epochs = np.arange(1, epoch_count + 1)
    out_dir = OUTPUTS_ROOT / 'vision' / dataset_name / f'comparison_{"_".join([p.lower() for p in pair])}'
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_labels = {
        "test_acc": "Test Accuracy",
        "train_acc": "Train Accuracy",
        "train_loss": "Train Loss",
        "test_loss": "Test Loss",
    }

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    # Load results for each strategy
    method_results = []
    for idx, method in enumerate(pair):
        run_dir = find_latest_run_path(method, dataset_name)
        means, cis = load_summary(run_dir)
        method_results.append((method, means, cis, colors[idx]))

    # Create comparison plots
    for metric in ["test_acc", "train_acc", "train_loss"]:
        plt.figure(figsize=(7, 5))

        for (name, means, cis, col) in method_results:
            if metric in means and len(means[metric]) > 0:
                m = np.array(means[metric])
                c = np.array(cis[metric])
                plt.plot(epochs[:len(m)], m, label=name, linewidth=2, color=col)
                plt.fill_between(epochs[:len(m)], m - c, m + c, alpha=0.2, color=col)

        plt.xlabel('Epoch')
        plt.ylabel(metric_labels[metric])
        plt.title(f'{metric_labels[metric]} Comparison - {dataset_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_cmp.png")
        plt.close()

    print(f"Comparison plots for {pair[0]} vs {pair[1]} saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Compare batch strategies')
    parser.add_argument('--dataset', type=str, default=ACTIVE_DATASET,
                       help=f'Dataset to compare (default: {ACTIVE_DATASET})')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"COMPARING BATCH STRATEGIES")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Comparison pairs: {VISION_STRATEGY_COMPARISON_PAIRS}")
    print(f"{'='*60}\n")

    for pair in VISION_STRATEGY_COMPARISON_PAIRS:
        try:
            plot_comparison(pair, args.dataset)
        except Exception as e:
            print(f"Error comparing {pair}: {e}")
            print("Make sure you've run experiments first with:")
            print(f"  python -m tasks.vision.run_experiment --dataset {args.dataset}")


if __name__ == "__main__":
    main()
