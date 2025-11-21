#!/usr/bin/env python3
"""
Main experiment runner for vision batch selection strategies.

Usage:
    python -m tasks.vision.run_experiment
    python -m tasks.vision.run_experiment --dataset cifar10_csv
"""
import importlib
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import stats

from config.base import (
    BASE_DIR, DATASETS_ROOT, OUTPUTS_ROOT, DEVICE, USE_GPU,
    NUM_WORKERS, PIN_MEMORY, RANDOM_SEED
)
from config.vision import (
    EPOCHS, BATCH_SIZE, N_RUNS, MOVING_AVG_DECAY, ACTIVE_DATASET
)
from config.datasets import DATASET_SPECS
from config.batch_strategies import VISION_BATCH_STRATEGIES

from tasks.vision.models.mlp import SimpleMLP
from tasks.vision.datasets.factory import build_dataset, build_model_for
from tasks.vision.train import train_model
from tasks.vision.evaluate import evaluate


def create_run_dir(strategy_name, dataset_name):
    """Create output directory for this experiment run."""
    out_path = OUTPUTS_ROOT / 'vision' / dataset_name / f'batching_{strategy_name.lower()}'
    out_path.mkdir(parents=True, exist_ok=True)

    # Find existing runs and increment
    existing = [d for d in out_path.iterdir() if d.is_dir() and d.name.startswith('run-')]
    if len(existing) > 0:
        next_num = max([int(d.name.split('-')[1]) for d in existing]) + 1
    else:
        next_num = 1

    run_dir = out_path / f'run-{next_num:03d}'
    run_dir.mkdir()
    return run_dir


def run_experiment(batch_strategy, strategy_label, train_ds, test_ds, model_constructor,
                   epochs, batch_size, n_runs):
    """Run multiple training runs with a given batch strategy."""
    results = []
    for run_idx in range(n_runs):
        print(f"\n[{strategy_label}] Run {run_idx+1}/{n_runs}")

        # Create fresh model for each run
        model = model_constructor()
        start = time.time()

        train_acc, test_acc, train_loss, test_loss = train_model(
            model, train_ds, test_ds, epochs, batch_size, batch_strategy,
            seed=RANDOM_SEED + run_idx
        )

        elapsed = time.time() - start

        results.append({
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "time": elapsed
        })

    return results


def mean_and_ci(arrays, axis=0):
    """Calculate mean and 95% confidence intervals."""
    arr = np.array(arrays)
    mean = arr.mean(axis=0)
    sem = stats.sem(arr, axis=0)
    # 95% confidence interval
    ci = sem * stats.t.ppf((1 + 0.95) / 2., arr.shape[0] - 1)
    return mean, ci


def aggregate_results(results):
    """Aggregate results across multiple runs."""
    means = {}
    cis = {}

    for metric in ["train_acc", "test_acc", "train_loss", "test_loss"]:
        mean, ci = mean_and_ci([run[metric] for run in results])
        means[metric] = mean
        cis[metric] = ci

    # Also get mean/ci for time
    mean_time, ci_time = mean_and_ci([run["time"] for run in results], axis=0)
    means["time"] = mean_time
    cis["time"] = ci_time

    return means, cis


def save_plots(run_dir, strategy_label, means, cis, epochs):
    """Save all metric plots."""
    epochs_range = np.arange(1, epochs + 1)

    def plot_metric(metric, ylabel, title, filename):
        plt.figure(figsize=(7, 5))
        plt.plot(epochs_range, means[metric], label=strategy_label, linewidth=2)
        plt.fill_between(epochs_range, means[metric] - cis[metric],
                        means[metric] + cis[metric], alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{title} - {strategy_label}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(run_dir / filename)
        plt.close()

    plot_metric('test_acc', 'Test Accuracy', "Test Accuracy vs Epoch", "test_acc.png")
    plot_metric('train_acc', 'Train Accuracy', "Train Accuracy vs Epoch", "train_acc.png")
    plot_metric('train_loss', 'Train Loss', "Train Loss vs Epoch", "train_loss.png")
    plot_metric('test_loss', 'Test Loss', "Test Loss vs Epoch", "test_loss.png")


def save_summary(run_dir, means, cis, epochs):
    """Save summary statistics to text file."""
    with open(run_dir / "summary.txt", "w") as f:
        for i in range(epochs):
            f.write(f"Epoch {i+1}: "
                   f"train_acc={means['train_acc'][i]:.4f}±{cis['train_acc'][i]:.4f}, "
                   f"test_acc={means['test_acc'][i]:.4f}±{cis['test_acc'][i]:.4f}, "
                   f"train_loss={means['train_loss'][i]:.4f}±{cis['train_loss'][i]:.4f}\n")
        f.write(f"Time: {means['time']:.2f}±{cis['time']:.2f} sec\n")


def main():
    # Display GPU/CPU information
    print(f"\n{'='*60}")
    print(f"DEVICE CONFIGURATION")
    print(f"{'='*60}")
    print(f"Device: {DEVICE.upper()}")
    if USE_GPU:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Running on CPU (GPU not available)")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Number of workers for data loading: {NUM_WORKERS}")
    print(f"Pin memory: {PIN_MEMORY}")
    print(f"{'='*60}\n")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train models with different batch strategies')
    parser.add_argument('--dataset', type=str, default=ACTIVE_DATASET,
                       help=f'Dataset to use (default: {ACTIVE_DATASET})')
    args = parser.parse_args()

    dataset_name = args.dataset

    # Validate dataset
    if dataset_name not in DATASET_SPECS:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Available datasets: {list(DATASET_SPECS.keys())}")
        exit(1)

    # Load dataset
    print(f"\n{'='*60}")
    print(f"Loading dataset: {dataset_name}")
    print(f"{'='*60}")
    train_ds, test_ds = build_dataset(str(DATASETS_ROOT), dataset_name)
    print(f"Dataset loaded: {len(train_ds)} train, {len(test_ds)} test samples")

    # Get dataset info
    dataset_spec = DATASET_SPECS[dataset_name]
    print(f"Input dimension: {dataset_spec['input_dim']}")
    print(f"Number of classes: {dataset_spec['num_classes']}")
    print(f"{'='*60}\n")

    # Build model constructor
    model_constructor = lambda: build_model_for(dataset_name, train_ds, SimpleMLP)
    print(f"Using SimpleMLP model (input_dim={dataset_spec['input_dim']}, "
          f"num_classes={dataset_spec['num_classes']})")

    # Loop through all batch strategies
    for strategy_label, strategy_module_name in VISION_BATCH_STRATEGIES.items():
        print(f"\n==== Batching strategy: {strategy_label} ====")

        # Dynamically load the batch strategy module
        module_name = f"tasks.vision.batch_strategies.{strategy_module_name}"
        batch_module = importlib.import_module(module_name)
        batch_strategy = batch_module.batch_sampler

        # Create directory for this run
        run_dir = create_run_dir(strategy_label, dataset_name)

        # Run all experiments for this strategy
        results = run_experiment(
            batch_strategy, strategy_label, train_ds, test_ds, model_constructor,
            EPOCHS, BATCH_SIZE, N_RUNS
        )

        # Aggregate results
        means, cis = aggregate_results(results)

        # Save plots and summary
        save_plots(run_dir, strategy_label, means, cis, EPOCHS)
        save_summary(run_dir, means, cis, EPOCHS)

        print(f"\nResults for {strategy_label} saved to: {run_dir}")


if __name__ == '__main__':
    main()
