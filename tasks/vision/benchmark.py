#!/usr/bin/env python3
"""
Benchmark a single strategy across multiple datasets.

Usage:
    python -m tasks.vision.benchmark
    python -m tasks.vision.benchmark --strategy Smart
    python -m tasks.vision.benchmark --datasets mnist_csv qmnist_csv cifar10_csv
"""
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config.base import DATASETS_ROOT, OUTPUTS_ROOT
from config.vision import EPOCHS, BATCH_SIZE, N_RUNS
from config.datasets import DATASET_SPECS
from config.batch_strategies import VISION_BATCH_STRATEGIES

from tasks.vision.models.mlp import SimpleMLP
from tasks.vision.datasets.factory import build_dataset, build_model_for
from tasks.vision.run_experiment import run_experiment, aggregate_results


def save_summary(dataset_name, means, cis, file):
    """Write summary statistics to file."""
    file.write(f"\n{dataset_name}:\n")
    for i in range(len(means["train_acc"])):
        file.write(
            f"Epoch {i+1}: "
            f"train_acc={means['train_acc'][i]:.4f}±{cis['train_acc'][i]:.4f}, "
            f"test_acc={means['test_acc'][i]:.4f}±{cis['test_acc'][i]:.4f}, "
            f"train_loss={means['train_loss'][i]:.4f}±{cis['train_loss'][i]:.4f}\n"
        )
    file.write(f"Time: {means['time']:.2f}±{cis['time']:.2f} sec\n")


def plot_metric(metric, ylabel, title, filename, means, cis, epochs_axis, run_dir, dataset_name):
    """Create plot for a metric."""
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_axis, means[metric], label=dataset_name, linewidth=2)
    plt.fill_between(
        epochs_axis, means[metric] - cis[metric],
        means[metric] + cis[metric], alpha=0.2
    )
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f"{title} - {dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / filename)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Benchmark strategy across datasets')
    parser.add_argument('--strategy', type=str, default='Random',
                       help='Batch strategy to benchmark (default: Random)')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['mnist_csv', 'qmnist_csv', 'cifar10_csv'],
                       help='Datasets to benchmark')
    args = parser.parse_args()

    strategy_name = args.strategy
    datasets = args.datasets

    # Validate strategy
    if strategy_name not in VISION_BATCH_STRATEGIES:
        print(f"Error: Unknown strategy '{strategy_name}'")
        print(f"Available: {list(VISION_BATCH_STRATEGIES.keys())}")
        return

    # Validate datasets
    for ds in datasets:
        if ds not in DATASET_SPECS:
            print(f"Error: Unknown dataset '{ds}'")
            print(f"Available: {list(DATASET_SPECS.keys())}")
            return

    print(f"\n{'='*60}")
    print(f"BENCHMARKING ACROSS DATASETS")
    print(f"{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"Datasets: {datasets}")
    print(f"Epochs: {EPOCHS}, Runs: {N_RUNS}")
    print(f"{'='*60}\n")

    # Load strategy
    module_name = f"tasks.vision.batch_strategies.{VISION_BATCH_STRATEGIES[strategy_name]}"
    batch_module = importlib.import_module(module_name)
    batch_strategy = batch_module.batch_sampler

    # Create output directory
    out_dir = OUTPUTS_ROOT / 'vision' / 'benchmarks' / f'{strategy_name.lower()}_multi_dataset'
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_file = open(out_dir / 'summary.txt', 'w')
    summary_file.write(f"Benchmark: {strategy_name} across datasets\n")
    summary_file.write(f"Epochs: {EPOCHS}, Runs: {N_RUNS}\n")
    summary_file.write(f"Datasets: {', '.join(datasets)}\n")
    summary_file.write("="*60 + "\n")

    # Run experiments for each dataset
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        try:
            # Load dataset
            train_ds, test_ds = build_dataset(str(DATASETS_ROOT), dataset_name)
            print(f"Loaded: {len(train_ds)} train, {len(test_ds)} test samples")

            # Build model
            model_constructor = lambda: build_model_for(dataset_name, train_ds, SimpleMLP)

            # Run experiments
            results = run_experiment(
                batch_strategy, strategy_name, train_ds, test_ds,
                model_constructor, EPOCHS, BATCH_SIZE, N_RUNS
            )

            # Aggregate results
            means, cis = aggregate_results(results)

            # Save summary
            save_summary(dataset_name, means, cis, summary_file)

            # Save plots
            epochs_axis = np.arange(1, EPOCHS + 1)
            plot_metric('test_acc', 'Test Accuracy', 'Test Accuracy',
                       f'{dataset_name}_test_acc.png', means, cis, epochs_axis,
                       out_dir, dataset_name)
            plot_metric('train_acc', 'Train Accuracy', 'Train Accuracy',
                       f'{dataset_name}_train_acc.png', means, cis, epochs_axis,
                       out_dir, dataset_name)
            plot_metric('train_loss', 'Train Loss', 'Train Loss',
                       f'{dataset_name}_train_loss.png', means, cis, epochs_axis,
                       out_dir, dataset_name)
            plot_metric('test_loss', 'Test Loss', 'Test Loss',
                       f'{dataset_name}_test_loss.png', means, cis, epochs_axis,
                       out_dir, dataset_name)

            print(f"✓ Completed {dataset_name}")

        except Exception as e:
            print(f"✗ Error with {dataset_name}: {e}")
            summary_file.write(f"\n{dataset_name}: ERROR - {e}\n")

    summary_file.close()
    print(f"\n{'='*60}")
    print(f"Benchmark complete! Results saved to:")
    print(f"  {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
