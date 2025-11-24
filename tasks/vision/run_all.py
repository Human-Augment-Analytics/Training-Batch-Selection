#!/usr/bin/env python3
"""
Unified runner for complete vision batch selection pipeline.

This single command runs:
1. Main experiments for all batch strategies
2. Comparison plots between strategies
3. Optional benchmark across multiple datasets

Usage:
    # Run everything on default dataset
    python -m tasks.vision.run_all

    # Run on specific dataset
    python -m tasks.vision.run_all --dataset cifar10_csv

    # Run experiments + comparisons only (skip benchmark)
    python -m tasks.vision.run_all --no-benchmark

    # Run benchmark on specific datasets
    python -m tasks.vision.run_all --benchmark-datasets mnist_csv qmnist_csv

    # Quick test mode (1 epoch, 1 run)
    python -m tasks.vision.run_all --quick
"""
import importlib
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from config.base import (
    BASE_DIR, DATASETS_ROOT, OUTPUTS_ROOT, DEVICE, USE_GPU,
    NUM_WORKERS, PIN_MEMORY, RANDOM_SEED
)
from config.vision import (
    EPOCHS, BATCH_SIZE, N_RUNS, ACTIVE_DATASET
)
from config.datasets import DATASET_SPECS
from config.batch_strategies import VISION_BATCH_STRATEGIES, VISION_STRATEGY_COMPARISON_PAIRS

from tasks.vision.models.mlp import SimpleMLP
from tasks.vision.datasets.factory import build_dataset, build_model_for
from tasks.vision.run_experiment import (
    run_experiment, aggregate_results, save_plots, save_summary, create_run_dir
)
from tasks.vision.compare import load_summary, find_latest_run_path
from scipy import stats


def display_header(title):
    """Display formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def display_device_info():
    """Display GPU/CPU configuration."""
    display_header("DEVICE CONFIGURATION")
    print(f"Device: {DEVICE.upper()}")
    if USE_GPU:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Running on CPU (GPU not available)")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Number of workers: {NUM_WORKERS}")
    print(f"Pin memory: {PIN_MEMORY}")


def run_all_experiments(dataset_name, epochs, batch_size, n_runs):
    """Run experiments for all batch strategies."""
    display_header(f"RUNNING EXPERIMENTS - {dataset_name}")

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    train_ds, test_ds = build_dataset(str(DATASETS_ROOT), dataset_name)
    dataset_spec = DATASET_SPECS[dataset_name]
    print(f"✓ Dataset loaded: {len(train_ds)} train, {len(test_ds)} test samples")
    print(f"  Input dim: {dataset_spec['input_dim']}, Classes: {dataset_spec['num_classes']}")

    # Build model constructor
    model_constructor = lambda: build_model_for(dataset_name, train_ds, SimpleMLP)

    # Track completed strategies
    completed = []

    # Run each strategy
    for strategy_label, strategy_module_name in VISION_BATCH_STRATEGIES.items():
        print(f"\n{'-'*70}")
        print(f"Strategy: {strategy_label}")
        print(f"{'-'*70}")

        try:
            # Load strategy
            module_name = f"tasks.vision.batch_strategies.{strategy_module_name}"
            batch_module = importlib.import_module(module_name)
            batch_strategy = batch_module.batch_sampler

            # Create output directory
            run_dir = create_run_dir(strategy_label, dataset_name)

            # Run experiments
            results = run_experiment(
                batch_strategy, strategy_label, train_ds, test_ds, model_constructor,
                epochs, batch_size, n_runs
            )

            # Aggregate and save
            means, cis = aggregate_results(results)
            save_plots(run_dir, strategy_label, means, cis, epochs)
            save_summary(run_dir, means, cis, epochs)

            print(f"✓ Results saved to: {run_dir}")
            completed.append(strategy_label)

        except Exception as e:
            print(f"✗ Error running {strategy_label}: {e}")

    return completed


def run_all_comparisons(dataset_name, epochs):
    """Run comparisons between strategy pairs."""
    display_header(f"GENERATING COMPARISONS - {dataset_name}")

    if not VISION_STRATEGY_COMPARISON_PAIRS:
        print("No comparison pairs configured.")
        return

    print(f"Comparison pairs: {VISION_STRATEGY_COMPARISON_PAIRS}")

    for pair in VISION_STRATEGY_COMPARISON_PAIRS:
        print(f"\n{'-'*70}")
        print(f"Comparing: {pair[0]} vs {pair[1]}")
        print(f"{'-'*70}")

        try:
            # Create output directory
            out_dir = OUTPUTS_ROOT / 'vision' / dataset_name / f'comparison_{"_".join([p.lower() for p in pair])}'
            out_dir.mkdir(parents=True, exist_ok=True)

            # Load results for both strategies
            epochs_range = np.arange(1, epochs + 1)
            colors = ["tab:blue", "tab:orange"]
            method_results = []

            for idx, method in enumerate(pair):
                run_dir = find_latest_run_path(method, dataset_name)
                means, cis = load_summary(run_dir)
                method_results.append((method, means, cis, colors[idx]))

            # Create comparison plots
            metric_labels = {
                "test_acc": "Test Accuracy",
                "train_acc": "Train Accuracy",
                "train_loss": "Train Loss",
            }

            for metric in ["test_acc", "train_acc", "train_loss"]:
                plt.figure(figsize=(7, 5))

                for (name, means, cis, col) in method_results:
                    if metric in means and len(means[metric]) > 0:
                        m = np.array(means[metric])
                        c = np.array(cis[metric])
                        plt.plot(epochs_range[:len(m)], m, label=name, linewidth=2, color=col)
                        plt.fill_between(epochs_range[:len(m)], m - c, m + c, alpha=0.2, color=col)

                plt.xlabel('Epoch')
                plt.ylabel(metric_labels[metric])
                plt.title(f'{metric_labels[metric]} Comparison')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(out_dir / f"{metric}_cmp.png")
                plt.close()

            print(f"✓ Comparison plots saved to: {out_dir}")

        except Exception as e:
            print(f"✗ Error comparing {pair}: {e}")


def run_benchmark(datasets, strategy='Random', epochs=EPOCHS, batch_size=BATCH_SIZE, n_runs=N_RUNS):
    """Run benchmark of one strategy across multiple datasets."""
    display_header(f"BENCHMARKING {strategy} ACROSS DATASETS")

    print(f"Strategy: {strategy}")
    print(f"Datasets: {datasets}")
    print(f"Epochs: {epochs}, Runs: {n_runs}")

    # Validate strategy
    if strategy not in VISION_BATCH_STRATEGIES:
        print(f"✗ Error: Unknown strategy '{strategy}'")
        return

    # Load strategy
    module_name = f"tasks.vision.batch_strategies.{VISION_BATCH_STRATEGIES[strategy]}"
    batch_module = importlib.import_module(module_name)
    batch_strategy = batch_module.batch_sampler

    # Create output directory
    out_dir = OUTPUTS_ROOT / 'vision' / 'benchmarks' / f'{strategy.lower()}_multi_dataset'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open summary file
    summary_file = open(out_dir / 'summary.txt', 'w')
    summary_file.write(f"Benchmark: {strategy} across datasets\n")
    summary_file.write(f"Epochs: {epochs}, Runs: {n_runs}\n")
    summary_file.write(f"Datasets: {', '.join(datasets)}\n")
    summary_file.write("="*70 + "\n")

    # Run for each dataset
    for dataset_name in datasets:
        print(f"\n{'-'*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'-'*70}")

        try:
            # Validate dataset
            if dataset_name not in DATASET_SPECS:
                print(f"✗ Skipping unknown dataset: {dataset_name}")
                continue

            # Load dataset
            train_ds, test_ds = build_dataset(str(DATASETS_ROOT), dataset_name)
            print(f"✓ Loaded: {len(train_ds)} train, {len(test_ds)} test samples")

            # Build model
            model_constructor = lambda: build_model_for(dataset_name, train_ds, SimpleMLP)

            # Run experiments
            results = run_experiment(
                batch_strategy, strategy, train_ds, test_ds,
                model_constructor, epochs, batch_size, n_runs
            )

            # Aggregate results
            means, cis = aggregate_results(results)

            # Save to summary file
            summary_file.write(f"\n{dataset_name}:\n")
            for i in range(len(means["train_acc"])):
                summary_file.write(
                    f"Epoch {i+1}: "
                    f"train_acc={means['train_acc'][i]:.4f}±{cis['train_acc'][i]:.4f}, "
                    f"test_acc={means['test_acc'][i]:.4f}±{cis['test_acc'][i]:.4f}, "
                    f"train_loss={means['train_loss'][i]:.4f}±{cis['train_loss'][i]:.4f}\n"
                )
            summary_file.write(f"Time: {means['time']:.2f}±{cis['time']:.2f} sec\n")

            # Save plots
            epochs_range = np.arange(1, epochs + 1)

            for metric, ylabel in [('test_acc', 'Test Accuracy'), ('train_acc', 'Train Accuracy'),
                                   ('train_loss', 'Train Loss'), ('test_loss', 'Test Loss')]:
                plt.figure(figsize=(7, 5))
                plt.plot(epochs_range, means[metric], label=dataset_name, linewidth=2)
                plt.fill_between(epochs_range, means[metric] - cis[metric],
                               means[metric] + cis[metric], alpha=0.2)
                plt.xlabel('Epoch')
                plt.ylabel(ylabel)
                plt.title(f"{ylabel} - {dataset_name}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(out_dir / f'{dataset_name}_{metric}.png')
                plt.close()

            print(f"✓ Completed {dataset_name}")

        except Exception as e:
            print(f"✗ Error with {dataset_name}: {e}")
            summary_file.write(f"\n{dataset_name}: ERROR - {e}\n")

    summary_file.close()
    print(f"\n✓ Benchmark results saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Run complete vision batch selection pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything (experiments + comparisons + benchmark)
  python -m tasks.vision.run_all

  # Run on specific dataset
  python -m tasks.vision.run_all --dataset cifar10_csv

  # Skip benchmark
  python -m tasks.vision.run_all --no-benchmark

  # Custom benchmark datasets
  python -m tasks.vision.run_all --benchmark-datasets mnist_csv qmnist_csv

  # Quick test (1 epoch, 1 run)
  python -m tasks.vision.run_all --quick
        """
    )

    parser.add_argument('--dataset', type=str, default=ACTIVE_DATASET,
                       help=f'Primary dataset for experiments (default: {ACTIVE_DATASET})')
    parser.add_argument('--datasets', type=str, nargs='+',
                       help='Run on multiple specific datasets (e.g., --datasets mnist_csv cifar10_csv qmnist_csv)')
    parser.add_argument('--all-datasets', action='store_true',
                       help='Run experiments on all vision datasets sequentially')
    parser.add_argument('--no-benchmark', action='store_true',
                       help='Skip benchmark across datasets')
    parser.add_argument('--benchmark-datasets', type=str, nargs='+',
                       default=['mnist_csv', 'qmnist_csv'],
                       help='Datasets for benchmark (default: mnist_csv qmnist_csv)')
    parser.add_argument('--benchmark-strategy', type=str, default='Random',
                       help='Strategy to use for benchmark (default: Random)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode: 1 epoch, 1 run')

    args = parser.parse_args()

    # Handle quick mode
    if args.quick:
        epochs = 1
        n_runs = 1
        print("\n⚡ QUICK TEST MODE: 1 epoch, 1 run")
    else:
        epochs = EPOCHS
        n_runs = N_RUNS

    # Start timer
    start_time = time.time()

    # Display device info
    display_device_info()

    # Determine which datasets to run
    if args.all_datasets:
        # Get all vision datasets
        vision_datasets = [name for name in DATASET_SPECS.keys() if '_csv' in name]
        datasets_to_run = vision_datasets
        print(f"\n🔄 Running on ALL vision datasets: {', '.join(datasets_to_run)}")
    elif args.datasets:
        # Run on user-specified list of datasets
        datasets_to_run = args.datasets
        # Validate all datasets
        invalid = [d for d in datasets_to_run if d not in DATASET_SPECS]
        if invalid:
            print(f"\n✗ Error: Unknown datasets: {', '.join(invalid)}")
            print(f"Available datasets: {list(DATASET_SPECS.keys())}")
            return 1
        print(f"\n🔄 Running on specified datasets: {', '.join(datasets_to_run)}")
    else:
        # Validate single dataset
        if args.dataset not in DATASET_SPECS:
            print(f"\n✗ Error: Unknown dataset '{args.dataset}'")
            print(f"Available datasets: {list(DATASET_SPECS.keys())}")
            return 1
        datasets_to_run = [args.dataset]

    # Track all completed experiments across datasets
    all_completed = {}

    # Step 1: Run all experiments for each dataset
    for dataset_name in datasets_to_run:
        print(f"\n{'='*70}")
        print(f"PROCESSING DATASET: {dataset_name}")
        print(f"{'='*70}")

        completed = run_all_experiments(dataset_name, epochs, BATCH_SIZE, n_runs)

        if not completed:
            print(f"\n⚠️  No experiments completed successfully for {dataset_name}.")
            continue

        all_completed[dataset_name] = completed

        # Step 2: Run comparisons for this dataset
        run_all_comparisons(dataset_name, epochs)

    if not all_completed:
        print("\n✗ No experiments completed successfully on any dataset.")
        return 1

    # Step 3: Run benchmark (optional)
    if not args.no_benchmark:
        run_benchmark(args.benchmark_datasets, args.benchmark_strategy, epochs, BATCH_SIZE, n_runs)
    else:
        print("\n⏭️  Skipping benchmark (--no-benchmark flag)")

    # Display summary
    elapsed = time.time() - start_time
    display_header("PIPELINE COMPLETE")
    print(f"✓ Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"✓ Results saved to: {OUTPUTS_ROOT / 'vision'}")

    print(f"\nCompleted datasets and strategies:")
    for dataset, strategies in all_completed.items():
        print(f"  {dataset}: {', '.join(strategies)}")

    print(f"\nOutput structure:")
    for dataset in all_completed.keys():
        print(f"  {OUTPUTS_ROOT / 'vision' / dataset}/")
        for strategy in all_completed[dataset]:
            print(f"    ├── batching_{strategy.lower()}/")
        if VISION_STRATEGY_COMPARISON_PAIRS:
            print(f"    ├── comparison_*/")
    if not args.no_benchmark:
        print(f"  {OUTPUTS_ROOT / 'vision' / 'benchmarks'}/")

    return 0


if __name__ == '__main__':
    exit(main())
