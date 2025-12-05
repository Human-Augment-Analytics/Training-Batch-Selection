import importlib
from trainer.batching.vision_batching.rho_loss_batch import compute_irreducible_losses, train_irreducible_loss_model
from trainer.constants import (
    BASE_DIR, TRAIN_CSV, TEST_CSV, OUTPUT_DIR,
    EPOCHS, BATCH_SIZE, N_RUNS, RANDOM_SEED, DEVICE, MOVING_AVG_DECAY
)
from trainer.model.vision.model import SimpleMLP
from trainer.dataloader.vision_dataloader import MNISTCsvDataset

# NEW import
from trainer.constants_batch_strategy import BATCH_STRATEGIES

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from scipy import stats

# Load dataset
train_ds = MNISTCsvDataset(TRAIN_CSV)
test_ds = MNISTCsvDataset(TEST_CSV)

# Print device info
print(f"Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============ Helpers ============

def split_train_holdout(train_ds, holdout_frac=0.1, seed=None):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(train_ds))
    rng.shuffle(indices)
    n_holdout = max(1, int(len(train_ds) * holdout_frac))
    holdout_indices = indices[:n_holdout]
    train_indices = indices[n_holdout:]
    return Subset(train_ds, train_indices), Subset(train_ds, holdout_indices)


# ============ Train Function (Batch strategy as argument) =============
def train_model(model, train_ds, test_ds, epochs, batch_size, batch_strategy,
                loss_kwargs={}, batch_kwargs={}, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Move model to device
    model = model.to(DEVICE)

    # By default train on the provided training set; RHO-Loss overrides this to a split
    il_losses = None
    if "rho_loss" in batch_strategy.__module__:
        train_ds, holdout_ds = split_train_holdout(train_ds, holdout_frac=0.1, seed=seed)
        irreducible_loss_model = train_irreducible_loss_model(model, holdout_ds, epochs=5, device=DEVICE)
        il_losses = compute_irreducible_losses(irreducible_loss_model, train_ds, device=DEVICE)

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss(**loss_kwargs)

    # For gradient-based batching, we need a loss function with reduction='none'
    loss_fn_per_sample = nn.CrossEntropyLoss(reduction='none')

    # Prepare loss history for smart batch
    per_sample_loss = np.zeros(len(train_ds))
    train_accs, test_accs, train_losses, test_losses = [],[],[],[]

    for epoch in range(epochs):
        correct, n, running_loss = 0, 0, 0
        model.train()
        if batch_strategy.__name__ == "batch_sampler":
            # Check what parameters the batch strategy needs
            params = batch_strategy.__code__.co_varnames
            if "loss_history" in params:
                # Smart batching (loss-based)
                batch_iter = batch_strategy(train_ds, batch_size, loss_history=per_sample_loss)
            elif "irreducible_losses" in params:
                # RHO-LOSS
                batch_iter = batch_strategy(train_ds, batch_size, model=model, loss_fn=loss_fn_per_sample, irreducible_losses=il_losses, device=DEVICE)
            elif "model" in params and "loss_fn" in params:
                # Gradient-based batching (GraND)
                batch_iter = batch_strategy(train_ds, batch_size, model=model, loss_fn=loss_fn_per_sample, device=DEVICE)
            else:
                # Default batching (fixed/random)
                batch_iter = batch_strategy(train_ds, batch_size)
        elif batch_strategy.__name__ == "great_batch_sampler":
            # GREAT batch sampler
            batch_iter = batch_strategy(train_ds, batch_size, model=model, loss_fn=loss_fn_per_sample, device=DEVICE)
        else:
            batch_iter = batch_strategy(train_ds, batch_size)

        # Main batch loop
        for idxs in batch_iter:
            x, y = zip(*[train_ds[i] for i in idxs])
            x = torch.stack(x).view(len(idxs), -1).to(DEVICE)
            y = torch.tensor(y).to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(x)
            losses = loss_fn(y_pred, y)
            if len(losses.shape)>0: # smart batch: reduction=none
                loss = losses.mean()
            else:
                loss = losses
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (y_pred.argmax(1) == y).sum().item()
            n += x.size(0)

            # Update per-sample loss for smart batching
            if "loss_history" in batch_strategy.__code__.co_varnames:
                for k, idx in enumerate(idxs):
                    per_sample_loss[idx] = MOVING_AVG_DECAY * per_sample_loss[idx] + (1 - MOVING_AVG_DECAY) * \
                                          (losses[k].item() if len(losses.shape) > 0 else losses.item())

        train_accs.append(correct / n)
        train_losses.append(running_loss / n)
        test_acc, test_loss = evaluate(model, test_ds)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}: train_acc={train_accs[-1]:.4f}, test_acc={test_acc:.4f}")

    return train_accs, test_accs, train_losses, test_losses

# ============ Evaluation Function ============

def evaluate(model, ds):
    loader = DataLoader(ds, batch_size=256)
    model.eval()
    correct, n, total_loss = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.size(0), -1).to(DEVICE)
            y = y.to(DEVICE)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item() * x.size(0)
            correct += (y_pred.argmax(1) == y).sum().item()
            n += x.size(0)
    return correct / n, total_loss / n

# ============ Output Directory ============

def create_run_dir(strategy_name):
    out_path = os.path.join(OUTPUT_DIR, f'batching_{strategy_name.lower()}')
    os.makedirs(out_path, exist_ok=True)
    existing = [d for d in os.listdir(out_path) if d.startswith('run-')]
    next_num = max([int(d.split('-')[1]) for d in existing], default=0) + 1
    run_dir = os.path.join(out_path, f'run-{next_num:03d}')
    os.makedirs(run_dir)
    return run_dir

# ============ Experiment Runner ============

def run_experiment(batch_strategy, strategy_label, train_ds, test_ds, model_constructor, epochs, batch_size, n_runs):
    results = []
    for seed in range(n_runs):
        print(f"\n[{strategy_label}] Run {seed+1}/{n_runs}")
        model = model_constructor()
        start = time.time()
        train_acc, test_acc, train_loss, test_loss = train_model(
            model, train_ds, test_ds, epochs, batch_size, batch_strategy, seed=seed
        )
        elapsed = time.time() - start
        results.append({
            "train_acc": train_acc, "test_acc": test_acc,
            "train_loss": train_loss, "test_loss": test_loss,
            "time": elapsed
        })
    return results

# ============ CI calculation ============

def mean_and_ci(arrays, axis=0):
    arr = np.array(arrays)
    mean = arr.mean(axis=0)
    sem = stats.sem(arr, axis=0)
    ci = sem * stats.t.ppf((1 + 0.95) / 2., arr.shape[0] - 1)
    return mean, ci

def aggregate_results(results):
    means, cis = {}, {}
    for metric in ["train_acc", "test_acc", "train_loss", "test_loss"]:
        mean, ci = mean_and_ci([run[metric] for run in results])
        means[metric], cis[metric] = mean, ci
    mean_time, ci_time = mean_and_ci([run["time"] for run in results], axis=0)
    means["time"], cis["time"] = mean_time, ci_time
    return means, cis

# ============ Comparison Plots ============

def create_comparison_plots(all_results, epochs):
    """Create comparison plots for all batch strategies."""
    comparison_dir = os.path.join(OUTPUT_DIR, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    # Find latest run number across all strategies
    existing = [d for d in os.listdir(comparison_dir) if d.startswith('run-')]
    next_num = max([int(d.split('-')[1]) for d in existing], default=0) + 1
    comp_run_dir = os.path.join(comparison_dir, f'run-{next_num:03d}')
    os.makedirs(comp_run_dir)

    epochs_range = np.arange(1, epochs+1)

    # Define metrics to plot
    metrics_info = [
        ('test_acc', 'Test Accuracy', 'Test Accuracy Comparison'),
        ('train_acc', 'Train Accuracy', 'Train Accuracy Comparison'),
        ('test_loss', 'Test Loss', 'Test Loss Comparison'),
        ('train_loss', 'Train Loss', 'Train Loss Comparison')
    ]

    for metric, ylabel, title in metrics_info:
        plt.figure(figsize=(10, 6))
        for strategy_label, (means, cis) in all_results.items():
            plt.plot(epochs_range, means[metric], label=strategy_label, linewidth=2)
            plt.fill_between(epochs_range,
                           means[metric] - cis[metric],
                           means[metric] + cis[metric],
                           alpha=0.2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(comp_run_dir, f'{metric}_comparison.png'), dpi=150)
        plt.close()

    # Create training time comparison bar chart
    plt.figure(figsize=(10, 6))
    strategies = list(all_results.keys())
    times = [all_results[s][0]['time'] for s in strategies]
    time_cis = [all_results[s][1]['time'] for s in strategies]

    bars = plt.bar(range(len(strategies)), times, yerr=time_cis, capsize=5, alpha=0.7)
    plt.xticks(range(len(strategies)), strategies, rotation=45, ha='right')
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.title('Training Time Comparison', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(comp_run_dir, 'training_time_comparison.png'), dpi=150)
    plt.close()

    # Create summary table
    with open(os.path.join(comp_run_dir, "comparison_summary.txt"), "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BATCH STRATEGY COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for strategy_label, (means, cis) in all_results.items():
            f.write(f"\n{strategy_label}:\n")
            f.write("-" * 40 + "\n")
            final_train_acc = means['train_acc'][-1]
            final_test_acc = means['test_acc'][-1]
            final_train_loss = means['train_loss'][-1]
            final_test_loss = means['test_loss'][-1]

            f.write(f"  Final Train Accuracy: {final_train_acc:.4f} ± {cis['train_acc'][-1]:.4f}\n")
            f.write(f"  Final Test Accuracy:  {final_test_acc:.4f} ± {cis['test_acc'][-1]:.4f}\n")
            f.write(f"  Final Train Loss:     {final_train_loss:.4f} ± {cis['train_loss'][-1]:.4f}\n")
            f.write(f"  Final Test Loss:      {final_test_loss:.4f} ± {cis['test_loss'][-1]:.4f}\n")
            f.write(f"  Training Time:        {means['time']:.2f} ± {cis['time']:.2f} sec\n")

        f.write("\n" + "=" * 80 + "\n")

    return comp_run_dir

# ============ MAIN ============

if __name__ == '__main__':
    # Dictionary to store all results for comparison
    all_results = {}

    # Loop over batch strategies from config/constants
    for strategy_label, strategy_path in BATCH_STRATEGIES.items():
        print(f"\n==== Running with batching strategy: {strategy_label} ====")
        # Import batching function dynamically
        module_name = f"trainer.batching.vision_batching.{strategy_path}".replace("/", ".").replace("\\", ".")
        batch_module = importlib.import_module(module_name)
        batch_strategy = batch_module.batch_sampler

        run_dir = create_run_dir(strategy_label)
        means, cis = None, None

        results = run_experiment(
            batch_strategy, strategy_label, train_ds, test_ds, SimpleMLP,
            EPOCHS, BATCH_SIZE, N_RUNS
        )
        means, cis = aggregate_results(results)

        # Store results for comparison
        all_results[strategy_label] = (means, cis)

        # Save plots and summaries
        epochs_range = np.arange(1, EPOCHS+1)
        def plot_metric(metric, ylabel, title, filename):
            plt.figure(figsize=(7, 5))
            plt.plot(epochs_range, means[metric], label=strategy_label, linewidth=2)
            plt.fill_between(epochs_range, means[metric] - cis[metric], means[metric] + cis[metric], alpha=0.2)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(f"{title} - {strategy_label}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, filename))
            plt.close()
        plot_metric('test_acc', 'Test Accuracy', "Test Accuracy vs Epoch", "test_acc.png")
        plot_metric('train_acc', 'Train Accuracy', "Train Accuracy vs Epoch", "train_acc.png")
        plot_metric('train_loss', 'Train Loss', "Train Loss vs Epoch", "train_loss.png")
        plot_metric('test_loss', 'Test Loss', "Test Loss vs Epoch", "test_loss.png")
        # Save summary
        with open(os.path.join(run_dir, "summary.txt"), "w") as f:
            for i in range(EPOCHS):
                f.write(f"Epoch {i+1}: train_acc={means['train_acc'][i]:.4f}±{cis['train_acc'][i]:.4f}, "
                        f"test_acc={means['test_acc'][i]:.4f}±{cis['test_acc'][i]:.4f}, "
                        f"train_loss={means['train_loss'][i]:.4f}±{cis['train_loss'][i]:.4f}\n")
            f.write(f"CPU Time: {means['time']:.2f}±{cis['time']:.2f} sec\n")
        print(f"\n✅ All results for {strategy_label} saved to: {run_dir}")

    # Create comparison plots
    print("\n==== Creating comparison plots ====")
    comp_dir = create_comparison_plots(all_results, EPOCHS)
    print(f"\n✅ Comparison plots saved to: {comp_dir}")
