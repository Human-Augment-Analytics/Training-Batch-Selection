import importlib
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
from torch.utils.data import DataLoader
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

# ============ Train Function (Batch strategy as argument) =============
def train_model(model, train_ds, test_ds, epochs, batch_size, batch_strategy,
                loss_kwargs={}, batch_kwargs={}, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Move model to device
    model = model.to(DEVICE)

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
        # Choose batch_sampler based on required parameters
        if batch_strategy.__name__ == "batch_sampler":
            # Check what parameters the batch strategy needs
            params = batch_strategy.__code__.co_varnames
            if "loss_history" in params:
                # Smart batching (loss-based)
                batch_iter = batch_strategy(train_ds, batch_size, loss_history=per_sample_loss)
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

# ============ MAIN ============

if __name__ == '__main__':
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

