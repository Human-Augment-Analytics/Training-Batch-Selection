import importlib
import argparse
from trainer.constants import (
    BASE_DIR, OUTPUT_DIR, DATASETS_ROOT, ACTIVE_DATASET,
    EPOCHS, BATCH_SIZE, N_RUNS, RANDOM_SEED, DEVICE, MOVING_AVG_DECAY
)
from trainer.model.vision.model import SimpleMLP
from trainer.dataloader.factory import build_dataset, build_model_for
from trainer.constants_batch_strategy import BATCH_STRATEGIES
from trainer.constants_datasets import DATASET_SPECS
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import stats

# trying to use the config system if it exists, not sure if this works properly
try:
    from trainer.constants_models import (
        ACTIVE_VISION_MODEL, ACTIVE_VISION_TRAINING, get_training_config
    )
    from trainer.factories import create_model, create_optimizer
    HAS_CONFIG_SYSTEM = True
except ImportError:
    HAS_CONFIG_SYSTEM = False

# Main training function - takes batch strategy as parameter
# TODO: maybe add learning rate scheduler later?
def train_model(model, train_ds, test_ds, epochs, batch_size, batch_strategy,
                loss_kwargs={}, batch_kwargs={}, seed=None, optimizer_name="Adam"):
    # set random seeds for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    # print(f"Training with seed: {seed}")  # debug

    # Use config system if available
    if HAS_CONFIG_SYSTEM and optimizer_name != "Adam":
        optimizer = create_optimizer(optimizer_name, model.parameters())
    else:
        optimizer = torch.optim.Adam(model.parameters())  # Adam usually works well

    loss_fn = nn.CrossEntropyLoss(**loss_kwargs)

    # need to track per-sample loss for smart batching strategies
    per_sample_loss = np.zeros(len(train_ds))
    train_accs, test_accs, train_losses, test_losses = [], [], [], []

    for epoch in range(epochs):
        correct = 0
        n = 0
        running_loss = 0
        model.train()

        # check if batch strategy needs loss_history parameter (for smart batching)
        # this took a while to figure out...
        if batch_strategy.__name__ == "batch_sampler" and "loss_history" in batch_strategy.__code__.co_varnames:
            batch_iter = batch_strategy(train_ds, batch_size, loss_history=per_sample_loss)
        else:
            batch_iter = batch_strategy(train_ds, batch_size)

        # iterate through batches
        for idxs in batch_iter:
            # get batch data
            x, y = zip(*[train_ds[i] for i in idxs])
            x = torch.stack(x).view(len(idxs), -1).to(DEVICE)
            y = torch.tensor(y).to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(x)
            losses = loss_fn(y_pred, y)

            # handle both reduction='none' and regular loss
            if len(losses.shape)>0:
                loss = losses.mean()
            else:
                loss = losses

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (y_pred.argmax(1) == y).sum().item()
            n += x.size(0)

            # update per-sample loss history (exponential moving average)
            if "loss_history" in batch_strategy.__code__.co_varnames:
                for k, idx in enumerate(idxs):
                    # EMA update: new = alpha * old + (1-alpha) * current
                    per_sample_loss[idx] = MOVING_AVG_DECAY * per_sample_loss[idx] + (1 - MOVING_AVG_DECAY) * \
                                          (losses[k].item() if len(losses.shape) > 0 else losses.item())

        # calculate metrics for this epoch
        train_accs.append(correct / n)
        train_losses.append(running_loss / n)
        test_acc, test_loss = evaluate(model, test_ds)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}: train_acc={train_accs[-1]:.4f}, test_acc={test_acc:.4f}")

    return train_accs, test_accs, train_losses, test_losses

# Evaluation function
def evaluate(model, ds):
    loader = DataLoader(ds, batch_size=256)  # can use larger batch for eval
    model.eval()
    correct, n, total_loss = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.size(0), -1)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item() * x.size(0)
            correct += (y_pred.argmax(1) == y).sum().item()
            n += x.size(0)

    return correct / n, total_loss / n

# helper to create output directory structure
def create_run_dir(strategy_name):
    out_path = os.path.join(OUTPUT_DIR, f'batching_{strategy_name.lower()}')
    os.makedirs(out_path, exist_ok=True)

    # find existing runs and increment
    existing = [d for d in os.listdir(out_path) if d.startswith('run-')]
    if len(existing) > 0:
        next_num = max([int(d.split('-')[1]) for d in existing]) + 1
    else:
        next_num = 1

    run_dir = os.path.join(out_path, f'run-{next_num:03d}')
    os.makedirs(run_dir)
    return run_dir

# Run experiments for a batch strategy
def run_experiment(batch_strategy, strategy_label, train_ds, test_ds, model_constructor, epochs, batch_size, n_runs):
    results = []
    for seed in range(n_runs):
        print(f"\n[{strategy_label}] Run {seed+1}/{n_runs}")

        # create fresh model for each run
        model = model_constructor()
        start = time.time()

        train_acc, test_acc, train_loss, test_loss = train_model(
            model, train_ds, test_ds, epochs, batch_size, batch_strategy, seed=seed
        )

        elapsed = time.time() - start
        # print(f"Run {seed+1} took {elapsed:.2f} seconds")  # debug

        results.append({
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "time": elapsed
        })

    return results

# Calculate mean and confidence intervals
# using scipy.stats for the t-distribution
def mean_and_ci(arrays, axis=0):
    arr = np.array(arrays)
    mean = arr.mean(axis=0)
    sem = stats.sem(arr, axis=0)  # standard error of mean
    # 95% confidence interval
    ci = sem * stats.t.ppf((1 + 0.95) / 2., arr.shape[0] - 1)
    return mean, ci

def aggregate_results(results):
    means = {}
    cis = {}

    for metric in ["train_acc", "test_acc", "train_loss", "test_loss"]:
        mean, ci = mean_and_ci([run[metric] for run in results])
        means[metric] = mean
        cis[metric] = ci

    # also get mean/ci for time
    mean_time, ci_time = mean_and_ci([run["time"] for run in results], axis=0)
    means["time"] = mean_time
    cis["time"] = ci_time

    return means, cis

# MAIN
if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train models with different batch strategies')
    parser.add_argument('--dataset', type=str, default=ACTIVE_DATASET,
                       help=f'Dataset to use (default: {ACTIVE_DATASET}). Options: mnist_csv, qmnist_csv, cifar10_csv, cifar100_csv')
    args = parser.parse_args()

    dataset_name = args.dataset

    # validate dataset
    if dataset_name not in DATASET_SPECS:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Available datasets: {list(DATASET_SPECS.keys())}")
        exit(1)

    # load dataset using factory pattern
    print(f"\n{'='*60}")
    print(f"Loading dataset: {dataset_name}")
    print(f"{'='*60}")
    train_ds, test_ds = build_dataset(DATASETS_ROOT, dataset_name)
    print(f"Dataset loaded: {len(train_ds)} train, {len(test_ds)} test samples")

    # get dataset info
    dataset_spec = DATASET_SPECS[dataset_name]
    print(f"Input dimension: {dataset_spec['input_dim']}")
    print(f"Number of classes: {dataset_spec['num_classes']}")
    print(f"{'='*60}\n")

    # check if we should use config-driven model selection
    if HAS_CONFIG_SYSTEM:
        model_constructor = lambda: create_model(ACTIVE_VISION_MODEL)
        print(f"Using config model: {ACTIVE_VISION_MODEL}")
    else:
        # build model dynamically based on dataset
        model_constructor = lambda: build_model_for(dataset_name, train_ds, SimpleMLP)
        print(f"Using SimpleMLP model (input_dim={dataset_spec['input_dim']}, num_classes={dataset_spec['num_classes']})")

    # loop through all batch strategies and run experiments
    for strategy_label, strategy_path in BATCH_STRATEGIES.items():
        print(f"\n==== Batching strategy: {strategy_label} ====")

        # dynamically load the batch strategy module
        module_name = f"trainer.batching.vision_batching.{strategy_path}".replace("/", ".").replace("\\", ".")
        batch_module = importlib.import_module(module_name)
        batch_strategy = batch_module.batch_sampler

        # create directory for this run
        run_dir = create_run_dir(strategy_label)

        # run all experiments for this strategy
        results = run_experiment(
            batch_strategy, strategy_label, train_ds, test_ds, model_constructor,
            EPOCHS, BATCH_SIZE, N_RUNS
        )

        # aggregate results across runs
        means, cis = aggregate_results(results)

        # save plots
        epochs_range = np.arange(1, EPOCHS+1)

        # helper function to plot metrics with confidence intervals
        def plot_metric(metric, ylabel, title, filename):
            plt.figure(figsize=(7, 5))
            plt.plot(epochs_range, means[metric], label=strategy_label, linewidth=2)
            # shaded region for confidence interval
            plt.fill_between(epochs_range, means[metric] - cis[metric], means[metric] + cis[metric], alpha=0.2)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(f"{title} - {strategy_label}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, filename))
            plt.close()

        # generate all plots
        plot_metric('test_acc', 'Test Accuracy', "Test Accuracy vs Epoch", "test_acc.png")
        plot_metric('train_acc', 'Train Accuracy', "Train Accuracy vs Epoch", "train_acc.png")
        plot_metric('train_loss', 'Train Loss', "Train Loss vs Epoch", "train_loss.png")
        plot_metric('test_loss', 'Test Loss', "Test Loss vs Epoch", "test_loss.png")

        # save summary text file
        with open(os.path.join(run_dir, "summary.txt"), "w") as f:
            for i in range(EPOCHS):
                f.write(f"Epoch {i+1}: train_acc={means['train_acc'][i]:.4f}±{cis['train_acc'][i]:.4f}, "
                        f"test_acc={means['test_acc'][i]:.4f}±{cis['test_acc'][i]:.4f}, "
                        f"train_loss={means['train_loss'][i]:.4f}±{cis['train_loss'][i]:.4f}\n")
            f.write(f"CPU Time: {means['time']:.2f}±{cis['time']:.2f} sec\n")

        print(f"\nResults for {strategy_label} saved to: {run_dir}")

