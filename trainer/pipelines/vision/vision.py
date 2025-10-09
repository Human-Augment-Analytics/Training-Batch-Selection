import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from scipy import stats

from trainer.model.vision.model import SimpleMLP
from trainer.dataloader.vision_dataloader import MNISTCsvDataset


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"BASE_DIR: {BASE_DIR}")

train_csv = os.path.join(BASE_DIR, 'data/vision/mnist_train.csv')
test_csv = os.path.join(BASE_DIR, 'data/vision/mnist_test.csv')

train_ds = MNISTCsvDataset(train_csv)
test_ds = MNISTCsvDataset(test_csv)

MOVING_AVG_DECAY = 0.9


# ---------------- SMART BATCH SELECTION ---------------- #
def get_smart_batch(loss_history, batch_size, explore_frac=0.5, top_k_frac=0.2):
    n_explore = int(batch_size * explore_frac)
    n_exploit = batch_size - n_explore
    n_total = len(loss_history)

    rand_idxs = np.random.choice(n_total, n_explore, replace=False)
    k = int(top_k_frac * n_total)
    exploit_candidates = np.argsort(-loss_history)[:k]
    exploit_idxs = np.random.choice(exploit_candidates, min(n_exploit, len(exploit_candidates)), replace=False)

    batch_idxs = np.concatenate([rand_idxs, exploit_idxs])
    np.random.shuffle(batch_idxs)
    return batch_idxs


# ---------------- TRAINING VARIANTS ---------------- #
def train_random(model, train_ds, test_ds, epochs=5, batch_size=64):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    train_accs, test_accs, train_losses, test_losses = [], [], [], []
    for epoch in range(epochs):
        correct, n, running_loss = 0, 0, 0
        model.train()
        for x, y in train_loader:
            x, y = x.to('cpu'), y.to('cpu')
            x = x.view(x.size(0), -1)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (y_pred.argmax(1) == y).sum().item()
            n += x.size(0)
        train_accs.append(correct / n)
        train_losses.append(running_loss / n)

        test_acc, test_loss = evaluate(model, test_ds)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        print(f"Rand-Epoch {epoch+1}: train_acc={train_accs[-1]:.4f}, test_acc={test_acc:.4f}")
    return train_accs, test_accs, train_losses, test_losses


def train_fixed(model, train_ds, test_ds, epochs=5, batch_size=64):
    # Baseline: no shuffling
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    train_accs, test_accs, train_losses, test_losses = [], [], [], []

    for epoch in range(epochs):
        correct, n, running_loss = 0, 0, 0
        model.train()
        for x, y in train_loader:
            x, y = x.to('cpu'), y.to('cpu')
            x = x.view(x.size(0), -1)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (y_pred.argmax(1) == y).sum().item()
            n += x.size(0)
        train_accs.append(correct / n)
        train_losses.append(running_loss / n)

        test_acc, test_loss = evaluate(model, test_ds)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        print(f"Fixed-Epoch {epoch+1}: train_acc={train_accs[-1]:.4f}, test_acc={test_acc:.4f}")
    return train_accs, test_accs, train_losses, test_losses


def train_smart(model, train_ds, test_ds, epochs=5, batch_size=64):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters())

    per_sample_loss = np.zeros(len(train_ds))
    train_accs, test_accs, train_losses, test_losses = [], [], [], []

    for epoch in range(epochs):
        n_batches = len(train_ds) // batch_size
        correct, n, running_loss = 0, 0, 0
        model.train()
        for _ in range(n_batches):
            idxs = get_smart_batch(per_sample_loss, batch_size)
            x, y = zip(*[train_ds[j] for j in idxs])
            x = torch.stack(x).view(batch_size, -1)
            y = torch.tensor(y)
            y_pred = model(x)
            losses = loss_fn(y_pred, y)
            loss = losses.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (y_pred.argmax(1) == y).sum().item()
            n += x.size(0)

            for k, idx in enumerate(idxs):
                per_sample_loss[idx] = MOVING_AVG_DECAY * per_sample_loss[idx] + (1 - MOVING_AVG_DECAY) * losses[k].item()

        train_accs.append(correct / n)
        train_losses.append(running_loss / n)
        test_acc, test_loss = evaluate(model, test_ds)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        print(f"Smart-Epoch {epoch+1}: train_acc={train_accs[-1]:.4f}, test_acc={test_acc:.4f}")
    return train_accs, test_accs, train_losses, test_losses


# ---------------- EVALUATION ---------------- #
def evaluate(model, ds):
    loader = DataLoader(ds, batch_size=256)
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



# ---------------- EXPERIMENTS ---------------- #
def run_experiment(train_fn, train_ds, test_ds, model_constructor, epochs=5, batch_size=64, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    model = model_constructor()
    start = time.time()
    train_acc, test_acc, train_loss, test_loss = train_fn(model, train_ds, test_ds, epochs=epochs, batch_size=batch_size)
    elapsed = time.time() - start
    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "time": elapsed
    }


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


# ---------------- OUTPUT SETUP ---------------- #
def create_run_dir():
    output_path = os.path.join(BASE_DIR, 'pipelines/vision/output')
    os.makedirs(output_path, exist_ok=True)
    existing = [d for d in os.listdir(output_path) if d.startswith('run-')]
    next_num = max([int(d.split('-')[1]) for d in existing], default=0) + 1
    run_dir = os.path.join(output_path, f'run-{next_num:03d}')
    os.makedirs(run_dir)
    return run_dir


# ---------------- MAIN ---------------- #
if __name__ == '__main__':
    run_dir = create_run_dir()
    print(f"Saving results to {run_dir}")

    EPOCHS, BATCH_SIZE, N_RUNS = 5, 64, 5
    results_random, results_fixed, results_smart = [], [], []

    for seed in range(N_RUNS):
        print(f"\n=== RUN {seed+1}/{N_RUNS} ===")
        results_random.append(run_experiment(train_random, train_ds, test_ds, SimpleMLP, EPOCHS, BATCH_SIZE, seed))
        results_fixed.append(run_experiment(train_fixed, train_ds, test_ds, SimpleMLP, EPOCHS, BATCH_SIZE, seed))
        results_smart.append(run_experiment(train_smart, train_ds, test_ds, SimpleMLP, EPOCHS, BATCH_SIZE, seed))

    means_random, cis_random = aggregate_results(results_random)
    means_fixed, cis_fixed = aggregate_results(results_fixed)
    means_smart, cis_smart = aggregate_results(results_smart)

    # Plot comparisons
    epochs = np.arange(1, EPOCHS + 1)
    methods = {"Random": (means_random, cis_random),
               "Fixed": (means_fixed, cis_fixed),
               "Smart": (means_smart, cis_smart)}

    def plot_metric(metric, ylabel, title, filename):
        plt.figure(figsize=(7, 5))
        for name, (means, cis) in methods.items():
            plt.plot(epochs, means[metric], label=name, linewidth=2)
            plt.fill_between(epochs, means[metric] - cis[metric], means[metric] + cis[metric], alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, filename))
        plt.close()

    plot_metric("test_acc", "Test Accuracy", "Test Accuracy vs Epoch", "test_acc.png")
    plot_metric("train_acc", "Train Accuracy", "Train Accuracy vs Epoch", "train_acc.png")
    plot_metric("train_loss", "Train Loss", "Train Loss vs Epoch", "train_loss.png")
    plot_metric("test_loss", "Test Loss", "Test Loss vs Epoch", "test_loss.png")


    # Save text summaries
    def save_summary(name, means, cis, file):
        file.write(f"\n{name}:\n")
        for i, (tr, te, lo) in enumerate(zip(means["train_acc"], means["test_acc"], means["train_loss"])):
            file.write(f"Epoch {i+1}: train_acc={tr:.4f}±{cis['train_acc'][i]:.4f}, "
                       f"test_acc={te:.4f}±{cis['test_acc'][i]:.4f}, "
                       f"train_loss={lo:.4f}±{cis['train_loss'][i]:.4f}\n")
        file.write(f"CPU Time: {means['time']:.2f}±{cis['time']:.2f} sec\n")

    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        save_summary("Random", means_random, cis_random, f)
        save_summary("Fixed", means_fixed, cis_fixed, f)
        save_summary("Smart", means_smart, cis_smart, f)

    print(f"\n✅ All results saved to: {run_dir}")
