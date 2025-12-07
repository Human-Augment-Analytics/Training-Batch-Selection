import importlib
import os
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from scipy import stats

from trainer.constants_nlp import (
    NLP_OUTPUT_DIR, MODEL_NAME, NUM_LABELS, MAX_LENGTH,
    BATCH_SIZE, EPOCHS, N_RUNS, LEARNING_RATE, LOSS_THRESHOLD
)
from trainer.dataloader.nlp_dataloader import IMDBDataset


# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ============ Training Function ============

def train_model(model, train_ds, test_ds, epochs, batch_size, batch_strategy,
                loss_threshold=LOSS_THRESHOLD, seed=None):
    """
    Train BERT model with specified batch sampling strategy.

    Args:
        model: BERT model instance
        train_ds: Training dataset
        test_ds: Test dataset
        epochs: Number of training epochs
        batch_size: Batch size
        batch_strategy: Batch sampling strategy function
        loss_threshold: Threshold for loss-based strategies
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_accs, test_accs, train_losses, test_losses, samples_per_epoch)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Calculate total training steps for scheduler
    n_samples = len(train_ds)
    steps_per_epoch = n_samples // batch_size
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Track per-sample loss
    per_sample_loss = np.zeros(len(train_ds))

    # Metrics storage
    train_accs, test_accs, train_losses, test_losses = [], [], [], []
    samples_per_epoch = []

    for epoch in range(epochs):
        model.train()
        correct, n_samples_epoch, running_loss = 0, 0, 0

        # Get batch iterator
        if 'loss_history' in batch_strategy.__code__.co_varnames:
            batch_iter = batch_strategy(
                train_ds, batch_size,
                loss_history=per_sample_loss,
                threshold=loss_threshold
            )
        else:
            batch_iter = batch_strategy(train_ds, batch_size)

        # Training loop
        for batch_indices in batch_iter:
            # Prepare batch
            batch = [train_ds[i] for i in batch_indices]
            input_ids = torch.stack([item['input_ids'] for item in batch]).to(DEVICE)
            attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(DEVICE)
            labels = torch.stack([item['label'] for item in batch]).to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Compute loss per sample
            logits = outputs.logits
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            losses_per_sample = loss_fn(logits, labels)
            loss = losses_per_sample.mean()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Update metrics
            running_loss += loss.item() * len(batch_indices)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            n_samples_epoch += len(batch_indices)

            # Update per-sample loss history
            if 'loss_history' in batch_strategy.__code__.co_varnames:
                for k, idx in enumerate(batch_indices):
                    per_sample_loss[idx] = losses_per_sample[k].item()

        # Epoch metrics
        train_acc = correct / n_samples_epoch if n_samples_epoch > 0 else 0
        train_loss = running_loss / n_samples_epoch if n_samples_epoch > 0 else 0
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        samples_per_epoch.append(n_samples_epoch)

        # Evaluation
        test_acc, test_loss = evaluate(model, test_ds, batch_size)
        test_accs.append(test_acc)
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{epochs}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, "
              f"train_loss={train_loss:.4f}, samples={n_samples_epoch}")

    return train_accs, test_accs, train_losses, test_losses, samples_per_epoch


# ============ Evaluation Function ============

def evaluate(model, ds, batch_size):
    """
    Evaluate model on dataset.

    Args:
        model: BERT model
        ds: Dataset to evaluate on
        batch_size: Batch size for evaluation

    Returns:
        Tuple of (accuracy, loss)
    """
    model.eval()
    correct, total, total_loss = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()

    indices = list(range(len(ds)))

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = [ds[i] for i in batch_indices]

            input_ids = torch.stack([item['input_ids'] for item in batch]).to(DEVICE)
            attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(DEVICE)
            labels = torch.stack([item['label'] for item in batch]).to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            logits = outputs.logits
            loss = loss_fn(logits, labels)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * len(batch_indices)
            correct += (predictions == labels).sum().item()
            total += len(batch_indices)

    return correct / total, total_loss / total


# ============ Output Directory Management ============

def create_run_dir(strategy_name):
    """
    Create output directory for a run.

    Args:
        strategy_name: Name of the batch selection strategy

    Returns:
        Path to the created run directory
    """
    out_path = os.path.join(NLP_OUTPUT_DIR, f'batching_{strategy_name.lower()}')
    os.makedirs(out_path, exist_ok=True)
    existing = [d for d in os.listdir(out_path) if d.startswith('run-')]
    next_num = max([int(d.split('-')[1]) for d in existing], default=0) + 1
    run_dir = os.path.join(out_path, f'run-{next_num:03d}')
    os.makedirs(run_dir)
    return run_dir


# ============ Experiment Runner ============

def run_experiment(batch_strategy, strategy_label, train_ds, test_ds,
                   model_constructor, epochs, batch_size, n_runs):
    """
    Run multiple training runs with specified strategy.

    Args:
        batch_strategy: Batch sampling strategy function
        strategy_label: Label for the strategy
        train_ds: Training dataset
        test_ds: Test dataset
        model_constructor: Function that creates a new model instance
        epochs: Number of epochs
        batch_size: Batch size
        n_runs: Number of runs for statistical significance

    Returns:
        List of results dictionaries
    """
    assert callable(model_constructor), "model_constructor must be a callable factory function"

    results = []
    for seed in range(n_runs):
        print(f"\n[{strategy_label}] Run {seed+1}/{n_runs}")
        model = model_constructor()
        start = time.time()
        train_acc, test_acc, train_loss, test_loss, samples_per_epoch = train_model(
            model, train_ds, test_ds, epochs, batch_size, batch_strategy, seed=seed
        )
        elapsed = time.time() - start
        results.append({
            "train_acc": train_acc, "test_acc": test_acc,
            "train_loss": train_loss, "test_loss": test_loss,
            "samples_per_epoch": samples_per_epoch,
            "time": elapsed
        })
    return results


# ============ Statistical Analysis ============

def mean_and_ci(arrays, axis=0):
    """
    Compute mean and confidence interval.

    Args:
        arrays: List of arrays
        axis: Axis along which to compute statistics

    Returns:
        Tuple of (mean, confidence_interval)
    """
    arr = np.array(arrays)
    mean = arr.mean(axis=0)
    sem = stats.sem(arr, axis=0)
    ci = sem * stats.t.ppf((1 + 0.95) / 2., arr.shape[0] - 1)
    return mean, ci


def aggregate_results(results):
    """
    Aggregate metrics across runs.

    Args:
        results: List of result dictionaries

    Returns:
        Tuple of (means, confidence_intervals) dictionaries
    """
    means, cis = {}, {}
    for metric in ["train_acc", "test_acc", "train_loss", "test_loss", "samples_per_epoch"]:
        mean, ci = mean_and_ci([run[metric] for run in results])
        means[metric], cis[metric] = mean, ci
    mean_time, ci_time = mean_and_ci([run["time"] for run in results], axis=0)
    means["time"], cis["time"] = mean_time, ci_time
    return means, cis


# ============ Plotting Functions ============

def plot_metric(metric, ylabel, title, filename, means, cis, strategy_label, run_dir, epochs):
    """
    Plot a single metric.

    Args:
        metric: Metric name
        ylabel: Y-axis label
        title: Plot title
        filename: Output filename
        means: Dictionary of mean values
        cis: Dictionary of confidence intervals
        strategy_label: Strategy name
        run_dir: Output directory
        epochs: Number of epochs
    """
    epochs_range = np.arange(1, epochs + 1)
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
    plt.savefig(os.path.join(run_dir, filename))
    plt.close()


# ============ MAIN ============

if __name__ == '__main__':
    # Import batch strategies from config
    from trainer.constants_batch_strategy import NLP_BATCH_STRATEGIES

    # Load datasets
    from trainer.constants_nlp import USE_SUBSET, TRAIN_SUBSET_SIZE, TEST_SUBSET_SIZE

    train_ds = IMDBDataset(
        split='train',
        max_length=MAX_LENGTH,
        subset_size=TRAIN_SUBSET_SIZE if USE_SUBSET else None
    )
    test_ds = IMDBDataset(
        split='test',
        max_length=MAX_LENGTH,
        subset_size=TEST_SUBSET_SIZE if USE_SUBSET else None
    )

    # Model constructor
    def create_model():
        return BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS
        )

    # Loop over batch strategies
    for strategy_label, strategy_path in NLP_BATCH_STRATEGIES.items():
        print(f"\n==== Running with batching strategy: {strategy_label} ====")

        # Import batching function dynamically
        module_name = f"trainer.batching.nlp_batching.{strategy_path}".replace("/", ".").replace("\\", ".")
        batch_module = importlib.import_module(module_name)
        batch_strategy = batch_module.batch_sampler

        run_dir = create_run_dir(strategy_label)

        # Run experiments
        results = run_experiment(
            batch_strategy, strategy_label, train_ds, test_ds,
            create_model, EPOCHS, BATCH_SIZE, N_RUNS
        )
        means, cis = aggregate_results(results)

        # Save plots
        plot_metric('test_acc', 'Test Accuracy', "Test Accuracy vs Epoch",
                   "test_acc.png", means, cis, strategy_label, run_dir, EPOCHS)
        plot_metric('train_acc', 'Train Accuracy', "Train Accuracy vs Epoch",
                   "train_acc.png", means, cis, strategy_label, run_dir, EPOCHS)
        plot_metric('train_loss', 'Train Loss', "Train Loss vs Epoch",
                   "train_loss.png", means, cis, strategy_label, run_dir, EPOCHS)
        plot_metric('samples_per_epoch', 'Samples', "Samples per Epoch",
                   "samples_per_epoch.png", means, cis, strategy_label, run_dir, EPOCHS)

        # Save summary
        with open(os.path.join(run_dir, "summary.txt"), "w") as f:
            for i in range(EPOCHS):
                f.write(f"Epoch {i+1}: train_acc={means['train_acc'][i]:.4f}±{cis['train_acc'][i]:.4f}, "
                       f"test_acc={means['test_acc'][i]:.4f}±{cis['test_acc'][i]:.4f}, "
                       f"train_loss={means['train_loss'][i]:.4f}±{cis['train_loss'][i]:.4f}, "
                       f"samples={int(means['samples_per_epoch'][i])}\n")
            f.write(f"Training Time: {means['time']:.2f}±{cis['time']:.2f} sec\n")

        print(f"\n✅ All results for {strategy_label} saved to: {run_dir}")
