import os
import numpy as np
import matplotlib.pyplot as plt
from trainer.constants import OUTPUT_DIR, EPOCHS
from trainer.constants_batch_strategy import COMPARE_BATCH_STRATEGY_PAIRS

def load_summary(run_dir):
    summary_file = os.path.join(run_dir, "summary.txt")
    means, cis = {}, {}
    lines = open(summary_file).readlines()
    for metric in ["train_acc", "test_acc", "train_loss"]:
        means[metric] = []
        cis[metric] = []
    for line in lines:
        if 'Epoch' in line:
            for metric in ["train_acc", "test_acc", "train_loss"]:
                try:
                    # Split by metric name, then by ±
                    parts = line.split(f"{metric}=")[1].split("±")
                    val = float(parts[0])
                    # CI value might have comma after it, so take only the first part
                    ci_str = parts[1].split(",")[0].strip()
                    ci = float(ci_str)
                    means[metric].append(val)
                    cis[metric].append(ci)
                except Exception:
                    continue
    return means, cis

def find_latest_run_path(strategy_name):
    strat_dir = os.path.join(OUTPUT_DIR, f"batching_{strategy_name.lower()}")
    runs = sorted([d for d in os.listdir(strat_dir) if d.startswith('run-')])
    if not runs:
        raise RuntimeError(f"No run-XXX directories found for {strategy_name}")

    # Find the latest run that has a summary.txt file (completed run)
    for run in reversed(runs):
        run_path = os.path.join(strat_dir, run)
        summary_file = os.path.join(run_path, "summary.txt")
        if os.path.exists(summary_file):
            return run_path

    raise RuntimeError(f"No completed runs found for {strategy_name} (no summary.txt files)")

def plot_comparison(pair, epoch_count=EPOCHS):
    epochs = np.arange(1, epoch_count+1)
    out_dir = os.path.join(OUTPUT_DIR, f'comparison_{"_".join([p.lower() for p in pair])}')
    os.makedirs(out_dir, exist_ok=True)
    metric_labels = {
        "test_acc": "Test Accuracy", "train_acc":"Train Accuracy",
        "train_loss": "Train Loss", "test_loss": "Test Loss",
    }
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    method_results = []
    for idx, method in enumerate(pair):
        run_dir = find_latest_run_path(method)
        means, cis = load_summary(run_dir)
        method_results.append((method, means, cis, colors[idx]))

    for metric in ["test_acc", "train_acc", "train_loss"]:
        plt.figure(figsize=(7,5))
        for (name, means, cis, col) in method_results:
            m = np.array(means[metric])
            c = np.array(cis[metric])
            plt.plot(epochs, m, label=name, linewidth=2, color=col)
            plt.fill_between(epochs, m-c, m+c, alpha=0.2, color=col)
        plt.xlabel('Epoch')
        plt.ylabel(metric_labels[metric])
        plt.title(f'{metric_labels[metric]} Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}_cmp.png"))
        plt.close()
    print(f"Comparison plots for {pair[0]} vs {pair[1]} saved to {out_dir}")

if __name__ == "__main__":
    for pair in COMPARE_BATCH_STRATEGY_PAIRS:
        plot_comparison(pair)
