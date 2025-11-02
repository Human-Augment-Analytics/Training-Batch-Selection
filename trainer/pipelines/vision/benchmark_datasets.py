import os
import importlib
import numpy as np
import matplotlib.pyplot as plt

from trainer.model.vision.model import SimpleMLP, SimpleCNN
from trainer.pipelines.vision.vision import (
    train_model, aggregate_results, create_run_dir, run_experiment
)

from trainer.dataloader.factory import build_dataset, build_model_for
from trainer.constants_datasets import DATASET_SPECS
from trainer.constants import SHARED_DATA_DIR

# -------- config to tweak --------
EPOCHS = 5
BATCH_SIZE = 64
N_RUNS = 2

#specify the list of datasets to benchmark.  All dataset keys must exist in DATASET_SPECS
#DATASETS = ["mnist_csv", "mnist", "qmnist", "cifar10_flat"]  # for MLP
DATASETS = ["cifar10", "cifar100"]
#MODEL_CLS =  SimpleMLP
MODEL_CLS =  SimpleCNN
# -------------------------------------------

def dataset_root(ds_name: str) -> str:
    """Resolve the on-disk root for a dataset from SHARED_DATA_DIR + spec['subdir']."""
    spec = DATASET_SPECS[ds_name]
    subdir = spec["subdir"]  # e.g. "vision/MNIST"
    return os.path.join(SHARED_DATA_DIR, subdir)

def get_random_strategy():
    """Hard-code Random batching strategy."""
    mod = importlib.import_module("trainer.batching.vision_batching.random_batch")
    return mod.batch_sampler  # must yield batches of indices

def save_summary(name, means, cis, file):
    file.write(f"\n{name}:\n")
    for i, (tr, te, lo) in enumerate(
        zip(means["train_acc"], means["test_acc"], means["train_loss"])
    ):
        file.write(
            f"Epoch {i+1}: train_acc={tr:.4f}±{cis['train_acc'][i]:.4f}, "
            f"test_acc={means['test_acc'][i]:.4f}±{cis['test_acc'][i]:.4f}, "
            f"train_loss={lo:.4f}±{cis['train_loss'][i]:.4f}\n"
        )
    file.write(f"CPU Time: {means['time']:.2f}±{cis['time']:.2f} sec\n")

# def plot_metric_benchmark_datasets(metric, ylabel, title, filename, means, cis, epochs_axis, run_dir):
#     plt.figure(figsize=(7, 5))
#     plt.plot(epochs_axis, means[metric], label="Random", linewidth=2)
#     plt.fill_between(
#         epochs_axis, means[metric] - cis[metric], means[metric] + cis[metric], alpha=0.2
#     )
#     plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title)
#     plt.legend(); plt.grid(True); plt.tight_layout()
#     plt.savefig(os.path.join(run_dir, filename)); plt.close()

def run_benchmark_experiment(datasets, epochs=EPOCHS, batch_size=BATCH_SIZE, n_runs=N_RUNS, model_cls=SimpleMLP):
    strategy_label="Random-Benchmark"
    random_strategy = get_random_strategy()

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

    run_dir = create_run_dir("Random")  # will create .../batching_random/run-###
    print(f"Saving results to {run_dir}")

    for ds_name in datasets:
        print(f"\n=== DATASET: {ds_name} ===")

        # Build datasets and model 
        train_ds, test_ds = build_dataset(shared_root=SHARED_DATA_DIR, name=ds_name)
        model_ctor = lambda: build_model_for(ds_name, train_ds, model_cls=model_cls)
        means, cis = None, None

        results = run_experiment(
            random_strategy, "random-benchmarking", train_ds, test_ds, model_ctor,
            EPOCHS, BATCH_SIZE, N_RUNS
        )
        means, cis = aggregate_results(results)

        # Copied from vision.py.  Ideally both scripts should call the same plotting/summary functions.
        # Save plots and summaries
        epochs_range = np.arange(1, EPOCHS+1)
        plot_metric('test_acc', 'Test Accuracy', "Test Accuracy vs Epoch", f"test_acc-{ds_name}.png")
        plot_metric('train_acc', 'Train Accuracy', "Train Accuracy vs Epoch", f"train_acc-{ds_name}.png")
        plot_metric('train_loss', 'Train Loss', "Train Loss vs Epoch", f"train_loss-{ds_name}.png")
        plot_metric('test_loss', 'Test Loss', "Test Loss vs Epoch", f"test_loss-{ds_name}.png")
 
       # Save summary
        with open(os.path.join(run_dir, "summary.txt"), "a", buffering=1) as f:
            f.write(f'{ds_name}\n')
            for i in range(EPOCHS):
                f.write(f"Epoch {i+1}: train_acc={means['train_acc'][i]:.4f}±{cis['train_acc'][i]:.4f}, "
                        f"test_acc={means['test_acc'][i]:.4f}±{cis['test_acc'][i]:.4f}, "
                        f"train_loss={means['train_loss'][i]:.4f}±{cis['train_loss'][i]:.4f}\n")
            f.write(f"CPU Time: {means['time']:.2f}±{cis['time']:.2f} sec\n\n")
        print(f"\n All results for {strategy_label} saved to: {run_dir}")


        

if __name__ == "__main__":
#    run_benchmark(DATASETS, model_cls=SimpleMLP)
#    run_benchmark(DATASETS, model_cls=MODEL_CLS)
    run_benchmark_experiment(DATASETS, model_cls=MODEL_CLS)
