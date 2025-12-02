import os
import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt

from trainer.model.vision.model import SimpleMLP, SimpleCNN, ResNet18
from trainer.pipelines.vision.vision import (
    train_model, aggregate_results, create_run_dir, run_experiment
)

from trainer.dataloader.factory import build_dataset, build_model_for
from trainer.constants_datasets import DATASET_SPECS
from trainer.constants import SHARED_DATA_DIR

# -------- config to tweak --------
EPOCHS = 20
BATCH_SIZE = 64
N_RUNS = 2

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"

print(f"[config] Using DEVICE = {DEVICE}, NAME = {DEVICE_NAME}")

#specify the list of datasets to benchmark.  All dataset keys must exist in DATASET_SPECS
#DATASETS = ["mnist_csv", "mnist", "qmnist", "cifar10_flat"]  # for MLP
DATASETS = ["cifar10", "cifar100"]
#DATASETS = ["cifar10"]
#MODEL_CLS =  SimpleMLP
#MODEL_CLS =  SimpleCNN
MODEL_CLS = ResNet18
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

def plot_combined_old(all_means, run_dir, epochs_range):
    """
    all_means: dict[ds_name] -> means dict from aggregate_results
               must contain keys: 'train_acc', 'test_acc',
                                   'train_loss', 'test_loss'
    run_dir: directory to save plots into
    epochs_range: array-like of epoch indices (e.g. np.arange(1, EPOCHS+1))
    """
    # one distinct color per dataset
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Combined accuracy plot
    plt.figure(figsize=(7, 5))
    for i, (ds_name, means_ds) in enumerate(all_means.items()):
        color = color_cycle[i % len(color_cycle)]
        # train: solid
        plt.plot(
            epochs_range,
            means_ds["train_acc"],
            linestyle="-",
            color=color,
            label=f"{ds_name} train",
        )
        # test: dotted
        plt.plot(
            epochs_range,
            means_ds["test_acc"],
            linestyle=":",
            color=color,
            label=f"{ds_name} test",
        )

    plt.xlabel(f"Epoch (averaged over {N_RUNS} trials)")
    plt.ylabel("Accuracy")
    plt.title("Train/Test Accuracy vs Epoch (all datasets)")
    cpu_text = f"CPU time: {all_means['time']:.2f} sec"
    plt.text(
        0.02, 0.98, cpu_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.3)
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "combined_accuracy.png"))
    plt.close()

    # Combined loss plot
    plt.figure(figsize=(7, 5))
    for i, (ds_name, means_ds) in enumerate(all_means.items()):
        color = color_cycle[i % len(color_cycle)]
        # train: solid
        plt.plot(
            epochs_range,
            means_ds["train_loss"],
            linestyle="-",
            color=color,
            label=f"{ds_name} train",
        )
        # test: dotted
        plt.plot(
            epochs_range,
            means_ds["test_loss"],
            linestyle=":",
            color=color,
            label=f"{ds_name} test",
        )

    plt.xlabel(f"Epoch (averaged over {N_RUNS} trials)")
    cpu_text = f"CPU time: {all_means['time']:.2f} sec"
    plt.text(
        0.02, 0.98, cpu_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.3)
    )
    plt.ylabel("Loss")
    plt.title("Train/Test Loss vs Epoch (all datasets)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "combined_loss.png"))
    plt.close()

def plot_combined(all_means, run_dir, epochs_range, model_cls_name):
    """
    all_means: dict[ds_name] -> means dict from aggregate_results
               must contain keys: 'train_acc', 'test_acc',
                                   'train_loss', 'test_loss', 'time'
    run_dir: directory to save plots into
    epochs_range: array-like of epoch indices (e.g. np.arange(1, EPOCHS+1))
    """
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Prepare a CPU-time summary string, e.g.:
    # "CPU time (sec): cifar10=12.34, cifar100=23.45"
    cpu_parts = []
    for ds_name, means_ds in all_means.items():
        if "time" in means_ds:
            cpu_parts.append(f"{ds_name}={means_ds['time']:.2f}")
    cpu_text = ""
    box_text = ""
    if cpu_parts:
        cpu_text = "CPU time (sec): " + ", ".join(cpu_parts)
        box_text = f"{cpu_text}\nDevice: {DEVICE_NAME}"

    # Combined loss plot
    plt.figure(figsize=(7, 5))
    for i, (ds_name, means_ds) in enumerate(all_means.items()):
        color = color_cycle[i % len(color_cycle)]
        plt.plot(
            epochs_range,
            means_ds["train_acc"],
            linestyle="-",
            color=color,
            label=f"{ds_name} train",
        )
        plt.plot(
            epochs_range,
            means_ds["test_acc"],
            linestyle=":",
            color=color,
            label=f"{ds_name} test",
        )

    if cpu_text:
        plt.text(
            0.02,
            0.98,
            box_text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.3),
        )

    plt.xlabel(f"Epoch (averaged over {N_RUNS} trials)")
    plt.ylabel("Accuracy")
    plt.title(f"Train/Test Accuracy vs Epoch ({model_cls_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "combined_accuracy.png"))
    plt.close()

    #  Combined loss plot 
    plt.figure(figsize=(7, 5))
    for i, (ds_name, means_ds) in enumerate(all_means.items()):
        color = color_cycle[i % len(color_cycle)]
        plt.plot(
            epochs_range,
            means_ds["train_loss"],
            linestyle="-",
            color=color,
            label=f"{ds_name} train",
        )
        plt.plot(
            epochs_range,
            means_ds["test_loss"],
            linestyle=":",
            color=color,
            label=f"{ds_name} test",
        )

    if cpu_text:
        plt.text(
            0.02,
            0.98,
            box_text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.3),
        )

    plt.xlabel(f"Epoch (averaged over {N_RUNS} trials)")
    plt.ylabel("Loss")
    plt.title(f"Train/Test Loss vs Epoch ({model_cls_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "combined_loss.png"))
    plt.close()

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

    all_means = {}
    
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
        all_means[ds_name] = means

        # Copied from vision.py.  Ideally both scripts should call the same plotting/summary functions.
        # Save plots and summaries
        epochs_range = np.arange(1, EPOCHS+1)
        plot_metric('test_acc', 'Test Accuracy', "Test Accuracy vs Epoch", f"test_acc-{ds_name}.png")
        plot_metric('train_acc', 'Train Accuracy', "Train Accuracy vs Epoch", f"train_acc-{ds_name}.png")
        plot_metric('train_loss', 'Train Loss', "Train Loss vs Epoch", f"train_loss-{ds_name}.png")
        plot_metric('test_loss', 'Test Loss', "Test Loss vs Epoch", f"test_loss-{ds_name}.png")

        epochs_range = np.arange(1, EPOCHS + 1)
        print(f"\nCombined plots saved to: {run_dir}")

        # Save summary
        with open(os.path.join(run_dir, "summary.txt"), "a", buffering=1) as f:
            f.write(f'{ds_name}\n')
            for i in range(EPOCHS):
                f.write(f"Epoch {i+1}: train_acc={means['train_acc'][i]:.4f}±{cis['train_acc'][i]:.4f}, "
                        f"test_acc={means['test_acc'][i]:.4f}±{cis['test_acc'][i]:.4f}, "
                        f"train_loss={means['train_loss'][i]:.4f}±{cis['train_loss'][i]:.4f}\n")
            f.write(f"CPU Time: {means['time']:.2f}±{cis['time']:.2f} sec\n\n")
        print(f"\n All results for {strategy_label} saved to: {run_dir}")

    plot_combined(all_means, run_dir, epochs_range, model_cls.__name__)

        

if __name__ == "__main__":
#    run_benchmark(DATASETS, model_cls=SimpleMLP)
#    run_benchmark(DATASETS, model_cls=MODEL_CLS)
    run_benchmark_experiment(DATASETS, model_cls=MODEL_CLS)
