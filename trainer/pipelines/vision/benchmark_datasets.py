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
from trainer.pipelines.vision.vision import train_random, run_experiment, aggregate_results, create_run_dir

###### HOW TO ADD A DATASET:
###    Step 1: write the subclass in trainer/dataloader/vision.py
###    Step 2: add the specs below in DATASET_SPECS: input dimensions and number of classes
###    Step 3: write a builder function below that calls the dataset constructor
###    Step 4: associate the builder function defined in Step 3 with a dataset key matching Step 2
###    Step 5: add the dataset to ROOTS below in __main__ with key matching the one in DATASET_SPECS and
###               value matching the vision datasets environment variables below
###    Step 6: specify the list of dataset names to test in __main__  DATASETS
#####


##### DATASET PATH VARIABLES AND ACCESS
# should be set in higher level configuration so vision and NLP stay in sync - maybe yaml

DATASETS_DIR = "/storage/ice-shared/cs8903onl/lw-batch-selection/datasets"
VISION_DATASETS_DIR = os.path.join(DATASETS_DIR, "vision")

# Vision datasets
CIFAR100_DIR   = os.path.join(VISION_DATASETS_DIR, "cifar100")
CIFAR10_DIR    = os.path.join(VISION_DATASETS_DIR, "cifar10")
MNIST_DIR      = os.path.join(VISION_DATASETS_DIR, "MNIST")
MNIST_CSV_DIR      = os.path.join(VISION_DATASETS_DIR, "MNIST/csv")
QMNIST_DIR     = os.path.join(VISION_DATASETS_DIR, "qmnist")
SVHN_DIR       = os.path.join(VISION_DATASETS_DIR, "svhn")
IMAGENET_DIR   = os.path.join(VISION_DATASETS_DIR, "imagenet-2012")
TINY_IMAGENET_DIR = os.path.join(VISION_DATASETS_DIR, "tiny-imagenet-200")
CLOTHING1M_DIR = os.path.join(VISION_DATASETS_DIR, "clothing-1m")
VOC2012_DIR    = os.path.join(VISION_DATASETS_DIR, "voc2012")
CINIC10_DIR    = os.path.join(VISION_DATASETS_DIR, "CINIC-10")
WIKI_DIR     = os.path.join(VISION_DATASETS_DIR, "wikipedia_dataset")

### Step 2: to add a new dataset, specify input dimensions and number of classes here,
### and create a builder key (for now just the same key as the DATASET_SPECS dict key)

# put in trainer/pipelines/vision/registry.py?
DATASET_SPECS = {
    "mnist":   {"input_dim": 28*28,   "num_classes": 10, "builder": "mnist"},
    "mnist_csv":   {"input_dim": 28*28,   "num_classes": 10, "builder": "mnist_csv"},
    "qmnist":  {"input_dim": 28*28,   "num_classes": 10, "builder": "qmnist"},
    "cifar10_flat": {"input_dim": 3*32*32, "num_classes": 10, "builder": "cifar10_flat"},
    # add more here…
}

##### DATASET BUILDING - CREATE SUBCLASSES

# Individual builder functions
# put in trainer/pipelines/vision/builders.py?
from trainer.dataloader.vision_dataloader import (
    MNISTCsvDataset, MNISTRawDataset, QMNISTDataset, CIFAR10Dataset
)

### Step 3: write a builder function that calls the daset constructor with appropriate specifications
def build_mnist(root):
    return (
        # For best results
#        MNISTRawDataset(root, train=True,  flatten=True,  download=False),
#        MNISTRawDataset(root, train=False, flatten=True,  download=False),
        # to make it match CSV results
        MNISTRawDataset(root, train=True,  flatten=False,  download=False),
        MNISTRawDataset(root, train=False, flatten=False,  download=False),
    )

def build_mnist_csv(root):
    train_csv = os.path.join(root, 'mnist_train.csv')
    test_csv = os.path.join(root, 'mnist_test.csv')
    return (
        MNISTCsvDataset(train_csv),
        MNISTCsvDataset(test_csv),
    )

def build_qmnist(root):
    return (
        QMNISTDataset(root, train=True,  flatten=True, download=False),
        QMNISTDataset(root, train=False, flatten=True, download=False),
    )

def build_cifar10_flat(root):
    # flattened 3*32*32 inputs to run with MLP
    return (
        CIFAR10Dataset(root, train=True,  flatten=True,  download=False),
        CIFAR10Dataset(root, train=False, flatten=True,  download=False),
    )

### Step 4: associate the builder function with the builder key defined in Step 3
BUILDER_FUNCS = {
    "mnist": build_mnist,
    "mnist_csv": build_mnist_csv,
    "qmnist": build_qmnist,
    "cifar10_flat": build_cifar10_flat,
}

# put in trainer/pipelines/vision/factory.py?
import torch

from trainer.model.vision.model import SimpleMLP
#from trainer.pipelines.vision.registry import DATASET_SPECS
#from trainer.pipelines.vision.builders import BUILDER_FUNCS

def infer_input_dim(dataset):
    x0, _ = dataset[0]
    return int(x0.numel())

def infer_num_classes(dataset):
    # Try torchvision-style; fallback to labels’ max+1
    if hasattr(dataset, "base") and hasattr(dataset.base, "classes"):
        return len(dataset.base.classes)
    if hasattr(dataset, "classes"):
        return len(dataset.classes)
    # rough fallback
    _, y0 = dataset[0]
    return int(max(y0.item(), 9) + 1)

def build_dataset(name, roots):
    spec = DATASET_SPECS[name]
    builder_key = spec["builder"]
    builder = BUILDER_FUNCS[builder_key]
    return builder(roots[name])

def build_model_for(name, train_ds, hidden_dim=128):
    spec = DATASET_SPECS[name]
    cfg_in = spec["input_dim"]
    cfg_nc = spec["num_classes"]

    inf_in = infer_input_dim(train_ds)
    inf_nc = infer_num_classes(train_ds)

    # confirm registry disagrees with reality and fail if not
    if cfg_in != inf_in:
        raise ValueError(
            f"[{name}] input_dim mismatch: registry={cfg_in}, inferred={inf_in}. "
            "Check flatten/resize/transforms."
        )
    if cfg_nc != inf_nc:
        print(f"Warning: [{name}] num_classes registry={cfg_nc}, inferred={inf_nc}")

    return SimpleMLP(input_dim=cfg_in, hidden_dim=hidden_dim, num_classes=cfg_nc)

##### MAIN PROCEDURE
# functions currently defined in vision.py:__main__

def save_summary(name, means, cis, file):
    file.write(f"\n{name}:\n")
    for i, (tr, te, lo) in enumerate(zip(means["train_acc"], means["test_acc"], means["train_loss"])):
        file.write(f"Epoch {i+1}: train_acc={tr:.4f}±{cis['train_acc'][i]:.4f}, "
                   f"test_acc={te:.4f}±{cis['test_acc'][i]:.4f}, "
                   f"train_loss={lo:.4f}±{cis['train_loss'][i]:.4f}\n")
    file.write(f"CPU Time: {means['time']:.2f}±{cis['time']:.2f} sec\n")

def plot_metric(metric, ylabel, title, filename, methods, epochs, run_dir):
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

# Main benchmarking code starts here
# uncomment the below if registry, factory move to separate files
# from trainer.pipelines.vision.factory import build_dataset, build_model_for
# from trainer.pipelines.vision.registry import DATASET_SPECS

import os, numpy as np
from trainer.pipelines.vision.vision import (
    train_random, run_experiment, aggregate_results, create_run_dir
)
#from trainer.pipelines.vision.benchmark_utils import plot_metric, save_summary  # or inline these

def run_benchmark(datasets, roots, epochs=5, batch_size=64, n_runs=5):
    run_dir = create_run_dir()

    for ds_name in datasets:
        print(f"\n=== DATASET: {ds_name} ===")
        train_ds, test_ds = build_dataset(ds_name, roots)
        model_ctor = lambda: build_model_for(ds_name, train_ds)

        results_random = []  # reset per dataset!
        for seed in range(n_runs):
            print(f"  Run {seed+1}/{n_runs}")
            results_random.append(
                run_experiment(train_random, train_ds, test_ds, model_ctor, epochs, batch_size, seed)
            )

        means_random, cis_random = aggregate_results(results_random)

        epochs_axis = np.arange(1, epochs + 1)
        methods = {"Random": (means_random, cis_random)}
        plot_metric("test_acc",  "Test Accuracy",  f"Test Accuracy vs Epoch ({ds_name})",
                    f"test_acc-{ds_name}.png",  methods, epochs_axis, run_dir)
        plot_metric("train_acc", "Train Accuracy", f"Train Accuracy vs Epoch ({ds_name})",
                    f"train_acc-{ds_name}.png", methods, epochs_axis, run_dir)
        plot_metric("train_loss","Train Loss",     f"Train Loss vs Epoch ({ds_name})",
                    f"train_loss-{ds_name}.png",  methods, epochs_axis, run_dir)
        plot_metric("test_loss", "Test Loss",      f"Test Loss vs Epoch ({ds_name})",
                    f"test_loss-{ds_name}.png",   methods, epochs_axis, run_dir)

        with open(os.path.join(run_dir, f"summary-{ds_name}.txt"), "w") as f:
            save_summary("Random", means_random, cis_random, f)

    print(f"\n All results saved to: {run_dir}")


if __name__ == "__main__":

    ### Step 5: add the dataset to ROOTS below in __main__ with key matching the one in DATASET_SPECS and
    ###               value matching the vision datasets environment variables at top

    ROOTS = {
            "mnist":   MNIST_DIR,
            "mnist_csv":   MNIST_CSV_DIR,
            "qmnist":  QMNIST_DIR,
            "cifar10_flat": CIFAR10_DIR,
        }
    ### Step 6: add the dataset key to the list of datasets to test
    DATASETS = ["mnist_csv", "qmnist", "mnist", "cifar10_flat"]  # or ["mnist", "cifar10", ...]

    run_benchmark(DATASETS, ROOTS, epochs=5, batch_size=64, n_runs=3)
