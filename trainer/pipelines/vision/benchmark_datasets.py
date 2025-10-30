import os
import importlib
import numpy as np
import matplotlib.pyplot as plt

from trainer.model.vision.model import SimpleMLP
from trainer.model.vision.cnn import SimpleCNN
from trainer.pipelines.vision.vision import (
    train_model, aggregate_results, create_run_dir
)

from trainer.dataloader.factory import build_dataset, build_model_for
from trainer.constants_datasets import DATASET_SPECS
from trainer.constants import SHARED_DATA_DIR

# -------- config to tweak --------
EPOCHS = 2
BATCH_SIZE = 64
N_RUNS = 2

#specify the list of datasets to benchmark.  All dataset keys must exist in DATASET_SPECS
DATASETS = ["mnist_csv", "mnist", "qmnist", "cifar10_flat"]  # for MLP
#DATASETS = ["cifar10"]  # for CNN
MODEL_CLS = SimpleMLP # SimpleCNN
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

def run_benchmark(datasets, epochs=EPOCHS, batch_size=BATCH_SIZE, n_runs=N_RUNS, model_cls=SimpleMLP):
    run_dir = create_run_dir("Random")  # will create .../batching_random/run-###
    print(f"Saving results to {run_dir}")

    random_strategy = get_random_strategy()

    for ds_name in datasets:
        print(f"\n=== DATASET: {ds_name} ===")

        # Build datasets and model 
        train_ds, test_ds = build_dataset(shared_root=SHARED_DATA_DIR, name=ds_name)

        #### tmp
        # benchmark_datasets.py, right after build_dataset(...)
        x0, y0 = train_ds[0]
        print(f"[DEBUG] dataset={ds_name} sample_shape={tuple(x0.shape)} "
              f"flatten_attr={getattr(train_ds, 'flatten', None)} "
              f"type={type(train_ds).__name__}")
        ### end tmp
        model_ctor = lambda: build_model_for(ds_name, train_ds, model_cls=model_cls)

        # Run N times
        results = []
        for seed in range(n_runs):
            print(f"  Run {seed+1}/{n_runs}")
            model = model_ctor()
            train_acc, test_acc, train_loss, test_loss = train_model(
                model, train_ds, test_ds,
                epochs=epochs,
                batch_size=batch_size,
                batch_strategy=random_strategy,
                seed=seed,
            )


            results.append({
                "train_acc": train_acc, "test_acc": test_acc,
                "train_loss": train_loss, "test_loss": test_loss,
                "time": 0.0  # optional: add timing if you want
            })

        means, cis = aggregate_results(results)

        # Plots & summary per dataset
        epochs_axis = np.arange(1, epochs + 1)
        plot_metric("test_acc",  "Test Accuracy",
                    f"Test Accuracy vs Epoch ({ds_name})",
                    f"test_acc-{ds_name}.png",  means, cis, epochs_axis, run_dir)
        plot_metric("train_acc", "Train Accuracy",
                    f"Train Accuracy vs Epoch ({ds_name})",
                    f"train_acc-{ds_name}.png", means, cis, epochs_axis, run_dir)
        plot_metric("train_loss","Train Loss",
                    f"Train Loss vs Epoch ({ds_name})",
                    f"train_loss-{ds_name}.png",  means, cis, epochs_axis, run_dir)
        plot_metric("test_loss", "Test Loss",
                    f"Test Loss vs Epoch ({ds_name})",
                    f"test_loss-{ds_name}.png",   means, cis, epochs_axis, run_dir)

        with open(os.path.join(run_dir, f"summary-{ds_name}.txt"), "w") as f:
            save_summary("Random", means, cis, f)

    print(f"\nAll results saved to: {run_dir}")

if __name__ == "__main__":
#    run_benchmark(DATASETS, model_cls=SimpleMLP)
    run_benchmark(DATASETS, model_cls=MODEL_CLS)
