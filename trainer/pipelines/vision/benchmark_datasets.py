# trainer/pipelines/vision/benchmark_datasets_v2.py
"""
Benchmark datasets with a batching strategy (NO plotting).

Purpose:
- Start clean, minimal, easy to reason about.
- Run short smoke tests (default: 2 epochs) to compare against old setup.
- Print and record aggregate_results() metric keys (means.keys()) to guide next refactor step.

This file is meant to be the "new home" you migrate functions out of later.
"""

from __future__ import annotations

import os
import json
import importlib
from typing import Any, Dict, Tuple, Callable
import argparse
from datetime import datetime
from collections import Counter

import torch
from torch.utils.data import Subset
import numpy as np


from trainer.pipelines.vision.vision import (
#    create_run_dir,
    run_experiment,
    aggregate_results,
)

from trainer.dataloader.factory import build_dataset, build_model_for
from trainer.constants import SHARED_DATA_DIR, OUTPUT_DIR
from trainer.model.vision.model import SimpleMLP, SimpleCNN, ResNet18


def create_run_dir(strategy_name):
    out_path = os.path.join(OUTPUT_DIR, f"batching_{strategy_name.lower()}")
    os.makedirs(out_path, exist_ok=True)

    next_num = 1
    while True:
        run_dir = os.path.join(out_path, f"run-{next_num:03d}")
        try:
            os.makedirs(run_dir)
            return run_dir
        except FileExistsError:
            next_num += 1
            
# =========================
# Config (can be set here but should come from sbatch
# =========================
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 64
DEFAULT_N_RUNS = 2
DEFAULT_DATASETS = ["iwildcam"]
DEFAULT_STRATEGIES = ["random"]

DEFAULT_MODEL = "ResNet18"  # string, resolved in code
EPOCHS = 10      
BATCH_SIZE = 64
N_RUNS = 2

# All dataset keys must be supported by build_dataset + DATASET_SPECS.
# Supports overrides like "newt:<task>" (see parse_dataset_key below).
DATASETS = ["newt:fgvcx_plant_pathology_healthy_vs_sick"]

# Strategy label used for folder naming + summary logging.
STRATEGY_LABEL = "random"

# Strategy registry: add more as teammates contribute.
STRATEGIES = {
    "random": "trainer.batching.vision_batching.random_batch:batch_sampler",
    "smart": "trainer.batching.vision_batching.smart_batch:batch_sampler",
    "milo": "trainer.batching.vision_batching.milo:batch_sampler",
    "corset": "trainer.batching.vision_batching.corset:batch_sampler",
}

# Model class
MODEL_CLS = ResNet18

MODEL_REGISTRY = {
    "SimpleMLP": SimpleMLP,
    "SimpleCNN": SimpleCNN,
    "ResNet18": ResNet18,
}



# =========================
# Device info (for logging)
# =========================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"


# =========================
# FUTURE: trainer/utils/import_utils.py
# =========================
def load_attr(path: str):
    """Load an attribute from 'package.module:attr'."""
    if ":" not in path:
        raise ValueError(f"Expected 'module.path:attr', got {path!r}")
    mod_path, attr = path.split(":", 1)
    mod = importlib.import_module(mod_path)
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise AttributeError(f"Module {mod_path!r} has no attribute {attr!r}") from e


# =========================
# FUTURE: trainer/datasets/key_parsing.py
# =========================
def parse_dataset_key(key: str) -> Tuple[str, Dict[str, Any]]:
    """
    Supports keys like:
      - "cifar10"
      - "iwildcam"
      - "newt:<task>"
    Returns: (base_name, overrides_dict)
    """
    if key.startswith("newt:"):
        task = key.split("newt:", 1)[1].strip()
        if not task:
            raise ValueError("NeWT key must look like 'newt:<task>'")
        return "newt", {"task": task}
    return key, {}


def safe_name(s: str) -> str:
    """Filesystem-safe name for dataset keys like 'newt:task'."""
    return s.replace("/", "-").replace(":", "__")


# =========================
# FUTURE: trainer/analysis/stats.py
# =========================
def quick_stats_generic(ds, name: str, n: int = 2000):
    """
    Task-agnostic sanity snapshot:
    - Works for binary + multiclass classification (scalar labels).
    - If labels aren't scalar, it will just print the label type and stop.
    """

    n = min(n, len(ds))
    ys = []
    for i in range(n):
        _, y = ds[i]
        try:
            ys.append(int(y))
        except Exception:
            print(f"[stats] {name}: non-scalar label type={type(y)} (skipping histogram)")
            return

    c = Counter(ys)
    total = sum(c.values()) or 1
    top10 = c.most_common(10)
    majority_frac = (top10[0][1] / total) if top10 else 0.0
    print(f"[stats] {name}: size={len(ds)} sampled={n} classes_seen={len(c)} top10={top10} majority_frac={majority_frac:.3f}")


# =========================
# FUTURE: trainer/analysis/reporting.py
# =========================
def append_summary_txt(
    run_dir: str,
    header: str,
    means: Dict[str, Any],
    cis: Dict[str, Any],
    epochs: int,
    *,
    means_keys: list[str] | None = None,
):
    """
    Append a human-readable summary that DOES NOT assume specific metric names.
    It logs any metric that:
      - exists in both means and cis
      - has length == epochs
    Also records means.keys() once per dataset (as requested).
    """
    path = os.path.join(run_dir, "summary.txt")
    with open(path, "a", buffering=1) as f:
        f.write(header.rstrip() + "\n")

        if means_keys is not None:
            f.write("means.keys(): " + ", ".join(means_keys) + "\n")

        # Determine per-epoch metric keys
        metric_keys = []
        for k, v in means.items():
            if k == "time":
                continue
            if k not in cis:
                continue
            try:
                if hasattr(v, "__len__") and len(v) == epochs and hasattr(cis[k], "__len__") and len(cis[k]) == epochs:
                    metric_keys.append(k)
            except Exception:
                pass

        metric_keys = sorted(metric_keys)

        if not metric_keys:
            f.write("No per-epoch metrics found in means/cis.\n")
        else:
            for i in range(epochs):
                parts = []
                for k in metric_keys:
                    parts.append(f"{k}={float(means[k][i]):.4f}±{float(cis[k][i]):.4f}")
                f.write(f"Epoch {i+1}: " + ", ".join(parts) + "\n")

        if "time" in means and "time" in cis:
            f.write(f"time={float(means['time']):.2f}±{float(cis['time']):.2f} sec\n")

        f.write("\n")
    return path


def write_dataset_json(
    run_dir: str,
    ds_key: str,
    base_name: str,
    overrides: Dict[str, Any],
    means: Dict[str, Any],
    cis: Dict[str, Any],
):
    """
    Machine-readable snapshot for later aggregation/debugging.
    Keeps means.keys() and final-epoch values where present.
    """
    out = {
        "dataset_key": ds_key,
        "base_name": base_name,
        "overrides": overrides,
        "means_keys": sorted(list(means.keys())),
        "final": {},
    }

    # Save final values for any epoch-curves we can index
    for k, v in means.items():
        if k == "time":
            out["final"][k] = float(v)
            continue
        try:
            out["final"][k] = float(v[-1])
        except Exception:
            # ignore non-indexables
            pass

    out_path = os.path.join(run_dir, f"final__{safe_name(ds_key)}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    return out_path

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark datasets with batching strategies (no plotting).")

    p.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS),
                   help="Comma-separated dataset keys, e.g. 'cifar10,cifar100,newt:task'")

    p.add_argument("--strategies", type=str, default=",".join(DEFAULT_STRATEGIES),
                   help="Comma-separated strategy labels, e.g. 'random,smart' (must exist in STRATEGIES dict)")

    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--runs", type=int, default=DEFAULT_N_RUNS)

    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   choices=sorted(MODEL_REGISTRY.keys()),
                   help="Model class to use")

    p.add_argument("--run-tag", type=str, default="",
                   help="Optional label appended to run folder name (e.g. 'smoke' or 'ablation1')")

    return p.parse_args()

def make_run_label(strategy_label: str, model_name: str, run_tag: str) -> str:
    parts = [strategy_label, model_name]
    if run_tag:
        parts.append(run_tag)
    return "_".join(parts)

def subsample_dataset(ds, n, seed=0):
    if n is None or n >= len(ds):
        return ds
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    return Subset(ds, sorted(idx.tolist()))


# =========================
# Experiment runner
# =========================
def run_one_dataset(
    ds_key: str,
    *,
    batch_sampler,
    strategy_label: str,
    model_cls,
    epochs: int,
    batch_size: int,
    n_runs: int,
):
    base_name, overrides = parse_dataset_key(ds_key)

    train_ds, test_ds = build_dataset(shared_root=SHARED_DATA_DIR, name=base_name, **overrides)

    if base_name == "iwildcam":
        train_ds = subsample_dataset(train_ds, 2000, seed=0)
        test_ds = subsample_dataset(test_ds, 1000, seed=0)

    # Optional sanity checks (helpful during migration)
    try:
        quick_stats_generic(train_ds, "train")
        quick_stats_generic(test_ds, "test")
        print("[stats] train class_names:", getattr(train_ds, "class_names", None))
        print("[stats] test  class_names:", getattr(test_ds, "class_names", None))
    except Exception as e:
        print(f"[warn] quick_stats failed: {e}")

    model_ctor: Callable[[], Any] = lambda: build_model_for(base_name, train_ds, model_cls=model_cls)

    results = run_experiment(
        batch_sampler,
        f"{strategy_label}-benchmarking",
        train_ds,
        test_ds,
        model_ctor,
        epochs,
        batch_size,
        n_runs,
    )
    means, cis = aggregate_results(results)

    # Requested: print means.keys() so we can use it for the next step
    mk = sorted(list(means.keys()))
    print(f"[metrics] {ds_key} means.keys() = {mk}")

    return base_name, overrides, means, cis, mk


def main():
    args = parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    if not datasets:
        raise ValueError("No datasets provided.")
    if not strategies:
        raise ValueError("No strategies provided.")

    model_cls = MODEL_REGISTRY[args.model]

    print(f"[config] device={DEVICE} ({DEVICE_NAME}) epochs={args.epochs} bs={args.batch_size} runs={args.runs}")
    print(f"[config] model={model_cls.__name__} datasets={datasets} strategies={strategies}")

    for strategy_label in strategies:
        if strategy_label not in STRATEGIES:
            raise KeyError(f"Unknown strategy {strategy_label!r}. Known: {sorted(STRATEGIES)}")

        batch_sampler = load_attr(STRATEGIES[strategy_label])

        run_label = make_run_label(strategy_label, model_cls.__name__, args.run_tag)
        run_dir = create_run_dir(run_label)

        print(f"\n[bench] Strategy={strategy_label} -> {run_dir}")

        # Save run-level config once per strategy run
        cfg_path = os.path.join(run_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(
                {
                    "strategy": strategy_label,
                    "strategy_path": STRATEGIES[strategy_label],
                    "datasets": datasets,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "n_runs": args.runs,
                    "model_cls": model_cls.__name__,
                    "device_name": DEVICE_NAME,
                    "shared_data_dir": SHARED_DATA_DIR,
                    "run_tag": args.run_tag,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"[bench] Wrote {cfg_path}")

        for ds_key in datasets:
            print(f"\n=== DATASET: {ds_key} ===")
            base_name, overrides, means, cis, means_keys = run_one_dataset(
                ds_key,
                batch_sampler=batch_sampler,
                strategy_label=strategy_label,
                model_cls=model_cls,
                epochs=args.epochs,
                batch_size=args.batch_size,
                n_runs=args.runs,
            )

            header = (
                f"{ds_key} | base={base_name} | strategy={strategy_label} | model={model_cls.__name__} | "
                f"epochs={args.epochs} bs={args.batch_size} runs={args.runs} device={DEVICE_NAME}"
            )
            append_summary_txt(run_dir, header, means, cis, args.epochs, means_keys=means_keys)
            write_dataset_json(run_dir, ds_key, base_name, overrides, means, cis)

        print(f"\n[bench] Done strategy={strategy_label}. Results: {run_dir}")

if __name__ == "__main__":
    main()
