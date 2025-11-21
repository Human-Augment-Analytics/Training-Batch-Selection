"""
Dataset factory for vision tasks.
"""
import importlib
import os
import torch
from config.datasets import DATASET_SPECS


def dataset_root(shared_root: str, name: str) -> str:
    """Get the on-disk path for a dataset."""
    spec = DATASET_SPECS[name]
    return os.path.join(shared_root, spec["subdir"])


def build_dataset(shared_root: str, name: str, **overrides):
    """
    Build a dataset using the factory pattern.

    Args:
        shared_root: Root directory for datasets (e.g., "./datasets")
        name: Dataset name key from DATASET_SPECS
        **overrides: Optional kwargs forwarded to the builder

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    spec = DATASET_SPECS[name]
    mod = importlib.import_module("tasks.vision.datasets.builders")
    builder = getattr(mod, spec["builder"])
    root = dataset_root(shared_root, name)
    return builder(root, **overrides)


def spec_for(name: str):
    """Get dataset specification."""
    return DATASET_SPECS[name]


def _infer_input_dim(dataset) -> int:
    """Infer input dimension from dataset."""
    x0, _ = dataset[0]
    return int(x0.numel())


def _infer_num_classes(dataset) -> int:
    """Infer number of classes from dataset."""
    # torchvision-style
    if hasattr(dataset, "base") and hasattr(dataset.base, "classes"):
        return len(dataset.base.classes)
    if hasattr(dataset, "classes"):
        return len(dataset.classes)
    # fallback
    _, y0 = dataset[0]
    return int(y0.max().item() + 1 if torch.is_tensor(y0) else int(y0) + 1)


def build_model_for(name: str, train_ds, model_cls, hidden_dim: int = 128, **model_kwargs):
    """
    Construct a model consistent with the dataset spec and sanity-check shapes.

    Args:
        name: Dataset name
        train_ds: Training dataset
        model_cls: Model class to instantiate
        hidden_dim: Hidden dimension for model
        **model_kwargs: Additional model arguments

    Returns:
        Initialized model
    """
    spec = DATASET_SPECS[name]
    cfg_in = int(spec["input_dim"])
    cfg_nc = int(spec["num_classes"])

    inf_in = _infer_input_dim(train_ds)
    inf_nc = _infer_num_classes(train_ds)

    if cfg_in != inf_in:
        raise ValueError(
            f"[{name}] input_dim mismatch: spec={cfg_in}, inferred={inf_in}. "
            "Check flatten/resize/transforms."
        )
    if cfg_nc != inf_nc:
        print(f"Warning: [{name}] num_classes spec={cfg_nc}, inferred={inf_nc}")

    return model_cls(input_dim=cfg_in, hidden_dim=hidden_dim, num_classes=cfg_nc, **model_kwargs)
