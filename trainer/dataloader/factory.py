import importlib
import os
import torch
from trainer.constants_datasets import DATASET_SPECS

# One shared place to compute the on-disk path for a dataset name
def dataset_root(shared_root: str, name: str) -> str:
    spec = DATASET_SPECS[name]
    return os.path.join(shared_root.rstrip("/"), spec["subdir"])

def build_dataset(shared_root: str, name: str, **overrides):
    """
    shared_root: e.g. "/storage/ice-shared/cs8903onl/lw-batch-selection/datasets"
    name: key from DATASET_SPECS
    overrides: optional kwargs forwarded to the builder (e.g., as_flat=False for CNN)
    """
    spec = DATASET_SPECS[name]
    mod = importlib.import_module("trainer.dataloader.builders")
    builder = getattr(mod, spec["builder"])
    root = dataset_root(shared_root, name)
    return builder(root, **overrides)

def spec_for(name: str):
    return DATASET_SPECS[name]

def _infer_input_dim(dataset) -> int:
    x0, _ = dataset[0]
    return int(x0.numel())

def _infer_num_classes(dataset) -> int:
    # torchvision-style
    if hasattr(dataset, "base") and hasattr(dataset.base, "classes"):
        return len(dataset.base.classes)
    if hasattr(dataset, "classes"):
        return len(dataset.classes)
    # fallback
    _, y0 = dataset[0]
    return int(y0.max().item() + 1 if torch.is_tensor(y0) else int(y0) + 1)

def build_model_for_org(name: str, train_ds, model_cls, hidden_dim: int = 128, **model_kwargs):
    """Construct a model consistent with the model/dataset spec and sanity-check shapes."""
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

#    return SimpleMLP(input_dim=cfg_in, hidden_dim=hidden_dim, num_classes=cfg_nc)
    # allow for model specification
    return model_cls(input_dim=cfg_in, hidden_dim=hidden_dim, num_classes=cfg_nc, **model_kwargs)

def build_model_for_a(name: str, train_ds, model_cls, hidden_dim: int = 128, **model_kwargs):
    """Construct a model consistent with the model/dataset spec and sanity-check shapes."""
    spec   = DATASET_SPECS[name]
    cfg_in = int(spec["input_dim"])
    cfg_nc = int(spec["num_classes"])

    # infer from dataset
    x0, _  = train_ds[0]
    inf_nc = _infer_num_classes(train_ds)
    flat_inferred = x0.numel()

    # sanity checks (use flattened size so this works for both flat/CNN datasets)
    if cfg_in != flat_inferred:
        raise ValueError(
            f"[{name}] input_dim mismatch: spec={cfg_in}, inferred(flattened)={flat_inferred}. "
            "Check flatten/resize/transforms."
        )
    if cfg_nc != inf_nc:
        print(f"Warning: [{name}] num_classes spec={cfg_nc}, inferred={inf_nc}")

    # Route by model type
    if getattr(model_cls, "expects_flat", True):
        # MLP-style models get input_dim/hidden_dim
        if x0.ndim != 1:
            # Not fatal—your dataloader/model may flatten internally—but make it explicit.
            print(f"Note: [{name}] dataset returns shape {tuple(x0.shape)}; model expects flat. "
                  "Ensure flattening happens in dataset or forward().")
        return model_cls(input_dim=cfg_in, hidden_dim=hidden_dim, num_classes=cfg_nc, **model_kwargs)
    else:
        # CNN-style models get in_channels only
        if x0.ndim != 3:
            raise ValueError(
                f"[{name}] model expects images (C,H,W) but dataset returns shape {tuple(x0.shape)}. "
                "Use a non-flattened image dataset/builder."
            )
        in_channels = int(x0.shape[0])  # (C,H,W)
        # Optional: guard against mismatched explicit kwargs
        if "in_channels" in model_kwargs and model_kwargs["in_channels"] != in_channels:
            print(f"Warning: overriding provided in_channels={model_kwargs['in_channels']} with {in_channels} "
                  f"based on dataset for [{name}].")
            model_kwargs.pop("in_channels", None)
        return model_cls(in_channels=in_channels, num_classes=cfg_nc, **model_kwargs)

import inspect

def build_model_for(name: str, train_ds, model_cls, hidden_dim: int = 128, **model_kwargs):
    """Construct a model consistent with the dataset spec and the model's constructor signature."""
    spec   = DATASET_SPECS[name]
    cfg_in = int(spec["input_dim"])     # keep flattened for consistency in specs
    cfg_nc = int(spec["num_classes"])

    # Infer from actual dataset sample
    x0, _ = train_ds[0]
    flat_inferred = x0.numel()

    # Sanity checks
    if cfg_in != flat_inferred:
        raise ValueError(
            f"[{name}] input_dim mismatch: spec={cfg_in}, inferred(flattened)={flat_inferred}. "
            "Check flatten/resize/transforms or use the correct dataset name."
        )

    # Decide how to call the model constructor by inspecting its signature
    params = set(inspect.signature(model_cls).parameters.keys())

    if "input_dim" in params:
        # MLP-style
        if x0.ndim != 1:
            # Not fatal, but warn if the dataset is images and model expects flat
            print(f"Note: [{name}] dataset returns shape {tuple(x0.shape)}; model ctor has input_dim. "
                  "Ensure flattening occurs in dataset or model.forward().")
        return model_cls(input_dim=cfg_in, hidden_dim=hidden_dim, num_classes=cfg_nc, **model_kwargs)

    if "in_channels" in params:
        # CNN-style
        if x0.ndim != 3:
            raise ValueError(
                f"[{name}] model ctor has in_channels (CNN), but dataset sample is {tuple(x0.shape)}. "
                "Use the image builder (flatten=False), i.e., dataset name 'cifar10'."
            )
        in_channels = int(x0.shape[0])
        return model_cls(in_channels=in_channels, num_classes=cfg_nc, **model_kwargs)

    # Fallbacks if neither param name exists (some authors use different names)
    # Try CNN-style first if input looks like images; else try MLP-style.
    try:
        if x0.ndim == 3:
            in_channels = int(x0.shape[0])
            return model_cls(in_channels=in_channels, num_classes=cfg_nc, **model_kwargs)
        else:
            return model_cls(input_dim=cfg_in, hidden_dim=hidden_dim, num_classes=cfg_nc, **model_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Unable to construct {model_cls.__name__}. Expected ctor with either "
            "`input_dim` (for MLP) or `in_channels` (for CNN). Original error: {e}"
        )
