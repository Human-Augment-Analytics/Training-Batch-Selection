import importlib
import os
import torch
import inspect
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

def build_model_for_works(name: str, train_ds, model_cls, hidden_dim: int = 128, **model_kwargs):
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


def build_model_for(
    name: str,
    train_ds,
    model_cls,
    *,
    hidden_dim: int = 128,
    verify_sample: bool = True,   # set False to skip runtime verification
    **model_kwargs
):
    """
    Construct a model using DATASET_SPECS as the ground truth.
    We do not infer; we only verify (optionally) and give loud feedback.
    """
    spec = DATASET_SPECS[name]  # must contain num_classes; plus input_dim (MLP) or in_channels (CNN)
    print(f"[build_model_for] Constructing model: {model_cls.__name__}")

    # ---- Required spec fields ----
    if "num_classes" not in spec:
        raise KeyError(f"[{name}] DATASET_SPECS must define 'num_classes'.")
    cfg_nc = int(spec["num_classes"])

    params = set(inspect.signature(model_cls).parameters.keys())
    is_mlp = "input_dim" in params
    is_cnn = "in_channels" in params

    if is_mlp and "input_dim" not in spec:
        raise KeyError(f"[{name}] model expects 'input_dim' but DATASET_SPECS lacks it.")
    if is_cnn and "in_channels" not in spec:
        raise KeyError(f"[{name}] model expects 'in_channels' but DATASET_SPECS lacks it.")

    # ---- Build strictly from spec ----
    if is_mlp:
        cfg_in = int(spec["input_dim"])
        print(f"[build_model_for] {name}: MLP -> input_dim={cfg_in}, num_classes={cfg_nc}")
        model = model_cls(input_dim=cfg_in, hidden_dim=hidden_dim, num_classes=cfg_nc, **model_kwargs)

    elif is_cnn:
        cfg_c = int(spec["in_channels"])
#        print(f"[build_model_for] {name}: CNN -> in_channels={cfg_c}, num_classes={cfg_nc}")
#        model = model_cls(in_channels=cfg_c, num_classes=cfg_nc, **model_kwargs)
        input_size = spec.get("image_size", None)
        print(f"[build_model_for] {name}: CNN -> in_channels={cfg_c}, num_classes={cfg_nc}, input_size={input_size}")
        model = model_cls(in_channels=cfg_c, num_classes=cfg_nc,
                  input_size=input_size, **model_kwargs)

    else:
        # Fallback: choose based on spec keys, still with no inference
        if "in_channels" in spec:
            cfg_c = int(spec["in_channels"])
            print(f"[build_model_for] {name}: CNN (fallback) -> in_channels={cfg_c}, num_classes={cfg_nc}")
            model = model_cls(in_channels=cfg_c, num_classes=cfg_nc, **model_kwargs)
        elif "input_dim" in spec:
            cfg_in = int(spec["input_dim"])
            print(f"[build_model_for] {name}: MLP (fallback) -> input_dim={cfg_in}, num_classes={cfg_nc}")
            model = model_cls(input_dim=cfg_in, hidden_dim=hidden_dim, num_classes=cfg_nc, **model_kwargs)
        else:
            raise TypeError(
                f"[{name}] DATASET_SPECS must include either 'in_channels' (for CNNs) "
                f"or 'input_dim' (for MLPs)."
            )

    # ---- Optional verification (diagnostics only, no inference) ----
    if verify_sample:
        try:
            x0, y0 = train_ds[0]
            flat_inferred = int(x0.numel())
            if is_mlp:
                cfg_in = int(spec["input_dim"])
                if flat_inferred != cfg_in:
                    raise ValueError(
                        f"[{name}] VERIFY: dataset sample flattened={flat_inferred}, "
                        f"spec.input_dim={cfg_in}. "
                        "Mismatch: check transforms/flatten setting or spec."
                    )
                if x0.ndim != 1:
                    print(f"[{name}] NOTE: dataset sample shape {tuple(x0.shape)} but MLP expects flat; "
                          "ensure flattening happens in shaper or dataset.")
            else:  # CNN path
                cfg_c = int(spec["in_channels"])
                if x0.ndim != 3:
                    raise ValueError(
                        f"[{name}] VERIFY: dataset sample shape {tuple(x0.shape)} but CNN requires (C,H,W). "
                        "Likely the dataset is flattened; build with flatten=False."
                    )
                if int(x0.shape[0]) != cfg_c:
                    raise ValueError(
                        f"[{name}] VERIFY: dataset channels={int(x0.shape[0])} but spec.in_channels={cfg_c}. "
                        "Mismatch: adjust spec or dataset transforms."
                    )
        except Exception as e:
            # Make verification failures loud and actionable
            print(f"[build_model_for] VERIFICATION FAILED for {name}: {e}")
            raise

    return model
