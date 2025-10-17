import importlib
import os
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

def build_model_for(name: str, train_ds, model_cls, hidden_dim: int = 128, **model_kwargs):
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
    return model_cls(input_dim=cfg_in, hidden_dim=hidden_dim, num_classes=cfg_nc, **model_kwargs)
