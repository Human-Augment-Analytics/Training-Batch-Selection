import importlib
import inspect

from trainer.constants_datasets import DATASET_SPECS


def build_dataset(shared_root: str, name: str, **overrides):
    """
    Build a dataset pair (train_ds, test_ds) from DATASET_SPECS.

    Parameters
    ----------
    shared_root : str
        Root shared data directory.
    name : str
        Dataset key in DATASET_SPECS.
    overrides : dict
        Extra dataset-specific settings, e.g. task=... for NeWT.
    """
    if name not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(DATASET_SPECS)}")

    spec = DATASET_SPECS[name]

    if "builder" not in spec:
        raise KeyError(f"[{name}] DATASET_SPECS must define 'builder'.")
    if "subdir" not in spec:
        raise KeyError(f"[{name}] DATASET_SPECS must define 'subdir'.")

    builder_name = spec["builder"]
    root = f"{shared_root}/{spec['subdir']}"

    builders_mod = importlib.import_module("trainer.dataloader.builders")
    if not hasattr(builders_mod, builder_name):
        raise AttributeError(f"trainer.dataloader.builders has no builder '{builder_name}'")
    builder = getattr(builders_mod, builder_name)

    # Pass through spec fields except the ones used only by the factory itself.
    builder_kwargs = {
        k: v
        for k, v in spec.items()
        if k not in {"builder", "subdir", "num_classes", "input_dim"}
    }

    # Explicit overrides win.
    builder_kwargs.update(overrides)

    return builder(root, **builder_kwargs)


def build_model_for(
    name: str,
    train_ds,
    model_cls,
    *,
    hidden_dim: int = 128,
    verify_sample: bool = True,
    **model_kwargs,
):
    """
    Construct a model using DATASET_SPECS as the ground truth.
    We do not infer; we only verify (optionally) and give loud feedback.
    """
    spec = DATASET_SPECS[name]
    model_name = model_cls.__name__
    print(f"[build_model_for] Constructing model: {model_name}")

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

    # Optional model kwargs that may be present in DATASET_SPECS for some datasets.
    optional_model_args = {
        k: spec[k]
        for k in ["pretrained", "freeze_backbone"]
        if k in spec and k in params
    }

    if is_mlp:
        cfg_in = int(spec["input_dim"])
        print(
            f"[build_model_for] {name}: {model_name} (vector model) "
            f"-> input_dim={cfg_in}, num_classes={cfg_nc}"
        )
        model = model_cls(
            input_dim=cfg_in,
            hidden_dim=hidden_dim,
            num_classes=cfg_nc,
            **model_kwargs,
        )
    ###
    elif is_cnn:
        cfg_c = int(spec["in_channels"])
        input_size = spec.get("image_size", None)

        optional_model_args = {}

        if model_name == "ResNet18":
            if "resnet18_pretrained" in spec:
                optional_model_args["pretrained"] = spec["resnet18_pretrained"]
            if "resnet18_freeze_backbone" in spec:
                optional_model_args["freeze_backbone"] = spec["resnet18_freeze_backbone"]

        print(
            f"[build_model_for] {name}: {model_name} (image model) "
            f"-> in_channels={cfg_c}, num_classes={cfg_nc}, input_size={input_size}, "
            f"extras={optional_model_args}"
        )

        model = model_cls(
            in_channels=cfg_c,
            num_classes=cfg_nc,
            input_size=input_size,
            **optional_model_args,
            **model_kwargs
        )
        ###
        
    elif is_cnnx: #save old one
        cfg_c = int(spec["in_channels"])
        input_size = spec.get("image_size", None)

        print(
            f"[build_model_for] {name}: {model_name} (image model) "
            f"-> in_channels={cfg_c}, num_classes={cfg_nc}, input_size={input_size}, "
            f"extras={optional_model_args}"
        )

        model = model_cls(
            in_channels=cfg_c,
            num_classes=cfg_nc,
            input_size=input_size,
            **optional_model_args,
            **model_kwargs,
        )

    else:
        if "in_channels" in spec:
            cfg_c = int(spec["in_channels"])
            print(
                f"[build_model_for] {name}: {model_name} (fallback image model) "
                f"-> in_channels={cfg_c}, num_classes={cfg_nc}, extras={optional_model_args}"
            )
            model = model_cls(
                in_channels=cfg_c,
                num_classes=cfg_nc,
                **optional_model_args,
                **model_kwargs,
            )

        elif "input_dim" in spec:
            cfg_in = int(spec["input_dim"])
            print(
                f"[build_model_for] {name}: {model_name} (fallback vector model) "
                f"-> input_dim={cfg_in}, num_classes={cfg_nc}"
            )
            model = model_cls(
                input_dim=cfg_in,
                hidden_dim=hidden_dim,
                num_classes=cfg_nc,
                **model_kwargs,
            )

        else:
            raise TypeError(
                f"[{name}] DATASET_SPECS must include either 'in_channels' "
                f"(for CNNs) or 'input_dim' (for MLPs)."
            )

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
                    print(
                        f"[{name}] NOTE: dataset sample shape {tuple(x0.shape)} but MLP expects flat; "
                        "ensure flattening happens in shaper or dataset."
                    )

            else:
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
            print(f"[build_model_for] VERIFICATION FAILED for {name}: {e}")
            raise

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[build_model_for] trainable params: {trainable} / {total}")


    return model

