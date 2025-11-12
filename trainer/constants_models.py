"""
Model and training configs - edit these to switch between different setups.
"""

def _get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


MODEL_SPECS = {
    "SimpleMLP": {
        "module": "trainer.model.vision.model",
        "class": "SimpleMLP",
        "type": "vision",
        "params": {"input_dim": 784, "hidden_dim": 128, "num_classes": 10},
    },
    "TinyLLM": {
        "module": "trainer.model.nlp.model",
        "class": "TinyLLM",
        "type": "nlp",
        "params": {
            "vocab_size": 128256,
            "d_model": 1024,
            "num_layers": 12,
            "num_heads": 16,
            "seq_length": 1024,
            "dropout": 0.1,
            "device": _get_device()
        },
    },
}

OPTIMIZER_SPECS = {
    "Adam": {
        "class": "torch.optim.Adam",
        "params": {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0},
    },
    "AdamW": {
        "class": "torch.optim.AdamW",
        "params": {"lr": 3e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-2},
    },
    "SGD": {
        "class": "torch.optim.SGD",
        "params": {"lr": 1e-2, "momentum": 0.9, "weight_decay": 1e-4, "nesterov": True},
    },
}

SCHEDULER_SPECS = {
    "none": {"enabled": False},
    "cosine_warmup": {
        "enabled": True,
        "class": "custom",
        "params": {"warmup_steps": 1000, "total_steps": 10000, "eta_min": 1e-6},
    },
}

TRAINING_CONFIGS = {
    "vision_default": {
        "epochs": 5,
        "batch_size": 64,
        "device": _get_device(),
        "optimizer": "Adam",
        "scheduler": "none",
        "grad_clip_norm": None,
        "loss_fn": "CrossEntropyLoss",
    },
    "nlp_pretraining": {
        "epochs": 2,
        "batch_size": 8,
        "device": _get_device(),
        "optimizer": "AdamW",
        "scheduler": "cosine_warmup",
        "grad_clip_norm": 1.0,
        "warmup_steps": 1000,
    },
}

# Change these to switch models/configs
ACTIVE_VISION_MODEL = "SimpleMLP"
ACTIVE_VISION_TRAINING = "vision_default"

ACTIVE_NLP_MODEL = "TinyLLM"
ACTIVE_NLP_TRAINING = "nlp_pretraining"


def get_model_spec(model_name):
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(MODEL_SPECS.keys())}")
    return MODEL_SPECS[model_name]


def get_optimizer_spec(optimizer_name):
    if optimizer_name not in OPTIMIZER_SPECS:
        raise ValueError(f"Optimizer '{optimizer_name}' not found. Available: {list(OPTIMIZER_SPECS.keys())}")
    return OPTIMIZER_SPECS[optimizer_name]


def get_scheduler_spec(scheduler_name):
    if scheduler_name not in SCHEDULER_SPECS:
        raise ValueError(f"Scheduler '{scheduler_name}' not found. Available: {list(SCHEDULER_SPECS.keys())}")
    return SCHEDULER_SPECS[scheduler_name]


def get_training_config(config_name):
    if config_name not in TRAINING_CONFIGS:
        raise ValueError(f"Config '{config_name}' not found. Available: {list(TRAINING_CONFIGS.keys())}")
    return TRAINING_CONFIGS[config_name]
