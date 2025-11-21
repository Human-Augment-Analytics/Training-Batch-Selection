"""
Model configurations registry.
"""

# Vision models
VISION_MODELS = {
    "SimpleMLP": {
        "module": "tasks.vision.models.mlp",
        "class": "SimpleMLP",
    },
}

# NLP models
NLP_MODELS = {
    "TinyLLM": {
        "module": "tasks.nlp.models.transformer",
        "class": "TinyLLM",
    },
}

# Combined registry
MODEL_CONFIGS = {
    "vision": VISION_MODELS,
    "nlp": NLP_MODELS,
}

# Active models
ACTIVE_VISION_MODEL = "SimpleMLP"
ACTIVE_NLP_MODEL = "TinyLLM"
