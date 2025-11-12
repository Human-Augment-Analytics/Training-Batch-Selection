# Config-Driven Model System

Quick guide for switching models and optimizers without code changes.

## How to Switch Models

Edit `trainer/constants_models.py` and change:

```python
ACTIVE_VISION_MODEL = "SimpleMLP"  # Change this
```

Available models:
- `SimpleMLP` - 2-layer MLP for vision tasks
- `TinyLLM` - Transformer for NLP

## How to Switch Optimizers

In `trainer/constants_models.py`, edit the training config:

```python
TRAINING_CONFIGS = {
    "vision_default": {
        "optimizer": "Adam",  # Change to "AdamW" or "SGD"
        ...
    }
}
```

Available optimizers: `Adam`, `AdamW`, `SGD`

## Usage in Code

```python
from trainer.factories import create_model, create_optimizer

# Create model
model = create_model("SimpleMLP")

# Create optimizer
optimizer = create_optimizer("Adam", model.parameters())
```

## Adding New Models

1. Add to `MODEL_SPECS` in `trainer/constants_models.py`:

```python
"MyModel": {
    "module": "trainer.model.vision.my_model",
    "class": "MyModel",
    "type": "vision",
    "params": {"param1": value1},
}
```

2. Use it: `ACTIVE_VISION_MODEL = "MyModel"`

That's it! No code changes needed in training scripts.
