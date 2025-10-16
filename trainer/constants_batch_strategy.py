# Key: str label for directories/graphing, Value: python module path (relative to trainer.batching)
BATCH_STRATEGIES = {
    "Random": "random_batch",
    "Fixed": "fixed_batch",
    "Smart": "smart_batch"
}

# For comparison tool:
COMPARE_BATCH_STRATEGY_PAIRS = [
    ("Fixed", "Smart"),
    ("Random", "Smart"),
]