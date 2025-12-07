# ========== Vision Batch Strategies ==========
# Key: str label for directories/graphing, Value: python module path (relative to trainer.batching.vision_batching)
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

# ========== NLP Batch Strategies ==========
# Key: str label for directories/graphing, Value: python module path (relative to trainer.batching.nlp_batching)
NLP_BATCH_STRATEGIES = {
    "Random": "random_batch",
    "LossBased": "loss_based_batch"
}

# For NLP comparison tool:
COMPARE_NLP_BATCH_STRATEGY_PAIRS = [
    ("Random", "LossBased"),
]