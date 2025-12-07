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
# Note: BERT training script (bert.py) runs both strategies in same execution
# and generates comparison plots directly (like the original notebook).
# These are defined here for reference but not used for dynamic loading.
NLP_BATCH_STRATEGIES = {
    "Random": "random_batch",
    "LossBased": "loss_based_batch"
}