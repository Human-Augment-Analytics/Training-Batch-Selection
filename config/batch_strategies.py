"""
Batch strategy registry for all supported strategies.
"""

# Vision batch strategies
VISION_BATCH_STRATEGIES = {
    "Random": "random",
    "Fixed": "fixed",
    "Smart": "smart",
}

# NLP batch strategies (future)
NLP_BATCH_STRATEGIES = {
    "Random": "random",
    # Add more as implemented
}

# Combined registry
BATCH_STRATEGIES = {
    "vision": VISION_BATCH_STRATEGIES,
    "nlp": NLP_BATCH_STRATEGIES,
}

# Comparison pairs for vision
VISION_STRATEGY_COMPARISON_PAIRS = [
    ("Fixed", "Smart"),
    ("Random", "Smart"),
]
