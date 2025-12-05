# ========== Toggle Gradient-Based Strategies ==========
# Set to True to enable GraND and GREAT batch strategies
ENABLE_GRAND = True
ENABLE_GREAT = True
ENABLE_RHO_LOSS = True

# Key: str label for directories/graphing, Value: python module path (relative to trainer.batching)
BATCH_STRATEGIES = {
    "Random": "random_batch",
    "Fixed": "fixed_batch",
    "Smart": "smart_batch",
}

# Add gradient-based strategies if enabled
if ENABLE_GRAND:
    BATCH_STRATEGIES["GraND"] = "vision_batching.gradient_batch"
if ENABLE_GREAT:
    BATCH_STRATEGIES["GREAT"] = "vision_batching.gradient_batch"
if ENABLE_RHO_LOSS:
    BATCH_STRATEGIES["RHO-Loss"] = "vision_batching.rho_loss_batch"

# For comparison tool:
COMPARE_BATCH_STRATEGY_PAIRS = [
    ("Fixed", "Smart"),
    ("Random", "Smart"),
]
