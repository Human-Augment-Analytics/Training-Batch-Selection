# ========== Toggle Gradient-Based Strategies ==========
# Set to True to enable GraND and GREAT batch strategies
from trainer.batching.vision_batching import rho_loss_batch


ENABLE_GRAND = False
ENABLE_GREAT = False
ENABLE_RHO_LOSS = True

# Key: str label for directories/graphing, Value: python module path (relative to trainer.batching)
BATCH_STRATEGIES = {
    "RHO-Loss": "rho_loss_batch",
    "Random": "random_batch",
    "Fixed": "fixed_batch",
    "Smart": "smart_batch",
}

# Add gradient-based strategies if enabled
if ENABLE_GRAND:
    BATCH_STRATEGIES["GraND"] = "gradient_batch"
if ENABLE_GREAT:
    BATCH_STRATEGIES["GREAT"] = "gradient_batch"
if ENABLE_RHO_LOSS:
    BATCH_STRATEGIES["RHO-Loss"] = "rho_loss_batch"

# For comparison tool:
COMPARE_BATCH_STRATEGY_PAIRS = [
    ("Fixed", "Smart"),
    ("Random", "Smart"),
    ("RHO-Loss", "Smart")
]
