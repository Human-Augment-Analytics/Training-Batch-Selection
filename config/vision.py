"""
Vision-specific configuration.
"""

# ========== Training Hyperparameters ==========
EPOCHS = 5
BATCH_SIZE = 64
N_RUNS = 5  # Number of runs for statistical significance
LEARNING_RATE = 1e-3

# ========== Model Hyperparameters ==========
INPUT_DIM = 784  # 28x28 MNIST images (will be overridden by dataset)
HIDDEN_DIM = 128
NUM_CLASSES = 10  # Will be overridden by dataset

# ========== Smart Batch Parameters ==========
MOVING_AVG_DECAY = 0.9  # For exponential moving average of losses
EXPLORE_FRAC = 0.5  # Fraction of batch for exploration
TOP_K_FRAC = 0.2  # Top k% of hardest samples to consider

# ========== Active Dataset ==========
ACTIVE_DATASET = "mnist_csv"  # Default dataset for vision pipeline
