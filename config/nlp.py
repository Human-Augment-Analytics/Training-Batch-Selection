"""
NLP-specific configuration for pretraining.
"""

# ========== Training Hyperparameters ==========
EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 1000
GRAD_CLIP_NORM = 1.0
SAVE_EVERY = 1  # Save checkpoint every N epochs

# ========== Model Hyperparameters ==========
VOCAB_SIZE = 128256
D_MODEL = 1024
NUM_LAYERS = 12
NUM_HEADS = 16
SEQ_LENGTH = 1024
DROPOUT = 0.1

# ========== Data Settings ==========
TOKENIZED_ROOT = 'trainer/data/tokenized/tiiuae_falcon-refinedweb'
MAX_SAMPLES = 10_000_000
TOKENIZE_STRIDE = 1
EOS_TOKEN = None
