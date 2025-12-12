import os

# ========== Directory Paths ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLP_OUTPUT_DIR = os.path.join(BASE_DIR, 'trainer/pipelines/nlp/bert/output')

# ========== Model Configuration ==========
MODEL_NAME = 'bert-base-uncased'
NUM_LABELS = 2  # Binary classification for IMDB

# ========== Training Configuration ==========
MAX_LENGTH = 128  # Sequence length for BERT tokenization
BATCH_SIZE = 8
EPOCHS = 3
N_RUNS = 1  # Set to 3+ for statistical significance
LEARNING_RATE = 2e-5
DEVICE = 'cuda'  # Will fallback to 'cpu' if CUDA not available

# ========== Loss-Based Batch Selection ==========
LOSS_THRESHOLD = 0.5  # Threshold for loss-based filtering

# ========== Dataset Configuration ==========
USE_SUBSET = True  # Set to False for full dataset
TRAIN_SUBSET_SIZE = 1000  # Use subset for quick testing
TEST_SUBSET_SIZE = 500
