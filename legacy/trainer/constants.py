import os
import yaml
from pathlib import Path

# Load configuration from YAML file
def _load_config():
    """Load dataset configuration from YAML file"""
    config_file = Path(__file__).parent.parent / "dataset_config_enhanced.yaml"
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return None

_config = _load_config()

# ========== Directory Paths ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION_DATA_DIR = os.path.join(BASE_DIR, 'trainer/data/vision')
OUTPUT_DIR = os.path.join(BASE_DIR, 'trainer/pipelines/vision/output')

# get dataset root from config file or environment variable
config_root = _config['paths'].get('datasets_root', 'datasets') if _config and 'paths' in _config else 'datasets'
DATASETS_ROOT = os.environ.get('DATASETS_ROOT', config_root)

# convert to absolute path if relative
DATASETS_ROOT = os.path.abspath(os.path.join(BASE_DIR, DATASETS_ROOT)) if not os.path.isabs(DATASETS_ROOT) else DATASETS_ROOT

# backward compatibility
SHARED_DATA_DIR = DATASETS_ROOT

# ========== Data Files ==========
TRAIN_CSV = os.path.join(VISION_DATA_DIR, 'mnist_train.csv')
TEST_CSV  = os.path.join(VISION_DATA_DIR, 'mnist_test.csv')

# ========== Model hyperparameters ==========
INPUT_DIM = 784  # 28x28 MNIST images
HIDDEN_DIM = 128  # hidden layer size
NUM_CLASSES = 10  # digits 0-9

# ========== Training hyperparameters ==========
EPOCHS = 5
BATCH_SIZE = 64
N_RUNS = 5  # number of runs for statistical significance

# Auto-detect GPU availability and use it if available
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_GPU = torch.cuda.is_available()
NUM_WORKERS = 4 if USE_GPU else 0  # parallel data loading for GPU
PIN_MEMORY = USE_GPU  # faster CPU->GPU transfer

# ========== Smart Batch parameters ==========
MOVING_AVG_DECAY = 0.9  # for exponential moving average of losses
EXPLORE_FRAC = 0.5  # fraction of batch for exploration
TOP_K_FRAC = 0.2  # top k% of hardest samples to consider

# ========== Random Seed ==========
RANDOM_SEED = 2024

# ========== Active Dataset ==========
# Default dataset for vision pipeline
# Can be overridden with --dataset command-line argument
# Supported: mnist_csv, qmnist_csv, cifar10_csv, cifar100_csv
ACTIVE_DATASET = "mnist_csv"

# Config-driven system for model/optimizer selection
# see trainer/constants_models.py for details
