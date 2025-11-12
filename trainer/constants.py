import os
import yaml
from pathlib import Path

# ========== Load Configuration from YAML ==========
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

# Get dataset root from config, with environment variable override
config_root = _config['paths'].get('datasets_root', 'datasets') if _config and 'paths' in _config else 'datasets'
DATASETS_ROOT = os.environ.get('DATASETS_ROOT', config_root)

# Make path absolute if it's relative (relative to project root)
DATASETS_ROOT = os.path.abspath(os.path.join(BASE_DIR, DATASETS_ROOT)) if not os.path.isabs(DATASETS_ROOT) else DATASETS_ROOT

# Legacy alias for backward compatibility
SHARED_DATA_DIR = DATASETS_ROOT

# ========== Data Files ==========
TRAIN_CSV = os.path.join(VISION_DATA_DIR, 'mnist_train.csv')
TEST_CSV  = os.path.join(VISION_DATA_DIR, 'mnist_test.csv')

# ========== Model ==========
INPUT_DIM = 784
HIDDEN_DIM = 128
NUM_CLASSES = 10

# ========== Training ==========
EPOCHS = 5
BATCH_SIZE = 64
N_RUNS = 5
DEVICE = 'cpu'

# ========== Smart Batch ==========
MOVING_AVG_DECAY = 0.9
EXPLORE_FRAC = 0.5
TOP_K_FRAC = 0.2

# ========== Random Seed ==========
RANDOM_SEED = 2024

# ========== Config-Driven System ==========
# For model/optimizer switching, see trainer/constants_models.py
# Usage:
#   from trainer.factories import create_model, create_optimizer
#   model = create_model("SimpleMLP")
#   optimizer = create_optimizer("Adam", model.parameters())
