"""
Base configuration for device, paths, and general settings.
"""
import os
import yaml
import torch
from pathlib import Path

# ========== Project Paths ==========
BASE_DIR = Path(__file__).parent.parent.absolute()
DATASETS_ROOT = BASE_DIR / 'datasets'
OUTPUTS_ROOT = BASE_DIR / 'outputs'

# Load YAML config if exists
def _load_yaml_config():
    config_file = BASE_DIR / "dataset_config_enhanced.yaml"
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return None

_yaml_config = _load_yaml_config()

# Override datasets root from YAML or environment variable
if _yaml_config and 'paths' in _yaml_config:
    config_root = _yaml_config['paths'].get('datasets_root', 'datasets')
    DATASETS_ROOT = Path(config_root) if Path(config_root).is_absolute() else BASE_DIR / config_root

# Environment variable takes highest priority
if 'DATASETS_ROOT' in os.environ:
    DATASETS_ROOT = Path(os.environ['DATASETS_ROOT'])

# ========== Device Configuration (GPU/CPU Auto-Detection) ==========
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_GPU = torch.cuda.is_available()
NUM_WORKERS = 4 if USE_GPU else 0  # Parallel data loading for GPU
PIN_MEMORY = USE_GPU  # Faster CPU->GPU transfer

# ========== Random Seed for Reproducibility ==========
RANDOM_SEED = 2024

# ========== Create output directories ==========
OUTPUTS_ROOT.mkdir(exist_ok=True)
(OUTPUTS_ROOT / 'vision').mkdir(exist_ok=True)
(OUTPUTS_ROOT / 'nlp').mkdir(exist_ok=True)
