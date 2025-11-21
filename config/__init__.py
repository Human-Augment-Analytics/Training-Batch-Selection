"""
Configuration module for Training Batch Selection project.

All configuration settings are centralized here for easy management.
"""
from config.base import *
from config.datasets import DATASET_SPECS
from config.batch_strategies import BATCH_STRATEGIES
from config.models import MODEL_CONFIGS

__all__ = [
    # Base config
    'DEVICE', 'USE_GPU', 'NUM_WORKERS', 'PIN_MEMORY',
    'BASE_DIR', 'DATASETS_ROOT', 'OUTPUTS_ROOT',
    'RANDOM_SEED',

    # Dataset specs
    'DATASET_SPECS',

    # Batch strategies
    'BATCH_STRATEGIES',

    # Models
    'MODEL_CONFIGS',
]
