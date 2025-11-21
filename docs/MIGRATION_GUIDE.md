# Project Restructuring - Migration Guide

## ✅ What Was Done

Complete project overhaul to create a clean, task-oriented structure while preserving all functionality including GPU optimizations.

### Major Changes:

1. **Created Clean Structure**
   - `config/` - All configuration centralized
   - `tasks/` - Task-specific code (vision/nlp)
   - `core/` - Shared utilities
   - `scripts/` - Utility scripts
   - `outputs/` - Clean output organization
   - `legacy/` - Old code preserved for reference

2. **Migrated All Code**
   - ✅ Vision models → `tasks/vision/models/`
   - ✅ Vision datasets → `tasks/vision/datasets/`
   - ✅ Vision batch strategies → `tasks/vision/batch_strategies/`
   - ✅ Vision training/eval → `tasks/vision/train.py`, `tasks/vision/evaluate.py`
   - ✅ NLP models → `tasks/nlp/models/`
   - ✅ NLP datasets → `tasks/nlp/datasets/`
   - ✅ Factories → `core/factories/`
   - ✅ Scripts → `scripts/`

3. **Preserved Features**
   - ✅ GPU auto-detection (in `config/base.py`)
   - ✅ CSV auto-conversion for all datasets
   - ✅ Dataset factory pattern
   - ✅ Batch strategy plugin system
   - ✅ Multi-run statistical analysis
   - ✅ Config-driven design

4. **Old Code**
   - ✅ Moved `trainer/` → `legacy/trainer/` (for reference)

## 📁 File Mapping

### Configuration Files

| Old Location | New Location | Notes |
|--------------|-------------|-------|
| `trainer/constants.py` | `config/base.py` + `config/vision.py` | Split by purpose |
| `trainer/constants_datasets.py` | `config/datasets.py` | Enhanced |
| `trainer/constants_batch_strategy.py` | `config/batch_strategies.py` | Task-specific |
| `trainer/constants_models.py` | `config/models.py` | Registry |

### Vision Task

| Old Location | New Location |
|--------------|-------------|
| `trainer/model/vision/model.py` | `tasks/vision/models/mlp.py` |
| `trainer/dataloader/vision_dataloader.py` | `tasks/vision/datasets/loaders.py` |
| `trainer/dataloader/builders.py` | `tasks/vision/datasets/builders.py` |
| `trainer/dataloader/factory.py` | `tasks/vision/datasets/factory.py` |
| `trainer/dataloader/auto_convert_csv.py` | `tasks/vision/datasets/auto_convert_csv.py` |
| `trainer/batching/vision_batching/*.py` | `tasks/vision/batch_strategies/*.py` |
| `trainer/pipelines/vision/vision.py` | `tasks/vision/run_experiment.py` + `train.py` + `evaluate.py` |

### NLP Task

| Old Location | New Location |
|--------------|-------------|
| `trainer/model/nlp/` | `tasks/nlp/models/` |
| `trainer/dataloader/text_dataloader.py` | `tasks/nlp/datasets/tokenized.py` |
| `trainer/pipelines/pretraining/run_pretraining.py` | `tasks/nlp/run_pretraining.py` |

### Core & Scripts

| Old Location | New Location |
|--------------|-------------|
| `trainer/factories/` | `core/factories/` |
| `simple_dataset_loader.py` | `scripts/download_datasets.py` |
| `check_device.py` | `scripts/check_device.py` |
| `convert_mnist_to_csv.py` | `scripts/convert_to_csv.py` |

## 🚀 How to Use New Structure

### Running Vision Experiments

**Old way:**
```bash
python -m trainer.pipelines.vision.vision --dataset cifar10_csv
```

**New way:**
```bash
python -m tasks.vision.run_experiment --dataset cifar10_csv
```

### Importing Configuration

**Old way:**
```python
from trainer.constants import DEVICE, EPOCHS, BATCH_SIZE
from trainer.constants_datasets import DATASET_SPECS
```

**New way:**
```python
from config.base import DEVICE
from config.vision import EPOCHS, BATCH_SIZE
from config.datasets import DATASET_SPECS
```

### Importing Models

**Old way:**
```python
from trainer.model.vision.model import SimpleMLP
```

**New way:**
```python
from tasks.vision.models.mlp import SimpleMLP
```

### Importing Datasets

**Old way:**
```python
from trainer.dataloader.factory import build_dataset
```

**New way:**
```python
from tasks.vision.datasets.factory import build_dataset
```

## 🔧 Output Directory Changes

**Old structure:**
```
trainer/pipelines/vision/output/
├── batching_random/run-001/
├── batching_fixed/run-001/
└── batching_smart/run-001/
```

**New structure:**
```
outputs/
└── vision/
    ├── mnist_csv/
    │   ├── batching_random/run-001/
    │   ├── batching_fixed/run-001/
    │   └── batching_smart/run-001/
    └── cifar10_csv/
        ├── batching_random/run-001/
        └── ...
```

## 🎯 Benefits of New Structure

1. **Clarity** - Easy to see what's vision vs NLP code
2. **Modularity** - Each task is self-contained
3. **Scalability** - Easy to add new tasks
4. **Clean Imports** - More intuitive import paths
5. **Better Organization** - Config, core, tasks, scripts separated
6. **Dataset-Organized Outputs** - Easier to compare across datasets

## ⚠️ Breaking Changes

If you have existing scripts that import from `trainer.*`, you'll need to update them:

1. Update imports to use `config.*`, `tasks.*`, or `core.*`
2. Update script commands to use new module paths
3. Update output directory references

## 🔄 Backwards Compatibility

- Old code is preserved in `legacy/trainer/` (untouched)
- Dataset folder unchanged
- Can run old code with: `python -m legacy.trainer.pipelines.vision.vision`

## 📝 Next Steps

1. Test new structure with a simple run:
   ```bash
   python -m tasks.vision.run_experiment
   ```

2. Update any custom scripts you have

3. Old code in `legacy/` can be deleted after verification (or kept as reference)

## ✨ Features Preserved

- ✅ GPU/CPU auto-switching
- ✅ Multi-dataset support (MNIST, QMNIST, CIFAR-10, CIFAR-100)
- ✅ CSV auto-conversion
- ✅ Smart batching with loss history
- ✅ Multi-run statistical analysis with confidence intervals
- ✅ Automated plotting
- ✅ Config-driven design
- ✅ Factory patterns
