## 🚀 Project Overhaul - Clean Structure

Based on your feedback, here's what I've done to the project structure:

### Key Changes:
1. **Task-Oriented**: Separated vision and NLP into `tasks/vision/` and `tasks/nlp/`
2. **Centralized Config**: All configuration in `config/` directory
3. **Clean Outputs**: Results organized in `outputs/{task}/{dataset}/{strategy}/`
4. **Legacy Code**: Old `trainer/` moved to `legacy/` for reference
5. **Scripts**: All utility scripts in `scripts/`

### How to Run (Examples):

```bash
# Vision experiments
python -m tasks.vision.run_experiment
python -m tasks.vision.run_experiment --dataset cifar10_csv

# Check device configuration
python scripts/check_device.py

# Download datasets
python scripts/download_datasets.py download mnist --yes
```

### New Directory Structure:

```
Training-Batch-Selection/
├── config/                    # ✨ All configuration
│   ├── base.py               # Device, paths (GPU auto-detect)
│   ├── vision.py             # Vision hyperparameters
│   ├── nlp.py                # NLP hyperparameters
│   ├── datasets.py           # Dataset specifications
│   ├── batch_strategies.py   # Strategy registry
│   └── models.py             # Model registry
│
├── tasks/                     # ✨ Task-specific code
│   ├── vision/               # Vision classification
│   │   ├── models/           # SimpleMLP
│   │   ├── datasets/         # MNIST, CIFAR loaders
│   │   ├── batch_strategies/ # Random, Fixed, Smart
│   │   ├── train.py          # Training loop
│   │   ├── evaluate.py       # Evaluation
│   │   └── run_experiment.py # Main runner
│   │
│   └── nlp/                  # Language modeling
│       ├── models/           # TinyLLM Transformer
│       ├── datasets/         # Tokenized data
│       └── run_pretraining.py
│
├── core/                      # ✨ Shared utilities
│   ├── factories/            # Model/optimizer factories
│   ├── metrics/              # Statistics & plotting
│   └── utils/                # General utilities
│
├── scripts/                   # ✨ Utility scripts
│   ├── download_datasets.py
│   ├── check_device.py
│   └── convert_to_csv.py
│
├── datasets/                  # Raw datasets (unchanged)
├── outputs/                   # Training outputs
│   ├── vision/
│   └── nlp/
│
└── legacy/                    # ✨ Old code (reference)
    └── trainer/
```

### Benefits:

✅ **Clearer separation** - Vision vs NLP code isolated
✅ **Better imports** - `from config.vision import EPOCHS`
✅ **GPU Auto-detection** - Automatically uses GPU if available
✅ **Cleaner outputs** - `outputs/vision/mnist/batching_smart/run-001/`
✅ **Easy to extend** - Add new tasks, strategies, datasets
✅ **Legacy preserved** - Old code in `legacy/` for reference

### Migration Notes:

- All GPU optimizations preserved
- Dataset folder untouched
- Config-driven design maintained
- Factory patterns intact
- Backward compatible (old code in legacy/)
