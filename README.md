# Training Batch Selection Research Project

A research project for evaluating and comparing different **batch selection strategies** during neural network training. Instead of using standard random batching, this project experiments with different ways to select which samples go into each training batch.

[![Python](https://img.shields.io/badge/Python-3.10--3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Usage Examples](#-usage-examples)
- [Configuration](#️-configuration)
- [Batch Strategies](#-batch-strategies)
- [Datasets](#-datasets)
- [GPU Optimization](#-gpu-optimization)
- [Output Management](#-output-management)
- [Adding New Components](#-adding-new-components)
- [Documentation](#-documentation)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

### Core Capabilities
- **🔄 Multiple Batch Strategies**: Random, Fixed, Smart (loss-based prioritization)
- **🚀 GPU Auto-Detection**: Automatically uses CUDA if available, falls back to CPU seamlessly
- **📊 Multi-Dataset Support**: MNIST, QMNIST, CIFAR-10, CIFAR-100 with automatic CSV conversion
- **📈 Statistical Analysis**: Multi-run experiments with mean ± 95% confidence intervals
- **📉 Automated Plotting**: Training/test accuracy and loss curves with CI bands
- **🔌 Plugin Architecture**: Easy to add new strategies, datasets, and models
- **⚙️ Config-Driven**: All hyperparameters centralized in configuration files

### Task Pipelines
1. **Vision Pipeline** (MNIST/CIFAR) - ✅ Fully implemented
2. **NLP Pipeline** (Transformer pretraining) - 🚧 Under development

---

## 🚀 Quick Start

### Requirements

- **Python 3.10 or 3.11** (enforced by `pyproject.toml`)
- CUDA 12.x (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/Human-Augment-Analytics/Training-Batch-Selection.git
cd Training-Batch-Selection

# Specifically for GA TECH PACE
module load python/3.10

# Create virtual environment (use Python 3.10 or 3.11)
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip

# Install dependencies (choose one):

# Option A: Exact pinned versions (recommended for reproducibility)
pip install -r requirements.txt

# Option B: Flexible versions from pyproject.toml
pip install .
```

### Download Datasets

```bash
# Auto-converting datasets (MNIST, QMNIST, CIFAR-10, CIFAR-100)
# List available datasets
python scripts/download_datasets.py list

# Download specific dataset
python scripts/download_datasets.py download mnist --yes
python scripts/download_datasets.py download cifar10 --yes

# CINIC-10 (requires separate setup - see docs/CINIC10_SETUP.md)
python scripts/setup_cinic10.py --convert-csv
```

### Run Your First Experiment

```bash
# Run vision experiments with all batch strategies (default: MNIST)
python -m tasks.vision.run_experiment

# Run with different datasets
python -m tasks.vision.run_experiment --dataset cifar10_csv
python -m tasks.vision.run_experiment --dataset qmnist_csv
```

**That's it!** The code will:
- ✅ Auto-detect GPU and use it if available
- ✅ Auto-convert datasets to CSV if needed
- ✅ Train models with all batch strategies
- ✅ Generate plots and statistics
- ✅ Save results to `outputs/vision/{dataset}/`

---

## 📁 Project Structure

```
Training-Batch-Selection/
├── config/                    # 🔧 All configuration
│   ├── base.py               # Device, paths, GPU auto-detect
│   ├── vision.py             # Vision hyperparameters
│   ├── nlp.py                # NLP hyperparameters
│   ├── datasets.py           # Dataset specifications
│   ├── batch_strategies.py   # Strategy registry
│   ├── models.py             # Model registry
│   └── dataset_config_enhanced.yaml  # Dataset download configs
│
├── tasks/                     # 🎯 Task-specific code
│   ├── vision/               # Vision classification
│   │   ├── models/           # SimpleMLP
│   │   ├── datasets/         # MNIST, CIFAR loaders + factory
│   │   ├── batch_strategies/ # Random, Fixed, Smart
│   │   ├── train.py          # Training loop
│   │   ├── evaluate.py       # Evaluation
│   │   └── run_experiment.py # Main runner
│   │
│   └── nlp/                  # Language modeling (under development)
│       ├── models/           # TinyLLM Transformer
│       ├── datasets/         # Tokenized data loaders
│       └── run_pretraining.py
│
├── core/                      # 🛠️ Shared utilities
│   ├── factories/            # Model/optimizer factories
│   ├── metrics/              # Statistics & plotting (future)
│   └── utils/                # General utilities (future)
│
├── scripts/                   # 📜 Utility scripts
│   ├── download_datasets.py  # Dataset downloader
│   ├── check_device.py       # GPU/CPU checker
│   └── download_engines.py   # Download engines
│
├── datasets/                  # 💾 Raw datasets (auto-created)
├── outputs/                   # 📊 Training outputs (auto-created)
├── docs/                      # 📚 Documentation
├── legacy/                    # 🗄️ Old code (reference)
│
├── .env                       # Environment variables
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT License
├── README.md                  # This file
├── pyproject.toml             # Project config & Python version constraint
├── requirements.txt           # Pinned dependencies for reproducibility
└── setup.py                   # Package setup (legacy)
```

---

## 💡 Usage Examples

### Vision Experiments - Single Command for Everything! 🎯

**NEW: Run Everything at Once**
```bash
# One command for complete pipeline:
# - Run experiments with all strategies
# - Generate comparison plots
# - Benchmark across datasets
python -m tasks.vision.run_all

# Quick test mode (1 epoch, 1 run)
python -m tasks.vision.run_all --quick

# Run on specific dataset
python -m tasks.vision.run_all --dataset cifar10_csv

# Skip benchmark
python -m tasks.vision.run_all --no-benchmark

# Custom benchmark datasets
python -m tasks.vision.run_all --benchmark-datasets mnist_csv qmnist_csv cifar10_csv
```

**Individual Commands (if needed)**
```bash
# Run experiments only
python -m tasks.vision.run_experiment
python -m tasks.vision.run_experiment --dataset cifar10_csv

# Compare strategies
python -m tasks.vision.compare --dataset mnist_csv

# Benchmark across datasets
python -m tasks.vision.benchmark --strategy Random --datasets mnist_csv qmnist_csv

# Check device configuration
python scripts/check_device.py
```

### Customizing Experiments

Edit `config/vision.py`:
```python
EPOCHS = 10          # Increase epochs
N_RUNS = 10          # More runs for better statistics
BATCH_SIZE = 128     # Larger batches
ACTIVE_DATASET = "cifar10_csv"  # Change default dataset
```

---

## ⚙️ Configuration

All configuration is centralized in the `config/` directory. Edit these files to change hyperparameters:

- **`config/base.py`** - Device settings, paths, GPU configuration
- **`config/vision.py`** - Vision training hyperparameters
- **`config/nlp.py`** - NLP pretraining hyperparameters
- **`config/datasets.py`** - Dataset specifications
- **`config/batch_strategies.py`** - Batch strategy registry
- **`config/models.py`** - Model configurations

---

## 🎲 Batch Strategies

### 1. Random Batching (Baseline)
Standard approach: shuffle dataset at epoch start, iterate sequentially.

### 2. Fixed Batching
No shuffling - always processes data in same order. Baseline to measure shuffling effect.

### 3. Smart Batching (Loss-Based)
Prioritizes samples based on recent loss history:
- **Exploration**: Randomly sample examples
- **Exploitation**: Focus on high-loss (difficult) examples
- Uses exponential moving average for per-sample loss tracking

---

## 📊 Datasets

| Dataset | Features | Classes | Samples | CSV Size | Auto-Convert Time |
|---------|----------|---------|---------|----------|-------------------|
| MNIST | 784 (28×28 grayscale) | 10 | 60K | ~123 MB | ~30 seconds |
| QMNIST | 784 (28×28 grayscale) | 10 | 120K | ~314 MB | ~1 minute |
| CIFAR-10 | 3,072 (32×32×3 RGB) | 10 | 60K | ~615 MB | ~2-3 minutes |
| CIFAR-100 | 3,072 (32×32×3 RGB) | 100 | 60K | ~615 MB | ~2-3 minutes |
| **CINIC-10** | 3,072 (32×32×3 RGB) | 10 | **270K** | ~3.6 GB | ~5-10 minutes |

**MNIST/QMNIST/CIFAR**: CSV files are automatically generated on first use - no manual steps required!

**CINIC-10**: Requires manual download first. See [CINIC-10 Setup Guide](docs/CINIC10_SETUP.md) for instructions.

### Setting up CINIC-10

CINIC-10 is an augmented extension of CIFAR-10 with 4.5× more data (270K images).

```bash
# Quick setup: Download + Convert to CSV
python scripts/setup_cinic10.py --convert-csv

# Then run experiments
python -m tasks.vision.run_all --dataset cinic10_csv
```

For detailed instructions, see [docs/CINIC10_SETUP.md](docs/CINIC10_SETUP.md)

---

## 🚀 GPU Optimization

The codebase automatically detects and optimizes for GPU:

### Auto-Detection Features
✅ Auto-switches between CUDA and CPU
✅ Non-blocking data transfers
✅ Pin memory for faster CPU→GPU transfer
✅ Multi-worker data loading (4 workers on GPU)
✅ Reproducible CUDA seeds

### Performance
- **GPU**: ~6.5 seconds per epoch (70-95% utilization)
- **CPU**: ~30-40 seconds per epoch
- **Expected Speedup**: 10-50× on GPU

Check GPU status:
```bash
python scripts/check_device.py
```

---

## 📈 Output Management

Results are saved to: `outputs/vision/{dataset}/{strategy}/run-{number}/`

Each run directory contains:
- `summary.txt` - Mean ± 95% CI statistics
- `test_acc.png` - Test accuracy plot with CI bands
- `train_acc.png` - Training accuracy plot
- `test_loss.png` - Test loss plot
- `train_loss.png` - Training loss plot

---

## 🔧 Adding New Components

### Add a Batch Strategy

1. Create `tasks/vision/batch_strategies/my_strategy.py`
2. Register in `config/batch_strategies.py`
3. Run: `python -m tasks.vision.run_experiment`

### Add a Dataset

1. Add spec to `config/datasets.py`
2. Add builder to `tasks/vision/datasets/builders.py`
3. Create dataset class in `tasks/vision/datasets/loaders.py`
4. Run: `python -m tasks.vision.run_experiment --dataset my_dataset`

See [docs/CLAUDE.md](docs/CLAUDE.md) for detailed instructions.

---

## 📚 Documentation

Additional documentation in `docs/`:

- **[CLAUDE.md](docs/CLAUDE.md)** - Complete developer guide
- **[MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md)** - Migration from old structure
- **[RESTRUCTURING_STATUS.md](docs/RESTRUCTURING_STATUS.md)** - Test results

---

## 🐛 Troubleshooting

### ModuleNotFoundError
```bash
source tbs/bin/activate
pip install -r requirements.txt
```

### CUDA out of memory
Reduce `BATCH_SIZE` in `config/vision.py`

### Dataset not found
```bash
python scripts/download_datasets.py download mnist --yes
```

For more help, see [docs/CLAUDE.md](docs/CLAUDE.md)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 🎓 Citation

```bibtex
@software{training_batch_selection,
  title={Training Batch Selection Research Project},
  author={Human Augment Analytics Group},
  organization={Georgia Institute of Technology},
  year={2024},
  url={https://github.com/Human-Augment-Analytics/Training-Batch-Selection}
}
```

---

## 🏆 Acknowledgments

- Georgia Institute of Technology
- Human Augment Analytics Group (HAAG)

---

**Made with ❤️ for deep learning research**
