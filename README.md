# Training Batch Selection Research Project

A research project for evaluating and comparing different **batch selection strategies** during neural network training. Instead of using standard random batching, this project experiments with different ways to select which samples go into each training batch (e.g., prioritizing high-loss samples).

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Vision Pipeline (MNIST)](#vision-pipeline-mnist)
- [Dataset Management](#dataset-management)
- [Configuration System](#configuration-system)
- [Adding Custom Components](#adding-custom-components)
- [NLP Pipeline (Pretraining)](#nlp-pipeline-pretraining)
- [Requirements](#requirements)

---

## Overview

The codebase supports two main pipelines:

1. **Vision Pipeline** (MNIST) - Complete implementation with multiple batch strategies
2. **Text/NLP Pipeline** (Pretraining) - Under development

### Key Features

- **Plugin-based Batch Strategy System** - Add new strategies without touching core code
- **Config-driven Design** - Change hyperparameters via config files
- **Automated Experiment Tracking** - All metrics and plots saved automatically
- **Dataset Factory Pattern** - Unified dataset loading interface
- **Multi-dataset Benchmarking** - Compare performance across different datasets

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Human-Augment-Analytics/Training-Batch-Selection.git

cd Training-Batch-Selection

# Install dependencies
pip install -r requirements.txt
```

### Download Datasets

```bash
# List available datasets
python simple_dataset_loader.py list

# Download specific dataset
python simple_dataset_loader.py download mnist --yes
```

### Run Vision Experiments

```bash
# Train with all batch strategies (Random, Fixed, Smart)
python -m trainer.pipelines.vision.vision

# Compare two strategies
python -m trainer.pipelines.vision.comparison_vision_batch

# Benchmark across multiple datasets
python -m trainer.pipelines.vision.benchmark_datasets
```

**Important Notes:**
- Always use `-m` flag when running modules (treats code as package)
- Omit `.py` extension (use `vision` not `vision.py`)
- MNIST CSVs are **automatically generated** when needed - no manual conversion required!

### Check Results

Results are saved to `trainer/pipelines/vision/output/`:
- Each strategy gets its own folder: `batching_{strategy}/run-XXX/`
- Contains: `summary.txt`, accuracy/loss plots

---

## Project Structure

```
Training-Batch-Selection/
├── simple_dataset_loader.py       # Dataset downloader (config-driven)
├── download_engines.py             # Generic download engines
├── dataset_config_enhanced.yaml    # Dataset configurations
├── trainer/
│   ├── batching/
│   │   └── vision_batching/        # Batch selection strategies
│   │       ├── random_batch.py     # Baseline: random sampling
│   │       ├── fixed_batch.py      # Sequential batching
│   │       └── smart_batch.py      # Loss-based prioritization
│   ├── dataloader/
│   │   ├── vision_dataloader.py    # Vision dataset classes
│   │   ├── factory.py              # Dataset factory functions
│   │   └── builders.py             # Dataset builders
│   ├── model/
│   │   ├── vision/model.py         # SimpleMLP for MNIST
│   │   └── nlp/                    # Transformer architecture
│   ├── pipelines/
│   │   ├── vision/
│   │   │   ├── vision.py           # Main training loop
│   │   │   ├── comparison_vision_batch.py
│   │   │   └── benchmark_datasets.py
│   │   └── pretraining/
│   │       └── run_pretraining.py
│   ├── constants.py                # Hyperparameters
│   ├── constants_batch_strategy.py # Batch strategy registry
│   └── constants_datasets.py       # Dataset specifications
└── datasets/                       # Downloaded datasets
    ├── vision/
    └── nlp/
```

---

## Vision Pipeline (MNIST)

### What It Does

- **Loads MNIST data** from CSVs or torchvision
- **Trains a SimpleMLP model** using different batch strategies:
  - **Random Batching** - Standard random sampling
  - **Fixed Batching** - Sequential batching
  - **Smart Batching** - Prioritizes high-loss samples
- **Logs metrics** - Training/test accuracy and loss for each epoch
- **Saves results** automatically to organized output directories
- **Computes statistics** - Mean ± 95% confidence intervals across multiple runs

### Running Experiments

#### Train All Strategies

```bash
python -m trainer.pipelines.vision.vision
```

This will:
- Train SimpleMLP on MNIST for each strategy in `constants_batch_strategy.py`
- Run `N_RUNS` independent trials (default: 5) with different random seeds
- Compute mean ± 95% CI for accuracy/loss metrics
- Save plots and summary to `trainer/pipelines/vision/output/batching_{strategy}/run-XXX/`

Each run directory contains:
- `summary.txt` - Mean and confidence intervals
- `test_acc.png`, `train_acc.png` - Accuracy curves
- `test_loss.png`, `train_loss.png` - Loss curves

#### Compare Two Strategies

```bash
python trainer.pipelines.vision.comparison_vision_batch.py
```

- Edit `COMPARE_BATCH_STRATEGY_PAIRS` in `trainer/constants_batch_strategy.py`
- Generates overlay plots comparing two strategies
- Saves to `trainer/pipelines/vision/output/comparison_{stratA}_{stratB}/`

#### Benchmark Across Datasets

```bash
python -m trainer.pipelines.vision.benchmark_datasets
```

- Edit `DATASETS` list at top of `benchmark_datasets.py`
- Currently supports: `mnist_csv`, `mnist`, `qmnist`, `cifar10_flat`
- Generates per-dataset plots and summaries

### Output Example

```
trainer/pipelines/vision/output/
├── batching_Random/
│   └── run-001/
│       ├── summary.txt
│       ├── test_acc.png
│       ├── train_acc.png
│       ├── test_loss.png
│       └── train_loss.png
├── batching_Fixed/
│   └── run-001/...
├── batching_Smart/
│   └── run-001/...
└── comparison_Fixed_Smart/
    ├── train_acc_cmp.png
    ├── test_acc_cmp.png
    ├── train_loss_cmp.png
    └── test_loss_cmp.png
```

---

## Dataset Management

### Config-Driven Dataset Downloader

The project includes a fully config-driven dataset downloader. Add new datasets by editing YAML only - no Python code needed!

### Available Commands

**List all datasets:**
```bash
python simple_dataset_loader.py list
```

**Download specific dataset:**
```bash
python simple_dataset_loader.py download cifar10
python simple_dataset_loader.py download mnist --yes  # skip confirmation
```

**Download with auto-confirmation:**
```bash
python simple_dataset_loader.py download mnist --yes
```

### Available Datasets

#### Vision Datasets
- **CIFAR-10** (577 MB) - 60k 32×32 color images in 10 classes
- **CIFAR-100** (576 MB) - 60k 32×32 color images in 100 classes
- **MNIST** (340 MB) - 70k 28×28 grayscale handwritten digits
- **QMNIST** (586 MB) - Extended MNIST with additional test sets
- **SVHN** (4 GB) - Street View House Numbers dataset
- **CINIC-10** (1.2 GB) - CIFAR-10 extended with ImageNet images
- **Tiny ImageNet** (481 MB) - 200 classes, 500 images each
- **VOC 2012** (3.7 GB) - Object detection and segmentation

#### NLP Datasets
- **CoLA** (5 MB) - Corpus of Linguistic Acceptability
- **SST-2** (10 MB) - Stanford Sentiment Treebank
- **E2E NLG** (15 MB) - End-to-End NLG Challenge dataset

### Dataset Storage Location

Default location: `./datasets/`

**Change storage location:**

1. **Symbolic link (recommended for shared storage):**
   ```bash
   ln -s /path/to/shared/datasets ./datasets
   ```

2. **Edit code:**
   Modify `DATASETS_ROOT` in `simple_dataset_loader.py` line 32

3. **Environment variable:**
   Set `DATASETS_ROOT=/path/to/datasets` and modify code to read from `os.environ`

**For college cloud setup:**
```bash
cd /home/hice1/pawate3/scratch/Training-Batch-Selection
ln -s /storage/ice-shared/cs8903onl/lw-batch-selection/datasets ./datasets
```

### Directory Structure After Download

```
datasets/
├── vision/
│   ├── cifar10/
│   ├── cifar100/
│   ├── mnist/
│   ├── qmnist/
│   └── ...
└── nlp/
    ├── cola/
    ├── sst2/
    └── e2e_nlg/
```

---

## Configuration System

All configuration is centralized in Python constants files. Change settings without touching core logic!

### 1. Training Hyperparameters (`trainer/constants.py`)

```python
# Model architecture
INPUT_DIM = 784        # 28×28 flattened
HIDDEN_DIM = 128
NUM_CLASSES = 10

# Training
EPOCHS = 5
BATCH_SIZE = 64
N_RUNS = 5             # Number of independent trials
DEVICE = 'cpu'         # or 'cuda'

# Smart batch hyperparameters
MOVING_AVG_DECAY = 0.9
EXPLORE_FRAC = 0.5     # 50% random samples
TOP_K_FRAC = 0.2       # Top 20% high-loss samples

# Reproducibility
RANDOM_SEED = 2024
```

### 2. Batch Strategy Registry (`trainer/constants_batch_strategy.py`)

```python
# Key: Display name for plots/dirs
# Value: Module name (relative to trainer.batching.vision_batching)
BATCH_STRATEGIES = {
    "Random": "random_batch",
    "Fixed": "fixed_batch",
    "Smart": "smart_batch"
}

# Pairs to compare
COMPARE_BATCH_STRATEGY_PAIRS = [
    ("Fixed", "Smart"),
    ("Random", "Smart"),
]
```

### 3. Dataset Specifications (`trainer/constants_datasets.py`)

```python
DATASET_SPECS = {
    "mnist": {
        "builder": "build_mnist",
        "input_dim": 28 * 28,
        "num_classes": 10,
        "subdir": "vision/MNIST",
    },
    "cifar10_flat": {
        "builder": "build_cifar10_flat",
        "input_dim": 3 * 32 * 32,
        "num_classes": 10,
        "subdir": "vision/cifar10",
    },
    # Add more datasets here...
}
```

### 4. Dataset Download Configuration (`dataset_config_enhanced.yaml`)

```yaml
vision_datasets:
  mnist:
    download:
      method: "torchvision"
      class_name: "MNIST"
      splits:
        train:
          param_name: "train"
          param_value: true
    # Add more configuration...
```

---

## Adding Custom Components

### Adding a New Batch Strategy

**1. Create strategy module** in `trainer/batching/vision_batching/my_strategy.py`:

```python
import numpy as np

def batch_sampler(dataset, batch_size, **kwargs):
    """
    Yield batches of indices from the dataset.

    Args:
        dataset: PyTorch Dataset object
        batch_size: Number of samples per batch
        **kwargs: Optional args (e.g., loss_history for smart batching)

    Yields:
        np.ndarray: Batch indices
    """
    n = len(dataset)
    n_batches = n // batch_size

    for _ in range(n_batches):
        # Your custom logic here
        batch_idxs = np.random.choice(n, batch_size, replace=False)
        yield batch_idxs
```

**2. Register strategy** in `trainer/constants_batch_strategy.py`:

```python
BATCH_STRATEGIES = {
    "Random": "random_batch",
    "Fixed": "fixed_batch",
    "Smart": "smart_batch",
    "MyStrategy": "my_strategy",  # Add this line
}
```

**3. Run experiments:**
```bash
python -m trainer.pipelines.vision.vision
```

Your strategy will automatically be evaluated alongside others!

### Adding a New Dataset

**Using the Dataset Factory Pattern (Recommended):**

**1. Add specifications** to `trainer/constants_datasets.py`:

```python
DATASET_SPECS = {
    "my_dataset": {
        "builder": "build_my_dataset",
        "input_dim": 28 * 28,
        "num_classes": 10,
        "subdir": "vision/my_dataset",
    },
}
```

**2. Add builder function** to `trainer/dataloader/builders.py`:

```python
def build_my_dataset(root, *, as_flat=True, normalize=True, download=False, **kwargs):
    train = MyDataset(root, train=True, flatten=as_flat, normalize=normalize, download=download)
    test = MyDataset(root, train=False, flatten=as_flat, normalize=normalize, download=download)
    return train, test
```

**3. Create dataset class** in `trainer/dataloader/vision_dataloader.py`:

```python
from trainer.dataloader.base_dataloader import BaseDataset

class MyDataset(BaseDataset):
    def __len__(self):
        return ...

    def __getitem__(self, idx):
        return x, y  # (tensor, label)
```

**4. Add to dataset downloader** (optional) in `dataset_config_enhanced.yaml`:

```yaml
vision_datasets:
  my_dataset:
    download:
      method: "torchvision"
      class_name: "MyDataset"
```

**5. Use it immediately:**

```python
from trainer.dataloader.factory import build_dataset, build_model_for
from trainer.constants import SHARED_DATA_DIR

train_ds, test_ds = build_dataset(SHARED_DATA_DIR, "my_dataset")
model = build_model_for("my_dataset", train_ds, SimpleMLP)
```

### Adding a New Dataset to Downloader

Edit `dataset_config_enhanced.yaml` only - no Python code needed!

```yaml
vision_datasets:
  fashion_mnist:
    download:
      method: "torchvision"
      class_name: "FashionMNIST"
      splits:
        train:
          param_name: "train"
          param_value: true
        test:
          param_name: "train"
          param_value: false
    metadata:
      description: "Fashion-MNIST: 70k 28x28 grayscale fashion images"
      size_mb: 340
      num_samples: 70000
      url: "https://github.com/zalandoresearch/fashion-mnist"
```

Then run:
```bash
python simple_dataset_loader.py download fashion_mnist
```

---

## NLP Pipeline (Pretraining)

The NLP pipeline is currently under development for transformer-based language model pretraining.

### Running Pretraining

```bash
python trainer/pipelines/pretraining/run_pretraining.py
```

### Configuration

Edit configuration sections at the top of `run_pretraining.py`:

- `MODEL_CFG` - Model architecture (depth, width, vocab size, seq length)
- `TRAIN_CFG` - Training hyperparameters (epochs, lr, warmup, grad clip)
- `DATA_CFG` - Tokenized data location and batch settings
- `OUTPUT_DIR` - Where to save checkpoints and metrics

### Architecture

- Model: TinyLLM transformer in `trainer/model/nlp/`
- Data: Tokenized text expected in `trainer/data/pretraining/tokenized/`

---

## Requirements

### Required Dependencies

```bash
pip install torch torchvision Pillow
```

### Optional Dependencies (for NLP)

```bash
pip install datasets transformers
```

### Full Requirements

```
torch
torchvision
datasets
transformers
matplotlib
python-dotenv
pyyaml
Pillow
```

Note: scipy, numpy, and pandas are included with the above packages.

Install all at once:
```bash
pip install -r requirements.txt
```

### CUDA Support

Check CUDA availability:
```bash
python check_cuda.py
```

Set device in `trainer/constants.py`:
```python
DEVICE = 'cuda'  # or 'cpu'
```

---

## Key Implementation Details

### Batch Strategy Interface

All batch strategies must implement:

```python
def batch_sampler(dataset, batch_size, **kwargs) -> Iterator[np.ndarray]:
    """Yield batches of sample indices."""
```

The training loop automatically:
- Dynamically imports strategy modules
- Detects if `loss_history` parameter is needed
- Tracks per-sample losses with exponential moving average
- Passes loss history to samplers that need it

### Smart Batching Pattern

Smart batching uses per-sample loss history:

```python
def batch_sampler(dataset, batch_size, loss_history=None, **kwargs):
    # loss_history is a np.array of shape (len(dataset),)
    # Updated automatically by the training loop
    if loss_history is None:
        loss_history = np.zeros(len(dataset))

    # Use loss_history to prioritize high-loss samples
    # ...
```

### Metrics Aggregation

For each strategy:
- Run `N_RUNS` independent trials with different random seeds
- Collect train_acc, test_acc, train_loss, test_loss for each epoch
- Compute mean and 95% confidence interval using scipy.stats
- Save aggregated metrics to `summary.txt`

### Random Seeds

For reproducibility:
- Training uses `RANDOM_SEED` from `constants.py` as base
- Each run gets sequential seed: `RANDOM_SEED + run_number`
- Seeds affect both NumPy and PyTorch RNG

---

## Automatic MNIST CSV Conversion

MNIST CSV files are **automatically generated** when needed - no manual conversion required!

### How It Works

1. **Checks** if CSV files already exist
2. If **not found**: Automatically converts from torchvision MNIST format
3. **Saves** CSVs to both required locations:
   - `datasets/vision/MNIST/csv/` (for benchmark_datasets)
   - `trainer/data/vision/` (for vision.py)
4. **Loads** the data seamlessly

### Example Output

```bash
# First time use (no CSVs exist)
python -m trainer.pipelines.vision.vision

# Output:
# ⚠️  CSV not found: trainer/data/vision/mnist_train.csv
# 🔄 Auto-converting MNIST to CSV format...
# ✔️ Loaded MNIST: 60000 train, 10000 test
# ✔️ Saved CSVs to: datasets/vision/MNIST/csv
# ✔️ Saved CSVs to: trainer/data/vision
# 🎉 CSV conversion complete!
```

### Benefits

- **Zero Configuration**: No manual setup required
- **Intelligent**: Only converts when necessary
- **Fast**: Skips conversion if CSVs exist
- **Transparent**: Shows what it's doing

---

## Troubleshooting

### Command Syntax Errors

**Wrong:**
```bash
python trainer.pipelines.vision.comparison_vision_batch.py
# Error: ModuleNotFoundError: No module named 'trainer'
```

**Correct:**
```bash
python -m trainer.pipelines.vision.comparison_vision_batch
# Use -m flag and omit .py extension
```

**Rule**: When running Python modules as scripts, use `-m` flag + module path (dots) + no `.py` extension

### Import Errors

If you see `ImportError: No module named 'torchvision'`:
```bash
pip install torch torchvision Pillow
```

For NLP datasets:
```bash
pip install datasets transformers
```

### Dataset Download Failures

- Check your internet connection
- Try running with `--yes` to auto-confirm
- Some datasets have mirror URLs tried automatically
- Check if you're behind a firewall or proxy

### Comparison Script Errors

If comparison fails with "No completed runs found":
- Ensure you've run `python -m trainer.pipelines.vision.vision` first
- Check that `summary.txt` exists in run directories
- Verify runs completed successfully (not interrupted)

### MNIST Path Errors

If you see "Dataset not found" for MNIST:
- The auto-converter will download and convert automatically
- Ensure you have ~340MB free disk space
- If using shared storage, verify symbolic link works: `ls -la datasets/`

### Permission Denied on Shared Storage

```bash
# Check permissions
ls -la /storage/ice-shared/cs8903onl/lw-batch-selection/datasets/

# Verify you can read files
cat /storage/ice-shared/cs8903onl/lw-batch-selection/datasets/vision/cifar10/...
```

### Symbolic Link Not Working

```bash
# Remove the symbolic link
rm datasets

# Recreate it
ln -s /path/to/shared/datasets ./datasets

# Verify
ls -la datasets/
readlink -f datasets  # Shows actual path
```

---

## Common Commands Reference

### Dataset Management

```bash
# List all datasets with availability status
python simple_dataset_loader.py list

# Download specific datasets
python simple_dataset_loader.py download mnist --yes
python simple_dataset_loader.py download cifar10 --yes
python simple_dataset_loader.py download qmnist --yes
python simple_dataset_loader.py download cifar100 --yes

# Download NLP datasets
python simple_dataset_loader.py download cola --yes
python simple_dataset_loader.py download sst2 --yes
python simple_dataset_loader.py download e2e_nlg --yes
```

### Training & Experiments

```bash
# Run all batch strategies on MNIST (default: 5 runs per strategy)
python -m trainer.pipelines.vision.vision

# Compare two strategies (edit COMPARE_BATCH_STRATEGY_PAIRS first)
python -m trainer.pipelines.vision.comparison_vision_batch

# Benchmark across multiple datasets (edit DATASETS list first)
python -m trainer.pipelines.vision.benchmark_datasets
```

### Testing & Verification

```bash
# Check configuration
python -c "from trainer.constants import DATASETS_ROOT; print(f'Dataset root: {DATASETS_ROOT}')"

# Test dataset loading
python -c "from trainer.dataloader.factory import build_dataset; from trainer.constants import DATASETS_ROOT; train, test = build_dataset(DATASETS_ROOT, 'mnist'); print(f'✔️ MNIST: {len(train)} train, {len(test)} test')"

# Check batch strategies
python -c "from trainer.constants_batch_strategy import BATCH_STRATEGIES; print('Strategies:', list(BATCH_STRATEGIES.keys()))"

# Test model building
python -c "from trainer.dataloader.factory import build_dataset, build_model_for; from trainer.model.vision.model import SimpleMLP; from trainer.constants import DATASETS_ROOT; train, _ = build_dataset(DATASETS_ROOT, 'mnist'); model = build_model_for('mnist', train, SimpleMLP); print(f'✔️ Model: {model.__class__.__name__}')"

# Check MNIST CSVs exist
ls -lh datasets/vision/MNIST/csv/ trainer/data/vision/mnist_*.csv
```

### Output Management

```bash
# View latest run for a strategy
ls -lrt trainer/pipelines/vision/output/batching_random/run-*/

# View comparison plots
ls trainer/pipelines/vision/output/comparison_*/

# Check summary statistics
cat trainer/pipelines/vision/output/batching_random/run-001/summary.txt
```

---

## License

This project is part of the HAAG Batch Selection Research Project. Individual datasets have their own licenses - please refer to the original dataset sources for licensing information.

---

## Project Status

### Fully Implemented
- ✅ Vision pipeline with multiple batch strategies
- ✅ Config-driven dataset downloader (100%)
- ✅ Dataset factory pattern for easy dataset loading
- ✅ Multi-dataset benchmarking
- ✅ Automated experiment tracking and plotting

### Partially Config-Driven (~60%)
- ⚠️ Trainer hyperparameters (100% config-driven)
- ⚠️ Batch strategy registry (100% config-driven)
- ⚠️ Batch strategy logic (requires Python functions)
- ⚠️ Dataset builders (requires Python functions)

### Under Development
- 🚧 NLP pretraining pipeline
- 🚧 Additional batch selection strategies
- 🚧 More dataset integrations

---

## Support

For issues or questions:
1. Check this README first
2. Run `python simple_dataset_loader.py` to see available commands
3. Check the code comments in relevant files
4. Verify file permissions and paths

---

## Contributors

Human Augment Analytics Group (HAAG) - Georgia Tech
