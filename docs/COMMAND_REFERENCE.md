# Command Reference - Old vs New

## 🎯 Quick Comparison

### Old Structure (3 separate commands)
```bash
# 1. Run all strategies on one dataset
python -m trainer.pipelines.vision.vision

# 2. Compare two strategies
python trainer.pipelines.vision.comparison_vision_batch.py

# 3. Benchmark across datasets
python -m trainer.pipelines.vision.benchmark_datasets.py
```

### New Structure (Cleaner commands)
```bash
# 1. Run all strategies on one dataset
python -m tasks.vision.run_experiment

# 2. Compare strategies
python -m tasks.vision.compare

# 3. Benchmark across datasets
python -m tasks.vision.benchmark
```

---

## 📋 Complete Command Guide

### 1. Run Experiments (Main)

**Run all batch strategies on default dataset (MNIST):**
```bash
python -m tasks.vision.run_experiment
```

**Run on specific dataset:**
```bash
python -m tasks.vision.run_experiment --dataset mnist_csv
python -m tasks.vision.run_experiment --dataset qmnist_csv
python -m tasks.vision.run_experiment --dataset cifar10_csv
python -m tasks.vision.run_experiment --dataset cifar100_csv
```

**What it does:**
- Runs all 3 strategies (Random, Fixed, Smart)
- 5 runs per strategy (for statistical significance)
- 5 epochs per run
- Generates plots and summaries
- Output: `outputs/vision/{dataset}/batching_{strategy}/run-###/`

---

### 2. Compare Strategies

**Compare strategies (using pairs defined in config):**
```bash
python -m tasks.vision.compare
```

**Compare on specific dataset:**
```bash
python -m tasks.vision.compare --dataset cifar10_csv
```

**What it does:**
- Creates overlay plots comparing two strategies
- Uses pairs from `config/batch_strategies.py` → `VISION_STRATEGY_COMPARISON_PAIRS`
- Default pairs: `[("Fixed", "Smart"), ("Random", "Smart")]`
- Output: `outputs/vision/{dataset}/comparison_{stratA}_{stratB}/`

**Configure comparison pairs:**
Edit `config/batch_strategies.py`:
```python
VISION_STRATEGY_COMPARISON_PAIRS = [
    ("Fixed", "Smart"),
    ("Random", "Smart"),
    ("Random", "Fixed"),  # Add more pairs
]
```

---

### 3. Benchmark Across Datasets

**Benchmark Random strategy across multiple datasets:**
```bash
python -m tasks.vision.benchmark
```

**Benchmark specific strategy:**
```bash
python -m tasks.vision.benchmark --strategy Smart
python -m tasks.vision.benchmark --strategy Fixed
```

**Benchmark on custom dataset list:**
```bash
python -m tasks.vision.benchmark --datasets mnist_csv qmnist_csv cifar10_csv cifar100_csv
```

**Combine options:**
```bash
python -m tasks.vision.benchmark --strategy Smart --datasets mnist_csv cifar10_csv
```

**What it does:**
- Runs one strategy across multiple datasets
- Useful for comparing dataset difficulty
- Default datasets: `['mnist_csv', 'qmnist_csv', 'cifar10_csv']`
- Output: `outputs/vision/benchmarks/{strategy}_multi_dataset/`

---

### 4. Utility Commands

**Check GPU/CPU configuration:**
```bash
python scripts/check_device.py
```

**Download datasets:**
```bash
# List available datasets
python scripts/download_datasets.py list

# Download specific dataset
python scripts/download_datasets.py download mnist --yes
python scripts/download_datasets.py download cifar10 --yes
```

---

## 📊 Output Structure

```
outputs/vision/
├── {dataset}/                           # Per dataset
│   ├── batching_random/run-001/        # Run experiments output
│   │   ├── summary.txt
│   │   ├── test_acc.png
│   │   ├── train_acc.png
│   │   ├── test_loss.png
│   │   └── train_loss.png
│   ├── batching_fixed/run-001/
│   ├── batching_smart/run-001/
│   └── comparison_fixed_smart/          # Compare output
│       ├── test_acc_cmp.png
│       ├── train_acc_cmp.png
│       └── train_loss_cmp.png
└── benchmarks/                          # Benchmark output
    └── smart_multi_dataset/
        ├── summary.txt
        ├── mnist_csv_test_acc.png
        ├── cifar10_csv_test_acc.png
        └── ...
```

---

## 🎯 Common Workflows

### Workflow 1: Quick Test on MNIST
```bash
# Just run this!
python -m tasks.vision.run_experiment
```

### Workflow 2: Compare Strategies on CIFAR-10
```bash
# Step 1: Run experiments
python -m tasks.vision.run_experiment --dataset cifar10_csv

# Step 2: Generate comparison plots
python -m tasks.vision.compare --dataset cifar10_csv
```

### Workflow 3: Full Benchmark
```bash
# Benchmark Smart strategy across all datasets
python -m tasks.vision.benchmark --strategy Smart --datasets mnist_csv qmnist_csv cifar10_csv cifar100_csv
```

### Workflow 4: Test All Datasets with All Strategies
```bash
# Run on each dataset (can run in parallel in different terminals)
python -m tasks.vision.run_experiment --dataset mnist_csv
python -m tasks.vision.run_experiment --dataset qmnist_csv
python -m tasks.vision.run_experiment --dataset cifar10_csv
python -m tasks.vision.run_experiment --dataset cifar100_csv

# Then compare
python -m tasks.vision.compare --dataset mnist_csv
python -m tasks.vision.compare --dataset qmnist_csv
# ...etc
```

---

## ⚙️ Configuration

### Change Default Dataset
Edit `config/vision.py`:
```python
ACTIVE_DATASET = "cifar10_csv"  # Change from "mnist_csv"
```

### Change Training Hyperparameters
Edit `config/vision.py`:
```python
EPOCHS = 10      # More epochs
N_RUNS = 10      # More runs for better statistics
BATCH_SIZE = 128 # Larger batches
```

### Add New Comparison Pairs
Edit `config/batch_strategies.py`:
```python
VISION_STRATEGY_COMPARISON_PAIRS = [
    ("Fixed", "Smart"),
    ("Random", "Smart"),
    ("Random", "Fixed"),  # Add this
]
```

---

## 📝 Summary

| Task | Old Command | New Command |
|------|-------------|-------------|
| Run all strategies | `python -m trainer.pipelines.vision.vision` | `python -m tasks.vision.run_experiment` |
| Compare strategies | `python trainer.pipelines.vision.comparison_vision_batch.py` | `python -m tasks.vision.compare` |
| Benchmark datasets | `python -m trainer.pipelines.vision.benchmark_datasets.py` | `python -m tasks.vision.benchmark` |
| Check GPU | `python check_cuda.py` or `python check_device.py` | `python scripts/check_device.py` |
| Download data | `python simple_dataset_loader.py download mnist` | `python scripts/download_datasets.py download mnist --yes` |

**All commands now:**
- ✅ Use `-m` flag (proper module execution)
- ✅ Follow consistent naming
- ✅ Support `--help` for documentation
- ✅ Have sensible defaults
- ✅ Are well organized in `tasks/vision/`
