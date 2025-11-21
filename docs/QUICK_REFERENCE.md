# Quick Reference Guide

## Vision Experiments - Quick Commands

### 🎯 Recommended: Single Command Pipeline

```bash
# Run complete pipeline (experiments + comparisons + benchmark)
python -m tasks.vision.run_all

# Quick test mode (1 epoch, 1 run) - perfect for testing changes
python -m tasks.vision.run_all --quick

# Run on specific dataset
python -m tasks.vision.run_all --dataset cifar10_csv

# Skip benchmark if you only want experiments + comparisons
python -m tasks.vision.run_all --no-benchmark
```

**What does `run_all` do?**
1. ✅ Runs experiments with ALL batch strategies (Random, Fixed, Smart)
2. ✅ Generates comparison plots between strategy pairs
3. ✅ Benchmarks one strategy across multiple datasets
4. ✅ Saves all results to organized output folders

---

## Before vs After

### ❌ Old Way (3 separate commands)
```bash
# Step 1: Run experiments
python -m tasks.vision.run_experiment --dataset mnist_csv

# Step 2: Generate comparisons
python -m tasks.vision.compare --dataset mnist_csv

# Step 3: Benchmark
python -m tasks.vision.benchmark --strategy Random --datasets mnist_csv qmnist_csv
```

### ✅ New Way (1 command)
```bash
python -m tasks.vision.run_all
```

---

## Individual Commands (Still Available)

If you need more control, individual commands still work:

```bash
# Run experiments only
python -m tasks.vision.run_experiment --dataset mnist_csv

# Compare strategies only
python -m tasks.vision.compare --dataset mnist_csv

# Benchmark only
python -m tasks.vision.benchmark --strategy Random --datasets mnist_csv qmnist_csv
```

---

## Common Workflows

### Testing Code Changes
```bash
# Quick test with 1 epoch, 1 run
python -m tasks.vision.run_all --quick
```

### Full Production Run
```bash
# Edit config/vision.py first to set:
# EPOCHS = 10
# N_RUNS = 5

# Then run
python -m tasks.vision.run_all
```

### Experiment on New Dataset
```bash
# Run on CIFAR-10
python -m tasks.vision.run_all --dataset cifar10_csv

# Run on CIFAR-100
python -m tasks.vision.run_all --dataset cifar100_csv
```

### Skip Time-Consuming Benchmark
```bash
# Only run experiments and comparisons (faster)
python -m tasks.vision.run_all --no-benchmark
```

---

## Output Structure

After running `python -m tasks.vision.run_all`, you'll get:

```
outputs/vision/
├── mnist_csv/                          # Your primary dataset
│   ├── batching_random/
│   │   └── run-001/
│   │       ├── test_acc.png
│   │       ├── train_acc.png
│   │       ├── train_loss.png
│   │       ├── test_loss.png
│   │       └── summary.txt
│   ├── batching_fixed/
│   │   └── run-001/
│   ├── batching_smart/
│   │   └── run-001/
│   ├── comparison_random_fixed/        # Comparison plots
│   │   ├── test_acc_cmp.png
│   │   ├── train_acc_cmp.png
│   │   └── train_loss_cmp.png
│   └── comparison_random_smart/
│
└── benchmarks/                          # Benchmark results
    └── random_multi_dataset/
        ├── summary.txt
        ├── mnist_csv_test_acc.png
        ├── qmnist_csv_test_acc.png
        └── ...
```

---

## Configuration

Edit `config/vision.py` to customize:

```python
# Training parameters
EPOCHS = 5              # Number of epochs per run
N_RUNS = 5              # Number of runs for statistics
BATCH_SIZE = 64         # Batch size

# Default dataset
ACTIVE_DATASET = "mnist_csv"

# Model architecture
HIDDEN_DIM = 256
```

Edit `config/batch_strategies.py` to configure comparisons:

```python
# Which strategy pairs to compare
VISION_STRATEGY_COMPARISON_PAIRS = [
    ("Random", "Fixed"),
    ("Random", "Smart"),
]
```

---

## Tips

- **First time?** Use `--quick` to verify everything works
- **Debugging?** Individual commands give more control
- **Production?** Use `run_all` without flags for complete analysis
- **Time-constrained?** Use `--no-benchmark` to save time
- **GPU not detected?** Run `python scripts/check_device.py` to diagnose