# BERT Training with Batch Selection Strategies

This directory contains the BERT training pipeline integrated with the codebase structure for researching lightweight training batch selection strategies.

## Quick Start

### 1. Install Dependencies

Dependencies are managed in the project root `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

### 2. Configure Experiment

Edit `trainer/constants_nlp.py` to set training parameters:

```python
# ========== Training Configuration ==========
MAX_LENGTH = 128        # Sequence length for BERT
BATCH_SIZE = 8          # Batch size (adjust based on GPU memory)
EPOCHS = 3              # Number of training epochs
N_RUNS = 1              # Number of runs (set to 3+ for statistical significance)
LEARNING_RATE = 2e-5
DEVICE = 'cuda'         # 'cuda' or 'cpu'

# ========== Dataset Configuration ==========
USE_SUBSET = True           # Set to False for full IMDB dataset
TRAIN_SUBSET_SIZE = 1000    # Smaller for quick testing
TEST_SUBSET_SIZE = 500

# ========== Loss-Based Batch Selection ==========
LOSS_THRESHOLD = 0.5    # Examples with loss > threshold are kept
```

### 3. Run Training

From the project root directory:

```bash
# Run all strategies defined in constants_batch_strategy.py
python -m trainer.pipelines.nlp.bert.bert
```

Trains BERT on IMDB with all configured strategies (Random, LossBased) and saves results to `trainer/pipelines/nlp/bert/output/batching_<strategy>/run-XXX/`.

### 4. Compare Results

```bash
python -m trainer.pipelines.nlp.bert.comparison_bert_batch
```

Generates comparison plots in `output/comparison_random_lossbased/`.

## Understanding the Output

### Output Structure

```
trainer/pipelines/nlp/bert/output/
├── batching_random/
│   └── run-001/
│       ├── test_acc.png              # Test accuracy over epochs
│       ├── train_acc.png             # Train accuracy over epochs
│       ├── train_loss.png            # Training loss over epochs
│       ├── samples_per_epoch.png     # Samples processed per epoch
│       └── summary.txt               # Numerical results with CI
├── batching_lossbased/
│   └── run-001/
│       └── (same files as above)
└── comparison_random_lossbased/
    ├── test_acc_cmp.png              # Side-by-side accuracy comparison
    ├── train_acc_cmp.png
    ├── train_loss_cmp.png
    └── samples_per_epoch_cmp.png     # Shows compute savings!
```

### Reading Results

Each `summary.txt` contains per-epoch metrics with confidence intervals:
```
Epoch 1: train_acc=0.8450±0.0120, test_acc=0.8320±0.0095, train_loss=0.3421±0.0234, samples=1000
Epoch 2: train_acc=0.9010±0.0087, test_acc=0.8650±0.0078, train_loss=0.2156±0.0189, samples=650
Epoch 3: train_acc=0.9234±0.0065, test_acc=0.8789±0.0072, train_loss=0.1834±0.0145, samples=420
Training Time: 45.23±2.31 sec
```

Note: Samples decrease in epochs 2-3 for loss-based strategy, indicating compute savings.

## File Structure

```
trainer/
├── constants_nlp.py                          # NLP configuration
├── constants_batch_strategy.py               # Strategy registry
├── batching/
│   └── nlp_batching/
│       ├── random_batch.py                   # Baseline strategy
│       └── loss_based_batch.py               # Loss-based filtering
├── dataloader/
│   └── nlp_dataloader.py                     # IMDB dataset wrapper
└── pipelines/
    └── nlp/bert/
        ├── bert.py                           # Main training script
        ├── comparison_bert_batch.py          # Comparison tool
        ├── bert_trainer.py                   # Legacy standalone
        └── BERT_Training_Colab.ipynb         # Original notebook
```

## Configuration Guide

### Quick Testing (Fast)
```python
USE_SUBSET = True
TRAIN_SUBSET_SIZE = 1000
TEST_SUBSET_SIZE = 500
EPOCHS = 3
N_RUNS = 1
```

### Statistical Significance (Recommended for Research)
```python
USE_SUBSET = True
TRAIN_SUBSET_SIZE = 5000
TEST_SUBSET_SIZE = 1000
EPOCHS = 5
N_RUNS = 3  # Generates confidence intervals
```

### Full Dataset (Publication-Ready)
```python
USE_SUBSET = False  # Uses all 25,000 train + 25,000 test
EPOCHS = 5
N_RUNS = 5
BATCH_SIZE = 16  # If you have enough GPU memory
```

## Available Batch Strategies

Edit `trainer/constants_batch_strategy.py` to control which strategies run:

```python
NLP_BATCH_STRATEGIES = {
    "Random": "random_batch",        # Baseline: all examples every epoch
    "LossBased": "loss_based_batch"  # Smart: filter low-loss examples
}
```

### Strategy Comparison

| Strategy | Description | Compute | Accuracy |
|----------|-------------|---------|----------|
| **Random** | Train on all examples every epoch | 100% | Baseline |
| **LossBased** | Epoch 1: all, Epoch 2+: only high-loss | 50-70% | Similar or better |

## Adding New Batch Selection Strategies

### Step 1: Create Strategy File

Create `trainer/batching/nlp_batching/my_strategy.py`:

```python
import numpy as np

def batch_sampler(dataset, batch_size, loss_history=None, **kwargs):
    """
    Custom batch selection strategy.

    Args:
        dataset: Training dataset
        batch_size: Number of samples per batch
        loss_history: Array of per-sample losses from previous epoch
        **kwargs: Additional arguments

    Yields:
        Lists of indices for each batch
    """
    n = len(dataset)
    indices = np.arange(n)

    # Implement custom logic here
    # Examples: gradient-based selection, uncertainty sampling,
    # curriculum learning, active learning

    np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size].tolist()
```

### Step 2: Register Strategy

Edit `trainer/constants_batch_strategy.py`:

```python
NLP_BATCH_STRATEGIES = {
    "Random": "random_batch",
    "LossBased": "loss_based_batch",
    "MyStrategy": "my_strategy"
}

COMPARE_NLP_BATCH_STRATEGY_PAIRS = [
    ("Random", "LossBased"),
    ("Random", "MyStrategy"),
    ("LossBased", "MyStrategy")
]
```

### Step 3: Run and Analyze

```bash
python -m trainer.pipelines.nlp.bert.bert
python -m trainer.pipelines.nlp.bert.comparison_bert_batch
```

Results saved in `output/batching_mystrategy/run-001/`.

## Example Strategy Ideas

### 1. Confidence-Based Sampling
```python
# Only train on examples where model is uncertain (probability close to 0.5)
def batch_sampler(dataset, batch_size, confidence_history=None, **kwargs):
    if confidence_history is not None:
        uncertainty = 1 - np.abs(confidence_history - 0.5) * 2
        top_uncertain = np.argsort(-uncertainty)[:int(len(dataset) * 0.7)]
        indices = np.random.choice(top_uncertain,
                                   min(len(top_uncertain), len(dataset)),
                                   replace=False)
    else:
        indices = np.arange(len(dataset))
    # ... yield batches
```

### 2. Curriculum Learning
```python
# Start with easy examples, gradually add harder ones
def batch_sampler(dataset, batch_size, loss_history=None, epoch=0, **kwargs):
    if loss_history is not None:
        percentile = min(100, 50 + epoch * 10)  # 50%, 60%, 70%...
        threshold = np.percentile(loss_history, percentile)
        indices = np.where(loss_history <= threshold)[0]
    # ... yield batches
```

### 3. Diversity Sampling
```python
# Select batches that maximize diversity (e.g., using embeddings)
# This could reduce redundancy in training
```

## Loss-Based Strategy Implementation

The loss-based strategy filters examples based on training loss:

**Mechanism:**
- Epoch 1: Train on all examples, record per-sample loss
- Epoch 2+: Only train on examples with loss > threshold

**Rationale:**
- Low-loss examples are already learned
- Focus compute on hard/misclassified examples
- Typically reduces training time 40-60% with minimal accuracy impact

**Configuration:**
Adjust `LOSS_THRESHOLD` in `trainer/constants_nlp.py`:
- 0.3: Aggressive filtering (fewer examples, faster)
- 0.5: Moderate filtering (default)
- 0.7: Conservative filtering (more examples, safer)

## Evaluation Metrics

### Efficiency
- Training time with confidence intervals (from `summary.txt`)
- Samples processed per epoch (from `samples_per_epoch.png`)
- Total compute savings (compare cumulative samples across strategies)

### Accuracy
- Final test accuracy comparison
- Convergence rate across epochs
- Statistical significance via confidence intervals (requires N_RUNS >= 3)

### Trade-offs
- Accuracy vs efficiency curves
- Strategy comparison plots
- Per-epoch sample reduction

## Experimental Guidelines

1. Start with `TRAIN_SUBSET_SIZE=1000` for rapid iteration
2. Use `N_RUNS=3` minimum for confidence intervals
3. Increase to 5000 samples once strategy is validated
4. Use full dataset (`USE_SUBSET=False`) for final results
5. Adjust `BATCH_SIZE` or `MAX_LENGTH` if GPU memory issues occur

## Troubleshooting

### CUDA out of memory
Reduce `BATCH_SIZE` or `MAX_LENGTH` in `constants_nlp.py`:
```python
BATCH_SIZE = 4
MAX_LENGTH = 64
```

### Missing dependencies
Install from project root:
```bash
pip install -r requirements.txt
```

### Strategy not found
- Verify `NLP_BATCH_STRATEGIES` in `constants_batch_strategy.py`
- Check file exists in `trainer/batching/nlp_batching/`
- Ensure function is named `batch_sampler`

### Comparison fails
- Run `bert.py` first to generate results
- Verify `output/batching_<strategy>/run-001/` exists
