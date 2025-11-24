# CINIC-10 Dataset Setup Guide

## What is CINIC-10?

CINIC-10 (CINIC-10 Is Not ImageNet or CIFAR-10) is an augmented extension of CIFAR-10 that includes:
- **270,000 images** total (vs CIFAR-10's 60,000)
- Same 10 classes as CIFAR-10: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Same resolution: 32×32 RGB images
- Images from CIFAR-10 + downsampled ImageNet images
- Split: 90,000 train + 90,000 validation + 90,000 test

**Paper**: Darlow et al., "CINIC-10 is Not ImageNet or CIFAR-10" (2018)
**Source**: https://github.com/BayesWatch/cinic-10

---

## Quick Setup (Automated)

### Option 1: Download + Convert to CSV in One Command

```bash
# Download and convert CINIC-10 to CSV format
python scripts/setup_cinic10.py --convert-csv
```

This will:
1. Download ~1.7 GB compressed archive
2. Extract to `datasets/vision/cinic-10/`
3. Convert to CSV format (~2 GB CSV files)
4. Ready to use with experiments

### Option 2: Download First, Convert Later

```bash
# Step 1: Download only
python scripts/setup_cinic10.py

# Step 2: Convert to CSV when ready
python scripts/setup_cinic10.py --csv-only
```

---

## Manual Setup

### Step 1: Download

```bash
# Download CINIC-10 (1.7 GB)
cd datasets/vision/
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz

# Extract
tar -xzf CINIC-10.tar.gz
rm CINIC-10.tar.gz
```

### Step 2: Verify Structure

After extraction, you should have:

```
datasets/vision/cinic-10/
├── train/
│   ├── airplane/
│   ├── automobile/
│   ├── bird/
│   ├── cat/
│   ├── deer/
│   ├── dog/
│   ├── frog/
│   ├── horse/
│   ├── ship/
│   └── truck/
├── valid/
│   └── (same structure)
└── test/
    └── (same structure)
```

### Step 3: Convert to CSV

```bash
# From project root
python -m tasks.vision.datasets.auto_convert_csv cinic10 datasets
```

This creates:
- `datasets/vision/cinic-10/csv/cinic10_train.csv` (~1.8 GB)
- `datasets/vision/cinic-10/csv/cinic10_test.csv` (~1.8 GB)

**Note**: CSV conversion takes 5-10 minutes and processes 270,000 images.

---

## Running Experiments

### Single Dataset

```bash
# Run all batch strategies on CINIC-10
python -m tasks.vision.run_all --dataset cinic10_csv

# Quick test (1 epoch, 1 run)
python -m tasks.vision.run_all --dataset cinic10_csv --quick
```

### Multiple Datasets

```bash
# Run on CINIC-10 + CIFAR-10
python -m tasks.vision.run_all --datasets cinic10_csv cifar10_csv

# Run on all vision datasets
python -m tasks.vision.run_all --all-datasets
```

---

## Storage Requirements

| Component | Size | Location |
|-----------|------|----------|
| Download (compressed) | 1.7 GB | Temporary |
| Extracted images | 2.5 GB | `datasets/vision/cinic-10/` |
| CSV files | ~3.6 GB | `datasets/vision/cinic-10/csv/` |
| **Total** | **~6.1 GB** | |

**Recommendation**: If storage is limited, you can delete the image directories after CSV conversion:

```bash
# Keep only CSVs (saves ~2.5 GB)
rm -rf datasets/vision/cinic-10/train
rm -rf datasets/vision/cinic-10/valid
rm -rf datasets/vision/cinic-10/test
```

---

## Troubleshooting

### Error: "CINIC-10 dataset not found"

**Cause**: Raw CINIC-10 images not downloaded

**Fix**:
```bash
python scripts/setup_cinic10.py
```

### Error: "CINIC-10 CSV not found"

**Cause**: CSV files not generated

**Fix**:
```bash
python scripts/setup_cinic10.py --csv-only
```

### Error: "Neither wget nor curl found"

**Fix on Ubuntu/Debian**:
```bash
sudo apt-get install wget
```

**Fix on macOS**:
```bash
brew install wget
```

**Manual download**:
1. Download from: https://datashare.ed.ac.uk/handle/10283/3192
2. Extract to: `datasets/vision/cinic-10/`

### Out of Memory During CSV Conversion

**Cause**: Processing 270k images requires significant RAM

**Fix**: The conversion script processes images in batches by class (10 classes). If still running out of memory, process manually:

```python
from tasks.vision.datasets.auto_convert_csv import ensure_cinic10_csv

# Reduce memory usage by processing fewer classes at once
ensure_cinic10_csv("datasets")
```

---

## CINIC-10 vs CIFAR-10

| Dataset | Train | Valid | Test | Total | Source |
|---------|-------|-------|------|-------|--------|
| CIFAR-10 | 50,000 | - | 10,000 | 60,000 | Original CIFAR |
| CINIC-10 | 90,000 | 90,000 | 90,000 | 270,000 | CIFAR + ImageNet |

**Advantages of CINIC-10**:
- 4.5× more training data
- More diverse images (includes ImageNet)
- Better for testing generalization
- Same input dimensions as CIFAR-10

**Considerations**:
- Larger storage requirements
- Longer training times
- May be easier than CIFAR-10 due to more data

---

## Using CINIC-10 Validation Set

By default, experiments use `train/` for training and `test/` for evaluation. To use the validation set:

```python
from tasks.vision.datasets.factory import build_dataset

# Load with validation set
train_ds, test_ds = build_dataset("datasets", "cinic10_csv")

# To use validation instead of test, modify builders.py:
# train = train + valid  # Combine for training
# test = test             # Keep test for final evaluation
```

---

## Citation

If you use CINIC-10 in your research, please cite:

```bibtex
@article{darlow2018cinic,
  title={CINIC-10 is not ImageNet or CIFAR-10},
  author={Darlow, Luke N and Crowley, Elliot J and Antoniou, Antreas and Storkey, Amos J},
  journal={arXiv preprint arXiv:1810.03505},
  year={2018}
}
```

---

## Additional Resources

- **Paper**: https://arxiv.org/abs/1810.03505
- **GitHub**: https://github.com/BayesWatch/cinic-10
- **Dataset**: https://datashare.ed.ac.uk/handle/10283/3192
- **Discussion**: https://github.com/BayesWatch/cinic-10/issues

---

## Support

For dataset-specific issues, see: https://github.com/BayesWatch/cinic-10/issues

For project-specific issues, see: [Project README](../README.md)
