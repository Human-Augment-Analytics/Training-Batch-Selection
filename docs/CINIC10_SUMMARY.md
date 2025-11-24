# CINIC-10 Support - Implementation Summary

## Problem

You encountered a `FileNotFoundError` when trying to run experiments with CINIC-10:

```
FileNotFoundError: CINIC-10 CSV not found: /storage/ice1/6/1/pawate3/Training-Batch-Selection/datasets/vision/cinic-10/csv/cinic10_train.csv
```

## Root Cause

CINIC-10 is different from other datasets (MNIST, CIFAR-10/100) because:
1. **Not in torchvision**: Can't auto-download like MNIST/CIFAR
2. **Must download manually**: Requires downloading from Edinburgh DataShare
3. **Large dataset**: 270K images (vs CIFAR-10's 60K)
4. **Structured differently**: Image files in class folders, not binary batches

## Solution Implemented

### 1. Created CINIC-10 Auto-Conversion Function

**File**: [tasks/vision/datasets/auto_convert_csv.py](../tasks/vision/datasets/auto_convert_csv.py)

Added `ensure_cinic10_csv()` function that:
- Checks if CSV files already exist
- If not, looks for raw CINIC-10 image data
- Converts PNG images to CSV format
- Provides clear error message with download instructions if data not found

### 2. Updated Dataset Builder

**File**: [tasks/vision/datasets/builders.py](../tasks/vision/datasets/builders.py)

Modified `build_cinic10_csv()` to:
- Auto-detect if CSV files are missing
- Automatically call `ensure_cinic10_csv()` for conversion
- Seamless integration with existing pipeline

### 3. Created Automated Setup Script

**File**: [scripts/setup_cinic10.py](../scripts/setup_cinic10.py)

New script that:
- Downloads CINIC-10 from Edinburgh DataShare (1.7 GB compressed)
- Extracts to proper directory structure
- Optionally converts to CSV format
- Verifies dataset integrity
- Provides clear progress messages

Usage:
```bash
# Download + Convert in one command
python scripts/setup_cinic10.py --convert-csv

# Download only
python scripts/setup_cinic10.py

# Convert existing data
python scripts/setup_cinic10.py --csv-only
```

### 4. Created Comprehensive Documentation

**File**: [docs/CINIC10_SETUP.md](CINIC10_SETUP.md)

Complete guide covering:
- What is CINIC-10 and why use it
- Automated setup instructions
- Manual setup instructions
- Storage requirements (~6 GB total)
- Troubleshooting common issues
- CINIC-10 vs CIFAR-10 comparison
- How to run experiments
- Citation information

### 5. Updated Main README

**File**: [README.md](../README.md)

Added:
- CINIC-10 to datasets table
- Quick setup instructions
- Link to detailed CINIC10_SETUP.md guide

## Files Changed

1. ✅ `tasks/vision/datasets/auto_convert_csv.py` - Added `ensure_cinic10_csv()` function
2. ✅ `tasks/vision/datasets/builders.py` - Updated `build_cinic10_csv()` with auto-conversion
3. ✅ `scripts/setup_cinic10.py` - NEW automated download/setup script
4. ✅ `docs/CINIC10_SETUP.md` - NEW comprehensive setup guide
5. ✅ `README.md` - Updated with CINIC-10 information

## How to Use Now

### On PACE (or any system)

```bash
# Step 1: Download and convert CINIC-10
python scripts/setup_cinic10.py --convert-csv

# Step 2: Run experiments
python -m tasks.vision.run_all --dataset cinic10_csv --quick

# Or run on multiple datasets
python -m tasks.vision.run_all --datasets cifar10_csv cinic10_csv
```

### What Happens Behind the Scenes

1. **First time running with CINIC-10**:
   - Script checks for CSV files
   - If not found, checks for raw image data
   - If images exist, converts to CSV (~5-10 minutes)
   - If images don't exist, shows download instructions

2. **Subsequent runs**:
   - CSV files found, loads instantly
   - No re-conversion needed

## Storage Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Download (compressed) | 1.7 GB | Can delete after extraction |
| Extracted images | 2.5 GB | Can delete after CSV conversion |
| CSV files | 3.6 GB | Required for training |
| **Total (with images)** | **6.1 GB** | |
| **Total (CSV only)** | **3.6 GB** | After cleaning up images |

## Benefits of This Implementation

1. **Seamless Integration**: Works exactly like other datasets once set up
2. **Auto-Conversion**: Converts to CSV automatically if raw data exists
3. **Clear Error Messages**: Tells you exactly what to do if data is missing
4. **Automated Script**: One command to download and set up
5. **Comprehensive Docs**: Step-by-step guide for any situation
6. **Storage Efficient**: Optional cleanup to save space

## Alternative: Skip CINIC-10

If you don't need CINIC-10 right now, you can run experiments on the other 4 datasets that work automatically:

```bash
# Run on all auto-converting datasets
python -m tasks.vision.run_all --datasets mnist_csv qmnist_csv cifar10_csv cifar100_csv
```

These will download and convert automatically without any manual steps.

## Next Steps

1. **To use CINIC-10 on PACE**:
   ```bash
   cd /storage/ice1/6/1/pawate3/Training-Batch-Selection
   python scripts/setup_cinic10.py --convert-csv
   ```

2. **To test the working datasets first**:
   ```bash
   python -m tasks.vision.run_all --datasets mnist_csv cifar10_csv --quick
   ```

3. **To run full pipeline on all datasets**:
   ```bash
   # After setting up CINIC-10
   python -m tasks.vision.run_all --all-datasets
   ```

## Support

- **CINIC-10 Setup Issues**: See [CINIC10_SETUP.md](CINIC10_SETUP.md)
- **Dataset Issues**: See [Project README](../README.md)
- **CINIC-10 Source**: https://github.com/BayesWatch/cinic-10
