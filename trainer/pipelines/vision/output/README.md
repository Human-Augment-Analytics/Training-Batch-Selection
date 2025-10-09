## Vision Pipeline — MNIST Training

This pipeline trains and evaluates vision models on the MNIST dataset using 3 batch selection strategies: **Random Batching**, **Smart batching** and **Fixed batching**

### What It Does
- Loads MNIST data from CSVs via `MNISTCsvDataset`.
- Trains a `SimpleMLP` model using 3 batch strategies.  
- Logs training/test accuracy and loss across epochs.
- Automatically saves all outputs (metrics, plots, configs) under `/output/run-XXX`.
- `/output/run-001[sample]` contains sample summary files, accuracy and loss curves

---

### Output Directory Structure
Each new run creates a folder under `/trainer/pipelines/vision/output/` as follows:

```
/output
    /runs-001
        ├── summary.txt                     # Epoch-wise metrics
        ├── test_acc.png               
        ├── train_acc.png            
        ├── train_loss.png 
    /runs-002
    ...
```

### How to Run

```bash
python trainer/pipelines/vision/vision.py
```

- The script will automatically:
  - Train 3 batch strategies
  - Evaluate their performance.
  - Save metrics, plots, and configurations to `/output/runs-XXX`.