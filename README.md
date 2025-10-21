## The Code consists of 2 pipelines, text and vision

### Text Pipeline — [TODO: Ishan]

### Vision Pipeline — MNIST Training

This pipeline trains and evaluates vision models on the MNIST dataset using three modular, plug-and-play batch selection strategies: **Random Batching**, **Fixed Batching**, and **Smart Batching**.

### What It Does

- **Loads MNIST data** from CSVs via `MNISTCsvDataset`.
- **Trains a `SimpleMLP` model** on MNIST for each batch strategy:
  - Random Batching (`random_batch.py`)
  - Fixed Batching  (`fixed_batch.py`)
  - Smart Batching  (`smart_batch.py`)
- **Logs training and test accuracy and loss** for each epoch.
- **Saves all results automatically** to subdirectories under:
trainer/pipelines/vision/output/batching_<strategy>/run-XXX/

- Each `run-XXX` subdirectory contains:
  - `summary.txt`: Table of mean and confidence interval for all metrics.
  - `test_acc.png`, `train_acc.png`: Accuracy curves.
  - `test_loss.png`, `train_loss.png`: Loss curves.

- **Easy comparison**: Use `comparison_vision_batch.py` to overlay metric plots for any two strategies.

---

### How to Run

In the root directory, run:
```bash
python -m trainer.pipelines.vision.vision
```
The script will:
- Train and evaluate using all configured batch strategies.
- Save all metrics and plots, grouped by batch strategy, to /output/batching_<strategy>/run-XXX.

### To Compare Strategies
You can generate visual comparisons (accuracy/loss curves) between any two strategies using:
```bash
python trainer.pipelines.vision.comparison_vision_batch.py
```
- Edit trainer/constants_batch_strategy.py to specify which batch strategies to compare.
- Overlaid result plots will be saved to:
`trainer/pipelines/vision/output/comparison_<stratA>_<stratB>/`

### How to Add or Change a Batch Strategy
- Add a batching module in `trainer/batching/vision_batching/` (e.g., my_batch.py).
- Register your strategy in `trainer/constants_batch_strategy.py`:
```
BATCH_STRATEGIES = {
    "Random": "vision_batching/random_batch",
    "Fixed":  "vision_batching/fixed_batch",
    "Smart":  "vision_batching/smart_batch",
    "MyBatch": "vision_batching/my_batch",  # <--- add this line
}
```
- Re-run `vision.py`. 
- Your new strategy will be evaluated and outputs saved automatically.
- Output Example
```
trainer/pipelines/vision/output/
  batching_random/
    run-001/
      summary.txt
      test_acc.png
      train_acc.png
      test_loss.png
      train_loss.png
  batching_fixed/
    run-001/...
  batching_smart/
    run-001/...
  comparison_fixed_smart/
    train_acc_cmp.png
    test_acc_cmp.png
    ...
```

### To Compare Datasets

You can run the same experiment on different datasets by running
```bash
python -m trainer.pipelines.vision.benchmark_datasets.py
```
- This script calls the experiment specified in `trainer/pipelines/vision/vision.py`
- Specify the list of datasets to compare in the `DATASETS` variable at the top of `trainer/pipelines/vision/benchmark_datasets.py`, e.g. `DATASETS = ["mnist_csv", "mnist", "qmnist", "cifar10_flat"]`
- result plots will be saved to the directory specified in the experiment implementation

### To Add a Dataset

- Add specifications (input dimensions, number of classes, etc.) to `trainer/constants_datasets.py`
- Add a builder to `trainer/dataloader/builders.py`


### About the Constants Files

- All major configuration for the pipeline is controlled via two key files:

1. `trainer/constants.py`: Holds paths, data locations, all model dimensions, training hyperparameters, and smart-batch tuning constants.
Change anything here (e.g., epochs, batch size, hidden layer size), and your pipeline adjusts! Sample format of the file below
```
BASE_DIR = ...
SHARED_DATA_DIR = ...
TRAIN_CSV = ...
INPUT_DIM = 784
HIDDEN_DIM = 128
NUM_CLASSES = 10
EPOCHS = 5
BATCH_SIZE = 64
... 
```
2. `trainer/constants_batch_strategy.py`: All available batch strategies (labels and module paths) are defined here.

3. `trainer/constants_datasets.py`: Holds specifications of different datasets, such as input dimensions and number of classes. Sample format of the file below
```
DATASET_SPECS = {
     "mnist": {
         "builder": "build_mnist",
         "input_dim": 28 * 28,
         "num_classes": 10,
         "subdir": "vision/MNIST",
    },
    # add more …
}

... 
```
