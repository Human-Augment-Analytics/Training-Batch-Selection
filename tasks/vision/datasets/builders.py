from tasks.vision.datasets.loaders import (
    MNISTRawDataset, MNISTCsvDataset, QMNISTDataset, QMNISTCsvDataset,
    CIFAR10Dataset, CIFAR10CsvDataset, CIFAR100CsvDataset, CINIC10CsvDataset
)

# Each builder returns (train_ds, test_ds).
# Defaults for preprocessing live here (not in the registry).
# Override at the call site if you want different behavior for an experiment.

def build_mnist(root, *, as_flat=True, normalize=True, download=False, **kwargs):
    train = MNISTRawDataset(root, train=True,  flatten=as_flat, download=download, normalize=normalize)
    test  = MNISTRawDataset(root, train=False, flatten=as_flat, download=download, normalize=normalize)
    return train, test

def build_mnist_csv(root, **kwargs):
    # root should contain mnist_train.csv and mnist_test.csv
    import os
    train = MNISTCsvDataset(os.path.join(root, "mnist_train.csv"))
    test  = MNISTCsvDataset(os.path.join(root, "mnist_test.csv"))
    return train, test

def build_qmnist(root, *, as_flat=True, normalize=True, download=False, **kwargs):
    train = QMNISTDataset(root, train=True,  flatten=as_flat, download=download, normalize=normalize)
    test  = QMNISTDataset(root, train=False, flatten=as_flat, download=download, normalize=normalize)
    return train, test

def build_qmnist_csv(root, **kwargs):
    # root should contain qmnist_train.csv and qmnist_test.csv
    import os
    train = QMNISTCsvDataset(os.path.join(root, "qmnist_train.csv"))
    test  = QMNISTCsvDataset(os.path.join(root, "qmnist_test.csv"))
    return train, test

def build_cifar10_flat(root, *, normalize=True, augment=True, download=False, **kwargs):
    # flattened 3*32*32 inputs to fit the MLP
    train = CIFAR10Dataset(root, train=True,  flatten=True,  download=download, normalize=normalize, augment=augment)
    test  = CIFAR10Dataset(root, train=False, flatten=True,  download=download, normalize=normalize, augment=False)
    return train, test

def build_cifar10_csv(root, **kwargs):
    # root should contain cifar10_train.csv and cifar10_test.csv
    import os
    train = CIFAR10CsvDataset(os.path.join(root, "cifar10_train.csv"))
    test  = CIFAR10CsvDataset(os.path.join(root, "cifar10_test.csv"))
    return train, test

def build_cifar100_csv(root, **kwargs):
    # root should contain cifar100_train.csv and cifar100_test.csv
    import os
    train = CIFAR100CsvDataset(os.path.join(root, "cifar100_train.csv"))
    test  = CIFAR100CsvDataset(os.path.join(root, "cifar100_test.csv"))
    return train, test

def build_cinic10_csv(root, **kwargs):
    # root should contain cinic10_train.csv and cinic10_test.csv
    # Auto-convert if needed
    import os
    from pathlib import Path

    train_csv = os.path.join(root, "cinic10_train.csv")
    test_csv = os.path.join(root, "cinic10_test.csv")

    # If CSVs don't exist, try to auto-convert
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print(f"CINIC-10 CSVs not found. Attempting auto-conversion...")
        # Extract the root dir (going up from .../vision/cinic-10/csv to datasets/)
        root_path = Path(root)
        if root_path.name == "csv":
            datasets_root = root_path.parent.parent.parent
        else:
            datasets_root = root_path.parent

        from tasks.vision.datasets.auto_convert_csv import ensure_cinic10_csv
        train_csv, test_csv = ensure_cinic10_csv(str(datasets_root))

    train = CINIC10CsvDataset(train_csv)
    test  = CINIC10CsvDataset(test_csv)
    return train, test

