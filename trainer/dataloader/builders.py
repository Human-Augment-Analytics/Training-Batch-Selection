from trainer.dataloader.vision_dataloader import (
    MNISTRawDataset, MNISTCsvDataset, QMNISTDataset, CIFAR10Dataset
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

def build_cifar10_flat(root, *, normalize=True, augment=True, download=False, **kwargs):
    # flattened 3*32*32 inputs to fit the MLP
    train = CIFAR10Dataset(root, train=True,  flatten=True,  download=download, normalize=normalize, augment=augment)
    test  = CIFAR10Dataset(root, train=False, flatten=True,  download=download, normalize=normalize, augment=False)
    return train, test

