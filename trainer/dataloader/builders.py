from trainer.dataloader.vision_dataloader import (
    MNISTRawDataset, MNISTCsvDataset, QMNISTDataset, CIFARDatasetUnified, TinyImageNetDataset
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

def build_tinyimagenet(root, *, as_flat=True, normalize=True, augment=False, download=False, **kwargs):
    import os
    train = TinyImageNetDataset(root, train=True,  flatten=as_flat, download=download, normalize=normalize, augment=False, **kwargs)
    test  = TinyImageNetDataset(root, train=False,  flatten=as_flat, download=download, normalize=normalize, augment=False, **kwargs)
    return train, test

def build_cifar10_flat(root, *, normalize=True, augment=True, download=False, **kwargs):
    # flattened 3*32*32 inputs to fit the MLP
    GRAY_MEAN, GRAY_STD = [0.5], [0.5]
    in_channels=1
#    train = CIFAR10Dataset(root, train=True,  flatten=True,  download=download, normalize=normalize, augment=augment)
#    test  = CIFAR10Dataset(root, train=False, flatten=True,  download=download, normalize=normalize, augment=False)
    train = CIFARDatasetUnified(root, dataset='cifar10', train=True,  flatten=True,
                           download=download, normalize=normalize, augment=augment,
                           in_channels=in_channels, mean=GRAY_MEAN, std=GRAY_STD, **kwargs)
    test  = CIFARDatasetUnified(root, dataset='cifar10', train=False, flatten=True,
                           download=download, normalize=normalize, augment=False,
                           in_channels=in_channels, mean=GRAY_MEAN, std=GRAY_STD, **kwargs)
    return train, test

def build_cifar10(root, *, normalize=True, augment=True, download=False, in_channels=3, **kwargs):

    if kwargs.get("flatten", None) is True:
        raise ValueError("build_cifar10 received flatten=True; remove that override.")
    print (f'building a dataset with in_channels={in_channels}')
    CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR10_STD  = [0.2023, 0.1994, 0.2010]

    train = CIFARDatasetUnified(root, dataset='cifar10', train=True,  flatten=False,
                           download=download, normalize=normalize, augment=augment,
                           in_channels=in_channels, mean=CIFAR10_MEAN, std=CIFAR10_STD, **kwargs)
    test  = CIFARDatasetUnified(root, dataset='cifar10', train=False, flatten=False,
                           download=download, normalize=normalize, augment=False,
                           in_channels=in_channels, mean=CIFAR10_MEAN, std=CIFAR10_STD, **kwargs)
    return train, test

def build_cifar100(root, *, normalize=True, augment=True, download=False, in_channels=3, **kwargs):

    if kwargs.get("flatten", None) is True:
        raise ValueError("build_cifar100 received flatten=True; remove that override.")
    CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR100_STD  = [0.2675, 0.2565, 0.2761]

    train = CIFARDatasetUnified(root, dataset='cifar100', train=True,  flatten=False,
                           download=download, normalize=normalize, augment=augment,
                           in_channels=in_channels, mean=CIFAR100_MEAN, std=CIFAR100_STD, **kwargs)
    test  = CIFARDatasetUnified(root, dataset='cifar100', train=False, flatten=False,
                           download=download, normalize=normalize, augment=False,
                           in_channels=in_channels, mean=CIFAR100_MEAN, std=CIFAR100_STD, **kwargs)
    return train, test

