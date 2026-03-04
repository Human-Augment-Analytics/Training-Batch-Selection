from trainer.dataloader.vision_dataloader import (
    MNISTRawDataset, MNISTCsvDataset, QMNISTDataset, CIFARDatasetUnified, NeWTDatasetUnified, WILDSXY
)
from torchvision import transforms
from wilds import get_dataset

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
    GRAY_MEAN, GRAY_STD = [0.5], [0.5]
    in_channels=1

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

def build_newt(
    root,
    *,
    task: str,
    normalize=True,
    augment=True,
    in_channels=3,
    image_size=224,
    mean=None,
    std=None,
    **kwargs
):
    """                                                                                                           
    Build NeWT (binary) datasets for ONE task.                                     Expects:                                                                         root/                                                                            newt2021_labels.csv                                                            newt2021_images/<id>.jpg                                                   """

    if kwargs.get("flatten", None) is True:
        raise ValueError("build_newt received flatten=True; remove that override.")

    if mean is None:
        mean = [0.485, 0.456, 0.406] if in_channels == 3 else [0.5]
    if std is None:
        std = [0.229, 0.224, 0.225] if in_channels == 3 else [0.5]

    train = NeWTDatasetUnified(
        root,
        task=task,
        split="train",
        flatten=False,
        normalize=normalize,
        augment=augment,
        in_channels=in_channels,
        img_size=image_size,
        mean=mean,
        std=std,
        **kwargs,
    )
    test = NeWTDatasetUnified(
        root,
        task=task,
        split="test",
        flatten=False,
        normalize=normalize,
        augment=False,
        in_channels=in_channels,
        img_size=image_size,
        mean=mean,
        std=std,
        **kwargs,
    )
    return train, test


import torch
from torchvision import transforms
from torch.utils.data import Dataset
from wilds import get_dataset

class WILDSXY(Dataset):
    def __init__(self, wilds_subset, *, flatten=False, target_transform=None):
        self.ds = wilds_subset
        self.flatten = flatten
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y, meta = self.ds[idx]
        y = int(y)
        if self.target_transform is not None:
            y = self.target_transform(y)
        if self.flatten:
            x = torch.flatten(x)
        return x, torch.tensor(y, dtype=torch.int64)


def build_iwildcam(
    root,
    *,
    normalize=True,
    augment=True,
    download=False,
    in_channels=3,
    image_size=224,
    mean=None,
    std=None,
    # keep the same hook name torchvision uses
    target_transform=None,
    **kwargs
):
    # Mirror CIFAR behavior: ignore irrelevant spec keys if they appear
    kwargs.pop("task", None)
    kwargs.pop("img_size", None)  # just in case
    # If someone tries to force flatten via overrides, fail loudly like others
    if kwargs.get("flatten", None) is True:
        raise ValueError("build_iwildcam received flatten=True; remove that override.")

    if in_channels not in (1, 3):
        raise ValueError("build_iwildcam: in_channels must be 1 or 3")

    if mean is None:
        mean = [0.485, 0.456, 0.406] if in_channels == 3 else [0.5]
    if std is None:
        std = [0.229, 0.224, 0.225] if in_channels == 3 else [0.5]

    # train/eval transforms (similar spirit to NeWT)
    def make_tfm(is_train: bool):
        t = []
        if is_train and augment:
            t += [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            t += [
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
            ]
        if in_channels == 1:
            t.append(transforms.Grayscale(num_output_channels=1))
        t.append(transforms.ToTensor())
        if normalize:
            t.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(t)

    base = get_dataset("iwildcam", root_dir=root, download=download)

    # Choose which WILDS split to treat as "test"
    # Many people use 'val' for quick iteration; switch to 'test' if desired.
    train_subset = base.get_subset("train", transform=make_tfm(True))
    test_subset  = base.get_subset("val",   transform=make_tfm(False))

    train = WILDSXY(train_subset, flatten=False, target_transform=target_transform)
    test  = WILDSXY(test_subset,  flatten=False, target_transform=target_transform)

    # Optional parity fields (nice for prints)
    train.in_channels = in_channels
    test.in_channels = in_channels
    train.num_classes = getattr(base, "n_classes", None)
    test.num_classes = getattr(base, "n_classes", None)

    return train, test

