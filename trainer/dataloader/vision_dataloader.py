import torch
import pandas as pd
import numpy as np
from torchvision import transforms, datasets

from trainer.dataloader.base_dataloader import BaseDataset

print("vision_dataloader.py loaded from:", __file__)


## Data loader for MNIST CSV dataset
class MNISTCsvDataset(BaseDataset):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._load_data()

    def _load_data(self):
        # (Optional) assert the extension
        # assert self.csv_path.endswith(".csv"), "Expected a .csv file"
        data = pd.read_csv(self.csv_path).values
        self.X = data[:, 1:].astype(np.float32) / 255.0
        self.y = data[:, 0].astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


## Data loader for MNIST raw dataset
# Flatten will return imagess as 784-long vector. Call with flatten=False for CNN.
# Use normalize for reproducability
# Check return type: int64 for XE loss but may need float32 for BCE loss
class MNISTRawDataset(BaseDataset):
    def __init__(self, root:str, train:bool=True, flatten:bool=True, download:bool=False, normalize:bool=True):
        t = [transforms.ToTensor()]
        if normalize:
            t.append(transforms.Normalize((0.1307,), (0.3081,)))
        tfm = transforms.Compose(t)

        self.base = datasets.MNIST(root=root, train=train, download=download, transform=tfm)
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        x,y = self.base[idx]
        if self.flatten:
            x = torch.flatten(x)
        return x, torch.tensor(y, dtype=torch.int64)

class QMNISTDataset(BaseDataset):
    def __init__(self, root:str, train:bool=True, flatten:bool=True, download:bool=False, normalize:bool=True):
        t = [transforms.ToTensor()]
        if normalize:
            t.append(transforms.Normalize((0.1307,), (0.3081,)))
        tfm = transforms.Compose(t)

        self.base = datasets.QMNIST(root=root, train=train, download=download, transform=tfm)
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        x,y = self.base[idx]
        if self.flatten:
            x = torch.flatten(x)
        return x, torch.tensor(y, dtype=torch.int64)

class CIFARDatasetUnified(BaseDataset):
    """
    Unified CIFAR dataset.
    - dataset: 'cifar10' or 'cifar100'
    - in_channels: 3 (RGB) or 1 (grayscale)
    - augment: random crop + horizontal flip for train
    - normalize: uses canonical stats by default (overridable with mean/std)
    - flatten: default False (prefer shaping per model)
    Exposes: .num_classes, .class_names, .in_channels
    """
    def __init__(
        self,
        root: str,
        *,
        train: bool = True,
        dataset: str = "cifar10",
        download: bool = False,
        normalize: bool = True,
        augment: bool = False,
        in_channels: int = 3,
#        mean: Optional[Sequence[float]] = None,
#        std: Optional[Sequence[float]] = None,
        mean: list[float] = None,
        std: list[float] = None,
        flatten: bool = False,
        target_transform=None,  # keep hook for fine->coarse mapping, etc.
    ):

        print(f'[CIFARDatasetUnified]: constructing {dataset} dataset (train={train}) with in_channels={in_channels} and flatten={flatten}')
        dataset = dataset.lower()
        if dataset not in {"cifar10", "cifar100"}:
            raise ValueError("dataset must be 'cifar10' or 'cifar100'")
        if in_channels not in (1, 3):
            raise ValueError("in_channels must be 1 or 3")

        # --- Transforms ---
        t = []
        if augment and train:
            t += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
            ]
        if in_channels == 1:
            # do this before ToTensor so it outputs 1 channel
            t.append(transforms.Grayscale(num_output_channels=1))
        t.append(transforms.ToTensor())

        if normalize:
#            mean, std = _pick_stats(dataset, in_channels, mean, std)
            if len(mean) != in_channels or len(std) != in_channels:
                raise ValueError("mean/std length must match in_channels")
            t.append(transforms.Normalize(mean=mean, std=std))

        tfm = transforms.Compose(t)

        # --- Base dataset ---
        if dataset == "cifar10":
            base = datasets.CIFAR10(root=root, train=train, download=download, transform=tfm, target_transform=target_transform)
            self.num_classes = 10
            self.class_names = list(base.classes)  # ['airplane', 'automobile', ...]
        else:
            base = datasets.CIFAR100(root=root, train=train, download=download, transform=tfm, target_transform=target_transform)
            self.num_classes = 100
            self.class_names = list(base.classes)  # 100 fine labels

        self.base = base
        self.flatten = flatten
        self.in_channels = in_channels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]            # x: (C, 32, 32) float32
        if self.flatten:
            x = torch.flatten(x)         # -> (C*32*32,)
        return x, torch.tensor(y, dtype=torch.int64)
