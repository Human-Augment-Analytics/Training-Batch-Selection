import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path
from torchvision import transforms, datasets

from trainer.dataloader.base_dataloader import BaseDataset

print("vision_dataloader.py loaded from:", __file__)


## Data loader for MNIST CSV dataset
class MNISTCsvDataset(BaseDataset):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._load_data()

    def _load_data(self):
        # Auto-convert if CSV doesn't exist
        if not os.path.exists(self.csv_path):
            print(f"⚠️  CSV not found: {self.csv_path}")
            print("🔄 Auto-converting MNIST to CSV format...")

            # Import and run auto-converter
            from trainer.dataloader.auto_convert_csv import ensure_mnist_csv
            from trainer.constants import DATASETS_ROOT

            train_csv, test_csv = ensure_mnist_csv(DATASETS_ROOT)

            # Check if this is train or test CSV we need
            if "train" in self.csv_path:
                self.csv_path = train_csv
            else:
                self.csv_path = test_csv

            print(f"✔️ Using: {self.csv_path}")

        # Load CSV data
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

class CIFAR10Dataset(BaseDataset):
    def __init__(self, root:str, train:bool=True, flatten:bool=True, download:bool=False, normalize:bool=True, augment:bool=False):
        t = []
        if augment and train:
            t += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                ]
        t.append(transforms.ToTensor())
        if normalize:
            t.append(transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std = [0.2023, 0.1994, 0.2010],
            ))

        # collect all the transforms that will be applied
        tfm = transforms.Compose(t)

        self.base = datasets.CIFAR10(root=root, train=train, download=download, transform=tfm)
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        x,y = self.base[idx]
        if self.flatten:
            x = torch.flatten(x)
        return x.to(torch.float32), torch.tensor(y, dtype=torch.int64)
