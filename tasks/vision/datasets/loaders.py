import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path
from torchvision import transforms, datasets

from tasks.vision.datasets.base import BaseDataset

print("vision_dataloader.py loaded from:", __file__)


# Data loader for MNIST CSV dataset
class MNISTCsvDataset(BaseDataset):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._load_data()

    def _load_data(self):
        # check if CSV exists, if not auto-convert from torchvision format
        if not os.path.exists(self.csv_path):
            print(f"Warning: CSV not found: {self.csv_path}")
            print("Auto-converting MNIST to CSV format...")

            # import and run auto-converter
            from tasks.vision.datasets.auto_convert_csv import ensure_mnist_csv
            from config.base import DATASETS_ROOT

            train_csv, test_csv = ensure_mnist_csv(DATASETS_ROOT)

            # figure out if we need train or test
            if "train" in self.csv_path:
                self.csv_path = train_csv
            else:
                self.csv_path = test_csv

            print(f"Using: {self.csv_path}")

        # load the CSV file
        data = pd.read_csv(self.csv_path).values
        # first column is label, rest are pixel values
        self.X = data[:, 1:].astype(np.float32) / 255.0  # normalize to [0,1]
        self.y = data[:, 0].astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# Data loader for MNIST raw dataset (using torchvision)
# flatten=True returns images as 784-long vectors (for MLP)
# flatten=False keeps 28x28 shape (for CNN)
class MNISTRawDataset(BaseDataset):
    def __init__(self, root:str, train:bool=True, flatten:bool=True, download:bool=False, normalize:bool=True):
        # setup transforms
        t = [transforms.ToTensor()]
        if normalize:
            # MNIST mean and std
            t.append(transforms.Normalize((0.1307,), (0.3081,)))

        tfm = transforms.Compose(t)

        # load MNIST using torchvision
        self.base = datasets.MNIST(root=root, train=train, download=download, transform=tfm)
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.flatten:
            x = torch.flatten(x)  # flatten to 1D for MLP
        return x, torch.tensor(y, dtype=torch.int64)

# Data loader for QMNIST CSV dataset
class QMNISTCsvDataset(BaseDataset):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._load_data()

    def _load_data(self):
        # check if CSV exists, if not auto-convert from torchvision format
        if not os.path.exists(self.csv_path):
            print(f"Warning: CSV not found: {self.csv_path}")
            print("Auto-converting QMNIST to CSV format...")

            # import and run auto-converter
            from tasks.vision.datasets.auto_convert_csv import ensure_qmnist_csv
            from config.base import DATASETS_ROOT

            train_csv, test_csv = ensure_qmnist_csv(DATASETS_ROOT)

            # figure out if we need train or test
            if "train" in self.csv_path:
                self.csv_path = train_csv
            else:
                self.csv_path = test_csv

            print(f"Using: {self.csv_path}")

        # load the CSV file
        data = pd.read_csv(self.csv_path).values
        # first column is label, rest are pixel values
        self.X = data[:, 1:].astype(np.float32) / 255.0  # normalize to [0,1]
        self.y = data[:, 0].astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


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

# Data loader for CIFAR10 CSV dataset
class CIFAR10CsvDataset(BaseDataset):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._load_data()

    def _load_data(self):
        # check if CSV exists, if not auto-convert from torchvision format
        if not os.path.exists(self.csv_path):
            print(f"Warning: CSV not found: {self.csv_path}")
            print("Auto-converting CIFAR10 to CSV format...")

            # import and run auto-converter
            from tasks.vision.datasets.auto_convert_csv import ensure_cifar10_csv
            from config.base import DATASETS_ROOT

            train_csv, test_csv = ensure_cifar10_csv(DATASETS_ROOT)

            # figure out if we need train or test
            if "train" in self.csv_path:
                self.csv_path = train_csv
            else:
                self.csv_path = test_csv

            print(f"Using: {self.csv_path}")

        # load the CSV file
        print(f"Loading CIFAR10 CSV from {self.csv_path}...")
        data = pd.read_csv(self.csv_path).values
        # first column is label, rest are pixel values (3072 for RGB 32x32)
        self.X = data[:, 1:].astype(np.float32) / 255.0  # normalize to [0,1]
        self.y = data[:, 0].astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# Data loader for CIFAR100 CSV dataset
class CIFAR100CsvDataset(BaseDataset):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._load_data()

    def _load_data(self):
        # check if CSV exists, if not auto-convert from torchvision format
        if not os.path.exists(self.csv_path):
            print(f"Warning: CSV not found: {self.csv_path}")
            print("Auto-converting CIFAR100 to CSV format...")

            # import and run auto-converter
            from tasks.vision.datasets.auto_convert_csv import ensure_cifar100_csv
            from config.base import DATASETS_ROOT

            train_csv, test_csv = ensure_cifar100_csv(DATASETS_ROOT)

            # figure out if we need train or test
            if "train" in self.csv_path:
                self.csv_path = train_csv
            else:
                self.csv_path = test_csv

            print(f"Using: {self.csv_path}")

        # load the CSV file
        print(f"Loading CIFAR100 CSV from {self.csv_path}...")
        data = pd.read_csv(self.csv_path).values
        # first column is label, rest are pixel values (3072 for RGB 32x32)
        self.X = data[:, 1:].astype(np.float32) / 255.0  # normalize to [0,1]
        self.y = data[:, 0].astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class CINIC10CsvDataset(BaseDataset):
    """Dataset loader for CINIC-10 CSV format.
    CINIC-10 is an extension of CIFAR-10 with ImageNet data, 32x32 RGB images."""
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._load_data()

    def _load_data(self):
        # Check if CSV exists
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"CINIC-10 CSV not found: {self.csv_path}\n"
                f"Please ensure CINIC-10 data is downloaded and converted to CSV format.\n"
                f"Expected files: cinic10_train.csv and cinic10_test.csv in {Path(self.csv_path).parent}"
            )

        # Load the CSV file
        print(f"Loading CINIC-10 CSV from {self.csv_path}...")
        data = pd.read_csv(self.csv_path).values
        # First column is label, rest are pixel values (3072 for RGB 32x32)
        self.X = data[:, 1:].astype(np.float32) / 255.0  # normalize to [0,1]
        self.y = data[:, 0].astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class CIFAR10Dataset(BaseDataset):
    def __init__(self, root:str, train:bool=True, flatten:bool=True, download:bool=False, normalize:bool=True, augment:bool=False):
        t = []

        # data augmentation for training (if requested)
        if augment and train:
            t += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
            ]

        t.append(transforms.ToTensor())

        # normalization with CIFAR10 statistics
        if normalize:
            t.append(transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],  # CIFAR10 mean per channel
                std = [0.2023, 0.1994, 0.2010],  # CIFAR10 std per channel
            ))

        # compose all transforms
        tfm = transforms.Compose(t)

        # load CIFAR10 dataset
        self.base = datasets.CIFAR10(root=root, train=train, download=download, transform=tfm)
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.flatten:
            x = torch.flatten(x)  # flatten for MLP
        return x.to(torch.float32), torch.tensor(y, dtype=torch.int64)
