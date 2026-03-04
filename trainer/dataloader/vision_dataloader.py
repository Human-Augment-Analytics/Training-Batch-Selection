import torch
import os
import pandas as pd
import numpy as np
from torchvision import transforms, datasets
from PIL import Image
from trainer.dataloader.base_dataloader import BaseDataset
from wilds import get_dataset
from torch.utils.data import Dataset

print("vision_dataloader.py loaded from:", __file__)

_ALLOWED_EXTRA_KWARGS = {"task", "img_size", "image_size"}

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
        mean: list[float] = None,
        std: list[float] = None,
        flatten: bool = False,
        target_transform=None,  # keep hook for fine->coarse mapping, etc.
        image_size=None,   # accept it
        **kwargs,          # and anything else future builders pass
    ):

        unknown = set(kwargs) - _ALLOWED_EXTRA_KWARGS
        if unknown:
            raise TypeError(f"CIFARDatasetUnified got unexpected kwargs: {sorted(unknown)}")

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

class NeWTDatasetUnified(BaseDataset):
    """                                                                                                                            
    NeWT 2021 wrapper for ONE binary task.                                         Expects in root:                                                                                                               
      newt2021_labels.csv                                                                                                          
      newt2021_images/<id>.jpg                                                                                                     
    """

    def __init__(
        self,
        root: str,
        *,
        task: str,
        split: str = "train",           # "train" or "test"                                                                        
        normalize: bool = True,
        augment: bool = False,
        in_channels: int = 3,
        img_size: int = 224,
        mean: list[float] = None,
        std: list[float] = None,
        flatten: bool = False,
        target_transform=None,
    ):
        print(f"[NeWTDatasetUnified]: task={task} split={split} img_size={img_size} in_channels={in_channels} flatten={flatten}")

        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if in_channels not in (1, 3):
            raise ValueError("in_channels must be 1 or 3")

        labels_csv = os.path.join(root, "newt2021_labels.csv")
        images_dir = os.path.join(root, "newt2021_images")
        if not os.path.isfile(labels_csv):
            raise FileNotFoundError(f"Missing labels CSV: {labels_csv}")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Missing images dir: {images_dir}")

        df = pd.read_csv(labels_csv)
        df = df[(df["task"] == task) & (df["split"] == split)].copy()
        if len(df) == 0:
            raise ValueError(f"No rows for task='{task}' split='{split}'")

        df["filepath"] = df["id"].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))

        missing = df[~df["filepath"].apply(os.path.exists)]
        if len(missing) > 0:
            raise FileNotFoundError(f"{len(missing)} missing images. First: {missing.iloc[0]['filepath']}")

        # metadata (like CIFAR)                                                                                                    
        self.num_classes = 2
        self.in_channels = in_channels
        self.flatten = flatten

        pos = df[df["label"] == 1]["text_label"].unique().tolist()
        neg = df[df["label"] == 0]["text_label"].unique().tolist()
        pos_name = pos[0] if len(pos) else "pos"
        neg_name = neg[0] if len(neg) else "neg"
        self.class_names = [neg_name, pos_name]

        # transforms                                                                                                               
        t = []
        if augment and split == "train":
            t += [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            t.append(transforms.Resize((img_size, img_size)))

        if in_channels == 1:
            t.append(transforms.Grayscale(num_output_channels=1))

        t.append(transforms.ToTensor())

        if normalize:
            if mean is None:
                mean = [0.485, 0.456, 0.406] if in_channels == 3 else [0.5]
            if std is None:
                std = [0.229, 0.224, 0.225] if in_channels == 3 else [0.5]
            if len(mean) != in_channels or len(std) != in_channels:
                raise ValueError("mean/std length must match in_channels")
            t.append(transforms.Normalize(mean=mean, std=std))

        self.tfm = transforms.Compose(t)
        self.df = df.reset_index(drop=True)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        x = self.tfm(img)
        y = int(row["label"])
        if self.target_transform is not None:
            y = self.target_transform(y)
        if self.flatten:
            x = torch.flatten(x)
        return x, torch.tensor(y, dtype=torch.int64)


class WILDSXY(Dataset):
    """Wrap a WILDS subset so __getitem__ returns (x, y) only."""
    def __init__(self, wilds_subset, *, target_transform=None, flatten=False):
        self.ds = wilds_subset
        self.target_transform = target_transform
        self.flatten = flatten

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

class IWildCam(Dataset):
    def __init__(self, root, split, transform=None, download=False):
        # split names in WILDS: 'train', 'val', 'test', plus some versions have 'id_val', 'id_test'
        self.ds = get_dataset(
            dataset="iwildcam",
            root_dir=root,
            download=download,
        ).get_subset(split, transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y, meta = self.ds[idx]  # WILDS returns (x, y, metadata)
        return x, int(y)
