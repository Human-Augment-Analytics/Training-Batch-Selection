import torch
import pandas as pd
import numpy as np
from trainer.dataloader.base_dataloader import BaseDataset


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
