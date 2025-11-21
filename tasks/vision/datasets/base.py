from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class BaseDataset(Dataset, ABC):
    """Minimal abstract base: enforce PyTorch Dataset contract only."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass
