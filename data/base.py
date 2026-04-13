from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class BaseDataModule(Dataset, ABC):
    def __init__(self, data_path, transform=None):
        """
        Common initialization for all PDE datasets.
        """
        super().__init__()

        self.data_path = data_path
        self.transform = transform

        # Subclasses must populate these during their __init__
        self.c_channels = None  
        self.target_channels = None

    @abstractmethod
    def __len__(self):
        """Must return the total number of independent training pairs."""
        pass

    @abstractmethod
    def _fetch_data_pair(self, idx):
        """
        Subclasses implement this to handle their specific data structures.
        Must return a tuple of (Conditioning_Tensor, Target_Tensor).
        """
        pass

    def __getitem__(self, idx):
        """
        The Template Method: Enforces the dictionary structure for the training loop.
        Subclasses should NOT override this method.
        """
        C, X_target = self._fetch_data_pair(idx)

        if self.transform:
            C = self.transform(C)
            X_target = self.transform(X_target)

        return {
            "x": C,
            "y": X_target
        }