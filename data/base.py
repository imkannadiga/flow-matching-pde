from typing import Optional
from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class BaseDataModule(Dataset, ABC):
    def __init__(self, data_path, transform=None, eval: bool = False):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.eval = eval
        self.c_channels = None
        self.target_channels = None

    @abstractmethod
    def _fetch_data_pair(self, idx):
        """Return (C, X_target) where C is the full conditioning tensor for the model."""
        pass

    def _fetch_dataset_label(self, idx: int) -> Optional[str]:
        """Return a string label identifying which sub-dataset sample ``idx`` belongs to.

        Override in subclasses that expose multiple datasets (e.g. multiple beta
        files in Darcy).  Return ``None`` (the default) if there is only one
        dataset so that no ``dataset_id`` key is added to the batch.
        """
        return None

    def _trajectory_count(self) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} does not support eval mode. "
            "Implement _trajectory_count() and _fetch_trajectory()."
        )

    def _fetch_trajectory(self, idx: int) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} does not support eval mode. "
            "Implement _trajectory_count() and _fetch_trajectory()."
        )

    def __len__(self):
        if self.eval:
            return self._trajectory_count()
        raise NotImplementedError("Subclasses must implement __len__ for training mode.")

    def __getitem__(self, idx):
        if self.eval:
            return self._fetch_trajectory(idx)

        C, X_target = self._fetch_data_pair(idx)

        if self.transform:
            C = self.transform(C)
            X_target = self.transform(X_target)

        item = {"x": C, "y": X_target}
        label = self._fetch_dataset_label(idx)
        if label is not None:
            item["dataset_id"] = label
        return item
