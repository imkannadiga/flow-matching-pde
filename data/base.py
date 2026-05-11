from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class BaseDataModule(Dataset, ABC):
    def __init__(self, data_path, transform=None, eval: bool = False):
        """
        Common initialization for all PDE datasets.
        """
        super().__init__()

        self.data_path = data_path
        self.transform = transform
        self.eval = eval

        # Subclasses must populate these during their __init__
        self.c_channels = None
        self.target_channels = None

    @abstractmethod
    def _fetch_data_pair(self, idx):
        """
        Subclasses implement this to handle their specific data structures.
        Must return a tuple of (Conditioning_Tensor, Target_Tensor).
        """
        pass

    def _make_x0(self, idx: int, X_target):
        """
        Return the physical initial condition for this sample.
        Returning None signals the flow matching processor to fall back to Gaussian noise.
        Subclasses override this for datasets with a meaningful X_0.
        """
        return None

    def _trajectory_count(self) -> int:
        """Number of trajectories. Override in subclasses that support eval mode."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support eval mode. "
            "Implement _trajectory_count() and _fetch_trajectory()."
        )

    def _fetch_trajectory(self, idx: int) -> dict:
        """
        Return a full trajectory for eval-mode rollout.
        Must return a dict with keys:
          'x_0'          [1, H, W]           initial state
          'conditions'   [T-1, C_extra, H, W] pre-built extra conditioning (no state channel)
          'targets'      [T-1, C_out, H, W]  ground-truth states at each step
          'time_schedule'[T-1]               physical time at each step
        Override in subclasses that support eval mode.
        """
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

        # Training Template Method: enforces the dictionary structure for the training loop.
        C, X_target = self._fetch_data_pair(idx)

        if self.transform:
            C = self.transform(C)
            X_target = self.transform(X_target)

        result = {"x": C, "y": X_target}
        x_0 = self._make_x0(idx, X_target)
        if x_0 is not None:
            result["x_0"] = x_0
        return result