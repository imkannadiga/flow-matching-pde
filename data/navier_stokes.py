from data.base import BaseDataModule
import torch

class NSDataModule(BaseDataModule):
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """
        Expects [T, H, W, N] → returns [N, T, C, H, W]
        """
        data = data.permute(3, 0, 1, 2)      # [N, T, H, W]
        data = data.unsqueeze(2)             # [N, T, 1, H, W]
        return data
