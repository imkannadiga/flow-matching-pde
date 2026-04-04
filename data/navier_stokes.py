from typing import List, Optional

import torch

from data.base import BaseDataModule


class NSDataModule(BaseDataModule):
    """
    Navier-Stokes volumes on the unit square; preprocessed tensor shape **(N, T, C, H, W)**.

    ``domain_bounds`` lists ``(min, max)`` per spatial axis in physical coordinates
    (used later for coordinate conditioning / plotting).
    """

    # Unit square [0, 1]^2 for this dataset family (override in subclasses for other domains).
    domain_bounds = ((0.0, 1.0), (0.0, 1.0))

    def __init__(
        self,
        *args,
        domain_bounds: Optional[tuple] = None,
        default_param_vector: Optional[List[float]] = None,
        **kwargs,
    ):
        cp = None
        if default_param_vector is not None:
            cp = torch.tensor(list(default_param_vector), dtype=torch.float32)
        super().__init__(*args, constant_params=cp, **kwargs)
        if domain_bounds is not None:
            self.domain_bounds = domain_bounds

    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """
        Expects **[T, H, W, N]** → returns **[N, T, C, H, W]** with ``C=1``.
        """
        data = data.permute(3, 0, 1, 2)  # [N, T, H, W]
        data = data.unsqueeze(2)  # [N, T, 1, H, W]
        return data
