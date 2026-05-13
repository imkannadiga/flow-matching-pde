"""Shared base class for PDE surrogate models (time-conditioned fields)."""

from abc import abstractmethod

import torch.nn as nn


class PDEModel(nn.Module):
    """All models accept optional coordinate and parameter conditioning."""

    @abstractmethod
    def forward(self, t, u, coords=None, params=None):
        """
        Args:
            t: time tensor [B] or broadcastable scalar tensor
            u: state [B, C, H, W]
            coords: optional [B, n_dim, H, W] normalized spatial grid
            params: optional physical parameters (e.g. Re, nu)
        """
        raise NotImplementedError
