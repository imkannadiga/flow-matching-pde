"""Feature-wise linear modulation from global parameters (Re, nu, ...)."""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Maps ``params`` [B, param_dim] to channel-wise ``gamma``, ``beta``; returns ``gamma * x + beta``."""

    def __init__(self, param_dim: int, n_channels: int, hidden_mult: int = 2):
        super().__init__()
        if param_dim < 1:
            raise ValueError("param_dim must be >= 1 for FiLMLayer")
        h = max(n_channels, n_channels * hidden_mult // 2)
        self.net = nn.Sequential(
            nn.Linear(param_dim, h),
            nn.GELU(),
            nn.Linear(h, 2 * n_channels),
        )
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        gb = self.net(params)
        gamma, beta = gb.chunk(2, dim=1)
        if x.dim() == 4:
            gamma = gamma.view(-1, self.n_channels, 1, 1)
            beta = beta.view(-1, self.n_channels, 1, 1)
            return gamma * x + beta
        if x.dim() == 3:
            # [B, N, D] token layout (D == n_channels)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            return gamma * x + beta
        raise ValueError(f"FiLMLayer expects x of rank 3 or 4; got shape {tuple(x.shape)}")
