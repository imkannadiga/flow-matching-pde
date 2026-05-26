"""Feature-wise spatial modulation from spatial parameters (u_prev, Re, nu, tau...)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMLayer(nn.Module):
    """
    Maps spatial ``params`` [B, param_dim, X, Y] to spatially-varying channel-wise 
    ``gamma``, ``beta`` using 1x1 convolutions; returns ``x * (1 + gamma) + beta``.
    """

    def __init__(self, param_dim: int, n_channels: int, hidden_mult: int = 2):
        super().__init__()
        if param_dim < 1:
            raise ValueError("param_dim must be >= 1 for FiLMLayer")
            
        h = max(n_channels, n_channels * hidden_mult // 2)
        
        # Swapped nn.Linear for nn.Conv2d(..., kernel_size=1)
        self.net = nn.Sequential(
            nn.Conv2d(param_dim, h, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(h, 2 * n_channels, kernel_size=1),
        )
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or params.dim() != 4:
            raise ValueError(
                f"Spatial FiLM expects 4D inputs [B, C, X, Y]. "
                f"Got x: {tuple(x.shape)}, params: {tuple(params.shape)}"
            )

        # Robustness check: Ensure params spatial dimensions match x
        # (The padding logic in the refactored FNO guarantees this, but it is 
        # good practice to include an interpolate safety net)
        if params.shape[-2:] != x.shape[-2:]:
            params = F.interpolate(params, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # gb shape: [B, 2 * n_channels, X, Y]
        gb = self.net(params)
        
        # scale and shift are both [B, n_channels, X, Y]
        scale, shift = gb.chunk(2, dim=1)
        
        return x * (1 + scale) + shift