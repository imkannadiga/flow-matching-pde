"""
Patch ViT for 2D fields with time conditioning (global token-style bias on attention).

Uses full multi-head attention over patches (patch grid is modest at 64^2 / 4^2 = 256 tokens).
FiLM from ``params`` is optional per block (rank-3 FiLM in ``models.film``).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.base import PDEModel
from models.film import FiLMLayer


class _TimeMLP(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.view(1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(t.float())


class _ViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        film_param_dim: int = 0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.film = (
            FiLMLayer(film_param_dim, dim) if film_param_dim and film_param_dim > 0 else None
        )

    def forward(
        self,
        x: torch.Tensor,
        time_bias: torch.Tensor,
        params: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = self.norm1(x)
        # Additive time bias broadcast to tokens (acts like global conditioning)
        h = h + time_bias.unsqueeze(1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        if self.film is not None and params is not None:
            x = self.film(x, params)
        return x


class FieldViT(PDEModel):
    def __init__(
        self,
        vis_channels: int = 1,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        coord_channels: int = 0,
        film_param_dim: int = 0,
        t_scaling: float = 1.0,
        out_channels=None,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.vis_channels = int(vis_channels)
        self.out_channels = (
            int(out_channels) if out_channels is not None else self.vis_channels
        )
        self.coord_channels = int(coord_channels)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.t_scaling = float(t_scaling)
        in_ch = self.vis_channels + self.coord_channels
        self.patch_embed = nn.Conv2d(
            in_ch, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        # Enough for 128 / 4 = 32 → 1024 tokens; increase if you use larger grids.
        self.max_tokens = 1024
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.time_mlp = _TimeMLP(embed_dim)
        fpd = int(film_param_dim) if film_param_dim else 0
        self.blocks = nn.ModuleList(
            [
                _ViTBlock(embed_dim, num_heads, mlp_ratio, film_param_dim=fpd)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, self.out_channels * patch_size * patch_size)

    def forward(self, t, u, coords=None, params=None):
        del coords
        t = t / self.t_scaling
        b, c, h, w = u.shape
        if self.coord_channels > 0:
            if c != self.vis_channels + self.coord_channels:
                raise ValueError(
                    f"Expected {self.vis_channels + self.coord_channels} input channels; got {c}"
                )
        ps = self.patch_size
        if h % ps != 0 or w % ps != 0:
            raise ValueError(f"H,W must be divisible by patch_size={ps}")

        x = self.patch_embed(u)
        x = x.flatten(2).transpose(1, 2)
        n_p = x.shape[1]
        if n_p > self.max_tokens:
            raise ValueError(
                f"Patch count {n_p} exceeds max_tokens={self.max_tokens}; "
                "increase patch_size or max_tokens."
            )
        pos = self.pos_embed[:, :n_p, :]
        x = x + pos

        if t.dim() == 0 or t.numel() == 1:
            t = torch.full((b,), float(t.item()), device=u.device, dtype=u.dtype)
        tb = self.time_mlp(t)

        for blk in self.blocks:
            x = blk(x, tb, params)

        x = self.norm(x)
        x = self.head(x)
        gh, gw = h // ps, w // ps
        x = x.view(b, gh, gw, self.out_channels, ps, ps)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(b, self.out_channels, h, w)
        return x
