"""Spatial coordinate grids for conditioning (normalized to a fixed range)."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union

import torch

Bounds = Sequence[Tuple[float, float]]


def make_coord_grid(
    domain_bounds: Bounds,
    shape: Tuple[int, ...],
    *,
    normalize: str = "neg1_1",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build a coordinate grid for a tensor with spatial ``shape`` (H, W) or (H,).

    Args:
        domain_bounds: one ``(min, max)`` pair per spatial axis (physical units).
        shape: spatial dimensions only, e.g. ``(H, W)``.
        normalize: ``neg1_1`` maps each axis linearly to ``[-1, 1]``;
            ``zero_one`` maps to ``[0, 1]`` (matches legacy ``make_posn_embed``).

    Returns:
        Tensor of shape ``(n_dim, *shape)`` (no batch dimension).
    """
    if len(domain_bounds) != len(shape):
        raise ValueError(
            f"domain_bounds has {len(domain_bounds)} axes but shape has {len(shape)} dims"
        )
    axes: List[torch.Tensor] = []
    for (xmin, xmax), n in zip(domain_bounds, shape):
        lin = torch.linspace(xmin, xmax, n, device=device, dtype=dtype)
        if normalize == "neg1_1":
            mid = (xmax + xmin) / 2.0
            half = (xmax - xmin) / 2.0
            half = half if half != 0 else 1.0
            lin = (lin - mid) / half
        elif normalize == "zero_one":
            span = xmax - xmin
            span = span if span != 0 else 1.0
            lin = (lin - xmin) / span
        else:
            raise ValueError(f"Unknown normalize={normalize!r}")
        axes.append(lin)

    if len(shape) == 1:
        return axes[0].unsqueeze(0)
    if len(shape) == 2:
        h, w = shape
        # meshgrid: row = y, col = x → channels [x, y] to match common conv layout
        gy = axes[0].view(h, 1).expand(h, w)
        gx = axes[1].view(1, w).expand(h, w)
        return torch.stack((gx, gy), dim=0)
    raise NotImplementedError(f"Only 1D/2D grids supported; got shape {shape}")
