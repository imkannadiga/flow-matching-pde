"""
Stack optional conditioning onto a core field map so the model sees one wide ``u`` tensor.

The **loss** (e.g. flow-matching target) stays on the core field only; extra maps are not
supervised. Typical batch keys (all optional except the core tensor passed in):

- ``x_aux``: extra maps already on the grid ``(B, C_aux, *spatial)``
- ``channel_mins``, ``channel_maxs``: per-core-channel bounds ``(B, C_core)``, broadcast
  to ``(B, C_core, *spatial)`` and concatenated (2 * C_core channels)
- ``params``: ``(B, P)`` global scalars expanded to ``(B, P, *spatial)``
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def _spatial_dims(x: torch.Tensor) -> Tuple[int, ...]:
    return tuple(x.shape[2:])


def expand_params_to_spatial(
    params: torch.Tensor, spatial: Tuple[int, ...]
) -> torch.Tensor:
    """``(B, P)`` → ``(B, P, *spatial)`` with constant values on the grid."""
    if params.dim() != 2:
        raise ValueError(f"params must be (B, P); got {tuple(params.shape)}")
    b, p = params.shape
    v = params.reshape(b, p, *([1] * len(spatial)))
    return v.expand(b, p, *spatial)


def expand_channel_bounds(
    channel_mins: torch.Tensor,
    channel_maxs: torch.Tensor,
    spatial: Tuple[int, ...],
    n_core_channels: int,
) -> torch.Tensor:
    """
    ``channel_*`` of shape ``(B, C)`` with ``C == n_core_channels`` →
    ``(B, 2 * C, *spatial)`` (all mins then all maxs as constant maps).
    """
    if channel_mins.shape != channel_maxs.shape:
        raise ValueError(
            f"channel_mins {tuple(channel_mins.shape)} != channel_maxs {tuple(channel_maxs.shape)}"
        )
    if channel_mins.dim() != 2 or channel_mins.shape[1] != n_core_channels:
        raise ValueError(
            f"Expected channel_mins (B, {n_core_channels}); got {tuple(channel_mins.shape)}"
        )
    b, c = channel_mins.shape
    mins = channel_mins.reshape(b, c, *([1] * len(spatial))).expand(b, c, *spatial)
    maxs = channel_maxs.reshape(b, c, *([1] * len(spatial))).expand(b, c, *spatial)
    return torch.cat([mins, maxs], dim=1)


def _assert_spatial(x: torch.Tensor, spatial: Tuple[int, ...], name: str) -> torch.Tensor:
    if tuple(x.shape[2:]) != spatial:
        raise ValueError(
            f"{name} has spatial {tuple(x.shape[2:])} but core field has {spatial}"
        )
    return x


def stack_additional_input_channels(
    u_core: torch.Tensor,
    sample: Dict[str, Any],
    *,
    stack_x_aux: bool = True,
    stack_channel_bounds: bool = True,
    stack_params_into_u: bool = True,
) -> torch.Tensor:
    """
    Concatenate optional conditioning along channel dim after ``u_core`` (flow / noisy field).

    Parameters
    ----------
    u_core
        Typically ``x_τ`` or noisy state with shape ``(B, C_core, *spatial)``.
    sample
        Batch dict (already on device) that may contain ``x_aux``, ``channel_mins``,
        ``channel_maxs``, ``params``.
    """
    spatial = _spatial_dims(u_core)
    parts = [u_core]
    b = u_core.shape[0]
    c_core = u_core.shape[1]

    if stack_x_aux and sample.get("x_aux") is not None:
        xa = sample["x_aux"]
        if not torch.is_tensor(xa):
            raise TypeError("x_aux must be a tensor")
        xa = _assert_spatial(xa.to(device=u_core.device, dtype=u_core.dtype), spatial, "x_aux")
        if xa.shape[0] != b:
            raise ValueError(f"x_aux batch {xa.shape[0]} != {b}")
        parts.append(xa)

    if stack_channel_bounds and (
        "channel_mins" in sample or "channel_maxs" in sample
    ):
        if "channel_mins" not in sample or "channel_maxs" not in sample:
            raise ValueError("Both channel_mins and channel_maxs are required if either is set")
        cm = sample["channel_mins"]
        cx = sample["channel_maxs"]
        if not torch.is_tensor(cm) or not torch.is_tensor(cx):
            raise TypeError("channel_mins / channel_maxs must be tensors")
        cm = cm.to(device=u_core.device, dtype=u_core.dtype)
        cx = cx.to(device=u_core.device, dtype=u_core.dtype)
        parts.append(expand_channel_bounds(cm, cx, spatial, c_core))

    if stack_params_into_u and sample.get("params") is not None:
        p = sample["params"]
        if not torch.is_tensor(p):
            raise TypeError("params must be a tensor")
        p = p.to(device=u_core.device, dtype=u_core.dtype)
        if p.dim() != 2 or p.shape[0] != b:
            raise ValueError(f"params must be (B, P) with B={b}; got {tuple(p.shape)}")
        parts.append(expand_params_to_spatial(p, spatial))

    return torch.cat(parts, dim=1) if len(parts) > 1 else u_core
