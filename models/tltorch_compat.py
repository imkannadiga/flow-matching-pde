"""
tensorly-torch often calls ``torch.empty(..., device=device, dtype=dtype)`` with
``device=None`` / ``dtype=None``. Some PyTorch builds reject those explicit Nones.

We (1) wrap ``torch.empty`` to drop None ``dtype`` / ``device`` kwargs and
(2) patch ``DenseTensor.new`` so the rare ``dtype=None`` path defaults to
``torch.cfloat`` (spectral weights).

Call ``apply_spectral_weight_empty_compat()`` as early as possible (see ``models/__init__.py``
and ``import models`` in training entry points).
"""

from __future__ import annotations

import warnings

import torch
from torch import nn

_TORCH_EMPTY_PATCHED = False
_FACTORIZED_PATCHED = False


def _apply_torch_empty_drop_explicit_none() -> None:
    """Strip ``dtype=None`` / ``device=None`` so PyTorch uses its defaults."""
    global _TORCH_EMPTY_PATCHED
    if _TORCH_EMPTY_PATCHED:
        return
    _orig = torch.empty

    def _empty(*args, **kwargs):
        if kwargs:
            kwargs = dict(kwargs)
            if kwargs.get("dtype") is None and "dtype" in kwargs:
                del kwargs["dtype"]
            if kwargs.get("device") is None and "device" in kwargs:
                del kwargs["device"]
        return _orig(*args, **kwargs)

    torch.empty = _empty  # type: ignore[assignment]
    _TORCH_EMPTY_PATCHED = True


def _parameter_empty(shape, *, device=None, dtype=None) -> nn.Parameter:
    if dtype is None:
        dtype = torch.cfloat
    if device is None:
        t = torch.empty(shape, dtype=dtype)
    else:
        t = torch.empty(shape, device=device, dtype=dtype)
    return nn.Parameter(t)


def _patch_dense_tensor_new() -> None:
    global _FACTORIZED_PATCHED
    if _FACTORIZED_PATCHED:
        return
    try:
        from tltorch.factorized_tensors.factorized_tensors import DenseTensor
    except ImportError as e:
        warnings.warn(
            f"tltorch_compat: could not import DenseTensor ({e!r}); "
            "relying on torch.empty shim only.",
            stacklevel=2,
        )
        return

    if getattr(DenseTensor.new, "_flow_matching_pde_patched", False):
        _FACTORIZED_PATCHED = True
        return

    @classmethod
    def _dense_new(cls, shape, rank=None, device=None, dtype=None, **kwargs):
        return cls(_parameter_empty(shape, device=device, dtype=dtype))

    _dense_new._flow_matching_pde_patched = True
    DenseTensor.new = _dense_new
    _FACTORIZED_PATCHED = True


def apply_spectral_weight_empty_compat() -> None:
    """Idempotent: safe to call multiple times."""
    _apply_torch_empty_drop_explicit_none()
    _patch_dense_tensor_new()

