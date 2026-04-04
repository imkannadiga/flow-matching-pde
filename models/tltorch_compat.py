"""
tensorly-torch builds spectral weights via ``torch.empty(shape, device=device, dtype=dtype)``.
Some PyTorch versions reject explicit ``dtype=None`` / ``device=None``; older neuraloperator
releases may omit ``dtype=`` so both stay None.

We patch FactorizedTensor ``.new`` helpers used by neuralop's SpectralConv before the FNO stack
is imported. Idempotent and no-op if tltorch is missing.
"""

from __future__ import annotations

import torch
from torch import nn

_APPLIED = False


def _parameter_empty(shape, *, device=None, dtype=None) -> nn.Parameter:
    if dtype is None:
        dtype = torch.cfloat
    if device is None:
        t = torch.empty(shape, dtype=dtype)
    else:
        t = torch.empty(shape, device=device, dtype=dtype)
    return nn.Parameter(t)


def apply_spectral_weight_empty_compat() -> None:
    global _APPLIED
    if _APPLIED:
        return
    try:
        import tensorly as tl

        tl.set_backend("pytorch")
        from tensorly.cp_tensor import validate_cp_rank
        from tensorly.tucker_tensor import validate_tucker_rank
        from tensorly.tt_tensor import validate_tt_rank
        from tltorch.factorized_tensors.factorized_tensors import (
            CPTensor,
            DenseTensor,
            TuckerTensor,
            TTTensor,
        )
    except ImportError:
        return

    @classmethod
    def _dense_new(cls, shape, rank=None, device=None, dtype=None, **kwargs):
        return cls(_parameter_empty(shape, device=device, dtype=dtype))

    @classmethod
    def _cp_new(cls, shape, rank, device=None, dtype=None, **kwargs):
        rank = validate_cp_rank(shape, rank)
        w = _parameter_empty((rank,), device=device, dtype=dtype)
        factors = [
            _parameter_empty((s, rank), device=device, dtype=dtype) for s in shape
        ]
        return cls(w, factors)

    @classmethod
    def _tucker_new(cls, shape, rank, fixed_rank_modes=None, device=None, dtype=None, **kwargs):
        rank = validate_tucker_rank(shape, rank, fixed_modes=fixed_rank_modes)
        core = _parameter_empty(rank, device=device, dtype=dtype)
        factors = [
            _parameter_empty((s, r), device=device, dtype=dtype)
            for (s, r) in zip(shape, rank)
        ]
        return cls(core, factors)

    @classmethod
    def _tt_new(cls, shape, rank, device=None, dtype=None, **kwargs):
        rank = validate_tt_rank(shape, rank)
        factors = [
            _parameter_empty((rank[i], s, rank[i + 1]), device=device, dtype=dtype)
            for i, s in enumerate(shape)
        ]
        return cls(factors)

    DenseTensor.new = _dense_new
    CPTensor.new = _cp_new
    TuckerTensor.new = _tucker_new
    TTTensor.new = _tt_new
    _APPLIED = True
