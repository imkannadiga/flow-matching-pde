"""
Rollout and distribution metrics (thin wrappers around ``eval_utils`` where shared).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from evaluation.eval_utils import density_mse as eval_density_mse
from evaluation.eval_utils import spectra_mse


def rollout_mse(
    pred_seq: torch.Tensor, gt_seq: torch.Tensor
) -> torch.Tensor:
    """
    Per-step mean squared error over spatial dimensions.

    Parameters
    ----------
    pred_seq, gt_seq
        Shapes **[T, C, H, W]** or **[B, T, C, H, W]** (broadcasting batch if needed).
    """
    if pred_seq.dim() == 4:
        pred_seq = pred_seq.unsqueeze(0)
    if gt_seq.dim() == 4:
        gt_seq = gt_seq.unsqueeze(0)
    if pred_seq.shape != gt_seq.shape:
        raise ValueError(
            f"pred_seq and gt_seq must match; got {tuple(pred_seq.shape)} vs {tuple(gt_seq.shape)}"
        )
    # [B, T, C, H, W] -> [B, T]
    err = (pred_seq - gt_seq) ** 2
    return err.mean(dim=(2, 3, 4))


def spectrum_mse(
    pred: torch.Tensor, gt: torch.Tensor, s: Optional[int] = None
) -> float:
    """Energy-spectrum MSE (uses last spatial dim as resolution if ``s`` omitted)."""
    if s is None:
        s = int(pred.shape[-1])
    return float(spectra_mse(gt, pred))


def density_mse(pred: torch.Tensor, gt: torch.Tensor, **kwargs) -> float:
    return float(eval_density_mse(gt, pred, **kwargs))


def inference_time(
    sampler: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> Tuple[float, float]:
    """
    Wall-clock milliseconds per sampler call (CUDA events); returns ``(mean, std)``.
    """
    device = x0.device
    if device.type != "cuda":
        import time

        for _ in range(n_warmup):
            sampler(x0)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            sampler(x0)
            times.append((time.perf_counter() - t0) * 1000.0)
        t = torch.tensor(times)
        return float(t.mean()), float(t.std(unbiased=False))

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(n_warmup):
        sampler(x0)
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(n_runs):
        starter.record()
        sampler(x0)
        ender.record()
        torch.cuda.synchronize()
        elapsed.append(starter.elapsed_time(ender))
    t = torch.tensor(elapsed)
    return float(t.mean()), float(t.std(unbiased=False))
