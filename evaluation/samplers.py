"""
Inference samplers: all return predicted next state ``x1_hat`` with shape **[B, C, H, W]**.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np
import torch


def _as_batch_t(t_scalar: float, batch: int, device, dtype):
    return torch.full((batch,), t_scalar, device=device, dtype=dtype)


class EulerSampler:
    """Single Euler step for rectified flow: x1 = x0 + v(x0, t=0)."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    @torch.no_grad()
    def __call__(self, x0: torch.Tensor) -> torch.Tensor:
        b = x0.shape[0]
        t0 = torch.zeros(b, device=x0.device, dtype=x0.dtype)
        v = self.model(t=t0, u=x0)
        return x0 + v


class ODESampler:
    """Integrate dx/dt = v(x, t) from t=0 to t=1 with ``torchdiffeq``."""

    def __init__(
        self,
        model: torch.nn.Module,
        n_steps: int = 16,
        method: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ):
        self.model = model
        self.n_steps = max(2, int(n_steps))
        self.method = method
        self.rtol = rtol
        self.atol = atol

    @torch.no_grad()
    def __call__(self, x0: torch.Tensor) -> torch.Tensor:
        from torchdiffeq import odeint

        device, dtype = x0.device, x0.dtype

        def func(t_scalar, x):
            tb = _as_batch_t(float(t_scalar.item()), x.shape[0], device, dtype)
            return self.model(t=tb, u=x)

        t = torch.linspace(0.0, 1.0, self.n_steps, device=device, dtype=dtype)
        out = odeint(
            func,
            x0,
            t,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )
        return out[-1]


class ARSampler:
    """One deterministic forward pass at t=0 (autoregressive one-step)."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    @torch.no_grad()
    def __call__(self, x0: torch.Tensor) -> torch.Tensor:
        b = x0.shape[0]
        t0 = torch.zeros(b, device=x0.device, dtype=x0.dtype)
        return self.model(t=t0, u=x0)


class DDIMSampler:
    """
    DDIM-style deterministic update using an ε-predictor trained with VP diffusion.
    Uses the same cosine α(·) schedule as ``DiffusionDataProcessor`` (argument 1−τ).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_steps: int = 20,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.n_steps = max(2, int(n_steps))
        self._device = device

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos((t + 0.08) / 2.16 * np.pi).to(t)

    @torch.no_grad()
    def __call__(self, x0: torch.Tensor) -> torch.Tensor:
        # Start from nearly pure noise at effective τ=1 (matches training forward at high noise)
        dev = x0.device if self._device is None else self._device
        b, c = x0.shape[0], x0.shape[1]
        spatial = x0.shape[2:]
        n_sp = len(spatial)
        x = torch.randn(b, c, *spatial, device=dev, dtype=x0.dtype)
        taus = torch.linspace(1.0, 0.0, self.n_steps + 1, device=dev, dtype=x.dtype)
        for i in range(self.n_steps):
            tau_i = taus[i]
            tau_next = taus[i + 1]
            t_batch = torch.full((b,), tau_i, device=dev, dtype=x.dtype)
            eps_hat = self.model(t=t_batch, u=x)
            a_i = self._alpha(1.0 - tau_i)
            a_next = self._alpha(1.0 - tau_next)
            # Expand α to spatial dims
            for _ in range(n_sp):
                a_i = a_i.unsqueeze(-1)
                a_next = a_next.unsqueeze(-1)
            sig_i = torch.sqrt(torch.clamp(1.0 - a_i**2, min=1e-8))
            x0_pred = (x - sig_i * eps_hat) / torch.clamp(a_i, min=1e-8)
            sig_next = torch.sqrt(torch.clamp(1.0 - a_next**2, min=1e-8))
            x = a_next * x0_pred + sig_next * eps_hat
        return x


def build_sampler(
    paradigm: Literal["fm", "ar", "diffusion"],
    model: torch.nn.Module,
    sampler_type: Literal["euler", "ode", "ar", "ddim"] = "euler",
    ode_steps: int = 16,
    ddim_steps: int = 20,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if paradigm == "ar" or sampler_type == "ar":
        return ARSampler(model)
    if paradigm == "diffusion":
        sampler_type = "ddim"
    if sampler_type == "ddim":
        return DDIMSampler(model, n_steps=ddim_steps, device=next(model.parameters()).device)
    if sampler_type == "ode":
        return ODESampler(model, n_steps=ode_steps)
    return EulerSampler(model)
