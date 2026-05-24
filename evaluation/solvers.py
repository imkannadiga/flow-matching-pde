from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor


class ODESolver(ABC):
    """Integrates a learned vector field from t=0 to t=1."""

    @abstractmethod
    def integrate(self, model_fn: Callable[[float, Tensor], Tensor], x_0: Tensor) -> Tensor:
        """
        Parameters
        ----------
        model_fn : callable
            ``(t: float, x: Tensor) -> velocity Tensor`` — queries the model at a point.
        x_0 : Tensor
            Initial state at t=0, shape [B, C, H, W].

        Returns
        -------
        Tensor
            Final state at t=1, same shape as x_0.
        """


class FixedStepSolver(ODESolver):
    """Base for uniform fixed-step solvers. Subclasses only need to implement ``step()``."""

    def __init__(self, n_steps: int = 100):
        self.n_steps = int(n_steps)

    def get_time_steps(self) -> list[tuple[float, float]]:
        """Returns ``[(t_cur, t_next), ...]`` for a uniform grid over [0, 1]."""
        ts = torch.linspace(0.0, 1.0, self.n_steps + 1)
        return list(zip(ts[:-1].tolist(), ts[1:].tolist()))

    @abstractmethod
    def step(
        self,
        model_fn: Callable[[float, Tensor], Tensor],
        x: Tensor,
        t_cur: float,
        t_next: float,
    ) -> Tensor:
        """Advance state from t_cur to t_next."""

    def integrate(self, model_fn: Callable, x_0: Tensor) -> Tensor:
        x = x_0
        for t_cur, t_next in self.get_time_steps():
            x = self.step(model_fn, x, t_cur, t_next)
        return x


class EulerSolver(FixedStepSolver):
    """First-order Euler: x_next = x + v * dt. Fast baseline."""

    def step(self, model_fn, x, t_cur, t_next):
        return x + model_fn(t_cur, x) * (t_next - t_cur)


class RK4Solver(FixedStepSolver):
    """Classic 4th-order Runge-Kutta. 4x model evals per step, more accurate than Euler."""

    def step(self, model_fn, x, t_cur, t_next):
        dt = t_next - t_cur
        k1 = model_fn(t_cur, x)
        k2 = model_fn(t_cur + dt / 2, x + dt / 2 * k1)
        k3 = model_fn(t_cur + dt / 2, x + dt / 2 * k2)
        k4 = model_fn(t_next, x + dt * k3)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class DopriSolver(ODESolver):
    """Adaptive-step Dormand-Prince RK45 via torchdiffeq.

    Requires: ``pip install torchdiffeq``
    """

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-5):
        try:
            import torchdiffeq  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "DopriSolver requires torchdiffeq. Install it with: pip install torchdiffeq"
            ) from exc
        self.rtol = float(rtol)
        self.atol = float(atol)

    def integrate(self, model_fn: Callable, x_0: Tensor) -> Tensor:
        from torchdiffeq import odeint

        # dopri5 needs float32 precision — bf16/fp16 step sizes underflow to 0
        orig_dtype = x_0.dtype
        if orig_dtype != torch.float32:
            x_0 = x_0.float()

        t_span = torch.tensor([0.0, 1.0], device=x_0.device, dtype=torch.float32)

        def ode_fn(t, y):
            v = model_fn(t.item(), y.to(orig_dtype))
            return v.float()

        result = odeint(ode_fn, x_0, t_span, method="dopri5", rtol=self.rtol, atol=self.atol)
        return result[-1].to(orig_dtype)
