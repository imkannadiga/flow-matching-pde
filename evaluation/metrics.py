from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class MetricTracker(ABC):
    """Accumulates per-batch values and computes an aggregate summary at the end."""

    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, pred: Tensor, true: Tensor) -> None:
        """Accumulate metrics from one batch (already gathered across processes)."""

    @abstractmethod
    def compute_summary(self) -> dict[str, float]:
        """Return aggregated metric dict."""


class RelativeL2Tracker(MetricTracker):
    """Relative L2 error: ``||pred - true||_2 / ||true||_2``, averaged over the batch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def update(self, pred: Tensor, true: Tensor) -> None:
        diff_norm = (pred - true).flatten(1).norm(dim=1)
        true_norm = true.flatten(1).norm(dim=1).clamp(min=1e-8)
        self._sum += (diff_norm / true_norm).sum().item()
        self._count += pred.shape[0]

    def compute_summary(self) -> dict[str, float]:
        return {"rel_l2": self._sum / max(self._count, 1)}


class MSETracker(MetricTracker):
    """Mean squared error per sample, averaged over the dataset."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def update(self, pred: Tensor, true: Tensor) -> None:
        mse = ((pred - true) ** 2).flatten(1).mean(dim=1)
        self._sum += mse.sum().item()
        self._count += pred.shape[0]

    def compute_summary(self) -> dict[str, float]:
        return {"mse": self._sum / max(self._count, 1)}


class LinfTracker(MetricTracker):
    """Maximum absolute error per sample (L-infinity norm), averaged over the dataset."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def update(self, pred: Tensor, true: Tensor) -> None:
        linf = (pred - true).abs().flatten(1).amax(dim=1)
        self._sum += linf.sum().item()
        self._count += pred.shape[0]

    def compute_summary(self) -> dict[str, float]:
        return {"l_inf": self._sum / max(self._count, 1)}


class CompositeMetricTracker(MetricTracker):
    """Wraps a list of MetricTrackers into one.

    Hydra instantiates the inner trackers from a list of ``_target_`` dicts:

    .. code-block:: yaml

        metric_tracker:
          _target_: evaluation.metrics.CompositeMetricTracker
          trackers:
            - _target_: evaluation.metrics.RelativeL2Tracker
            - _target_: evaluation.metrics.MSETracker
            - _target_: evaluation.metrics.LinfTracker
    """

    def __init__(self, trackers: list[MetricTracker]):
        self.trackers = list(trackers)

    def reset(self) -> None:
        for t in self.trackers:
            t.reset()

    def update(self, pred: Tensor, true: Tensor) -> None:
        for t in self.trackers:
            t.update(pred, true)

    def compute_summary(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for t in self.trackers:
            result.update(t.compute_summary())
        return result
