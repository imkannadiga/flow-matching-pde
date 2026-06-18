from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

wandb_available = False
try:
    import wandb
    wandb_available = True
except ModuleNotFoundError:
    pass


def _to_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


class BaseTest(ABC):
    """Abstract base for all training-time evaluation tests.

    Subclasses close over their dataloader and any dependencies at construction
    time.  The only runtime argument is the current model, so tests are fully
    self-contained and the trainer does not need to know their internals.
    """

    @property
    @abstractmethod
    def prefix(self) -> str:
        """Metric key prefix, e.g. ``'val'`` or ``'rollout'``."""
        ...

    @abstractmethod
    def run(self, model: nn.Module) -> dict[str, float | list[float]]:
        """Evaluate with the current model weights.

        Returns a metric dict whose keys follow ``"{prefix}/{metric_name}"``.
        Values are either scalars (``float``) or per-timestep lists
        (``list[float]``) for rollout-style tests.
        """
        ...


class LossTest(BaseTest):
    """Evaluates validation loss using the training data processor.

    Mirrors the training forward pass exactly so the reported val loss is
    directly comparable to the training loss.
    """

    def __init__(
        self,
        loader: DataLoader,
        loss_fn: nn.Module,
        data_processor,
        device: torch.device,
        accelerator=None,
        prefix: str = "val",
    ):
        self._prefix = prefix
        self.loader = loader
        self.loss_fn = loss_fn
        self.data_processor = data_processor
        self.device = device
        self.accelerator = accelerator

    @property
    def prefix(self) -> str:
        return self._prefix

    def run(self, model: nn.Module) -> dict[str, float]:
        model.eval()
        if self.data_processor is not None:
            self.data_processor.eval()

        is_main = self.accelerator is None or self.accelerator.is_main_process
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            pbar = tqdm(
                self.loader,
                desc=f"Test [{self._prefix}]",
                leave=False,
                unit="batch",
                disable=not is_main,
            )
            for sample in pbar:
                sample = self._prepare(sample)
                out = model(**sample["x"])
                if self.data_processor is not None:
                    out, sample = self.data_processor.postprocess(out, sample)
                loss = self.loss_fn(out, sample["y"])
                total_loss += loss.item()
                n_samples += sample["y"].shape[0]

        return {f"{self._prefix}/loss": total_loss / max(n_samples, 1)}

    def _prepare(self, sample: dict) -> dict:
        if self.data_processor is not None:
            return self.data_processor.preprocess(sample)
        return {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}


class RolloutTest(BaseTest):
    """ODE rollout evaluation via FlowMatchingEvaluator.

    Iterates the full rollout dataloader (all ``rollout_val_samples``
    trajectories) and returns per-timestep metrics averaged over all samples.
    Metric values are ``list[float]`` where index ``t`` corresponds to
    physical timestep ``t``.
    """

    def __init__(
        self,
        loader: DataLoader,
        evaluator,   # FlowMatchingEvaluator — typed loosely to avoid circular import
        prefix: str = "rollout",
    ):
        self._prefix = prefix
        self.loader = loader
        self.evaluator = evaluator

    @property
    def prefix(self) -> str:
        return self._prefix

    def run(self, model: nn.Module) -> dict[str, list[float]]:
        self.evaluator.model = model
        raw: dict[str, list[float]] = self.evaluator.evaluate_dataset(self.loader)
        return {f"{self._prefix}/{k}": v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def test_model(
    model: nn.Module,
    tests: list[BaseTest],
    epoch: int,
    wandb_log: bool = False,
    is_main: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run all registered tests, log results, and return the full metric dict.

    Scalar metrics are logged directly.  Per-timestep list metrics (rollout)
    are logged as individual ``{key}/t{t:03d}`` scalars plus a ``{key}/mean``
    summary.  The returned dict preserves the original list values so the
    caller can use them for checkpointing decisions.
    """
    all_metrics: dict[str, Any] = {}
    for test in tests:
        all_metrics.update(test.run(model))

    if verbose and is_main:
        _print_metrics(all_metrics)

    if wandb_log and is_main and wandb_available and wandb.run is not None:
        _log_to_wandb(all_metrics, step=epoch + 1)

    return all_metrics


def _print_metrics(metrics: dict[str, Any]) -> None:
    parts = []
    for k, v in metrics.items():
        if isinstance(v, list):
            mean = sum(v) / len(v) if v else float("nan")
            parts.append(f"{k}/mean={mean:.4f}")
        else:
            try:
                parts.append(f"{k}={_to_float(v):.4f}")
            except (TypeError, ValueError):
                pass
    if parts:
        print("Test: " + ", ".join(parts))
        sys.stdout.flush()


def _log_to_wandb(metrics: dict[str, Any], step: int) -> None:
    payload: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, list):
            for t, val in enumerate(v):
                payload[f"{k}/t{t:03d}"] = float(val)
            if v:
                payload[f"{k}/mean"] = sum(v) / len(v)
        else:
            try:
                payload[k] = _to_float(v)
            except (TypeError, ValueError):
                pass
    if payload:
        wandb.log(payload, step=step, commit=False)
