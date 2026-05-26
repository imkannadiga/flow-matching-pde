from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from evaluation.metrics import MetricTracker
from evaluation.sample_store import SampleStore
from evaluation.solvers import FixedStepSolver, ODESolver

from tqdm import tqdm, trange


def _update_step_accum(
    accum: dict[int, dict[str, list]],
    t_step: int,
    pred: Tensor,
    true: Tensor,
) -> None:
    if t_step not in accum:
        accum[t_step] = {
            "rel_l2": [0.0, 0],
            "mse":    [0.0, 0],
            "l_inf":  [0.0, 0],
        }
    entry = accum[t_step]
    n = pred.shape[0]

    diff_norm = (pred - true).flatten(1).norm(dim=1)
    true_norm = true.flatten(1).norm(dim=1).clamp(min=1e-8)
    entry["rel_l2"][0] += (diff_norm / true_norm).sum().item()
    entry["rel_l2"][1] += n

    mse = ((pred - true) ** 2).flatten(1).mean(dim=1)
    entry["mse"][0] += mse.sum().item()
    entry["mse"][1] += n

    linf = (pred - true).abs().flatten(1).amax(dim=1)
    entry["l_inf"][0] += linf.sum().item()
    entry["l_inf"][1] += n


class FlowMatchingEvaluator:
    """Evaluates a trained flow matching model via ODE integration from Gaussian noise.

    For time-invariant PDEs (e.g. Darcy), ``evaluate_dataset`` runs one ODE integration
    per sample starting from fresh noise, conditioned on batch["x"].

    For time-variant PDEs (e.g. SWE), set ``time_variate=True`` to enable auto-regressive
    rollout: at each physical step, fresh noise is integrated with conditioning built by
    concatenating the current predicted state with the extra conditions from the dataset.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        solver: ODESolver,
        metric_tracker: MetricTracker,
        sample_store: Optional[SampleStore] = None,
        accelerator=None,
        store_rollout: bool = False,
        time_variate: bool = False,
    ):
        self.model = model
        self.model.eval()
        self.solver = solver
        self.metrics = metric_tracker
        self.sample_store = sample_store
        self.accelerator = accelerator
        self.store_rollout = store_rollout
        self.time_variate = time_variate

    @property
    def device(self) -> torch.device:
        if self.accelerator is not None:
            return self.accelerator.device
        p = next(iter(self.model.parameters()), None)
        return p.device if p is not None else torch.device("cpu")

    def _make_model_fn(self, condition: Optional[Tensor]):
        def model_fn(t: float, x: Tensor) -> Tensor:
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
            return self.model(u=x, cond=condition, t=t_tensor)
        return model_fn

    @torch.no_grad()
    def _integrate(
        self,
        condition: Optional[Tensor],
        noise: Tensor,
        collect_rollout: bool = False,
    ) -> tuple[Tensor, Optional[list[Tensor]]]:
        """Integrate from noise (t=0) to predicted state (t=1).

        Parameters
        ----------
        condition : Tensor or None
            Physical conditioning [B, C_cond, H, W].
        noise : Tensor
            Gaussian noise starting point [B, C_state, H, W].
        collect_rollout : bool
            Collect intermediate states (FixedStepSolver only).
        """
        model_fn = self._make_model_fn(condition)

        if collect_rollout and isinstance(self.solver, FixedStepSolver):
            x = noise
            rollout: list[Tensor] = []
            for t_cur, t_next in self.solver.get_time_steps():
                rollout.append(x.detach().cpu())
                x = self.solver.step(model_fn, x, t_cur, t_next)
            return x, rollout

        return self.solver.integrate(model_fn, noise), None

    @torch.no_grad()
    def evaluate_dataset(self, dataloader: DataLoader) -> dict:
        if self.time_variate:
            return self._evaluate_rollout(dataloader)
        return self._evaluate_one_step(dataloader)

    @torch.no_grad()
    def _evaluate_one_step(self, dataloader: DataLoader) -> dict:
        """One-step evaluation for time-invariant PDEs (e.g. Darcy).

        Batches must yield: "x" [B, C_cond, H, W], "y" [B, C_out, H, W].
        The ODE starts from fresh Gaussian noise with the same shape as "y".

        When the dataset provides a ``"dataset_id"`` field (a list of strings, one
        per sample), per-dataset metrics are accumulated alongside the aggregate
        metrics and returned under the ``"per_dataset"`` key:

        .. code-block:: python

            {
                "rel_l2": 0.042,        # aggregate
                "mse":    0.001,
                "per_dataset": {
                    "beta1.0": {"rel_l2": 0.038, "mse": 0.0009},
                    "beta0.5": {"rel_l2": 0.047, "mse": 0.0012},
                },
            }

        If no ``"dataset_id"`` is present the return value is the flat dict
        produced by ``metric_tracker.compute_summary()`` (unchanged behaviour).
        """
        self.metrics.reset()
        if self.sample_store is not None:
            self.sample_store.reset()

        collect = self.store_rollout and self.sample_store is not None

        # Per-dataset tracker registry: lazily populated on first encounter.
        per_dataset_trackers: dict[str, MetricTracker] = {}

        disable_pbar = self.accelerator is not None and not self.accelerator.is_local_main_process
        pbar = tqdm(dataloader, desc="Evaluating", unit="batch", disable=disable_pbar)

        for batch in pbar:
            condition = batch["x"].to(self.device)
            true_target = batch["y"].to(self.device)
            # dataset_ids is a list[str] of length B when the dataset supplies labels.
            dataset_ids: Optional[list[str]] = batch.get("dataset_id", None)
            noise = torch.randn_like(true_target)

            pred, rollout = self._integrate(condition, noise, collect_rollout=collect)

            if self.accelerator is not None:
                pred, true_target = self.accelerator.gather_for_metrics((pred, true_target))
                gathered_cond = (
                    self.accelerator.gather_for_metrics(condition)
                    if self.sample_store is not None
                    else None
                )
            else:
                gathered_cond = condition if self.sample_store is not None else None

            # --- aggregate metrics ---
            self.metrics.update(pred, true_target)

            # --- per-dataset metrics ---
            if dataset_ids is not None:
                # Group sample indices by label and update each tracker.
                groups: dict[str, list[int]] = {}
                for i, label in enumerate(dataset_ids):
                    groups.setdefault(label, []).append(i)

                for label, indices in groups.items():
                    if label not in per_dataset_trackers:
                        per_dataset_trackers[label] = copy.deepcopy(self.metrics)
                        per_dataset_trackers[label].reset()
                    idx_t = torch.tensor(indices, device=pred.device)
                    per_dataset_trackers[label].update(
                        pred[idx_t], true_target[idx_t]
                    )

            if self.sample_store is not None:
                if self.accelerator is None or self.accelerator.is_main_process:
                    self.sample_store.maybe_store(
                        gathered_cond, pred, true_target, rollout, dataset_ids
                    )

        if self.sample_store is not None:
            is_main = self.accelerator is None or self.accelerator.is_main_process
            if is_main:
                self.sample_store.save()

        result = self.metrics.compute_summary()

        if per_dataset_trackers:
            result["per_dataset"] = {
                label: tracker.compute_summary()
                for label, tracker in sorted(per_dataset_trackers.items())
            }

        return result

    @torch.no_grad()
    def _evaluate_rollout(self, dataloader: DataLoader) -> dict[str, list[float]]:
        """Auto-regressive rollout evaluation for time-variant PDEs (e.g. SWE).

        Batches must yield:
          "x_0"          [B, 1, H, W]             initial physical state
          "conditions"   [B, T-1, C_extra, H, W]  extra conditioning per step (no state)
          "targets"      [B, T-1, C_out, H, W]    ground-truth next states

        At each physical step t the model receives:
          u=x_tau (from noise), cond=cat([current_state, conditions[:,t]]), t=tau
        matching the conditioning layout produced by _fetch_data_pair during training.
        """
        self.metrics.reset()
        all_pred_trajs: list[Tensor] = []
        all_true_trajs: list[Tensor] = []
        per_step_accum: dict[int, dict[str, list]] = {}

        disable_pbar = self.accelerator is not None and not self.accelerator.is_local_main_process
        pbar = tqdm(dataloader, desc="Evaluating rollout", unit="batch", disable=disable_pbar)

        for batch in pbar:
            x_0        = batch["x_0"].to(self.device)         # [B, 1, H, W]
            conditions = batch["conditions"].to(self.device)   # [B, T-1, C_extra, H, W]
            targets    = batch["targets"].to(self.device)      # [B, T-1, C_out, H, W]
            T_minus_1  = conditions.shape[1]

            current = x_0
            pred_traj = [current]

            for t in trange(T_minus_1, desc="Rollout steps", leave=False, disable=disable_pbar):
                extra_cond = conditions[:, t]                          # [B, C_extra, H, W]
                cond_t = torch.cat([current, extra_cond], dim=1)       # [B, 1 + C_extra, H, W]
                noise = torch.randn_like(current)
                current, _ = self._integrate(cond_t, noise)
                pred_traj.append(current)

            pred_stack = torch.stack(pred_traj, dim=1)  # [B, T, C_out, H, W]

            for t_step, step_pred in enumerate(pred_traj[1:]):
                step_true = targets[:, t_step]

                if self.accelerator is not None:
                    step_pred, step_true = self.accelerator.gather_for_metrics(
                        (step_pred, step_true)
                    )

                _update_step_accum(per_step_accum, t_step, step_pred, step_true)

            if self.accelerator is None or self.accelerator.is_main_process:
                all_pred_trajs.append(pred_stack.cpu())
                all_true_trajs.append(targets.cpu())

        if self.sample_store is not None:
            is_main = self.accelerator is None or self.accelerator.is_main_process
            if is_main:
                save_dir = self.sample_store.save_dir
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "pred": torch.cat(all_pred_trajs, dim=0),
                        "true": torch.cat(all_true_trajs, dim=0),
                    },
                    save_dir / "traj.pt",
                )

        per_step_metrics: dict[str, list[float]] = {}
        for t_step in sorted(per_step_accum):
            for metric_key, (s, c) in per_step_accum[t_step].items():
                per_step_metrics.setdefault(metric_key, []).append(s / max(c, 1))

        return per_step_metrics
