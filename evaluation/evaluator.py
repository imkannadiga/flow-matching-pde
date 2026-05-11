from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from evaluation.metrics import MetricTracker
from evaluation.packers import ConditionPacker
from evaluation.sample_store import SampleStore
from evaluation.solvers import FixedStepSolver, ODESolver


def _update_step_accum(
    accum: dict[int, dict[str, list]],
    t_step: int,
    pred: Tensor,
    true: Tensor,
) -> None:
    """Accumulate inline (sum, count) per-step metrics without MetricTracker instances.

    Replicates the formulas from RelativeL2Tracker, MSETracker, and LinfTracker so
    that per-step metrics can be accumulated across batches without storing all
    predictions in memory.
    """
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
    """Evaluates a trained flow matching model by ODE-integrating from noise to solution.

    Components (solver, packer, metrics, sample storage) are all independently swappable
    via Hydra config. Distributed evaluation across multiple GPUs/nodes is handled through
    the Accelerate instance.

    For time-invariant PDEs (e.g. Darcy), ``evaluate_dataset`` runs one ODE integration
    per dataset sample (teacher-forced one-step evaluation).

    For time-variant PDEs (e.g. SWE), set ``time_variate=True`` to enable auto-regressive
    rollout: each model prediction is fed back as ``x_0`` and the state channel of the
    condition for the next physical time step. Per-step metrics are returned as lists.

    Usage
    -----
    Instantiated by ``evaluate.py`` via ``hydra.utils.instantiate``. The packer must match
    the conditioning strategy used during training, since it determines the model's input
    channel layout at each integration step.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        solver: ODESolver,
        metric_tracker: MetricTracker,
        packer: ConditionPacker,
        sample_store: Optional[SampleStore] = None,
        accelerator=None,
        store_rollout: bool = False,
        time_variate: bool = False,
    ):
        self.model = model
        self.model.eval()
        self.solver = solver
        self.metrics = metric_tracker
        self.packer = packer
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
        """Returns a closure ``(t: float, x: Tensor) -> velocity`` for use by solvers."""
        def model_fn(t: float, x: Tensor) -> Tensor:
            t_tensor = torch.full(
                (x.shape[0],), t, device=x.device, dtype=x.dtype
            )
            u = self.packer.pack(x, condition)
            return self.model(t_tensor, u)
        return model_fn

    @torch.no_grad()
    def generate_single_state(
        self,
        condition: Optional[Tensor],
        x_0: Tensor,
        collect_rollout: bool = False,
    ) -> tuple[Tensor, Optional[list[Tensor]]]:
        """Integrate from the physical initial condition (t=0) to a predicted solution (t=1).

        Parameters
        ----------
        condition : Tensor or None
            Physical conditioning field [B, C_cond, H, W].
        x_0 : Tensor
            Physical initial condition at t=0, supplied by the dataset via ``_make_x0()``.
        collect_rollout : bool
            If True and the solver is a FixedStepSolver, collect and return the
            intermediate state at the start of each step. Ignored for adaptive solvers.

        Returns
        -------
        final_state : Tensor
            Predicted solution at t=1, same shape as ``x_0``.
        rollout : list[Tensor] or None
            CPU-side intermediate states if ``collect_rollout=True`` and the solver
            supports step-by-step access, else None.
        """
        model_fn = self._make_model_fn(condition)

        if collect_rollout and isinstance(self.solver, FixedStepSolver):
            x = x_0
            rollout: list[Tensor] = []
            for t_cur, t_next in self.solver.get_time_steps():
                rollout.append(x.detach().cpu())
                x = self.solver.step(model_fn, x, t_cur, t_next)
            return x, rollout

        return self.solver.integrate(model_fn, x_0), None

    @torch.no_grad()
    def evaluate_dataset(self, dataloader: DataLoader) -> dict:
        """Run the full evaluation loop over a dataset.

        Dispatches to ``_evaluate_rollout`` when ``time_variate=True``, otherwise
        runs the standard one-step evaluation path.

        Returns
        -------
        dict
            ``time_variate=False``: ``{metric: float}`` aggregated over the dataset.
            ``time_variate=True``:  ``{metric: [float, ...]}`` one value per rollout step.
        """
        if self.time_variate:
            return self._evaluate_rollout(dataloader)
        return self._evaluate_one_step(dataloader)

    @torch.no_grad()
    def _evaluate_one_step(self, dataloader: DataLoader) -> dict[str, float]:
        """Standard one-step evaluation for time-invariant PDEs (e.g. Darcy).

        Batches must yield dicts with:
          - ``"x"``:   physical condition [B, C_cond, H, W]
          - ``"y"``:   ground-truth target [B, C, H, W]
          - ``"x_0"``: physical initial condition [B, C, H, W]
        """
        self.metrics.reset()
        if self.sample_store is not None:
            self.sample_store.reset()

        collect = self.store_rollout and self.sample_store is not None

        for batch in dataloader:
            condition = batch["x"].to(self.device)
            true_target = batch["y"].to(self.device)

            if "x_0" not in batch:
                raise KeyError(
                    "evaluate_dataset requires 'x_0' in each batch. "
                    "Override _make_x0() in your dataset class to provide the physical initial condition."
                )
            x_0 = batch["x_0"].to(self.device)

            pred, rollout = self.generate_single_state(
                condition, x_0, collect_rollout=collect
            )

            if self.accelerator is not None:
                pred, true_target = self.accelerator.gather_for_metrics((pred, true_target))
                gathered_cond = (
                    self.accelerator.gather_for_metrics(condition)
                    if self.sample_store is not None
                    else None
                )
            else:
                gathered_cond = condition if self.sample_store is not None else None

            self.metrics.update(pred, true_target)

            if self.sample_store is not None:
                if self.accelerator is None or self.accelerator.is_main_process:
                    self.sample_store.maybe_store(gathered_cond, pred, true_target, rollout)

        if self.sample_store is not None:
            is_main = self.accelerator is None or self.accelerator.is_main_process
            if is_main:
                self.sample_store.save()

        return self.metrics.compute_summary()

    @torch.no_grad()
    def _evaluate_rollout(self, dataloader: DataLoader) -> dict[str, list[float]]:
        """Auto-regressive rollout evaluation for time-variant PDEs (e.g. SWE).

        Batches must yield dicts with:
          - ``"x_0"``:          initial state [B, 1, H, W]
          - ``"conditions"``:   extra conditioning per step [B, T-1, C_extra, H, W]
                                (no state channel — evaluator prepends the prediction)
          - ``"targets"``:      ground-truth states [B, T-1, C_out, H, W]
          - ``"time_schedule"``: physical times [T-1] (unused in computation, stored for reference)

        At each physical step t the evaluator builds:
            full_condition = cat([current_pred, conditions[:, t]], dim=1)
        which matches the channel layout produced by _fetch_data_pair during training.
        The prediction becomes ``x_0`` for the next ODE integration.

        Returns
        -------
        dict[str, list[float]]
            Per-step metrics: ``{"rel_l2": [v_0, v_1, ...], "mse": [...], "l_inf": [...]}``.
            Each list has T-1 entries, one per physical time step.
        """
        self.metrics.reset()
        all_pred_trajs: list[Tensor] = []
        all_true_trajs: list[Tensor] = []
        per_step_accum: dict[int, dict[str, list]] = {}

        for batch in dataloader:
            x_0        = batch["x_0"].to(self.device)          # [B, 1, H, W]
            conditions = batch["conditions"].to(self.device)    # [B, T-1, C_extra, H, W]
            targets    = batch["targets"].to(self.device)       # [B, T-1, C_out, H, W]
            T_minus_1  = conditions.shape[1]

            current = x_0
            pred_traj = [current]

            for t in range(T_minus_1):
                cond_t = conditions[:, t]                               # [B, C_extra, H, W]
                full_condition = torch.cat([current, cond_t], dim=1)   # [B, C_cond, H, W]
                current, _ = self.generate_single_state(full_condition, current)
                pred_traj.append(current)

            pred_stack = torch.stack(pred_traj, dim=1)   # [B, T, C_out, H, W]

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
                        "pred": torch.cat(all_pred_trajs, dim=0),   # [N, T, C_out, H, W]
                        "true": torch.cat(all_true_trajs, dim=0),   # [N, T-1, C_out, H, W]
                    },
                    save_dir / "traj.pt",
                )

        per_step_metrics: dict[str, list[float]] = {}
        for t_step in sorted(per_step_accum):
            for metric_key, (s, c) in per_step_accum[t_step].items():
                per_step_metrics.setdefault(metric_key, []).append(s / max(c, 1))

        return per_step_metrics
