from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from evaluation.metrics import MetricTracker
from evaluation.packers import ConditionPacker
from evaluation.sample_store import SampleStore
from evaluation.solvers import FixedStepSolver, ODESolver


class FlowMatchingEvaluator:
    """Evaluates a trained flow matching model by ODE-integrating from noise to solution.

    Components (solver, packer, metrics, sample storage) are all independently swappable
    via Hydra config. Distributed evaluation across multiple GPUs/nodes is handled through
    the Accelerate instance.

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
    ):
        """
        Parameters
        ----------
        model : nn.Module
            Trained neural network. Set to eval mode on construction.
        solver : ODESolver
            Numerical integration algorithm (Euler, RK4, Dopri5, ...).
        metric_tracker : MetricTracker
            Accumulates and reports domain-specific errors.
        packer : ConditionPacker
            Builds the model input ``u`` from the current state + physical condition.
        sample_store : SampleStore, optional
            If provided, stores a random subset of predictions to disk.
        accelerator : Accelerator, optional
            HuggingFace Accelerate instance for distributed evaluation.
        store_rollout : bool
            Pass intermediate integration states to the sample store. Only meaningful
            when ``sample_store`` is set and ``sample_store.rollout_length`` is not None.
            Ignored (silently) for adaptive solvers that don't expose a ``step()`` method.
        """
        self.model = model
        self.model.eval()
        self.solver = solver
        self.metrics = metric_tracker
        self.packer = packer
        self.sample_store = sample_store
        self.accelerator = accelerator
        self.store_rollout = store_rollout

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
        target_shape: tuple,
        collect_rollout: bool = False,
    ) -> tuple[Tensor, Optional[list[Tensor]]]:
        """Integrate from Gaussian noise (t=0) to a predicted solution (t=1).

        Parameters
        ----------
        condition : Tensor or None
            Physical conditioning field [B, C_cond, H, W].
        target_shape : tuple
            Shape for the initial noise sample — should match the target tensor shape.
        collect_rollout : bool
            If True and the solver is a FixedStepSolver, collect and return the
            intermediate state at the start of each step. Ignored for adaptive solvers.

        Returns
        -------
        final_state : Tensor
            Predicted solution at t=1, shape ``target_shape``.
        rollout : list[Tensor] or None
            CPU-side intermediate states if ``collect_rollout=True`` and the solver
            supports step-by-step access, else None.
        """
        dtype = condition.dtype if condition is not None else torch.float32
        x_0 = torch.randn(target_shape, device=self.device, dtype=dtype)
        model_fn = self._make_model_fn(condition)

        # Step-by-step loop for rollout collection (fixed-step solvers only)
        if collect_rollout and isinstance(self.solver, FixedStepSolver):
            x = x_0
            rollout: list[Tensor] = []
            for t_cur, t_next in self.solver.get_time_steps():
                rollout.append(x.detach().cpu())
                x = self.solver.step(model_fn, x, t_cur, t_next)
            return x, rollout

        # Default: let solver own the full integration
        return self.solver.integrate(model_fn, x_0), None

    @torch.no_grad()
    def evaluate_dataset(self, dataloader: DataLoader) -> dict[str, float]:
        """Run the full evaluation loop over a dataset.

        Batches are pulled from ``dataloader`` which must yield dicts with:
          - ``"x"``: physical condition [B, C_cond, H, W]
          - ``"y"``: ground-truth target [B, C, H, W]

        Parameters
        ----------
        dataloader : DataLoader
            Prepared dataloader (already through ``accelerator.prepare``).

        Returns
        -------
        dict
            Aggregated metric summary across the full dataset.
        """
        self.metrics.reset()
        if self.sample_store is not None:
            self.sample_store.reset()

        collect = self.store_rollout and self.sample_store is not None

        for batch in dataloader:
            condition = batch["x"].to(self.device)
            true_target = batch["y"].to(self.device)

            pred, rollout = self.generate_single_state(
                condition, true_target.shape, collect_rollout=collect
            )

            # All gather_for_metrics calls must be made by every rank — they are collectives.
            # Gate only the *use* of gathered data on is_main_process, not the calls themselves.
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
