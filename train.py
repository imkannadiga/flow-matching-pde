from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import torch
from accelerate import Accelerator
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from training.testing import BaseTest, LossTest, RolloutTest


def _abspath_from_project_root(path_like: str) -> str:
    path = Path(path_like)
    if path.is_absolute():
        return str(path)
    return str(Path(get_original_cwd()) / path)


def _resolve_config(cfg: DictConfig) -> DictConfig:
    OmegaConf.resolve(cfg)
    return cfg


def _build_loaders(
    cfg: DictConfig,
    dataset: Dataset,
    trajectory: bool = False,
) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    """Build train and test dataloaders from a single dataset instance.

    When ``trajectory=True`` the function also instantiates a second copy of
    the dataset in eval mode (full trajectories) and returns an additional
    ``"val_traj"`` loader sized to ``cfg.data.rollout_val_samples`` trajectories
    with its own ``cfg.data.rollout_val_batch_size``.  The trajectory val split
    uses the same seed as the pair-based split so results are comparable, but
    the split is over trajectory count (not flattened pairs).
    """
    batch_size = int(cfg.data.batch_size)
    num_workers = int(cfg.data.num_workers)
    val_fraction = float(cfg.data.val_fraction)
    split_seed = int(cfg.data.split_seed)

    dataset_len = len(dataset)
    val_size = int(dataset_len * val_fraction)
    train_size = dataset_len - val_size

    if val_size > 0:
        generator = torch.Generator().manual_seed(split_seed)
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_set = dataset
        val_set = dataset

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loaders: Dict[str, DataLoader] = {"val": val_loader}

    if trajectory:
        # Instantiate a separate dataset in eval mode (returns full trajectories).
        # Force preload=False regardless of the main config — we only need a small
        # subset, so loading everything to RAM is wasteful.
        traj_data_cfg = OmegaConf.merge(cfg.data, {"eval": True, "preload": False})
        traj_dataset = instantiate(traj_data_cfg)

        # Split at the trajectory level with the same seed for consistency.
        traj_len = len(traj_dataset)
        traj_val_size = int(traj_len * val_fraction)
        traj_train_size = traj_len - traj_val_size

        traj_gen = torch.Generator().manual_seed(split_seed)
        _, traj_val_set = random_split(
            traj_dataset, [traj_train_size, traj_val_size], generator=traj_gen
        )

        # Cap at the configured number of rollout trajectories.
        n_rollout = min(int(cfg.data.rollout_val_samples), len(traj_val_set))
        rollout_set = Subset(traj_val_set, list(range(n_rollout)))

        rollout_batch_size = int(cfg.data.rollout_val_batch_size)
        traj_loader = DataLoader(
            rollout_set,
            batch_size=rollout_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loaders["val_traj"] = traj_loader

    return train_loader, test_loaders


def _build_tests(
    cfg: DictConfig,
    test_loaders: Dict[str, DataLoader],
    loss_fn: torch.nn.Module,
    pre_train_processor,
    accelerator: Accelerator,
    model: torch.nn.Module,
) -> List[BaseTest]:
    """Construct the list of tests to run at each eval interval.

    Tests are derived from which loaders are available:
    - ``"val"``      → LossTest (always present)
    - ``"val_traj"`` → RolloutTest (present when data.rollout_val=true)

    The rollout solver is read from ``cfg.trainer.rollout_eval.solver`` when
    present; otherwise defaults to EulerSolver with 20 steps.
    """
    from evaluation.evaluator import FlowMatchingEvaluator
    from evaluation.metrics import CompositeMetricTracker, LinfTracker, MSETracker, RelativeL2Tracker

    device = accelerator.device
    tests: List[BaseTest] = []

    # --- Loss test (always) ---
    tests.append(LossTest(
        loader=test_loaders["val"],
        loss_fn=loss_fn,
        data_processor=pre_train_processor,
        device=device,
        accelerator=accelerator,
    ))

    # --- Rollout test (when trajectory loader was built) ---
    if "val_traj" in test_loaders:
        rollout_eval_cfg = OmegaConf.select(cfg, "trainer.rollout_eval", default=None)
        if rollout_eval_cfg is not None:
            solver = instantiate(rollout_eval_cfg.solver)
        else:
            from evaluation.solvers import EulerSolver
            solver = EulerSolver(n_steps=20)

        metric_tracker = CompositeMetricTracker([
            RelativeL2Tracker(), MSETracker(), LinfTracker(),
        ])
        evaluator = FlowMatchingEvaluator(
            model=model,
            solver=solver,
            metric_tracker=metric_tracker,
            accelerator=accelerator,
            time_variate=True,
        )
        tests.append(RolloutTest(
            loader=test_loaders["val_traj"],
            evaluator=evaluator,
        ))

    return tests


def _infer_model_channels(processed_batch: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Infer model channel sizes from the batch shape after processor preprocessing."""
    y = processed_batch["y"]
    if y.dim() != 4:
        raise ValueError("Expected processed y to be 4D: [B, C, H, W].")

    x_processed = processed_batch["x"]

    if isinstance(x_processed, dict):
        if "u" not in x_processed:
            raise ValueError("Expected processed x dict to contain key 'u'.")
        u = x_processed["u"]
        if u.dim() != 4:
            raise ValueError("Expected processed x['u'] to be 4D: [B, C, H, W].")
        in_channels = int(u.shape[1])
        cond_channels = int(x_processed["cond"].shape[1]) if (
            "cond" in x_processed and torch.is_tensor(x_processed["cond"])
        ) else 0
    else:
        if x_processed.dim() != 4:
            raise ValueError("Expected processed x to be 4D: [B, C, H, W].")
        in_channels = int(x_processed.shape[1])
        cond_channels = 0

    return {
        "in_channels": in_channels,
        "vis_channels": in_channels,
        "out_channels": int(y.shape[1]),
        "cond_channels": cond_channels,
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = _resolve_config(cfg)
    accelerator_cfg = cfg.accelerate

    if "data_path" in cfg.data and cfg.data.data_path is not None:
        cfg.data.data_path = _abspath_from_project_root(cfg.data.data_path)
    if "data_dir" in cfg.data and cfg.data.data_dir is not None:
        cfg.data.data_dir = _abspath_from_project_root(cfg.data.data_dir)

    accelerator = Accelerator(
        mixed_precision=str(accelerator_cfg.mixed_precision),
        cpu=bool(accelerator_cfg.cpu),
        gradient_accumulation_steps=int(cfg.trainer.gradient_accumulation_steps),
        log_with=accelerator_cfg.log_with,
        project_dir=str(accelerator_cfg.project_dir),
        dynamo_backend=accelerator_cfg.dynamo_backend,
    )

    dataset = instantiate(cfg.data)

    # Determine whether a trajectory (eval-mode) loader is also needed.
    needs_trajectory = (
        bool(OmegaConf.select(cfg, "data.time_variate", default=False))
        and bool(OmegaConf.select(cfg, "data.rollout_val", default=False))
    )
    train_loader, test_loaders = _build_loaders(cfg, dataset, trajectory=needs_trajectory)

    pre_train_processor = instantiate(cfg.trainer.pre_train_processor)
    first_batch = next(iter(train_loader))
    processed_first_batch = pre_train_processor.preprocess(dict(first_batch))
    inferred = _infer_model_channels(processed_first_batch)
    accelerator.print(
        f"Inferred channels — in: {inferred['in_channels']}, "
        f"cond: {inferred['cond_channels']}, "
        f"out: {inferred['out_channels']}, vis: {inferred['vis_channels']}"
    )

    model = instantiate(cfg.model, **inferred)
    optimizer = instantiate(cfg.trainer.optimizer)(model.parameters())
    scheduler = instantiate(cfg.trainer.scheduler)(optimizer=optimizer)
    loss = instantiate(cfg.trainer.loss)

    # Prepare all objects (model, optimizer, scheduler, all loaders) with accelerate.
    prepared = accelerator.prepare(
        model,
        optimizer,
        scheduler,
        train_loader,
        *test_loaders.values(),
    )
    model, optimizer, scheduler, train_loader, *prepared_test_values = prepared
    test_loaders = dict(zip(test_loaders.keys(), prepared_test_values))

    if OmegaConf.select(cfg, "wandb.use_wandb", default=False) and accelerator.is_main_process:
        try:
            import wandb
            from util.reproducibility import wandb_run_id, wandb_run_name, wandb_group
            wandb.init(
                project=OmegaConf.select(cfg, "wandb.project", default="flow-matching"),
                name=wandb_run_name(cfg),
                id=wandb_run_id(cfg),
                group=wandb_group(cfg),
                mode=OmegaConf.select(cfg, "wandb.mode", default="online"),
                config=OmegaConf.to_container(cfg, resolve=True),
                resume="allow",
            )
        except ImportError:
            pass

    # Build tests after accelerator.prepare() so loaders and model are on the right device.
    tests = _build_tests(cfg, test_loaders, loss, pre_train_processor, accelerator, model)

    trainer = instantiate(
        cfg.trainer,
        model=model,
        pre_train_processor=pre_train_processor,
        accelerator=accelerator,
        tests=tests,
    )

    metrics = trainer.train(
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=loss,
        save_every=cfg.trainer.save_every,
        save_dir=cfg.trainer.save_path,
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        (output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    if OmegaConf.select(cfg, "wandb.use_wandb", default=False) and accelerator.is_main_process:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
