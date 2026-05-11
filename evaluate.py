from __future__ import annotations

import json
import statistics
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split

from evaluation.evaluator import FlowMatchingEvaluator
from training.training_state import load_training_state


def _abspath_from_project_root(path_like: str) -> str:
    path = Path(path_like)
    if path.is_absolute():
        return str(path)
    return str(Path(get_original_cwd()) / path)


def _resolve_config(cfg: DictConfig) -> DictConfig:
    OmegaConf.resolve(cfg)
    return cfg


def _build_eval_loader(cfg: DictConfig, dataset) -> DataLoader:
    """Build the val split using the same seed as training for a consistent held-out set."""
    batch_size = int(cfg.data.batch_size)
    num_workers = int(cfg.data.num_workers)
    val_fraction = float(cfg.data.val_fraction)
    split_seed = int(cfg.data.split_seed)

    dataset_len = len(dataset)
    val_size = int(dataset_len * val_fraction)
    train_size = dataset_len - val_size

    if val_size > 0:
        generator = torch.Generator().manual_seed(split_seed)
        _, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        val_set = dataset

    return DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    cfg = _resolve_config(cfg)
    accelerator_cfg = cfg.accelerate

    if "data_path" in cfg.data and cfg.data.data_path is not None:
        cfg.data.data_path = _abspath_from_project_root(cfg.data.data_path)
    if "data_dir" in cfg.data and cfg.data.data_dir is not None:
        cfg.data.data_dir = _abspath_from_project_root(cfg.data.data_dir)

    time_variate = bool(cfg.data.get("time_variate", False))

    # Auto-enable dataset eval mode when running time-variate evaluation.
    # The user only needs to set time_variate=true; eval=true is implied.
    if time_variate:
        OmegaConf.update(cfg, "data.eval", True, merge=True)

    accelerator = Accelerator(
        mixed_precision=str(accelerator_cfg.mixed_precision),
        cpu=bool(accelerator_cfg.cpu),
        log_with=accelerator_cfg.log_with,
        project_dir=str(accelerator_cfg.project_dir),
        dynamo_backend=accelerator_cfg.dynamo_backend,
    )

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # --- Dataset & loader ---
    dataset = instantiate(cfg.data)
    eval_loader = _build_eval_loader(cfg, dataset)

    # --- Packer-aware channel inference ---
    packer = instantiate(cfg.evaluator.packer)
    first_batch = next(iter(eval_loader))

    if time_variate:
        # Batch keys: "x_0", "conditions" [B, T-1, C_extra, H, W], "targets", "time_schedule"
        x_0_sample   = first_batch["x_0"]               # [B, 1, H, W]
        cond_sample  = first_batch["conditions"][:, 0]  # [B, C_extra, H, W]
        full_cond    = torch.cat([x_0_sample, cond_sample], dim=1)
        dummy_packed = packer.pack(torch.zeros_like(x_0_sample), full_cond)
        in_channels  = int(dummy_packed.shape[1])
        out_channels = int(x_0_sample.shape[1])
    else:
        x_sample     = first_batch["x"]                 # [B, C_cond, H, W]
        y_sample     = first_batch["y"]                 # [B, C_out, H, W]
        dummy_packed = packer.pack(torch.zeros_like(y_sample), x_sample)
        in_channels  = int(dummy_packed.shape[1])
        out_channels = int(y_sample.shape[1])

    # --- Model ---
    model = instantiate(
        cfg.model,
        in_channels=in_channels,
        out_channels=out_channels,
        vis_channels=in_channels,
    )

    checkpoint_dir = _abspath_from_project_root(str(cfg.checkpoint_dir))
    checkpoint_name = str(cfg.checkpoint_name)
    model, *_ = load_training_state(checkpoint_dir, checkpoint_name, model)
    model.eval()

    # --- Evaluator components ---
    solver = instantiate(cfg.evaluator.solver)
    metric_tracker = instantiate(cfg.evaluator.metric_tracker)
    sample_store_cfg = cfg.evaluator.get("sample_store", None)
    sample_store = instantiate(sample_store_cfg) if sample_store_cfg is not None else None
    store_rollout = bool(cfg.evaluator.get("store_rollout", False))

    # --- Accelerate prepare ---
    model, eval_loader = accelerator.prepare(model, eval_loader)

    evaluator = FlowMatchingEvaluator(
        model=model,
        solver=solver,
        metric_tracker=metric_tracker,
        packer=packer,
        sample_store=sample_store,
        accelerator=accelerator,
        store_rollout=store_rollout,
        time_variate=time_variate,
    )

    accelerator.print(f"Evaluating on {len(eval_loader.dataset)} samples...")
    metrics = evaluator.evaluate_dataset(eval_loader)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print("\n=== Evaluation Results ===")
        if time_variate:
            for k, v in metrics.items():
                accelerator.print(
                    f"  {k}: mean={statistics.mean(v):.6f}  steps={len(v)}"
                )
        else:
            for k, v in metrics.items():
                accelerator.print(f"  {k}: {v:.6f}")

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        (output_dir / "resolved_config.yaml").write_text(
            OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
        )
        accelerator.print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
