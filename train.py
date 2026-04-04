import json
import warnings
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from util.reproducibility import save_config_hash, set_seed


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    set_seed(int(cfg.seed))
    config_hash = save_config_hash(cfg, run_dir)

    data = instantiate(cfg.data)
    model = instantiate(cfg.model)

    param_count = sum(p.numel() for p in model.parameters())
    if param_count < 10_000_000 or param_count > 30_000_000:
        warnings.warn(
            f"Parameter count {param_count} is outside the suggested 10M–30M band "
            "for this benchmark (informational only).",
            stacklevel=1,
        )

    train_loader, test_loader = data.get_dataloaders()

    pre_train_processor = instantiate(cfg.trainer.pre_train_processor)

    _optimizer = instantiate(cfg.trainer.optimizer)
    optimizer = _optimizer(params=model.parameters())

    _scheduler = instantiate(cfg.trainer.scheduler)
    scheduler = _scheduler(optimizer=optimizer)

    loss_fn = instantiate(cfg.trainer.loss)
    regularizer = (
        instantiate(cfg.trainer.regularizer) if "regularizer" in cfg.trainer else None
    )

    if cfg.wandb.get("use_wandb", False):
        import wandb

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.model.name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            mode=cfg.wandb.mode,
        )

    trainer = instantiate(cfg.trainer, model=model)

    epoch_metrics = trainer.train(
        train_loader=train_loader,
        test_loaders={"64x64": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=regularizer,
        training_loss=loss_fn,
        eval_losses={"mse": loss_fn},
        save_every=cfg.trainer.save_every,
        save_dir=cfg.trainer.save_path,
        resume_from_dir=cfg.trainer.resume_from_dir
        if "resume_from_dir" in cfg.trainer
        else None,
    )

    if cfg.wandb.get("use_wandb", False):
        import wandb

        wandb.finish()

    def _json_safe(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        return obj

    eth = epoch_metrics.get("epoch_train_time")
    n_ep = int(cfg.trainer.n_epochs)
    train_gpu_hours = (
        float(eth) * n_ep / 3600.0 if eth is not None else None
    )

    results = {
        "config_hash": config_hash,
        "paradigm": OmegaConf.select(cfg, "paradigm.name", default="fm"),
        "model_name": cfg.model.name,
        "seed": int(cfg.seed),
        "param_count": int(param_count),
        "train_gpu_hours": train_gpu_hours,
        "final_metrics": _json_safe(epoch_metrics),
    }
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
