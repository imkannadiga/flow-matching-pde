from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import hydra
import torch
from accelerate import Accelerator
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split



def _abspath_from_project_root(path_like: str) -> str:
    path = Path(path_like)
    if path.is_absolute():
        return str(path)
    return str(Path(get_original_cwd()) / path)


def _resolve_config(cfg: DictConfig) -> DictConfig:
    OmegaConf.resolve(cfg)
    return cfg


def _build_loaders(cfg: DictConfig, dataset: Dataset) -> Tuple[DataLoader, Dict[str, DataLoader]]:
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
    return train_loader, {"val": val_loader}


def _infer_model_channels(raw_batch: Dict[str, torch.Tensor], processed_batch: Dict[str, torch.Tensor]) -> Dict[str, int]:
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
    else:
        if x_processed.dim() != 4:
            raise ValueError("Expected processed x to be 4D: [B, C, H, W].")
        in_channels = int(x_processed.shape[1])

    # Keep backward compatibility for models that still consume in/vis channels.
    if "x" not in raw_batch or not torch.is_tensor(raw_batch["x"]) or raw_batch["x"].dim() != 4:
        vis_channels = in_channels
    else:
        vis_channels = int(raw_batch["x"].shape[1])

    return {"in_channels": in_channels, "vis_channels": vis_channels, "out_channels": int(y.shape[1])}


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = _resolve_config(cfg)
    accelerator_cfg = cfg.accelerate

    if "data_path" in cfg.data:
        cfg.data.data_path = _abspath_from_project_root(cfg.data.data_path)

    accelerator = Accelerator(
        mixed_precision=str(accelerator_cfg.mixed_precision),
        cpu=bool(accelerator_cfg.cpu),
        gradient_accumulation_steps=int(cfg.trainer.gradient_accumulation_steps),
        log_with=accelerator_cfg.log_with,
        project_dir=str(accelerator_cfg.project_dir),
        dynamo_backend=accelerator_cfg.dynamo_backend,
    )

    dataset = instantiate(cfg.data)
    train_loader, test_loaders = _build_loaders(cfg, dataset)
    pre_train_processor = instantiate(cfg.trainer.pre_train_processor)
    first_batch = next(iter(train_loader))
    processed_first_batch = pre_train_processor.preprocess(dict(first_batch))
    inferred = _infer_model_channels(first_batch, processed_first_batch)
    model = instantiate(cfg.model, **inferred)
    optimizer = instantiate(cfg.trainer.optimizer)(model.parameters())
    scheduler = instantiate(cfg.trainer.scheduler)(optimizer=optimizer)
    loss = instantiate(cfg.trainer.loss)

    prepared = accelerator.prepare(
        model,
        optimizer,
        scheduler,
        train_loader,
        *test_loaders.values(),
    )
    model, optimizer, scheduler, train_loader, *prepared_test_values = prepared
    test_loaders = dict(zip(test_loaders.keys(), prepared_test_values))

    trainer = instantiate(
        cfg.trainer,
        model=model,
        pre_train_processor=pre_train_processor,
        accelerator=accelerator,
    )

    metrics = trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=loss,
        eval_losses={"loss": loss},
        save_every=cfg.trainer.save_every,
        save_dir=cfg.trainer.save_path,
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


if __name__ == "__main__":
    main()
