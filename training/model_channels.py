"""
Infer ``vis_channels`` / ``in_channels`` and ``out_channels`` from one preprocessed batch.

``out_channels`` is always ``y.shape[1]`` (supervised field width). Input width on ``u`` is
``u.shape[1]``; when ``coord_channels > 0``, the first ``u.shape[1] - coord_channels`` channels
are the non-coordinate part (matches FNO / ViT / U-Net conventions).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict


def _coord_channels(cfg_model: DictConfig) -> int:
    return int(OmegaConf.select(cfg_model, "coord_channels", default=0))


def _target(cfg_model: DictConfig) -> str:
    return str(cfg_model.get("_target_", ""))


def patch_model_config_io(cfg_model: DictConfig, u: torch.Tensor, y: torch.Tensor) -> None:
    """Write ``vis_channels`` or ``in_channels`` and ``out_channels`` into ``cfg_model``."""
    t = _target(cfg_model)
    u_ch, y_ch = int(u.shape[1]), int(y.shape[1])
    cc = _coord_channels(cfg_model)
    field_in = u_ch - cc if cc > 0 else u_ch

    with open_dict(cfg_model):
        if t.endswith("models.fno.FNO"):
            cfg_model["vis_channels"] = field_in
            cfg_model["out_channels"] = y_ch
        elif t.endswith("models.lno.LFNO"):
            cfg_model["vis_channels"] = field_in
            cfg_model["out_channels"] = y_ch
        elif t.endswith("models.amfno.AMFNO"):
            cfg_model["vis_channels"] = field_in
            cfg_model["out_channels"] = y_ch
        elif t.endswith("models.unet.UNet"):
            cfg_model["in_channels"] = field_in
            cfg_model["out_channels"] = y_ch
        elif t.endswith("models.vit.FieldViT"):
            cfg_model["vis_channels"] = field_in
            cfg_model["out_channels"] = y_ch
        else:
            raise ValueError(
                f"Cannot infer I/O channels for model _target_={t!r}; "
                f"extend training/model_channels.py if you add a new backbone."
            )


def infer_model_shapes_from_data(
    cfg: DictConfig,
    device: torch.device,
    train_loader: Optional[Any] = None,
) -> None:
    """
    Run one training batch through the data processor and patch ``cfg.model``.

    Uses ``cfg.data`` (unless ``train_loader`` is passed), and
    ``trainer.pre_train_processor`` or ``paradigm.pre_train_processor``.
    """
    proc_cfg: Optional[DictConfig] = OmegaConf.select(
        cfg, "trainer.pre_train_processor", default=None
    )
    if proc_cfg is None:
        proc_cfg = OmegaConf.select(cfg, "paradigm.pre_train_processor", default=None)
    if proc_cfg is None:
        raise ValueError(
            "Channel inference needs trainer.pre_train_processor or paradigm.pre_train_processor."
        )

    if train_loader is None:
        data = instantiate(cfg.data)
        train_loader, _ = data.get_dataloaders()
    processor = instantiate(proc_cfg).to(device)
    batch = next(iter(train_loader))
    sample = processor.preprocess(batch)
    u = sample["x"]["u"]
    yt = sample["y"]
    patch_model_config_io(cfg.model, u, yt)
