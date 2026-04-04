"""Reproducibility helpers: seeds, resolved-config hashing, and W&B run naming."""

from __future__ import annotations

import hashlib
import random
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch RNG seeds (CPU and CUDA if available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wandb_run_name(cfg: DictConfig) -> str:
    """
    Resolve a unique Weights & Biases run name.

    If ``cfg.wandb.run_name`` is set (non-null), returns that string (Hydra
    interpolations are resolved when the config is composed). Otherwise builds
    ``{model}-{data}-{experiment_or_task}-{timestamp}``, appending ``-j{num}`` in
    Hydra multirun when ``job.num`` is available.
    """
    from hydra.core.hydra_config import HydraConfig

    rn = OmegaConf.select(cfg, "wandb.run_name")
    if rn is not None:
        s = str(rn).strip()
        if s and s.lower() not in ("null", "~", "none"):
            return s

    mn = OmegaConf.select(cfg, "model.name", default="model")
    dn = OmegaConf.select(cfg, "data.name", default="data")
    ex = OmegaConf.select(cfg, "experiment_name")
    if ex is None:
        ex = OmegaConf.select(cfg, "task.name", default="task")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [str(mn), str(dn), str(ex), ts]
    if HydraConfig.initialized():
        jn = OmegaConf.select(HydraConfig.get(), "job.num", default=None)
        if jn is not None:
            parts.append(f"j{jn}")
    return "-".join(parts)


def wandb_group(cfg: DictConfig) -> Union[str, None]:
    """
    Optional W&B ``group`` so multirun jobs appear under one sweep in the UI.

    If ``cfg.wandb.group`` is set (non-null), returns that string. Otherwise, when
    Hydra multirun is active (``hydra.job.num`` is set), returns the resolved parent
    of ``runtime.output_dir`` (the directory shared by all jobs in that multirun).
    Single runs return ``None`` (no group).
    """
    from hydra.core.hydra_config import HydraConfig

    g = OmegaConf.select(cfg, "wandb.group")
    if g is not None:
        s = str(g).strip()
        if s and s.lower() not in ("null", "~", "none"):
            return s

    if not HydraConfig.initialized():
        return None
    hc = HydraConfig.get()
    jn = OmegaConf.select(hc, "job.num", default=None)
    if jn is None:
        return None
    return str(Path(hc.runtime.output_dir).resolve().parent)


def save_config_hash(cfg: DictConfig, run_dir: Union[str, Path]) -> str:
    """
    Write config_hash.txt containing sha256 of the resolved config YAML.

    Returns the hex digest.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml_str = OmegaConf.to_yaml(cfg, resolve=True)
    digest = hashlib.sha256(yaml_str.encode("utf-8")).hexdigest()
    (run_dir / "config_hash.txt").write_text(digest + "\n", encoding="utf-8")
    return digest
