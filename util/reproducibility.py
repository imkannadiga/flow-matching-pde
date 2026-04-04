"""Reproducibility helpers: seeds and resolved-config hashing."""

from __future__ import annotations

import hashlib
import random
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
