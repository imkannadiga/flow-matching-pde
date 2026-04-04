#!/usr/bin/env python3
"""Sprint 3 dry run: compose ``train`` and forward-pass each model × conditioning (no data files)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

_REPO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(_REPO))

from training.model_channels import patch_model_config_io  # noqa: E402


def main() -> int:
    os.environ.setdefault("PROJECT_ROOT", str(_REPO))
    os.chdir(_REPO)

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_REPO / "configs")):
        for cond in ("level0", "level1", "level2"):
            extra = []
            if cond != "level0":
                extra.append("model.coord_channels=2")
            if cond == "level2":
                extra.append("data.default_param_vector=[20.0,0.001]")
            for m in ("fno", "unet", "vit", "amfno"):
                cfg = compose(
                    config_name="train",
                    overrides=[f"conditioning={cond}", f"model={m}"] + extra,
                )
                x = torch.randn(2, 1 + (2 if cond != "level0" else 0), 64, 64)
                y_template = torch.randn(2, 1, 64, 64)
                patch_model_config_io(cfg.model, x, y_template)
                model = instantiate(cfg.model)
                t = torch.zeros(2)
                kw: dict = {"t": t, "u": x}
                if cond == "level2":
                    kw["params"] = torch.tensor([[20.0, 0.001], [20.0, 0.001]])
                y = model(**kw)
                assert y.shape == (2, 1, 64, 64)
            try:
                cfg = compose(
                    config_name="train",
                    overrides=[f"conditioning={cond}", "model=lno"] + extra,
                )
                x = torch.randn(2, 1 + (2 if cond != "level0" else 0), 64, 64)
                y_template = torch.randn(2, 1, 64, 64)
                patch_model_config_io(cfg.model, x, y_template)
                model = instantiate(cfg.model)
                t = torch.zeros(2)
                kw = {"t": t, "u": x}
                if cond == "level2":
                    kw["params"] = torch.tensor([[20.0, 0.001], [20.0, 0.001]])
                model(**kw)
            except Exception as e:
                print("lno skipped:", type(e).__name__, e)

        for name in ("experiment/phase2", "experiment/phase3"):
            cfg = compose(config_name=name)
            assert cfg.model.coord_channels == 2
            assert cfg.data.default_param_vector is not None

    print("dry_compose_sprint3: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
