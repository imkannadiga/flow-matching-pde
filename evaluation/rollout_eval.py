"""
Trajectory-level rollout evaluation with per-step MSE and CUDA timing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from data.base import _train_test_trajectory_indices
from evaluation.eval_utils import load_model_from_manifest
from evaluation.metrics import density_mse, inference_time, spectrum_mse
from evaluation.results_logger import RESULT_FIELDS, append_row, load_train_results_json
from evaluation.samplers import build_sampler


class RolloutEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        dev = OmegaConf.select(cfg, "evaluate.device", default="cuda")
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        self.device = torch.device(dev)

        self.model = instantiate(cfg.model)
        ckpt = Path(
            OmegaConf.select(cfg, "evaluate.checkpoint_dir")
            or OmegaConf.select(cfg, "eval.state_dict_path")
            or "checkpoints"
        )
        self.checkpoint_dir = ckpt
        if (ckpt / "manifest.pt").exists():
            self.model = load_model_from_manifest(ckpt, self.model)
        elif (ckpt / "model_state_dict.pt").exists():
            sd = torch.load(
                ckpt / "model_state_dict.pt", map_location="cpu", weights_only=False
            )
            self.model.load_state_dict(sd)
        self.model.to(self.device)
        self.model.eval()

        self.data = instantiate(cfg.data)
        self.paradigm = OmegaConf.select(cfg, "paradigm.name", default="fm")
        st = OmegaConf.select(cfg, "evaluate.sampler_type", default="euler")
        self.sampler = build_sampler(
            paradigm=self.paradigm,
            model=self.model,
            sampler_type=st,
            ode_steps=int(OmegaConf.select(cfg, "evaluate.ode_steps", default=16)),
            ddim_steps=int(OmegaConf.select(cfg, "evaluate.ddim_steps", default=20)),
        )

        self.n_rollout_steps = int(
            OmegaConf.select(cfg, "evaluate.n_rollout_steps", default=50)
        )
        try:
            self.results_path = Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            self.results_path = Path(
                OmegaConf.select(cfg, "evaluate.results_path", default=".")
            )

    def _test_trajectories(self) -> List[torch.Tensor]:
        raw = self.data.read_data()
        processed = self.data.preprocess(raw)
        n_traj = processed.shape[0]
        train_traj, test_traj = _train_test_trajectory_indices(
            n_traj,
            int(self.data.cfg.n_tr),
            int(self.data.cfg.n_te),
            self.data.seed,
        )
        del train_traj
        return [processed[tid].to(self.device) for tid in test_traj]

    def _train_metadata(self) -> Dict[str, Any]:
        candidates = [
            self.checkpoint_dir / "results.json",
            self.checkpoint_dir.parent / "results.json",
        ]
        for p in candidates:
            data = load_train_results_json(p)
            if data:
                return data
        return {}

    def run(self) -> Dict[str, Any]:
        trajs = self._test_trajectories()
        meta = self._train_metadata()
        config_hash = meta.get("config_hash", "")
        seed = int(OmegaConf.select(self.cfg, "seed", default=42))
        conditioning_level = OmegaConf.select(
            self.cfg, "conditioning.name", default="level0"
        )
        param_count = int(
            meta.get("param_count")
            or sum(p.numel() for p in self.model.parameters())
        )
        train_gpu_hours = meta.get("train_gpu_hours")

        step_mse_excl_t0: List[float] = []
        final_mses: List[float] = []
        last_preds: List[torch.Tensor] = []
        last_gts: List[torch.Tensor] = []

        if trajs:
            u_w = trajs[0][0:1]
            b = min(4, max(1, u_w.shape[0]))
            infer_mean, infer_std = inference_time(
                lambda x: self.sampler(x),
                u_w.expand(b, *u_w.shape[1:]).contiguous(),
                n_warmup=3,
                n_runs=10,
            )
        else:
            infer_mean, infer_std = float("nan"), float("nan")

        with torch.no_grad():
            for traj in trajs:
                T = min(self.n_rollout_steps, traj.shape[0])
                if T < 2:
                    continue
                preds: List[torch.Tensor] = []
                u = traj[0:1]
                preds.append(u.squeeze(0))
                for _ in range(T - 1):
                    u = self.sampler(u)
                    preds.append(u.squeeze(0))
                pred_seq = torch.stack(preds, dim=0)
                gt_seq = traj[:T]
                mse_t = (pred_seq - gt_seq).pow(2).mean(dim=(1, 2, 3))
                # Exclude t=0 (identity)
                step_mse_excl_t0.extend([float(x) for x in mse_t[1:]])
                final_mses.append(float(mse_t[-1].item()))
                last_preds.append(pred_seq[-1])
                last_gts.append(gt_seq[-1])

        rollout_mean = (
            sum(step_mse_excl_t0) / len(step_mse_excl_t0) if step_mse_excl_t0 else float("nan")
        )
        rollout_final = sum(final_mses) / len(final_mses) if final_mses else float("nan")

        if last_preds:
            pstack = torch.stack(last_preds, dim=0).cpu()
            gstack = torch.stack(last_gts, dim=0).cpu()
            spec = spectrum_mse(pstack, gstack)
            dens = density_mse(pstack, gstack)
        else:
            spec = float("nan")
            dens = float("nan")

        row = {
            "config_hash": config_hash,
            "paradigm": self.paradigm,
            "model_name": self.cfg.model.name,
            "conditioning_level": conditioning_level,
            "seed": seed,
            "rollout_mse_mean": rollout_mean,
            "rollout_mse_final": rollout_final,
            "spectrum_mse": spec,
            "density_mse": dens,
            "infer_ms_per_step_mean": infer_mean,
            "infer_ms_per_step_std": infer_std,
            "train_gpu_hours": train_gpu_hours,
            "param_count": param_count,
        }
        append_row(self.results_path, row)
        print(json.dumps({k: row[k] for k in RESULT_FIELDS}, indent=2, default=str))
        return row
