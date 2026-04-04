"""
Autoregressive one-step (or multi-step pushforward) supervision: predict u_{k+1} from u_k.

When ``unroll_steps`` > 1, the batch may include ``trajectory`` of shape **[B, L, C, H, W]**
with ``L > unroll_steps``; the processor stacks ``unroll_steps`` consecutive one-step pairs
along the batch dimension so the Trainer performs one forward with sum loss over steps.
"""

from __future__ import annotations

import random
from typing import Any, Dict

import torch

from training.data_processors import DefaultDataProcessor


class ARDataProcessor(DefaultDataProcessor):
    def __init__(
        self,
        device,
        unroll_steps: int = 1,
        unroll_prob: float = 0.0,
        append_coords: bool = False,
        domain_bounds=None,
        film_params: bool = False,
        param_keys=None,
        coord_normalize: str = "neg1_1",
    ):
        super().__init__(
            device,
            append_coords=append_coords,
            domain_bounds=domain_bounds,
            film_params=film_params,
            param_keys=param_keys,
            coord_normalize=coord_normalize,
        )
        self.unroll_steps = int(unroll_steps)
        self.unroll_prob = float(unroll_prob)

    def preprocess(self, sample: Dict[str, Any]):
        sample = super().preprocess(sample)
        p = max(0.0, min(1.0, self.unroll_prob))
        use_pushforward = (
            self.unroll_steps > 1
            and "trajectory" in sample
            and (p >= 1.0 or random.random() < p)
        )
        if use_pushforward:
            traj = sample["trajectory"].to(self.device)
            if traj.dim() != 5:
                raise ValueError(
                    f"trajectory must be [B,L,C,H,W]; got {tuple(traj.shape)}"
                )
            b, l, c, h, w = traj.shape
            k = min(self.unroll_steps, l - 1)
            if k < 1:
                raise ValueError("trajectory too short for AR pushforward")
            u_in = traj[:, :k].reshape(b * k, c, h, w)
            y_tar = traj[:, 1 : k + 1].reshape(b * k, c, h, w)
            t0 = torch.zeros(b * k, device=self.device, dtype=u_in.dtype)
            out = {"x": {"t": t0, "u": u_in}, "y": y_tar}
            if "params" in sample:
                p = sample["params"]
                if torch.is_tensor(p) and p.shape[0] == b:
                    out["params"] = p.repeat_interleave(k, dim=0)
                else:
                    out["params"] = p
            return self.apply_model_conditioning(out)

        t0 = torch.zeros(sample["x"].shape[0], device=self.device, dtype=sample["x"].dtype)
        out = {"x": {"t": t0, "u": sample["x"]}, "y": sample["y"]}
        if "params" in sample:
            out["params"] = sample["params"]
        return self.apply_model_conditioning(out)

    def postprocess(self, output, data_dict):
        return output, data_dict

    def forward(self, x):
        return self.preprocess(x)
