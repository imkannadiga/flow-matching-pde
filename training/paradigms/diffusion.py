"""
VP-style diffusion on the next state x1 = u_{t+1}: noisy mixture with ε prediction target.
Matches FM next-state semantics; uses the same cosine ``alpha`` schedule as legacy FM VP path.
"""

from __future__ import annotations

import numpy as np
import torch

from training.data_processors import DefaultDataProcessor
from util.util import reshape_for_batchwise


class DiffusionDataProcessor(DefaultDataProcessor):
    def __init__(
        self,
        device,
        vp: bool = True,
        sigma_min: float = 1e-4,
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
        self.device = device
        self.vp = vp
        self.sigma_min = sigma_min
        if self.vp:
            self.alpha, _ = self._construct_alpha()
        else:
            self.alpha = None

    def _construct_alpha(self):
        def alpha(t):
            return torch.cos((t + 0.08) / 2.16 * np.pi).to(self.device)

        def dalpha(t):
            return (
                -np.pi
                / 2.16
                * torch.sin((t + 0.08) / 2.16 * np.pi).to(self.device)
            )

        return alpha, dalpha

    def preprocess(self, sample):
        sample = super().preprocess(sample)
        x1 = sample["y"]
        batch_size = x1.shape[0]
        n_dims = len(x1.shape[2:])
        tau = torch.rand(batch_size, device=self.device, dtype=x1.dtype)
        eps = torch.randn_like(x1)

        tau_spatial = reshape_for_batchwise(tau, 1 + n_dims)
        if self.vp:
            a = self.alpha(1.0 - tau_spatial)
            sigma = torch.sqrt(torch.clamp(1.0 - a**2, min=1e-8))
            x_noisy = a * x1 + sigma * eps
            # ε-parameterization target is the noise used in the forward process
            target = eps
        else:
            mu = tau_spatial * x1
            sigma = 1.0 - (1.0 - self.sigma_min) * tau_spatial
            x_noisy = mu + sigma * eps
            target = eps

        out = {"x": {"u": x_noisy, "t": tau}, "y": target}
        if "params" in sample:
            out["params"] = sample["params"]
        return self.apply_model_conditioning(out)

    def postprocess(self, output, data_dict):
        return output, data_dict

    def forward(self, x):
        return self.preprocess(x)
