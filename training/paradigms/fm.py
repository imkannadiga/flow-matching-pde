"""
Rectified flow for one-step dynamics:

- **bridge** (default): x0 = u_t, x1 = u_{t+1}, x_τ = (1-τ)x0 + τ x1, target v = x1 − x0.
  Matches rollout: Euler from τ=0 gives u_t + v ≈ u_{t+1}.

- **noise prior** (``bridge=False``): x0 ~ GP / Gaussian, x1 = u_{t+1} (generative toward the
  next state without conditioning on u_t in the path — use only for ablations).
"""

from __future__ import annotations

import torch

from training.data_processors import DefaultDataProcessor
from training.input_conditioning import stack_additional_input_channels
from util.util import make_grid, reshape_for_batchwise


class FMDataProcessor(DefaultDataProcessor):
    def __init__(
        self,
        device,
        bridge: bool = True,
        noise_prior: str = "gp",
        kernel_length: float = 0.001,
        kernel_variance: float = 1.0,
        vp: bool = False,
        sigma_min: float = 1e-4,
        append_coords: bool = False,
        domain_bounds=None,
        film_params: bool = False,
        param_keys=None,
        coord_normalize: str = "neg1_1",
        stack_x_aux: bool = True,
        stack_channel_bounds: bool = True,
        stack_params_into_u: bool = True,
    ):
        super().__init__(
            device,
            append_coords=append_coords,
            domain_bounds=domain_bounds,
            film_params=film_params,
            param_keys=param_keys,
            coord_normalize=coord_normalize,
        )
        self.stack_x_aux = bool(stack_x_aux)
        self.stack_channel_bounds = bool(stack_channel_bounds)
        self.stack_params_into_u = bool(stack_params_into_u)
        self.bridge = bridge
        self.noise_prior = noise_prior
        self.vp = vp
        self.sigma_min = sigma_min
        self.device = device
        self.gp = None
        if not bridge and noise_prior == "gp":
            from util.gaussian_process import GPPrior

            self.gp = GPPrior(
                lengthscale=kernel_length, var=kernel_variance, device=device
            )

    def _sample_x0(self, x1: torch.Tensor) -> torch.Tensor:
        batch_size = x1.shape[0]
        n_channels = x1.shape[1]
        dims = list(x1.shape[2:])
        if self.noise_prior == "gaussian":
            return torch.randn_like(x1)
        if self.noise_prior != "gp" or self.gp is None:
            raise ValueError(f"Unknown noise_prior={self.noise_prior!r}")
        query_points = make_grid(dims)
        return self.gp.sample(
            query_points, dims, n_samples=batch_size, n_channels=n_channels
        )

    def preprocess(self, sample):
        sample = super().preprocess(sample)
        x1 = sample["y"]
        if self.bridge:
            x0 = sample["x"]
        else:
            x0 = self._sample_x0(x1)
        batch_size = x1.shape[0]
        tau = torch.rand(batch_size, device=self.device, dtype=x1.dtype)
        tau_b = reshape_for_batchwise(tau, 1 + len(x1.shape[2:]))
        x_tau = (1.0 - tau_b) * x0 + tau_b * x1
        target = x1 - x0
        u = stack_additional_input_channels(
            x_tau,
            sample,
            stack_x_aux=self.stack_x_aux,
            stack_channel_bounds=self.stack_channel_bounds,
            stack_params_into_u=self.stack_params_into_u,
        )
        out = {"x": {"u": u, "t": tau}, "y": target}
        if "params" in sample:
            out["params"] = sample["params"]
        return self.apply_model_conditioning(out)

    def postprocess(self, output, data_dict):
        return output, data_dict

    def forward(self, x):
        return self.preprocess(x)
