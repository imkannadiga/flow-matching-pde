import torch
from neuralop.models import FNO as _FNO

from models.base import PDEModel
from models.film import FiLMLayer

"""
Time-conditioned FNO: conditioning channels and time are concatenated into the input.
Spatial coordinates are not injected here; if needed they should be included in the
conditioning tensor C upstream (via spatial_conditioning=true in the data config).
"""


def t_allhot(t, like_u: torch.Tensor) -> torch.Tensor:
    """Broadcast per-batch time to ``[B, 1, *spatial]`` matching ``like_u`` (any channel count)."""
    batch_size = like_u.shape[0]
    dim = like_u.shape[2:]
    t = t.to(device=like_u.device, dtype=like_u.dtype)
    if t.dim() == 0:
        t = t.view(1).expand(batch_size)
    t = t.reshape(batch_size, *([1] * (1 + len(dim))))
    return t * torch.ones(batch_size, 1, *dim, device=like_u.device, dtype=like_u.dtype)


class FNO(PDEModel):
    def __init__(
        self,
        modes,
        vis_channels,
        hidden_channels,
        proj_channels,
        x_dim=1,
        t_scaling=1,
        film_param_dim=0,
        out_channels=None,
        **kwargs,
    ):
        super().__init__()
        kwargs.pop("name", None)
        # in_channels is inferred at runtime from the first preprocessed batch (train.py).
        # It reflects the full concatenated input: X_tau + C channels.
        actual_channels = kwargs.pop("in_channels", None)

        self.t_scaling = t_scaling
        self.vis_channels = int(actual_channels if actual_channels is not None else vis_channels)
        self.out_channels = (
            int(out_channels) if out_channels is not None else self.vis_channels
        )
        n_modes = (modes,) * x_dim
        in_channels = self.vis_channels + 1  # state+conditioning channels + time
        projection_channel_ratio = proj_channels / max(hidden_channels, 1)

        self.model = _FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            projection_channel_ratio=projection_channel_ratio,
            in_channels=in_channels,
            out_channels=self.out_channels,
            positional_embedding=None,
            **kwargs,
        )
        fpd = int(film_param_dim) if film_param_dim else 0
        self.film = FiLMLayer(fpd, self.out_channels) if fpd > 0 else None

    def forward(self, t, u, params=None):
        t = t / self.t_scaling

        if t.dim() == 0 or t.numel() == 1:
            t = torch.ones(u.shape[0], device=t.device, dtype=t.dtype) * t

        assert t.dim() == 1
        assert t.shape[0] == u.shape[0]

        t_ch = t_allhot(t, u)
        u_in = torch.cat((u, t_ch), dim=1).float().contiguous()

        out = self.model(u_in)
        if self.film is not None and params is not None:
            out = self.film(out, params)
        return out
