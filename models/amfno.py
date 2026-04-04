"""
Amortized FNO-style surrogate: spectral trunk (neuralop FNO) with **parameter context maps**.

We do **not** generate full spectral weights from ``params`` (that would be a large hypernetwork);
instead an MLP maps ``(Re, \\nu, \\ldots)`` to ``context_dim`` channels tiled on the grid and
concatenated with the state (and optional coord channels) before the FNO, following the same
amortized-conditioning idea as input-modulated spectral models (cf. amortized / meta PDE surrogates).
"""

from __future__ import annotations

import torch
from neuralop.models import FNO as _FNO

from models.base import PDEModel
from models.film import FiLMLayer
from models.fno import make_posn_embed, t_allhot


class AMFNO(PDEModel):
    def __init__(
        self,
        modes,
        vis_channels,
        hidden_channels,
        proj_channels,
        x_dim=2,
        t_scaling=1,
        coord_channels=0,
        film_param_dim=0,
        param_dim: int = 2,
        context_dim: int = 8,
        **kwargs,
    ):
        super().__init__()
        kwargs.pop("name", None)
        self.t_scaling = t_scaling
        self.vis_channels = int(vis_channels)
        self.coord_channels = int(coord_channels)
        self.param_dim = int(param_dim)
        self.context_dim = int(context_dim)
        n_modes = (modes,) * x_dim
        spatial_extra = self.coord_channels if self.coord_channels > 0 else x_dim
        in_channels = self.vis_channels + spatial_extra + self.context_dim + 1
        projection_channel_ratio = proj_channels / max(hidden_channels, 1)

        self.param_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.param_dim, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, self.context_dim),
        )

        self.model = _FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            projection_channel_ratio=projection_channel_ratio,
            in_channels=in_channels,
            out_channels=vis_channels,
            **kwargs,
        )
        fpd = int(film_param_dim) if film_param_dim else 0
        self.film = FiLMLayer(fpd, vis_channels) if fpd > 0 else None

    def forward(self, t, u, coords=None, params=None):
        t = t / self.t_scaling
        batch_size = u.shape[0]
        dims = u.shape[2:]
        b, _, h, w = u.shape

        if params is None:
            p = u.new_zeros(batch_size, self.param_dim)
        else:
            p = params

        ctx = self.param_encoder(p.float())
        ctx_map = ctx.view(batch_size, self.context_dim, 1, 1).expand(-1, -1, h, w)

        if t.dim() == 0 or t.numel() == 1:
            t = torch.ones(u.shape[0], device=t.device, dtype=t.dtype) * t

        assert t.dim() == 1
        assert t.shape[0] == u.shape[0]

        t_ch = t_allhot(t, u)
        if self.coord_channels > 0:
            if u.shape[1] != self.vis_channels + self.coord_channels:
                raise ValueError(
                    f"Expected u with {self.vis_channels + self.coord_channels} channels; got {u.shape[1]}"
                )
            u_core = torch.cat((u, ctx_map, t_ch), dim=1).float().contiguous()
        else:
            if coords is not None:
                posn = coords.float().contiguous()
            else:
                posn = make_posn_embed(batch_size, dims).to(u.device)
            u_core = torch.cat((u, posn, ctx_map, t_ch), dim=1).float().contiguous()

        out = self.model(u_core)
        if self.film is not None and params is not None:
            out = self.film(out, params)
        return out
