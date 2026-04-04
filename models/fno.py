import torch
from neuralop.models import FNO as _FNO

from models.base import PDEModel
from models.film import FiLMLayer

"""
Time-conditioned FNO: time and optional coordinates are extra input channels.
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


def make_posn_embed(batch_size, dims):
    if len(dims) == 1:
        emb = torch.linspace(0, 1, dims[0])
        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1)
    elif len(dims) == 2:
        x1 = torch.linspace(0, 1, dims[1]).repeat(dims[0], 1).unsqueeze(0)
        x2 = torch.linspace(0, 1, dims[0]).repeat(dims[1], 1).T.unsqueeze(0)
        emb = torch.cat((x1, x2), dim=0)
        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        raise NotImplementedError
    return emb


class FNO(PDEModel):
    def __init__(
        self,
        modes,
        vis_channels,
        hidden_channels,
        proj_channels,
        x_dim=1,
        t_scaling=1,
        coord_channels=0,
        film_param_dim=0,
        out_channels=None,
        **kwargs,
    ):
        super().__init__()
        kwargs.pop("name", None)

        self.t_scaling = t_scaling
        self.vis_channels = int(vis_channels)
        self.out_channels = (
            int(out_channels) if out_channels is not None else self.vis_channels
        )
        self.coord_channels = int(coord_channels)
        n_modes = (modes,) * x_dim
        # If coord_channels > 0, ``u`` already ends with coord grid channels (no separate posn).
        spatial_extra = self.coord_channels if self.coord_channels > 0 else x_dim
        in_channels = self.vis_channels + spatial_extra + 1
        projection_channel_ratio = proj_channels / max(hidden_channels, 1)

        self.model = _FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            projection_channel_ratio=projection_channel_ratio,
            in_channels=in_channels,
            out_channels=self.out_channels,
            **kwargs,
        )
        fpd = int(film_param_dim) if film_param_dim else 0
        self.film = FiLMLayer(fpd, self.out_channels) if fpd > 0 else None

    def forward(self, t, u, coords=None, params=None):
        t = t / self.t_scaling
        batch_size = u.shape[0]
        dims = u.shape[2:]

        if t.dim() == 0 or t.numel() == 1:
            t = torch.ones(u.shape[0], device=t.device, dtype=t.dtype) * t

        assert t.dim() == 1
        assert t.shape[0] == u.shape[0]

        t_ch = t_allhot(t, u)
        if self.coord_channels > 0:
            if u.shape[1] != self.vis_channels + self.coord_channels:
                raise ValueError(
                    f"Expected u with {self.vis_channels + self.coord_channels} channels "
                    f"(vis + coord); got {u.shape[1]}"
                )
            u_in = torch.cat((u, t_ch), dim=1).float().contiguous()
        else:
            if coords is not None:
                posn = coords.float().contiguous()
            else:
                posn = make_posn_embed(batch_size, dims).to(u.device)
            u_in = torch.cat((u, posn, t_ch), dim=1).float().contiguous()

        out = self.model(u_in)
        if self.film is not None and params is not None:
            out = self.film(out, params)
        return out
