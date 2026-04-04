import torch
from typing import Optional

from models.fno import make_posn_embed, t_allhot
from models.base import PDEModel
from models.film import FiLMLayer

_LocalNO = None
_LOCAL_NO_IMPORT_ERROR: Optional[Exception] = None

try:
    from neuralop.models.local_no import LocalNO as _LocalNO
except Exception as e:  # pragma: no cover - optional torch-harmonics / DISCO stack
    _LOCAL_NO_IMPORT_ERROR = e


class LFNO(PDEModel):
    def __init__(
        self,
        modes,
        vis_channels,
        hidden_channels,
        x_dim=1,
        default_in_shape=(64, 64),
        disco_kernel_shape=(5, 5),
        t_scaling=1,
        disco_layers=True,
        coord_channels=0,
        film_param_dim=0,
        **kwargs,
    ):
        super().__init__()
        if _LocalNO is None:
            raise RuntimeError(
                "LFNO requires `neuralop` LocalNO (often needs `pip install torch-harmonics` "
                "for DISCO layers). Original import error: "
                f"{_LOCAL_NO_IMPORT_ERROR!r}"
            ) from _LOCAL_NO_IMPORT_ERROR

        self.t_scaling = t_scaling
        self.vis_channels = int(vis_channels)
        self.coord_channels = int(coord_channels)
        n_modes = (modes,) * x_dim
        spatial_extra = self.coord_channels if self.coord_channels > 0 else x_dim
        in_channels = self.vis_channels + spatial_extra + 1
        dks = list(disco_kernel_shape)
        if len(dks) == 1:
            dks = [dks[0], dks[0]]

        self.model = _LocalNO(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=vis_channels,
            hidden_channels=hidden_channels,
            default_in_shape=list(default_in_shape),
            disco_kernel_shape=dks,
            disco_layers=disco_layers,
            **kwargs,
        )
        fpd = int(film_param_dim) if film_param_dim else 0
        self.film = FiLMLayer(fpd, vis_channels) if fpd > 0 else None

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
                    f"Expected u with {self.vis_channels + self.coord_channels} channels; got {u.shape[1]}"
                )
            u_in = torch.cat((u, t_ch), dim=1).float()
        else:
            if coords is not None:
                posn = coords.float().contiguous()
            else:
                posn = make_posn_embed(batch_size, dims).to(u.device)
            u_in = torch.cat((u, posn, t_ch), dim=1).float()

        out = self.model(u_in)
        if self.film is not None and params is not None:
            out = self.film(out, params)
        return out
