import torch
import torch.nn as nn
from typing import Optional

from models.fno import make_posn_embed
from models.base import PDEModel
from models.film import FiLMLayer

_LocalNO = None
_LOCAL_NO_IMPORT_ERROR: Optional[Exception] = None

try:
    from neuralop.models.local_no import LocalNO as _LocalNO
except Exception as e:
    _LOCAL_NO_IMPORT_ERROR = e


class LFNO(PDEModel):
    def __init__(
        self,
        modes,
        vis_channels,
        cond_channels,       # Physical parameters coming from the data pipeline
        hidden_channels,
        x_dim=2,
        default_in_shape=(64, 64),
        disco_kernel_shape=(5, 5),
        t_scaling=1,
        disco_layers=True,
        coord_channels=0,
        out_channels=None,
        n_layers=4,          # Explicitly track number of blocks for FiLM
        **kwargs,
    ):
        super().__init__()
        if _LocalNO is None:
            raise RuntimeError(
                f"LFNO requires `neuralop` LocalNO. Original import error: {_LOCAL_NO_IMPORT_ERROR!r}"
            )

        self.t_scaling = t_scaling
        self.vis_channels = int(vis_channels)
        self.out_channels = int(out_channels) if out_channels is not None else self.vis_channels
        self.coord_channels = int(coord_channels)
        
        n_modes = (modes,) * x_dim
        dks = list(disco_kernel_shape)
        if len(dks) == 1:
            dks = [dks[0], dks[0]]

        # in_channels is strictly spatial (physics state + coordinates)
        spatial_extra = self.coord_channels if self.coord_channels > 0 else x_dim
        in_channels = self.vis_channels + spatial_extra

        # Unified condition dimension: Flow Time (1) + Physical Conditioning
        self.cond_dim = 1 + int(cond_channels)

        self.model = _LocalNO(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=self.out_channels,
            hidden_channels=hidden_channels,
            default_in_shape=list(default_in_shape),
            disco_kernel_shape=dks,
            disco_layers=disco_layers,
            n_layers=n_layers,
            **kwargs,
        )
        
        # Build a FiLM layer for every LocalNO block
        self.film_layers = nn.ModuleList([
            FiLMLayer(self.cond_dim, hidden_channels) for _ in range(n_layers)
        ])

    def forward(self, u, cond, t):
        t = t / self.t_scaling
        batch_size = u.shape[0]
        dims = u.shape[2:]

        # 1. Format Time & Conditioning
        if t.dim() == 0 or t.numel() == 1:
            t = torch.ones(u.shape[0], 1, device=u.device, dtype=torch.float32) * t
        elif t.dim() == 1:
            t = t.unsqueeze(1).float()
            
        if cond.dim() == 1:
            cond = cond.unsqueeze(1).float()
            
        cond_vector = torch.cat([t, cond], dim=1).float()

        # 2. Format Pure Spatial Input
        if self.coord_channels > 0:
            if u.shape[1] != self.vis_channels + self.coord_channels:
                raise ValueError(f"Expected u with {self.vis_channels + self.coord_channels} channels.")
            u_in = u.float().contiguous()
        else:
            posn = make_posn_embed(batch_size, dims).to(u.device)
            u_in = torch.cat((u, posn), dim=1).float().contiguous()

        # 3. Inter-block Hijack pass through neuralop LocalNO
        x = self.model.lifting(u_in)
        
        if hasattr(self.model, 'domain_padding') and self.model.domain_padding is not None:
            x = self.model.domain_padding.pad(x)
            
        for i, block in enumerate(self.model.fno_blocks):
            x = block(x)
            x = self.film_layers[i](x, cond_vector) # Deep Fusion injection
            
        if hasattr(self.model, 'domain_padding') and self.model.domain_padding is not None:
            x = self.model.domain_padding.unpad(x)
            
        out = self.model.projection(x)
        return out