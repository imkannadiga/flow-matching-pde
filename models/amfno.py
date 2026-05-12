from __future__ import annotations
import torch
import torch.nn as nn
from neuralop.models import FNO as _FNO

from models.fno import make_posn_embed
from models.base import PDEModel
from models.film import FiLMLayer


class AMFNO(PDEModel):
    def __init__(
        self,
        modes,
        vis_channels,
        cond_channels,       # Raw physical parameters (e.g., Re, nu)
        hidden_channels,
        proj_channels,
        x_dim=2,
        t_scaling=1,
        coord_channels=0,
        context_dim: int = 64, # Size of the amortized deep representation
        out_channels=None,
        n_layers=4,
        **kwargs,
    ):
        super().__init__()
        kwargs.pop("name", None)
        kwargs.pop("in_channels", None) 
        
        self.t_scaling = t_scaling
        self.vis_channels = int(vis_channels)
        self.out_channels = int(out_channels) if out_channels is not None else self.vis_channels
        self.coord_channels = int(coord_channels)
        self.context_dim = int(context_dim)
        
        n_modes = (modes,) * x_dim
        projection_channel_ratio = proj_channels / max(hidden_channels, 1)

        # in_channels is strictly spatial
        spatial_extra = self.coord_channels if self.coord_channels > 0 else x_dim
        in_channels = self.vis_channels + spatial_extra 

        # Amortized embedding: Maps raw params to a rich context vector
        self.param_encoder = nn.Sequential(
            nn.Linear(int(cond_channels), 64),
            nn.GELU(),
            nn.Linear(64, self.context_dim),
        )
        
        # Unified condition dimension: Flow Time (1) + Amortized Context
        self.cond_dim = 1 + self.context_dim

        self.model = _FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            projection_channel_ratio=projection_channel_ratio,
            in_channels=in_channels,
            out_channels=self.out_channels,
            n_layers=n_layers,
            **kwargs,
        )
        
        self.film_layers = nn.ModuleList([
            FiLMLayer(self.cond_dim, hidden_channels) for _ in range(n_layers)
        ])

    def forward(self, u, cond, t):
        t = t / self.t_scaling
        batch_size = u.shape[0]
        dims = u.shape[2:]

        # 1. Generate Amortized Context Vector
        ctx = self.param_encoder(cond.float()) # [B, context_dim]

        # 2. Format Time & Unified Vector
        if t.dim() == 0 or t.numel() == 1:
            t = torch.ones(batch_size, 1, device=u.device, dtype=torch.float32) * t
        elif t.dim() == 1:
            t = t.unsqueeze(1).float()
            
        cond_vector = torch.cat([t, ctx], dim=1).float() # [B, 1 + context_dim]

        # 3. Format Pure Spatial Input
        if self.coord_channels > 0:
            if u.shape[1] != self.vis_channels + self.coord_channels:
                raise ValueError(f"Expected u with {self.vis_channels + self.coord_channels} channels.")
            u_in = u.float().contiguous()
        else:
            posn = make_posn_embed(batch_size, dims).to(u.device)
            u_in = torch.cat((u, posn), dim=1).float().contiguous()

        # 4. Inter-block Hijack pass through neuralop FNO
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