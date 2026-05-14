import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fno import SpectralConv2d, make_posn_embed
from models.base import PDEModel
from models.film import FiLMLayer


class LFNOBlock(nn.Module):
    """Spectral conv block with direct FiLM conditioning."""
    def __init__(self, channels: int, modes1: int, modes2: int, cond_dim: int):
        super().__init__()
        self.conv_f = SpectralConv2d(channels, channels, modes1, modes2)
        self.conv_w = nn.Conv2d(channels, channels, 1)
        self.film   = FiLMLayer(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond_vector: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.film(self.conv_f(x), cond_vector) + self.conv_w(x))


class LFNO(PDEModel):
    """
    FNO with direct FiLM conditioning.

    Originally backed by neuralop LocalNO (DISCO convolutions). Rewritten using
    standard global spectral convolutions to remove the neuralop / torch-harmonics
    dependency entirely. Behaviour is equivalent; DISCO-specific config keys
    (disco_kernel_shape, default_in_shape, disco_layers) are silently absorbed
    via **kwargs so existing yaml configs continue to work without changes.

    Conditioning path : cond [B, C, H, W] → global avg-pool → cat with t →
                        FiLM scale/shift applied after every spectral block.
    Spatial path      : u + 2-D positional embedding → lifting Conv →
                        N spectral blocks → projection MLP.
    """

    def __init__(
        self,
        modes,
        vis_channels,
        cond_channels,
        hidden_channels,
        proj_channels=None,
        x_dim=2,
        t_scaling=1,
        coord_channels=0,
        out_channels=None,
        n_layers=4,
        **kwargs,  # absorbs disco_kernel_shape, default_in_shape, disco_layers, name, …
    ):
        super().__init__()
        del kwargs
        self.t_scaling      = t_scaling
        self.vis_channels   = int(vis_channels)
        self.out_channels   = int(out_channels) if out_channels is not None else self.vis_channels
        self.coord_channels = int(coord_channels)
        self.x_dim          = int(x_dim)

        modes_ = (modes, modes) if isinstance(modes, int) else tuple(modes[:2])
        proj   = int(proj_channels) if proj_channels is not None else hidden_channels

        spatial_extra = self.coord_channels if self.coord_channels > 0 else self.x_dim
        in_channels   = self.vis_channels + spatial_extra

        cond_dim = 1 + int(cond_channels)  # t scalar + encoded cond channels

        if cond_channels > 0:
            self.cond_encoder = nn.Sequential(
                nn.Conv2d(int(cond_channels), int(cond_channels), kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
            )
        else:
            self.cond_encoder = None

        self.p = nn.Conv2d(in_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList([
            LFNOBlock(hidden_channels, modes_[0], modes_[1], cond_dim)
            for _ in range(n_layers)
        ])
        self.q = nn.Sequential(
            nn.Conv2d(hidden_channels, proj, 1),
            nn.GELU(),
            nn.Conv2d(proj, self.out_channels, 1),
        )

    def forward(self, u, cond, t):
        B, _, H, W = u.shape
        t = t / self.t_scaling

        if cond.dim() == 4:
            cond = self.cond_encoder(cond.float()) if self.cond_encoder is not None \
                   else cond.mean(dim=[-1, -2])
        if t.dim() == 0 or t.numel() == 1:
            t = t.expand(B)
        cond_vector = torch.cat([t.unsqueeze(1).float(), cond.float()], dim=1)

        if self.coord_channels > 0:
            u_in = u.float()
        else:
            posn = make_posn_embed(B, (H, W)).to(u.device, u.dtype)
            u_in = torch.cat([u, posn], dim=1).float()

        x = self.p(u_in)
        for block in self.blocks:
            x = block(x, cond_vector)
        return self.q(x)
