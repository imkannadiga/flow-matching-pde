import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import PDEModel
from models.film import FiLMLayer

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(b, self.weights1.size(1), h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        return torch.fft.irfft2(out_ft, s=(h, w))

class FNOBlock(nn.Module):
    def __init__(self, channels: int, modes1: int, modes2: int, cond_dim: int):
        super().__init__()
        self.conv_f = SpectralConv2d(channels, channels, modes1, modes2)
        self.conv_w = nn.Conv2d(channels, channels, 1)
        self.film = FiLMLayer(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.film(self.conv_f(x) + self.conv_w(x), cond))

class FNO(PDEModel):
    def __init__(
        self,
        modes: int | tuple,
        vis_channels: int,
        cond_channels: int,
        hidden_channels: int,
        proj_channels: int,
        x_dim: int = 2,
        coord_channels: int = 0,
        out_channels: int = None,
        n_layers: int = 4,
        periodic: bool = False,
        use_posn_embed: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.vis_channels = vis_channels
        self.out_channels = out_channels or vis_channels
        self.coord_channels = coord_channels
        self.cond_channels = cond_channels
        self.use_posn_embed = use_posn_embed
        self.modes = (modes, modes) if isinstance(modes, int) else modes[:2]
        
        self.cond_dim = self.cond_channels + 1
        
        spatial_extra = self.coord_channels or (x_dim if self.use_posn_embed else 0)
        # +1 added back to in_channels to account for the appended `t` tensor
        in_channels = self.vis_channels + self.cond_channels + spatial_extra + 1
        
        self.p = nn.Conv2d(in_channels, hidden_channels, 1)
        
        self.blocks = nn.ModuleList([
            FNOBlock(hidden_channels, self.modes[0], self.modes[1], self.cond_dim)
            for _ in range(n_layers)
        ])
        
        self.q = nn.Sequential(
            nn.Conv2d(hidden_channels, proj_channels, 1),
            nn.GELU(),
            nn.Conv2d(proj_channels, self.out_channels, 1)
        )
        
        self.periodic = periodic
        self.pad = 0 if periodic else 8
        self._pos_grid = None 

    def _get_pos_embed(self, shape: tuple, device: torch.device) -> torch.Tensor:
        b, _, nx, ny = shape
        if self._pos_grid is None or self._pos_grid.shape[-2:] != (nx, ny):
            grid_x = torch.linspace(0, 1, nx, device=device)
            grid_y = torch.linspace(0, 1, ny, device=device)
            pos_x, pos_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
            self._pos_grid = torch.stack([pos_x, pos_y], dim=0).unsqueeze(0)
        return self._pos_grid.expand(b, -1, -1, -1)

    def forward(self, u: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b, _, nx, ny = u.shape

        t_spatial = t.view(-1, 1, 1, 1).expand(b, 1, nx, ny).float()
        
        cond_parts = [t_spatial, cond.float()] if cond is not None else [t_spatial]
        cond_vector = torch.cat(cond_parts, dim=1)

        # Build raw components before padding
        u_parts = [u.float()]
        if self.coord_channels == 0 and self.use_posn_embed:
            u_parts.append(self._get_pos_embed(u.shape, u.device))
        if cond is not None:
            u_parts.append(cond.float())
        
        # Append t back to the spatial input stream
        u_parts.append(t_spatial)

        # Handle distinct padding modes
        if self.pad > 0:
            pad_tuple = (0, self.pad, 0, self.pad)
            
            # Standard zero-padding for the input field `u`
            u_padded = F.pad(u_parts[0], pad_tuple, mode="constant", value=0)
            
            # Replicate padding for positional grids, spatial conditions, and time 
            # to repeat boundary values over the padding region
            aux_padded = [F.pad(c, pad_tuple, mode="replicate") for c in u_parts[1:]]
            
            u_in = torch.cat([u_padded] + aux_padded, dim=1)
            cond_vector = F.pad(cond_vector, pad_tuple, mode="replicate")
        else:
            u_in = torch.cat(u_parts, dim=1)

        x = self.p(u_in)
        for block in self.blocks:
            x = block(x, cond_vector)

        if self.pad > 0:
            x = x[..., :-self.pad, :-self.pad]

        return self.q(x)