import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import PDEModel
from models.film import FiLMLayer  # Note: See below, this must be updated to Spatial FiLM

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        
        # Initialize output Fourier representation
        out_ft = torch.zeros(batchsize, self.weights1.size(1), h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Multiply relevant Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        return torch.fft.irfft2(out_ft, s=(h, w))

class FNOBlock(nn.Module):
    def __init__(self, channels, modes1, modes2, cond_dim):
        super().__init__()
        self.conv_f = SpectralConv2d(channels, channels, modes1, modes2)
        self.conv_w = nn.Conv2d(channels, channels, 1)
        # Deep Fusion layer: Must be a Spatial FiLM layer (1x1 Conv)
        self.film = FiLMLayer(cond_dim, channels)

    def forward(self, x, cond):
        x_f = self.conv_f(x)
        x_w = self.conv_w(x)
        return F.gelu(self.film(x_f + x_w, cond))

class FNO(PDEModel):
    def __init__(
        self,
        modes,
        vis_channels,
        cond_channels,
        hidden_channels,
        proj_channels,
        x_dim=2,
        coord_channels=0,
        out_channels=None,
        n_layers=4,
        periodic=False,
        use_posn_embed: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.vis_channels = int(vis_channels)
        self.out_channels = int(out_channels) if out_channels is not None else self.vis_channels
        self.coord_channels = int(coord_channels)
        self.cond_channels = int(cond_channels)
        self.use_posn_embed = bool(use_posn_embed)
        self.modes = (modes, modes) if isinstance(modes, int) else modes[:2]

        # Cond dim is now strictly the number of spatial cond channels + 1 (for τ)
        self.cond_dim = self.cond_channels + 1

        # Calculate input channels dynamically
        spatial_extra = self.coord_channels if self.coord_channels > 0 else (x_dim if self.use_posn_embed else 0)
        in_channels = self.vis_channels + self.cond_channels + spatial_extra 
        # Note: -1 removed here because t_allhot is no longer concatenated to the input

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
        self.padding_x = 8 if not periodic else 0
        self.padding_y = 8 if not periodic else 0
        
        # Cache for positional embedding to prevent memory leaks
        self._pos_grid = None 

    def _get_pos_embed(self, shape, device):
        """Generates and caches the 2D grid to save memory and compute."""
        batch_size, _, nx, ny = shape
        if self._pos_grid is None or self._pos_grid.shape[-2:] != (nx, ny):
            grid_x = torch.linspace(0, 1, nx, device=device)
            grid_y = torch.linspace(0, 1, ny, device=device)
            pos_x, pos_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
            # Cache shape: [1, 2, X, Y]
            self._pos_grid = torch.stack([pos_x, pos_y], dim=0).unsqueeze(0)
            
        return self._pos_grid.expand(batch_size, -1, -1, -1)

    def forward(self, u, cond, t):
        batch_size, _, nx, ny = u.shape

        # 1. Format flow-matching time τ -> Spatial broadcast [B, 1, X, Y]
        if t.dim() == 0 or t.numel() == 1:
            t = t.view(1).expand(batch_size)
        t_spatial = t.view(batch_size, 1, 1, 1).expand(-1, -1, nx, ny).float()

        # 2. Build the Spatial Conditioning Tensor for FiLM [B, 1 + cond_channels, X, Y]
        if cond is not None:
            cond_vector = torch.cat([t_spatial, cond.float()], dim=1)
        else:
            cond_vector = t_spatial

        # 3. Build Spatial Input (u_in) without time concatenation
        u_in = u.float()
        
        if self.coord_channels == 0 and self.use_posn_embed:
            posn = self._get_pos_embed(u.shape, u.device)
            u_in = torch.cat((u_in, posn), dim=1)
            
        if cond is not None:
            u_in = torch.cat((u_in, cond.float()), dim=1)

        # 4. Handle Padding (Pad both the input AND the spatial condition vector)
        if not self.periodic:
            pad_tuple = (0, self.padding_y, 0, self.padding_x)
            u_in = F.pad(u_in, pad_tuple)
            cond_vector = F.pad(cond_vector, pad_tuple)

        # 5. Network Pass
        x = self.p(u_in)

        for block in self.blocks:
            x = block(x, cond_vector)

        if not self.periodic:
            x = x[..., :-self.padding_x, :-self.padding_y]

        return self.q(x)