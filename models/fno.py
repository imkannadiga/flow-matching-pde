import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import PDEModel

def make_posn_embed(batch_size, dims):
    """Creates a standard 2D positional embedding [Batch, 2, X, Y]."""
    grid_x = torch.linspace(0, 1, dims[0])
    grid_y = torch.linspace(0, 1, dims[1])
    pos_x, pos_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
    pos = torch.stack([pos_x, pos_y], dim=0)
    return pos.unsqueeze(0).expand(batch_size, -1, -1, -1)

class FiLMLayer(nn.Module):
    def __init__(self, cond_dim, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 2)
        )

    def forward(self, x, cond):
        out = self.net(cond)
        scale, shift = out.chunk(2, dim=1)
        scale = scale.view(scale.shape[0], scale.shape[1], 1, 1)
        shift = shift.view(shift.shape[0], shift.shape[1], 1, 1)
        return x * (1 + scale) + shift

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNOBlock(nn.Module):
    def __init__(self, channels, modes1, modes2, cond_dim):
        super().__init__()
        self.conv_f = SpectralConv2d(channels, channels, modes1, modes2)
        self.conv_w = nn.Conv2d(channels, channels, 1)
        # Deep Fusion layer instantiated inside the block
        self.film = FiLMLayer(cond_dim, channels)

    def forward(self, x, cond):
        x_f = self.conv_f(x)
        x_w = self.conv_w(x)
        # Apply the physical/time conditions directly to the spectral features
        x_f = self.film(x_f, cond)
        return F.gelu(x_f + x_w)

class FNO(PDEModel):
    def __init__(
        self,
        modes,
        vis_channels,
        cond_channels,
        hidden_channels,
        proj_channels,
        x_dim=2,
        t_scaling=1,
        coord_channels=0,
        out_channels=None,
        n_layers=4,
        **kwargs,
    ):
        super().__init__()
        self.t_scaling = t_scaling
        self.vis_channels = int(vis_channels)
        self.out_channels = int(out_channels) if out_channels is not None else self.vis_channels
        self.coord_channels = int(coord_channels)
        
        if isinstance(modes, int):
            self.modes = (modes, modes)
        else:
            self.modes = modes[:2]
            
        # Unified deep condition dimension: Flow Time (1) + Physical params
        self.cond_dim = 1 + int(cond_channels)
        
        # Spatial input channels only (vis_channels + coords)
        spatial_extra = self.coord_channels if self.coord_channels > 0 else x_dim
        in_channels = self.vis_channels + spatial_extra

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

    def forward(self, u, cond, t):
        batch_size = u.shape[0]
        dims = u.shape[2:]

        # 1. Format Flow Time (t)
        t = t / self.t_scaling
        if t.dim() == 0 or t.numel() == 1:
            t = torch.ones(batch_size, 1, device=u.device, dtype=torch.float32) * t
        elif t.dim() == 1:
            t = t.unsqueeze(1).float()
            
        # 2. Format Conditioning Parameters (Extracting from spatial grid)
        if cond.dim() == 4:
            cond = cond.mean(dim=[-1, -2])
        elif cond.dim() == 1:
            cond = cond.unsqueeze(1)
            
        # 3. Create unified 1D conditioning vector
        cond_vector = torch.cat([t, cond], dim=1).float()

        # 4. Format Pure Spatial Input (Adding Positional Embeddings)
        if self.coord_channels > 0:
            if u.shape[1] != self.vis_channels + self.coord_channels:
                raise ValueError(f"Expected u with {self.vis_channels + self.coord_channels} channels.")
            u_in = u.float().contiguous()
        else:
            posn = make_posn_embed(batch_size, dims).to(u.device)
            u_in = torch.cat((u, posn), dim=1).float().contiguous()

        # 5. Lift the spatial state
        x = self.p(u_in)
        
        # 6. Pass through blocks with Deep Fusion conditioning
        for block in self.blocks:
            x = block(x, cond_vector)
            
        return self.q(x)