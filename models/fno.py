import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import PDEModel

class FiLMLayer(nn.Module):
    def __init__(self, cond_dim, num_features):
        super().__init__()
        # Takes the 1D conditioning vector and outputs scale and shift for the spatial channels
        self.net = nn.Sequential(
            nn.Linear(cond_dim, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 2)
        )

    def forward(self, x, cond):
        out = self.net(cond)
        scale, shift = out.chunk(2, dim=1)
        # Reshape to broadcast across the spatial dimensions (B, C, 1, 1)
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
        self.film = FiLMLayer(cond_dim, channels)

    def forward(self, x, cond):
        x_f = self.conv_f(x)
        x_w = self.conv_w(x)
        # Modulate the spectral features using the unified conditioning vector
        x_f = self.film(x_f, cond)
        return F.gelu(x_f + x_w)

class FNO(PDEModel):
    def __init__(
        self,
        modes,
        vis_channels,      # Number of pure physical channels in 'u' (e.g., 1 for height)
        cond_channels,     # Number of conditioning parameters coming from your pipeline
        hidden_channels,
        proj_channels,
        x_dim=2,
        t_scaling=1,
        out_channels=None,
        **kwargs,
    ):
        super().__init__()
        self.t_scaling = t_scaling
        self.vis_channels = int(vis_channels)
        self.out_channels = int(out_channels) if out_channels is not None else self.vis_channels
        
        if isinstance(modes, int):
            self.modes = (modes, modes)
        else:
            self.modes = modes[:2]
            
        # Total conditioning dimension = flow time (1) + physical conditioning params
        self.cond_dim = 1 + cond_channels
        
        # Lifting layer purely for the spatial state 'u'
        self.p = nn.Conv2d(self.vis_channels, hidden_channels, 1)
        
        self.blocks = nn.ModuleList([
            FNOBlock(hidden_channels, self.modes[0], self.modes[1], self.cond_dim)
            for _ in range(4)
        ])
        
        self.q = nn.Sequential(
            nn.Conv2d(hidden_channels, proj_channels, 1),
            nn.GELU(),
            nn.Conv2d(proj_channels, self.out_channels, 1)
        )

    def forward(self, u, cond, t, **kwargs):
        """
        u: The pure spatial state [Batch, vis_channels, X, Y]
        conditioning: The physical parameters [Batch, cond_channels] 
        t: The flow matching time [Batch] or [Batch, 1]
        """
        # 1. Format Flow Time (t)
        t = t / self.t_scaling
        if t.dim() == 0 or t.numel() == 1:
            t = torch.ones(u.shape[0], 1, device=u.device, dtype=torch.float32) * t
        elif t.dim() == 1:
            t = t.unsqueeze(1).float()
            
        # 2. Format Conditioning Parameters
        if cond.dim() == 1:
            cond = cond.unsqueeze(1).float()
            
        # 3. Create unified 1D conditioning vector [Batch, cond_dim]
        cond_vector = torch.cat([t, cond], dim=1).float()
        
        # 4. Lift the pure spatial state 'u'
        x = self.p(u)
        
        # 5. Pass through blocks, applying deep conditioning at every step
        for block in self.blocks:
            x = block(x, cond_vector)
            
        # 6. Project to output channels
        x = self.q(x)
        return x