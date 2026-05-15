import torch
import torch.nn as nn

from models.base import PDEModel
from models.film import FiLMLayer


class UNet(PDEModel):
    """U-Net for time-conditioned field maps; optional extra ``coord`` channels (Sprint 2+)."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        coord_channels=0,
        cond_channels=0,
        t_scaling=1,
        film_param_dim=0,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.t_scaling = t_scaling
        self.field_channels = int(in_channels)
        self.coord_channels = int(coord_channels)
        self.cond_channels = int(cond_channels)
        enc_in = self.field_channels + self.cond_channels + self.coord_channels + 1
        fpd = int(film_param_dim) if film_param_dim else 0
        if fpd > 0:
            chs = [base_channels * m for m in (1, 2, 4, 8, 4, 2, 1)]
            self.film = nn.ModuleList([FiLMLayer(fpd, c) for c in chs])
        else:
            self.film = None

        self.enc1 = nn.Sequential(
            nn.Conv2d(enc_in, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(),
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.ReLU(),
        )

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(),
        )
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
        )

        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, t, u, cond=None, coords=None, **kwargs):
        t_orig = t
        t = t / self.t_scaling
        B, _, H, W = u.shape

        # 1. Spatial Inputs Setup (u + coords)
        if self.coord_channels > 0:
            expected_u = self.field_channels + self.coord_channels
            if u.shape[1] == expected_u:
                x_in = u
            elif coords is not None:
                x_in = torch.cat([u, coords], dim=1)
            else:
                raise ValueError("Coords expected but not provided.")
        else:
            x_in = u

        # 2. Build Deep Fusion Parameter Vector (t + cond)
        # Format t to [B, 1]
        if t.dim() == 0 or t.numel() == 1:
            t_vec = torch.ones(B, 1, device=u.device, dtype=torch.float32) * t
        elif t.dim() == 1:
            t_vec = t.unsqueeze(1).float()
        else:
            t_vec = t

        # Format cond to [B, C] and combine with t
        if cond is not None:
            if cond.dim() == 4:
                cond_vec = cond.mean(dim=[-1, -2]) # Or add a cond_encoder here too!
            elif cond.dim() == 1:
                cond_vec = cond.unsqueeze(1)
            else:
                cond_vec = cond
            params = torch.cat([t_vec, cond_vec], dim=1).float()
        else:
            params = t_vec.float()

        # 3. (Optional) Legacy Early Fusion Fallback 
        # If your UNet relies on spatial t/cond alongside FiLM, keep this. 
        # Otherwise, x_in can just be passed straight to enc1.
        if cond is not None and cond.dim() == 4:
            x_in = torch.cat([x_in, cond], dim=1)
            
        t_expanded = t_orig.view(B, 1, 1, 1).expand(-1, 1, H, W)
        x = torch.cat([x_in, t_expanded], dim=1)

        # 4. Encoder pass with triggered FiLM layers
        e1 = self.enc1(x)
        if self.film is not None:
            e1 = self.film[0](e1, params)
        e2 = self.enc2(e1)
        if self.film is not None:
            e2 = self.film[1](e2, params)
        e3 = self.enc3(e2)
        if self.film is not None:
            e3 = self.film[2](e3, params)
        b = self.bottleneck(e3)
        if self.film is not None:
            b = self.film[3](b, params)

        # 5. Decoder pass with triggered FiLM layers
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        if self.film is not None:
            d3 = self.film[4](d3, params)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        if self.film is not None:
            d2 = self.film[5](d2, params)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        if self.film is not None:
            d1 = self.film[6](d1, params)

        return self.out_conv(d1)