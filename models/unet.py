import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import base_model

# --- U-Net Model for Flow Matching ---
class UNet(base_model.BaseModel):
    def __init__(self, in_channels=3, out_channels=2, base_channels=64, **kwargs):
        super().__init__()
        # Encoder
        # in_channels = input channels + time embedding
        in_channels = in_channels + 1
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.ReLU(),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1),
            nn.ReLU(),
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.ReLU(),
        )
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.ReLU(),
        )
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
        )

        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)  # predict velocity field

    def forward(self, t, u):
        """
        Forward pass for U-Net model.
        
        Args:
            t: Time tensor [B] or scalar
            u: Input tensor [B, C, H, W]
        """
        # Normalize time t to [0, 1] range
        # Use t_scaling if available, otherwise default to 50
        t_scaling = getattr(self, 't_scaling', 50)
        t = t / t_scaling
        # x_t shape: [B, C, H, W], t shape: [B] or scalar
        B, C, H, W = u.shape
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(B)
        elif t.numel() == 1:
            t = t.expand(B)
        t_expanded = t.view(B, 1, 1, 1).expand(-1, 1, H, W)
        x = torch.cat([u, t_expanded], dim=1)  # concat time as additional channel

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)  # [B, 2, H, W]
        return out

