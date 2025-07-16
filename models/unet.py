import torch
import torch.nn as nn
import torch.nn.functional as F

# --- U-Net Model for Flow Matching ---
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, base_channels=64, **kwargs):
        super().__init__()
        # Encoder
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

    def forward(self, x_t, t):
        # Normalize time t to [0, 1] range
        t = t / 50 # Hardcoded for 50 timesteps, adjust as needed 
        # x_t shape: [B, 2, H, W], t shape: [B,1]
        B, C, H, W = x_t.shape
        t_expanded = t.view(B, 1, 1, 1).expand(-1, 1, H, W)
        x = torch.cat([x_t, t_expanded], dim=1)  # concat time as 3rd channel

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

