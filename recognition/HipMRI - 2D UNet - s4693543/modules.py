"""
File: modules.py
Author: William Wright
Description: Source code for the components of the model. Each component is a class or function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downscale with MaxPool then DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """Upscale then concatenate skip, then DoubleConv"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

        self.bilinear = bilinear

    def forward(self, x, skip):
        if self.bilinear:
            x = self.up(x)
        else:
            x = self.up(x)

        # handle size mismatches
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 conv to logits"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class BasicUNet(nn.Module):
    """
    A vanilla U-Net with 4 levels (encoder-bottleneck-decoder),
    BatchNorm + ReLU, bilinear upsampling.
    """
    def __init__(self, in_channels=1, num_classes=6, base_features=64, bilinear=True):
        super().__init__()
        bf = base_features

        # Encoder
        self.inc   = DoubleConv(in_channels, bf)
        self.down1 = Down(bf, bf * 2)
        self.down2 = Down(bf * 2, bf * 4)
        self.down3 = Down(bf * 4, bf * 8)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(bf * 8, bf * 16 // factor)

        # Decoder
        self.up1 = Up(bf * 16, bf * 8 // factor, bilinear=bilinear)
        self.up2 = Up(bf * 8,  bf * 4 // factor, bilinear=bilinear)
        self.up3 = Up(bf * 4,  bf * 2 // factor, bilinear=bilinear)
        self.up4 = Up(bf * 2,  bf,               bilinear=bilinear)

        self.outc = OutConv(bf, num_classes)

    def forward(self, x):
        x1 = self.inc(x)          # (B, bf, H, W)
        x2 = self.down1(x1)       # (B, 2bf, H/2, W/2)
        x3 = self.down2(x2)       # (B, 4bf, H/4, W/4)
        x4 = self.down3(x3)       # (B, 8bf, H/8, W/8)
        x5 = self.down4(x4)       # (B, 16bf/f, H/16, W/16)

        u1 = self.up1(x5, x4)     # (B, 8bf/f, H/8, W/8)
        u2 = self.up2(u1, x3)     # (B, 4bf/f, H/4, W/4)
        u3 = self.up3(u2, x2)     # (B, 2bf/f, H/2, W/2)
        u4 = self.up4(u3, x1)     # (B, bf,    H,   W)

        logits = self.outc(u4)    # (B, num_classes, H, W)
        return logits

if __name__ == "__main__":
    model = BasicUNet(in_channels=1, 
                      num_classes=6, 
                      base_features=64, 
                      bilinear=True)
    param_sum = sum(p.numel() for p in model.parameters())
    print(f"----Total param sum: {param_sum}----")