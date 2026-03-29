"""
model.py — Convolutional U-Net for Spectrogram Mask Prediction

Architecture:
    Input  [B, 1, 1025, T]  (mono magnitude spectrogram)
    Encoder x4  → Bottleneck → Decoder x4  → Sigmoid soft mask
    Output [B, 1, 1025, T]  in range [0, 1]

The model predicts a soft mask applied directly to the mixture magnitude
spectrogram to isolate the drum source.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """
    Downsampling encoder block.
    Conv2d → BatchNorm2d → LeakyReLU → MaxPool2d(2,2)
    Returns (pooled_output, skip_connection).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.conv(x)
        return self.pool(skip), skip


class Bottleneck(nn.Module):
    """
    Central bottleneck; no spatial downsampling.
    Conv2d → BatchNorm2d → LeakyReLU (x2)
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Upsampling decoder block with skip connection.
    ConvTranspose2d(stride=2) → concat skip → Conv2d → BN → ReLU (x2)
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)

        # Align upsampled `x` to `skip` — ConvTranspose2d can be ±1 vs encoder
        # feature maps when dimensions are odd; pad if x is smaller, center-crop if larger.
        _, _, hs, ws = skip.shape
        _, _, hx, wx = x.shape

        if hx > hs:
            dh = hx - hs
            x = x[:, :, dh // 2 : dh // 2 + hs, :]
        if wx > ws:
            dw = wx - ws
            x = x[:, :, :, dw // 2 : dw // 2 + ws]

        _, _, hx, wx = x.shape
        diff_h = hs - hx
        diff_w = ws - wx
        if diff_h > 0 or diff_w > 0:
            x = F.pad(
                x,
                [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class DrumUNet(nn.Module):
    """
    U-Net spectrogram mask predictor for drum source separation.

    Input:  [B, 1, FREQ_BINS, T]   — mono mixture magnitude spectrogram
    Output: [B, 1, FREQ_BINS, T]   — soft mask in [0, 1]

    Channel progression:
        Encoder:     1 → 16 → 32 → 64 → 128
        Bottleneck:  128 → 256
        Decoder:     256 → 128 → 64 → 32 → 16
        Head:        16 → 1 + Sigmoid
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = EncoderBlock(1,   16)
        self.enc2 = EncoderBlock(16,  32)
        self.enc3 = EncoderBlock(32,  64)
        self.enc4 = EncoderBlock(64,  128)

        # Bottleneck
        self.bottleneck = Bottleneck(128, 256)

        # Decoder — in_ch matches bottleneck/previous decoder output;
        #            skip_ch matches the corresponding encoder skip output.
        self.dec4 = DecoderBlock(in_ch=256, skip_ch=128, out_ch=128)
        self.dec3 = DecoderBlock(in_ch=128, skip_ch=64,  out_ch=64)
        self.dec2 = DecoderBlock(in_ch=64,  skip_ch=32,  out_ch=32)
        self.dec1 = DecoderBlock(in_ch=32,  skip_ch=16,  out_ch=16)

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Encoder path — collect skip connections
        x, s1 = self.enc1(x)   # s1: [B, 16,  H/1,  T/1]
        x, s2 = self.enc2(x)   # s2: [B, 32,  H/2,  T/2]
        x, s3 = self.enc3(x)   # s3: [B, 64,  H/4,  T/4]
        x, s4 = self.enc4(x)   # s4: [B, 128, H/8,  T/8]

        # Bottleneck
        x = self.bottleneck(x) # x:  [B, 256, H/16, T/16]

        # Decoder path — fuse skip connections
        x = self.dec4(x, s4)   # [B, 128, H/8,  T/8]
        x = self.dec3(x, s3)   # [B, 64,  H/4,  T/4]
        x = self.dec2(x, s2)   # [B, 32,  H/2,  T/2]
        x = self.dec1(x, s1)   # [B, 16,  H/1,  T/1]

        return self.head(x)    # [B, 1,   H,    T]  — soft mask


def build_model() -> DrumUNet:
    """Factory returning an initialised DrumUNet."""
    return DrumUNet()
