"""Jansson-style U-Net mask network (Cohen–Hadria et al. 2019 §2.1, §3.6)."""

from __future__ import annotations

import torch
from torch import nn


class VoiceSeparationUNet(nn.Module):
    """Predict a multiplicative soft mask on the input magnitude spectrogram.

    Input shape: ``(N, 1, n_freq_bins, n_frames)`` with ``n_freq_bins=512``, ``n_frames=128``
    (see :class:`source_separation.stft.STFTConfig`).

    Output: mask in ``(0, 1)`` with same spatial size as input (sigmoid).
    """

    def __init__(self) -> None:
        super().__init__()
        # Encoder: 1 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512 (Jansson-style channel schedule)
        self.enc0 = _EncoderBlock(1, 16)
        self.enc1 = _EncoderBlock(16, 32)
        self.enc2 = _EncoderBlock(32, 64)
        self.enc3 = _EncoderBlock(64, 128)
        self.enc4 = _EncoderBlock(128, 256)
        self.enc5 = _EncoderBlock(256, 512)

        # Decoder: five merge steps + one final upsample without skip
        self.dec0 = _DecoderMergeBlock(512, 256, 256)
        self.dec1 = _DecoderMergeBlock(256, 128, 128)
        self.dec2 = _DecoderMergeBlock(128, 64, 64)
        self.dec3 = _DecoderMergeBlock(64, 32, 32)
        self.dec4 = _DecoderMergeBlock(32, 16, 16)

        self.dec_final = nn.Sequential(
            nn.ConvTranspose2d(
                16,
                16,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid mask matching ``x`` shape."""
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        bot = self.enc5(s4)

        x = self.dec0(bot, s4)
        x = self.dec1(x, s3)
        x = self.dec2(x, s2)
        x = self.dec3(x, s1)
        x = self.dec4(x, s0)
        x = self.dec_final(x)
        logits = self.out_conv(x)
        return torch.sigmoid(logits)


class _EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DecoderMergeBlock(nn.Module):
    """Transposed conv upsample, concat skip, 5x5 conv + BN + ReLU."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        self.merge = nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = _center_crop_or_pad_to(x, skip.shape[-2:])
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn(self.merge(x)))
        return x


def _center_crop_or_pad_to(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    H, W = hw
    h, w = x.shape[-2], x.shape[-1]
    if h == H and w == W:
        return x
    if h >= H and w >= W:
        dh, dw = h - H, w - W
        t, l = dh // 2, dw // 2
        return x[..., t : t + H, l : l + W]
    pad_h = max(0, H - h)
    pad_w = max(0, W - w)
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
