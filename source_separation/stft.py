"""STFT helpers for Cohen–Hadria et al. (2019) §3.6 (1024 / 256 / Hann, 16 kHz)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class STFTConfig:
    """Spectrogram parameters matching the paper."""

    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_freq_bins: int = 512
    """Use the first **512** magnitude bins (drop the Nyquist bin) so that after six
    stride-2 encoder stages, frequency size is divisible by ``2**6`` (512 / 64 = 8)."""
    n_frames: int = 128
    """Training patch length along time (paper §3.6)."""
    center: bool = True


def _trim_freq(S: torch.Tensor, n_keep: int) -> torch.Tensor:
    """S: (..., F_full, T) -> first n_keep bins."""
    return S[..., :n_keep, :]


def waveform_to_magnitude(
    waveform: torch.Tensor,
    config: STFTConfig,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute magnitude spectrogram(s).

    Parameters
    ----------
    waveform
        ``(samples,)`` or ``(batch, samples)``.
    Returns
    -------
        ``(F, T)`` or ``(batch, F, T)`` with ``F = n_freq_bins``.
    """
    squeeze_batch = False
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
        squeeze_batch = True
    if window is None:
        window = torch.hann_window(config.n_fft, device=waveform.device, dtype=waveform.dtype)

    mags = []
    for b in range(waveform.shape[0]):
        spec = torch.stft(
            waveform[b],
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.n_fft,
            window=window,
            center=config.center,
            return_complex=True,
            normalized=False,
        )
        mag = spec.abs()
        mag = _trim_freq(mag, config.n_freq_bins)
        mags.append(mag)
    out = torch.stack(mags, dim=0)
    if squeeze_batch:
        out = out.squeeze(0)
    return out


def magnitude_to_waveform(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    config: STFTConfig,
    length: int | None = None,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inverse STFT from magnitude and phase (same truncated freq layout as ``waveform_to_magnitude``)."""
    if window is None:
        window = torch.hann_window(config.n_fft, device=magnitude.device, dtype=magnitude.dtype)
    n_full = config.n_fft // 2 + 1
    pad_bins = n_full - magnitude.shape[-2]
    if pad_bins > 0:
        z = torch.zeros(
            *magnitude.shape[:-2],
            pad_bins,
            magnitude.shape[-1],
            device=magnitude.device,
            dtype=magnitude.dtype,
        )
        mag = torch.cat([magnitude, z], dim=-2)
        ph = torch.cat([phase, z], dim=-2)
    else:
        mag = magnitude
        ph = phase
    spec = torch.polar(mag, ph)
    return torch.istft(
        spec,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.n_fft,
        window=window,
        center=config.center,
        length=length,
    )


def crop_or_pad_time(
    mag: torch.Tensor,
    n_frames: int,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Crop or zero-pad along last dim (time frames) to exactly ``n_frames``."""
    t = mag.shape[-1]
    if t == n_frames:
        return mag
    if t > n_frames:
        start_max = t - n_frames
        if rng is not None:
            start = int(torch.randint(0, start_max + 1, (1,), generator=rng).item())
        else:
            start = (start_max + 1) // 2
        return mag[..., start : start + n_frames]
    pad = n_frames - t
    return torch.nn.functional.pad(mag, (0, pad))


def mag_batch_to_model_input(
    mag: torch.Tensor,
    config: STFTConfig,
    rng: torch.Generator | None = None,
    training: bool = True,
) -> torch.Tensor:
    """Magnitude ``(B, F, T)`` -> ``(B, 1, F, T)`` with fixed time length."""
    if mag.dim() == 2:
        mag = mag.unsqueeze(0)
    mag = crop_or_pad_time(mag, config.n_frames, rng=rng if training else None)
    return mag.unsqueeze(1)


def waveform_batch_to_model_input(
    waveform: torch.Tensor,
    config: STFTConfig,
    rng: torch.Generator | None = None,
    training: bool = True,
) -> torch.Tensor:
    """``(B, samples)`` -> ``(B, 1, n_freq_bins, n_frames)``."""
    mag = waveform_to_magnitude(waveform, config)
    return mag_batch_to_model_input(mag, config, rng=rng, training=training)
