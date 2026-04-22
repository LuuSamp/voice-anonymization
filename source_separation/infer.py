"""Inference helpers for VoiceSeparationUNet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import librosa

from source_separation.stft import STFTConfig, magnitude_to_waveform
from source_separation.unet import VoiceSeparationUNet


def resolve_device(device: str = "auto") -> torch.device:
    if device == "cuda":
        return torch.device("cuda")
    if device == "mps":
        return torch.device("mps")
    if device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_unet_checkpoint(
    ckpt_path: str | Path,
    device: str | torch.device = "auto",
) -> tuple[VoiceSeparationUNet, STFTConfig, torch.device]:
    dev = resolve_device(device) if isinstance(device, str) else device
    model = VoiceSeparationUNet().to(dev)
    ckpt = torch.load(str(ckpt_path), map_location=dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    cfg_dict = ckpt.get("stft_config", {})
    config = STFTConfig(
        sample_rate=cfg_dict.get("sample_rate", 16000),
        n_fft=cfg_dict.get("n_fft", 1024),
        hop_length=cfg_dict.get("hop_length", 256),
        n_freq_bins=cfg_dict.get("n_freq_bins", 512),
        n_frames=cfg_dict.get("n_frames", 128),
        center=cfg_dict.get("center", True),
    )
    return model, config, dev


def separate_voice(
    y_mix: np.ndarray,
    sr: int,
    model: VoiceSeparationUNet,
    config: STFTConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (voice_est, background_est, sr_out)."""
    y = np.asarray(y_mix, dtype=np.float32)
    if y.ndim > 1:
        y = librosa.to_mono(y)
    if sr != config.sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=config.sample_rate).astype(np.float32)
    sr_out = config.sample_rate

    wav = torch.from_numpy(y).to(device)
    window = torch.hann_window(config.n_fft, device=device, dtype=wav.dtype)
    stft_full = torch.stft(
        wav,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.n_fft,
        window=window,
        center=config.center,
        return_complex=True,
    )
    mag = stft_full.abs()[: config.n_freq_bins, :]
    phase = torch.angle(stft_full)[: config.n_freq_bins, :]
    T = mag.shape[-1]
    n_frames = config.n_frames
    n_chunks = (T + n_frames - 1) // n_frames
    T_pad = n_chunks * n_frames
    if T_pad > T:
        mag = torch.nn.functional.pad(mag, (0, T_pad - T))
        phase = torch.nn.functional.pad(phase, (0, T_pad - T))

    mag_chunks = mag.view(config.n_freq_bins, n_chunks, n_frames).permute(1, 0, 2).unsqueeze(1)
    with torch.no_grad():
        mask_chunks = model(mag_chunks)
    est_mag_chunks = (mask_chunks * mag_chunks).squeeze(1)
    est_mag = est_mag_chunks.permute(1, 0, 2).reshape(config.n_freq_bins, T_pad)[:, :T]
    phase = phase[:, :T]

    y_voice = magnitude_to_waveform(est_mag, phase, config, length=len(y), window=window)
    y_voice = y_voice.detach().cpu().numpy().astype(np.float32)
    y_bg = (y - y_voice[: len(y)]).astype(np.float32)
    return y_voice, y_bg, sr_out

