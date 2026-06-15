"""Inference helpers for VoiceSeparationUNet."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import librosa

from src.source_separation.stft import STFTConfig, magnitude_to_waveform
from src.source_separation.unet import VoiceSeparationUNet


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


def compute_voice_activity(
    y_mix: np.ndarray,
    sr: int,
    model: VoiceSeparationUNet,
    config: STFTConfig,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    """Run the same STFT + UNet patching as :func:`separate_voice`, but only return a per-frame scalar activity curve.

    For each STFT frame ``t``, ``activity[t] = mean_f ( mask[f,t] * mag[f,t] )`` using the UNet mask on the input magnitude.

    The full magnitude spectrogram is kept on **CPU** float32; only each ``(1, 1, F, n_frames)`` patch is moved to ``device`` for the forward pass, so GPU memory stays small. No ISTFT and no full-length ``voice_est`` are built.

    Returns
    -------
    activity
        Shape ``(T,)`` with ``T`` the number of STFT frames (same time axis as ``separate_voice`` would use before padding trim).
    sr_out
        Model sample rate (``config.sample_rate`` after resampling the mix).
    """
    y = np.asarray(y_mix, dtype=np.float32)
    if y.ndim > 1:
        y = librosa.to_mono(y)
    if sr != config.sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=config.sample_rate).astype(np.float32)
    sr_out = config.sample_rate

    wav = torch.from_numpy(y)
    window = torch.hann_window(config.n_fft, dtype=torch.float32)
    stft_full = torch.stft(
        wav,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.n_fft,
        window=window,
        center=config.center,
        return_complex=True,
    )
    mag = stft_full.abs()[: config.n_freq_bins, :].to(dtype=torch.float32)
    del stft_full
    T = int(mag.shape[-1])
    n_frames = config.n_frames
    n_chunks = (T + n_frames - 1) // n_frames
    T_pad = n_chunks * n_frames
    if T_pad > T:
        mag = torch.nn.functional.pad(mag, (0, T_pad - T))

    activity = torch.zeros(T_pad, dtype=torch.float32)
    with torch.no_grad():
        for c in range(n_chunks):
            sl = slice(c * n_frames, (c + 1) * n_frames)
            patch = mag[:, sl].to(device).unsqueeze(0).unsqueeze(0)
            mask = model(patch)
            # mean over frequency bins -> (n_frames,)
            act = (mask.squeeze(0).squeeze(0) * patch.squeeze(0).squeeze(0)).mean(dim=0).cpu()
            activity[sl] = act

    return activity[:T].numpy().astype(np.float32), sr_out


def crop_windows_from_activity(
    activity: np.ndarray,
    *,
    sr: int,
    hop_length: int,
    total_samples: int,
    window_sec: float = 10.0,
    quantile: float = 0.98,
    min_run_frames: int = 3,
    merge_gap_frames: int = 8,
    max_windows: int | None = None,
) -> list[tuple[int, int]]:
    """Threshold UNet activity, merge runs in frame index, build fixed-length sample windows.

    Returns a list of ``(start_sample, end_sample_exclusive)`` at ``sr``, sorted by start,
    with overlaps merged (union of intervals clipped to ``total_samples``).
    """
    if activity.size == 0 or total_samples <= 0:
        return []
    thr = float(np.quantile(activity.astype(np.float64), quantile))
    active = activity >= thr
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(active)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and active[j]:
            j += 1
        if j - i >= min_run_frames:
            runs.append((i, j))
        i = j

    if not runs:
        return []

    merged: list[tuple[int, int]] = []
    for a0, a1 in sorted(runs):
        if not merged:
            merged.append((a0, a1))
            continue
        p0, p1 = merged[-1]
        if a0 <= p1 + merge_gap_frames:
            merged[-1] = (p0, max(p1, a1))
        else:
            merged.append((a0, a1))

    half = int(round(0.5 * window_sec * sr))
    win_len = int(round(window_sec * sr))
    raw_spans: list[tuple[int, int, float]] = []
    for f0, f1 in merged:
        sub = activity[f0:f1]
        peak = float(np.max(sub)) if sub.size else 0.0
        center_frame = int(f0 + int(np.argmax(sub))) if sub.size else f0
        t_center = float(
            librosa.frames_to_time(np.array([center_frame], dtype=np.int64), sr=sr, hop_length=hop_length)[0]
        )
        center_sample = int(np.round(t_center * sr))
        center_sample = max(0, min(center_sample, total_samples - 1))
        s0 = center_sample - half
        s1 = s0 + win_len
        if s0 < 0:
            s0 = 0
            s1 = min(win_len, total_samples)
        if s1 > total_samples:
            s1 = total_samples
            s0 = max(0, s1 - win_len)
        s0 = max(0, s0)
        s1 = min(total_samples, s1)
        if s1 > s0:
            raw_spans.append((s0, s1, peak))

    raw_spans.sort(key=lambda x: -x[2])
    if max_windows is not None:
        raw_spans = raw_spans[: max(0, max_windows)]

    # merge overlapping sample intervals
    intervals = [(s0, s1) for s0, s1, _ in raw_spans]
    intervals.sort()
    out: list[tuple[int, int]] = []
    for s0, s1 in intervals:
        if not out:
            out.append((s0, s1))
            continue
        p0, p1 = out[-1]
        if s0 < p1:
            out[-1] = (p0, max(p1, s1))
        else:
            out.append((s0, s1))
    return out


def crop_windows_from_activity_with_config(
    activity: np.ndarray,
    config: STFTConfig,
    total_samples: int,
    **kwargs,
) -> list[tuple[int, int]]:
    """Like :func:`crop_windows_from_activity` but takes ``hop_length`` / ``sr`` from ``config``."""
    return crop_windows_from_activity(
        activity,
        sr=config.sample_rate,
        hop_length=config.hop_length,
        total_samples=total_samples,
        **kwargs,
    )


def splice_anonymized_segments(
    y_full: np.ndarray,
    segments: Sequence[tuple[int, np.ndarray]],
    *,
    crossfade_samples: int = 256,
) -> np.ndarray:
    """Paste each ``(start_sample, y_crop)`` into a copy of ``y_full``.

    Segments should be non-overlapping (e.g. after merging windows). Uses a short
    linear crossfade at the left and right edges of each pasted segment when possible.
    """
    out = np.asarray(y_full, dtype=np.float32).copy()
    cf = max(0, int(crossfade_samples))
    for s0, seg in segments:
        seg = np.asarray(seg, dtype=np.float32).reshape(-1)
        Lm = min(len(seg), len(out) - s0)
        if Lm <= 0:
            continue
        seg = seg[:Lm]
        if cf == 0 or Lm <= 1:
            out[s0 : s0 + Lm] = seg
            continue
        nfl = min(cf, s0, Lm // 2)
        nfr = min(cf, max(0, len(out) - (s0 + Lm)), Lm // 2)
        mid_l = s0 + nfl
        mid_r = s0 + Lm - nfr
        if nfl > 0:
            a = np.linspace(0.0, 1.0, nfl, endpoint=False, dtype=np.float32)
            out[s0:mid_l] = (1.0 - a) * out[s0:mid_l] + a * seg[:nfl]
        if mid_r > mid_l:
            out[mid_l:mid_r] = seg[nfl : Lm - nfr if nfr else Lm]
        if nfr > 0:
            a = np.linspace(0.0, 1.0, nfr, endpoint=False, dtype=np.float32)
            seg_tail = seg[Lm - nfr : Lm]
            out_tail = out[mid_r : s0 + Lm]
            out[mid_r : s0 + Lm] = (1.0 - a) * seg_tail + a * out_tail
    return out

