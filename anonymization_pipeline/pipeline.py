"""End-to-end anonymization pipeline: separation -> blur -> remix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from source_separation import STFTConfig, VoiceSeparationUNet, separate_voice
from voice_blurring import low_pass_blur, mfcc_inversion_blur


@dataclass
class AnonymizationResult:
    sr: int
    voice_est: np.ndarray
    background_est: np.ndarray
    blurred_voice: np.ndarray
    anonymized_mix: np.ndarray
    blur_mode: str


def _align_len(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) == n:
        return y
    if len(y) > n:
        return y[:n].astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    out[: len(y)] = y
    return out


def _blur_voice(
    voice_est: np.ndarray,
    sr: int,
    blur_mode: str,
    low_pass_kwargs: dict[str, Any] | None = None,
    mfcc_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, int]:
    low_pass_kwargs = low_pass_kwargs or {}
    mfcc_kwargs = mfcc_kwargs or {}
    mode = blur_mode.lower()
    if mode == "low_pass":
        return low_pass_blur(voice_est, sr, **low_pass_kwargs)
    if mode == "mfcc":
        return mfcc_inversion_blur(voice_est, sr, **mfcc_kwargs), sr
    if mode == "cascade":
        y_lp, sr_lp = low_pass_blur(voice_est, sr, **low_pass_kwargs)
        return mfcc_inversion_blur(y_lp, sr_lp, **mfcc_kwargs), sr_lp
    raise ValueError("blur_mode must be one of: cascade, low_pass, mfcc")


def anonymize_audio(
    y_mix: np.ndarray,
    sr: int,
    *,
    model: VoiceSeparationUNet,
    config: STFTConfig,
    device,
    blur_mode: str = "cascade",
    low_pass_kwargs: dict[str, Any] | None = None,
    mfcc_kwargs: dict[str, Any] | None = None,
) -> AnonymizationResult:
    """Run full pipeline and return all intermediate and final signals."""
    voice_est, background_est, sr_sep = separate_voice(y_mix, sr, model, config, device)
    blurred_voice, sr_blur = _blur_voice(
        voice_est,
        sr_sep,
        blur_mode=blur_mode,
        low_pass_kwargs=low_pass_kwargs,
        mfcc_kwargs=mfcc_kwargs,
    )
    if sr_blur != sr_sep:
        # Keep remixing sample rate consistent with separation output.
        import librosa

        blurred_voice = librosa.resample(blurred_voice.astype(np.float32), orig_sr=sr_blur, target_sr=sr_sep)
        sr_blur = sr_sep

    n = max(len(background_est), len(blurred_voice))
    bg = _align_len(background_est, n)
    bv = _align_len(blurred_voice, n)
    mix = (bg + bv).astype(np.float32)
    return AnonymizationResult(
        sr=sr_sep,
        voice_est=voice_est.astype(np.float32),
        background_est=background_est.astype(np.float32),
        blurred_voice=bv,
        anonymized_mix=mix,
        blur_mode=blur_mode,
    )

