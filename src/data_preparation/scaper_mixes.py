"""Scaper-based soundscape generation with first-class dB SNR control."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np

# Scaper 1.6.x uses np.Inf, removed in NumPy 2.0.
if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]

import soundfile as sf

try:
    import jams
    import scaper
except ImportError as e:
    raise ImportError(
        "Scaper is required for mix synthesis. Install with: pip install scaper\n"
        "Also ensure FFmpeg is installed; on Linux, libsox (sox) dev headers are "
        "needed to build the soxbindings dependency."
    ) from e

from src.data_preparation.soundbank.build import list_background_wavs

PresetName = Literal["train", "eval", "overlay"]

SNR_TIERS: dict[str, tuple[float, float]] = {
    "low": (0.0, 6.0),
    "medium": (6.0, 15.0),
    "high": (15.0, 30.0),
}

DEFAULT_REF_DB = -50.0
MANIFEST_FIELDS = [
    "seed",
    "preset",
    "mix_index",
    "mix_wav",
    "jams_path",
    "background_label",
    "background_source",
    "foreground_source",
    "n_events",
    "duration_sec",
    "snr_min_db",
    "snr_max_db",
    "snr_db_mean",
    "peak_normalized",
]


@dataclass(frozen=True)
class SnrRange:
    min_db: float
    max_db: float

    @classmethod
    def from_tier(cls, tier: str) -> SnrRange:
        if tier not in SNR_TIERS:
            raise ValueError(f"Unknown snr tier {tier!r}; choose from {list(SNR_TIERS)}")
        lo, hi = SNR_TIERS[tier]
        return cls(lo, hi)


@dataclass(frozen=True)
class PresetConfig:
    name: PresetName
    min_events: int
    max_events: int
    event_duration: float | None  # None => use full mix duration (eval)
    event_duration_min: float | None = None
    event_duration_max: float | None = None


PRESETS: dict[str, PresetConfig] = {
    "train": PresetConfig("train", min_events=3, max_events=5, event_duration=1.0),
    "eval": PresetConfig("eval", min_events=1, max_events=1, event_duration=None),
    "overlay": PresetConfig(
        "overlay",
        min_events=1,
        max_events=3,
        event_duration=None,
        event_duration_min=0.5,
        event_duration_max=4.0,
    ),
}


def peak_limit(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    m = float(np.max(np.abs(y)) + 1e-12)
    if m > peak:
        y = y * (peak / m)
    return y.astype(np.float32)


def _audio_duration(path: Path, target_sr: int) -> float:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return len(y) / sr


def _foreground_labels(soundbank_root: Path) -> list[str]:
    fg_root = soundbank_root / "foreground"
    if not fg_root.is_dir():
        return []
    return sorted(p.name for p in fg_root.iterdir() if p.is_dir())


def _snr_from_jams(jam: jams.JAMS) -> tuple[float, list[float]]:
    ann = jam.annotations.search(namespace="scaper")
    if not ann:
        return float("nan"), []
    snrs: list[float] = []
    for obs in ann[0].data:
        if obs.value.get("role") == "foreground" and "snr" in obs.value:
            snrs.append(float(obs.value["snr"]))
    if not snrs:
        return float("nan"), []
    return float(np.mean(snrs)), snrs


def _configure_scaper(
    duration: float,
    fg_folder: Path,
    bg_folder: Path,
    target_sr: int,
    ref_db: float,
) -> scaper.Scaper:
    sc = scaper.Scaper(float(duration), str(fg_folder), str(bg_folder))
    sc.sr = target_sr
    sc.ref_db = ref_db
    sc.n_channels = 1
    sc.fade_in_len = 0
    sc.fade_out_len = 0
    return sc


def _soundbank_audio_path(path: Path) -> str:
    """Absolute path without resolving symlinks (Scaper checks parent folder vs label)."""
    return str(path.absolute())


def _add_background(sc: scaper.Scaper, bg_label: str, bg_path: Path) -> None:
    sc.add_background(
        label=("const", bg_label),
        source_file=("const", _soundbank_audio_path(bg_path)),
        source_time=("const", 0.0),
    )


def _add_foreground_events(
    sc: scaper.Scaper,
    preset: PresetConfig,
    duration: float,
    snr: SnrRange,
    rng: np.random.Generator,
    fg_labels: list[str],
) -> int:
    if not fg_labels:
        raise ValueError("Soundbank has no foreground labels")

    n_events = int(rng.integers(preset.min_events, preset.max_events + 1))
    snr_tuple = ("uniform", snr.min_db, snr.max_db)

    for _ in range(n_events):
        if preset.event_duration is not None:
            event_dur = float(preset.event_duration)
            latest_start = max(0.0, duration - event_dur)
            event_time = ("uniform", 0.0, latest_start)
            event_duration = ("const", event_dur)
        elif preset.event_duration_min is not None and preset.event_duration_max is not None:
            event_dur = float(rng.uniform(preset.event_duration_min, preset.event_duration_max))
            event_dur = min(event_dur, duration)
            latest_start = max(0.0, duration - event_dur)
            event_time = ("uniform", 0.0, latest_start)
            event_duration = ("const", event_dur)
        else:
            event_time = ("const", 0.0)
            event_duration = ("const", float(duration))

        sc.add_event(
            label=("choose", fg_labels),
            source_file=("choose", []),
            source_time=("const", 0.0),
            event_time=event_time,
            event_duration=event_duration,
            snr=snr_tuple,
            pitch_shift=("const", 0.0),
            time_stretch=("const", 1.0),
        )
    return n_events


def generate_soundscape(
    *,
    soundbank_root: Path,
    preset_name: PresetName,
    bg_label: str,
    bg_path: Path,
    snr: SnrRange,
    out_mix_path: Path,
    out_jams_path: Path,
    rng: np.random.Generator,
    target_sr: int = 16000,
    ref_db: float = DEFAULT_REF_DB,
    peak_norm: bool = True,
) -> dict:
    """Generate one soundscape and return manifest row fields (without seed/index)."""
    preset = PRESETS[preset_name]
    duration = _audio_duration(bg_path, target_sr)
    if duration <= 0:
        raise ValueError(f"Background has zero duration: {bg_path}")

    fg_folder = soundbank_root / "foreground"
    bg_folder = soundbank_root / "background"
    fg_labels = _foreground_labels(soundbank_root)
    if not fg_labels:
        raise ValueError(f"No foreground labels under {fg_folder}")

    sc = _configure_scaper(duration, fg_folder, bg_folder, target_sr, ref_db)
    _add_background(sc, bg_label, bg_path)
    n_events = _add_foreground_events(sc, preset, duration, snr, rng, fg_labels)

    out_mix_path.parent.mkdir(parents=True, exist_ok=True)
    out_jams_path.parent.mkdir(parents=True, exist_ok=True)

    audio, jam, _, _ = sc.generate(
        audio_path=str(out_mix_path),
        jams_path=str(out_jams_path),
        allow_repeated_label=True,
        allow_repeated_source=True,
        reverb=None,
        fix_clipping=False,
        peak_normalization=False,
        save_isolated_events=False,
        disable_sox_warnings=True,
    )

    if peak_norm and audio is not None:
        limited = peak_limit(audio)
        sf.write(out_mix_path, limited, target_sr)

    snr_mean, _ = _snr_from_jams(jam)
    ann = jam.annotations.search(namespace="scaper")[0]
    fg_sources = [
        obs.value.get("source_file", "")
        for obs in ann.data
        if obs.value.get("role") == "foreground"
    ]
    fg_source = fg_sources[0] if len(fg_sources) == 1 else json.dumps(fg_sources)

    return {
        "preset": preset_name,
        "mix_wav": str(out_mix_path.resolve()),
        "jams_path": str(out_jams_path.resolve()),
        "background_label": bg_label,
        "background_source": _soundbank_audio_path(bg_path),
        "foreground_source": fg_source,
        "n_events": n_events,
        "duration_sec": f"{duration:.6f}",
        "snr_min_db": f"{snr.min_db:.4f}",
        "snr_max_db": f"{snr.max_db:.4f}",
        "snr_db_mean": f"{snr_mean:.4f}" if not np.isnan(snr_mean) else "",
        "peak_normalized": peak_norm,
    }


def generate_batch(
    *,
    soundbank_root: Path,
    preset_name: PresetName,
    out_dir: Path,
    snr: SnrRange,
    rng: np.random.Generator,
    target_sr: int = 16000,
    max_mixes: int | None = None,
    peak_norm: bool = True,
    out_subdir: str | None = None,
    seed: int = 0,
) -> list[dict]:
    """Generate one mix per background file (shuffled), return manifest rows."""
    backgrounds = list_background_wavs(soundbank_root)
    if not backgrounds:
        raise FileNotFoundError(f"No background WAVs under {soundbank_root / 'background'}")

    bg_list = list(backgrounds)
    rng.shuffle(bg_list)
    n_mix = len(bg_list) if max_mixes is None else min(len(bg_list), max_mixes)

    sub = out_subdir or preset_name
    wav_dir = out_dir / sub
    jams_dir = out_dir / sub / "jams"
    wav_dir.mkdir(parents=True, exist_ok=True)
    jams_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i in range(n_mix):
        bg_label, bg_path = bg_list[i]
        mix_path = wav_dir / f"mix_{i:06d}.wav"
        jams_path = jams_dir / f"mix_{i:06d}.jams"
        row = generate_soundscape(
            soundbank_root=soundbank_root,
            preset_name=preset_name,
            bg_label=bg_label,
            bg_path=bg_path,
            snr=snr,
            out_mix_path=mix_path,
            out_jams_path=jams_path,
            rng=rng,
            target_sr=target_sr,
            peak_norm=peak_norm,
        )
        row["seed"] = seed
        row["mix_index"] = i
        rows.append(row)
    return rows


def generate_voice_stem_from_jams(
    jams_path: Path,
    out_path: Path,
    *,
    target_sr: int | None = None,
) -> np.ndarray:
    """Regenerate foreground-only audio from a Scaper JAMS file."""
    jam = jams.load(str(jams_path))
    anns = jam.annotations.search(namespace="scaper")
    if not anns:
        raise ValueError(f"No scaper annotation in {jams_path}")

    ann = anns[0]
    fg_obs = [obs for obs in ann.data if obs.value.get("role") == "foreground"]
    if not fg_obs:
        duration = float(ann.sandbox.scaper.get("duration", 0))
        sr = int(ann.sandbox.scaper.get("sr", target_sr or 16000))
        silence = np.zeros(int(duration * sr), dtype=np.float32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, silence, sr)
        return silence

    fg_jam = jams.load(str(jams_path))
    fg_ann = fg_jam.annotations.search(namespace="scaper")[0]
    fg_ann.data = fg_obs

    with tempfile.NamedTemporaryFile(suffix=".jams", delete=False) as tmp:
        tmp_jams = Path(tmp.name)
    try:
        fg_jam.save(str(tmp_jams))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        audio, _, _, _ = scaper.generate_from_jams(
            str(tmp_jams),
            audio_outfile=str(out_path),
            disable_sox_warnings=True,
        )
    finally:
        tmp_jams.unlink(missing_ok=True)

    if audio is None:
        raise RuntimeError(f"Failed to generate voice stem from {jams_path}")

    if target_sr is not None:
        sr = int(fg_ann.sandbox.scaper.get("sr", target_sr))
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sf.write(out_path, audio, target_sr)

    return np.asarray(audio, dtype=np.float32)


def resolve_snr(
    snr_min: float | None,
    snr_max: float | None,
    snr_tier: str | None,
    preset_name: PresetName,
) -> SnrRange:
    if snr_min is not None and snr_max is not None:
        return SnrRange(float(snr_min), float(snr_max))
    if snr_tier is not None:
        return SnrRange.from_tier(snr_tier)
    # Sensible defaults when nothing specified
    if preset_name == "train":
        return SnrRange(20.0, 30.0)
    return SnrRange.from_tier("low")
