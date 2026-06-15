"""Batch preprocess AudioSet clips for the UNet spectrogram dashboard."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

MANIFEST_NAME = "manifest.csv"
MIN_WAV_BYTES = 1000
SPEC_N_FFT = 1024
SPEC_HOP = 256

MANIFEST_FIELDS = [
    "sub_label",
    "stem",
    "mix_path",
    "voice_path",
    "bg_path",
    "mix_spec_path",
    "voice_spec_path",
    "bg_spec_path",
    "duration_sec",
    "voice_energy_ratio",
    "mix_mtime",
    "ckpt_name",
]


@dataclass(frozen=True)
class ClipRecord:
    sub_label: str
    stem: str
    mix_path: str
    voice_path: str
    bg_path: str
    mix_spec_path: str
    voice_spec_path: str
    bg_spec_path: str
    duration_sec: float
    voice_energy_ratio: float
    mix_mtime: float
    ckpt_name: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> ClipRecord:
        return cls(
            sub_label=row["sub_label"],
            stem=row["stem"],
            mix_path=row["mix_path"],
            voice_path=row["voice_path"],
            bg_path=row["bg_path"],
            mix_spec_path=row["mix_spec_path"],
            voice_spec_path=row["voice_spec_path"],
            bg_spec_path=row["bg_spec_path"],
            duration_sec=float(row["duration_sec"]),
            voice_energy_ratio=float(row["voice_energy_ratio"]),
            mix_mtime=float(row["mix_mtime"]),
            ckpt_name=row["ckpt_name"],
        )


def _stem_from_mix_path(mix_path: Path) -> str:
    name = mix_path.stem
    if name.endswith("_mix"):
        return name[: -len("_mix")]
    return name


def discover_mix_wavs(audio_root: Path) -> list[tuple[str, Path]]:
    """Return (sub_label, mix_wav_path) for every valid clip under audio_root."""
    found: list[tuple[str, Path]] = []
    if not audio_root.is_dir():
        return found
    for label_dir in sorted(audio_root.iterdir()):
        if not label_dir.is_dir():
            continue
        for wav in sorted(label_dir.glob("*.wav")):
            if wav.stat().st_size > MIN_WAV_BYTES:
                found.append((label_dir.name, wav))
    return found


def _output_paths(processed_root: Path, sub_label: str, stem: str) -> dict[str, Path]:
    out_dir = processed_root / sub_label
    return {
        "mix": out_dir / f"{stem}_mix.wav",
        "voice": out_dir / f"{stem}_voice.wav",
        "bg": out_dir / f"{stem}_bg.wav",
        "mix_spec": out_dir / f"{stem}_mix_spec.png",
        "voice_spec": out_dir / f"{stem}_voice_spec.png",
        "bg_spec": out_dir / f"{stem}_bg_spec.png",
    }


def _outputs_complete(paths: dict[str, Path]) -> bool:
    wav_ok = all(
        paths[k].is_file() and paths[k].stat().st_size > MIN_WAV_BYTES
        for k in ("mix", "voice", "bg")
    )
    spec_ok = all(
        paths[k].is_file() and paths[k].stat().st_size > 100
        for k in ("mix_spec", "voice_spec", "bg_spec")
    )
    return wav_ok and spec_ok


def save_spec_png(y: np.ndarray, sr: int, path: Path, *, title: str = "") -> None:
    """Write spectrogram PNG without changing the process matplotlib backend."""
    import matplotlib

    path.parent.mkdir(parents=True, exist_ok=True)
    S = np.abs(librosa.stft(y, n_fft=SPEC_N_FFT, hop_length=SPEC_HOP))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    with matplotlib.rc_context({"backend": "Agg"}):
        fig, ax = plt.subplots(figsize=(4, 2.2))
        librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=SPEC_HOP,
            x_axis="time",
            y_axis="hz",
            ax=ax,
        )
        if title:
            ax.set_title(title, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=80)
        plt.close(fig)


def _process_one_clip(
    sub_label: str,
    mix_path: Path,
    processed_root: Path,
    ckpt_name: str,
    model,
    config,
    device,
) -> ClipRecord:
    from src.source_separation.infer import separate_voice

    stem = mix_path.stem
    paths = _output_paths(processed_root, sub_label, stem)

    y_in, sr_in = librosa.load(str(mix_path), sr=None, mono=True)
    y_in = y_in.astype(np.float32)
    y_voice, y_bg, sr = separate_voice(y_in, sr_in, model, config, device)

    if sr_in != sr:
        y_mix = librosa.resample(y_in, orig_sr=sr_in, target_sr=sr).astype(np.float32)
    else:
        y_mix = y_in

    mix_energy = float(np.mean(y_mix**2)) + 1e-12
    voice_ratio = float(np.mean(y_voice**2) / mix_energy)

    paths["mix"].parent.mkdir(parents=True, exist_ok=True)
    for key, audio in (("mix", y_mix), ("voice", y_voice), ("bg", y_bg)):
        sf.write(str(paths[key]), audio, sr)

    save_spec_png(y_mix, sr, paths["mix_spec"], title="mix")
    save_spec_png(y_voice, sr, paths["voice_spec"], title="voice")
    save_spec_png(y_bg, sr, paths["bg_spec"], title="bg")

    mtime = mix_path.stat().st_mtime
    return ClipRecord(
        sub_label=sub_label,
        stem=stem,
        mix_path=str(paths["mix"]),
        voice_path=str(paths["voice"]),
        bg_path=str(paths["bg"]),
        mix_spec_path=str(paths["mix_spec"]),
        voice_spec_path=str(paths["voice_spec"]),
        bg_spec_path=str(paths["bg_spec"]),
        duration_sec=len(y_mix) / sr,
        voice_energy_ratio=voice_ratio,
        mix_mtime=mtime,
        ckpt_name=ckpt_name,
    )


def load_manifest(processed_root: Path) -> list[ClipRecord]:
    manifest_path = processed_root / MANIFEST_NAME
    if not manifest_path.is_file():
        return []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [ClipRecord.from_row(row) for row in reader]


def write_manifest(processed_root: Path, records: list[ClipRecord]) -> None:
    processed_root.mkdir(parents=True, exist_ok=True)
    manifest_path = processed_root / MANIFEST_NAME
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "sub_label": rec.sub_label,
                    "stem": rec.stem,
                    "mix_path": rec.mix_path,
                    "voice_path": rec.voice_path,
                    "bg_path": rec.bg_path,
                    "mix_spec_path": rec.mix_spec_path,
                    "voice_spec_path": rec.voice_spec_path,
                    "bg_spec_path": rec.bg_spec_path,
                    "duration_sec": f"{rec.duration_sec:.4f}",
                    "voice_energy_ratio": f"{rec.voice_energy_ratio:.6f}",
                    "mix_mtime": f"{rec.mix_mtime:.6f}",
                    "ckpt_name": rec.ckpt_name,
                }
            )


def _manifest_index(records: list[ClipRecord]) -> dict[tuple[str, str], ClipRecord]:
    return {(r.sub_label, r.stem): r for r in records}


def labels_with_clips(processed_root: Path) -> list[tuple[str, int]]:
    records = load_manifest(processed_root)
    counts: dict[str, int] = {}
    for rec in records:
        counts[rec.sub_label] = counts.get(rec.sub_label, 0) + 1
    return sorted(counts.items(), key=lambda x: x[0])


def clips_for_label(processed_root: Path, sub_label: str) -> list[ClipRecord]:
    return sorted(
        (r for r in load_manifest(processed_root) if r.sub_label == sub_label),
        key=lambda r: r.stem,
    )


def paths_for_clip(record: ClipRecord) -> dict[str, Path]:
    return {
        "mix": Path(record.mix_path),
        "voice": Path(record.voice_path),
        "bg": Path(record.bg_path),
        "mix_spec": Path(record.mix_spec_path),
        "voice_spec": Path(record.voice_spec_path),
        "bg_spec": Path(record.bg_spec_path),
    }


def preprocess_all(
    audio_root: Path,
    processed_root: Path,
    ckpt_path: Path,
    *,
    force: bool = False,
    show_progress: bool = True,
) -> dict[str, int]:
    """Run UNet on all valid mixes; write stems, spec PNGs, and manifest."""
    audio_root = Path(audio_root)
    processed_root = Path(processed_root)
    ckpt_path = Path(ckpt_path)
    ckpt_name = ckpt_path.name

    mixes = discover_mix_wavs(audio_root)
    existing = load_manifest(processed_root)
    index = _manifest_index(existing)
    records: list[ClipRecord] = list(existing)

    from src.source_separation.infer import load_unet_checkpoint

    model = config = device = None
    stats = {"processed": 0, "skipped": 0, "failed": 0}

    iterator = mixes
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(mixes, desc="preprocess")
        except ImportError:
            pass

    for sub_label, mix_path in iterator:
        stem = mix_path.stem
        paths = _output_paths(processed_root, sub_label, stem)
        mtime = mix_path.stat().st_mtime

        if not force:
            prev = index.get((sub_label, stem))
            if (
                prev is not None
                and abs(prev.mix_mtime - mtime) < 1e-6
                and prev.ckpt_name == ckpt_name
                and _outputs_complete(paths)
            ):
                stats["skipped"] += 1
                continue

        try:
            if model is None:
                model, config, device = load_unet_checkpoint(ckpt_path)
            rec = _process_one_clip(
                sub_label, mix_path, processed_root, ckpt_name, model, config, device
            )
            index[(sub_label, stem)] = rec
            records = [r for r in records if (r.sub_label, r.stem) != (sub_label, stem)]
            records.append(rec)
            stats["processed"] += 1
        except Exception as exc:
            stats["failed"] += 1
            if show_progress:
                print(f"FAILED {sub_label}/{mix_path.name}: {exc}")

    write_manifest(processed_root, sorted(records, key=lambda r: (r.sub_label, r.stem)))
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess AudioSet clips for dashboard.")
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("datasets/audioset/audio"),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("datasets/audioset/processed"),
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("checkpoints/unet_run1/unet_voice_sep.pt"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = preprocess_all(
        args.audio_root,
        args.processed_root,
        args.ckpt,
        force=args.force,
    )
    print(summary)
