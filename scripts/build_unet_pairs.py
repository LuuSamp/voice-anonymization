#!/usr/bin/env python3
"""Build `mix_wav,voice_wav` pairs for `scripts/train_unet.py`.

This converter targets manifests produced by:
  `scripts/prepare_sonyc_vox_mixes.py --mode train`

It reconstructs a voice-only stem per mix using `segments_json`:
- load each `vox_wav`
- resample to target SR and mono
- RMS-normalize (same as data prep)
- slice 1 second at `source_start_sample`
- RMS-normalize segment
- add segment at `insert_sample` into a zero-initialized stem

Output:
- one stem wav per mix under `--stems-dir`
- one CSV with columns `mix_wav,voice_wav` for `train_unet.py`
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def rms_normalize(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    rms = np.sqrt(np.mean(y**2) + eps)
    return (y / rms).astype(np.float32)


def load_mono_resampled(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32), target_sr


def parse_segments(segments_json: str) -> list[dict]:
    try:
        data = json.loads(segments_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid segments_json: {e}") from e
    if not isinstance(data, list):
        raise ValueError("segments_json must decode to a list")
    return data


def build_voice_stem(
    *,
    mix_len: int,
    segments: list[dict],
    target_sr: int,
) -> np.ndarray:
    seg_len = target_sr  # 1 second as in prep script
    stem = np.zeros(mix_len, dtype=np.float32)

    for seg in segments:
        vox_wav = Path(seg["vox_wav"])
        src_start = int(seg["source_start_sample"])
        insert = int(seg["insert_sample"])

        voice, _ = load_mono_resampled(vox_wav, target_sr)
        voice = rms_normalize(voice)

        if len(voice) >= seg_len:
            chunk = voice[src_start : src_start + seg_len]
            if len(chunk) < seg_len:
                pad = np.zeros(seg_len, dtype=np.float32)
                pad[: len(chunk)] = chunk
                chunk = pad
        else:
            chunk = np.zeros(seg_len, dtype=np.float32)
            chunk[: len(voice)] = voice

        chunk = rms_normalize(chunk)

        end = min(mix_len, insert + seg_len)
        if end > insert:
            stem[insert:end] += chunk[: end - insert]

    return stem


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert train mixing manifest to unet pair manifest (mix_wav,voice_wav)."
    )
    p.add_argument("--train-manifest", type=Path, required=True, help="manifest_train.csv path")
    p.add_argument("--pairs-manifest-out", type=Path, required=True, help="output CSV for train_unet.py")
    p.add_argument("--stems-dir", type=Path, required=True, help="where generated voice stems are written")
    p.add_argument("--target-sr", type=int, default=16000)
    p.add_argument("--limit", type=int, default=None, help="optional max rows")
    args = p.parse_args()

    args.stems_dir.mkdir(parents=True, exist_ok=True)
    args.pairs_manifest_out.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, str]] = []
    n_done = 0

    with args.train_manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"mix_wav", "segments_json"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise SystemExit("Expected manifest with columns at least: mix_wav, segments_json")

        for i, row in enumerate(reader):
            if args.limit is not None and n_done >= args.limit:
                break

            mix_path = Path(row["mix_wav"]).expanduser()
            if not mix_path.is_file():
                raise FileNotFoundError(f"Missing mix_wav file: {mix_path}")

            mix, _ = load_mono_resampled(mix_path, args.target_sr)
            segments = parse_segments(row["segments_json"])
            stem = build_voice_stem(mix_len=len(mix), segments=segments, target_sr=args.target_sr)

            stem_path = args.stems_dir / f"voice_{i:06d}.wav"
            sf.write(stem_path, stem, args.target_sr)

            rows_out.append(
                {
                    "mix_wav": str(mix_path.resolve()),
                    "voice_wav": str(stem_path.resolve()),
                }
            )
            n_done += 1

    with args.pairs_manifest_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mix_wav", "voice_wav"])
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    print(f"Wrote {n_done} stems to {args.stems_dir}")
    print(f"Wrote pairs manifest: {args.pairs_manifest_out}")


if __name__ == "__main__":
    main()

