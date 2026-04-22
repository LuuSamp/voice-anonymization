#!/usr/bin/env python3
"""
Prepare SONYC + VoxCeleb mixtures per Cohen–Hadria et al. (2019), §3.1.4.

SONYC-UST: https://doi.org/10.5281/zenodo.3692954

Expected layout under --sonyc-dataset-root:
  audio-dev/train/*.wav     for annotations split \"train\"
  audio-dev/validate/*.wav  for split \"validate\"
  audio-eval/*.wav          for split \"test\" (flat)

VoxCeleb WAVs: any tree under --vox-root; speaker id folder (e.g. id10002) must match
vox1_meta.csv \"VoxCeleb1 ID\". Filter by \"Set\": dev (train mixes), test (eval mixes).
If eval mode fails with no test WAVs, the subset on disk may only include dev speakers;
use --use-training-vox to run eval mixes with dev voices.

Examples (from repo root, venv activated)::

  python scripts/prepare_sonyc_vox_mixes.py --mode train --out-dir data/mixes_train
  python scripts/prepare_sonyc_vox_mixes.py --mode eval --out-dir data/mixes_eval --eval-snr high
  python scripts/prepare_sonyc_vox_mixes.py --mode eval --out-dir data/mixes_eval --use-training-vox

Post-mix peak limiting to 0.99 max abs is applied by default (not in the paper); use
--no-peak-limit to disable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    import librosa
    import soundfile as sf
except ImportError as e:
    raise SystemExit(
        "Missing dependency. Install with: pip install librosa soundfile numpy"
    ) from e


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_sonyc_wav(dataset_root: Path, split: str, audio_filename: str) -> Path:
    """Map annotations (split, audio_filename) to a WAV path."""
    if split == "train":
        return dataset_root / "audio-dev" / "train" / audio_filename
    if split == "validate":
        return dataset_root / "audio-dev" / "validate" / audio_filename
    if split == "test":
        return dataset_root / "audio-eval" / audio_filename
    raise ValueError(f"Unknown SONYC split: {split!r}")


def load_unique_sonyc_files(
    annotations_path: Path, split_filter: str
) -> list[tuple[str, str]]:
    """Return sorted unique (split, audio_filename) rows for the given split."""
    seen: set[tuple[str, str]] = set()
    with annotations_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"].strip()
            if split != split_filter:
                continue
            fn = row["audio_filename"].strip()
            seen.add((split, fn))
    return sorted(seen)


def parse_vox_meta(meta_path: Path) -> dict[str, str]:
    """Map VoxCeleb1 ID -> Set (dev/test)."""
    out: dict[str, str] = {}
    with meta_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            vid = row["VoxCeleb1 ID"].strip()
            st = row["Set"].strip().lower()
            out[vid] = st
    return out


def collect_vox_wavs(vox_root: Path, meta: dict[str, str], wanted_set: str) -> list[Path]:
    """All *.wav files under vox_root whose path contains a speaker id with Set==wanted_set."""
    allowed = {vid for vid, st in meta.items() if st == wanted_set}
    if not allowed:
        raise SystemExit(f"No speaker IDs with Set={wanted_set!r} in vox meta.")

    paths: list[Path] = []
    for p in vox_root.rglob("*.wav"):
        parts_set = set(p.parts)
        if parts_set & allowed:
            paths.append(p)
    paths.sort()
    if not paths:
        raise SystemExit(
            f"No WAV files found under {vox_root} for Set={wanted_set!r}. "
            "Check --vox-root and extracted audio. "
            "Partial VoxCeleb subsets may omit test-set speakers entirely."
        )
    return paths


def rms_normalize(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale signal to unit RMS (Cohen–Hadria §3.1.4)."""
    y = np.asarray(y, dtype=np.float64)
    rms = np.sqrt(np.mean(y**2) + eps)
    return (y / rms).astype(np.float32)


def load_mono_16k(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Load audio as mono, resample to target_sr."""
    y, sr = librosa.load(path, mono=True, sr=None)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32), target_sr


def peak_limit(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Scale so max |y| <= peak (post-mix safety; not in paper, documented here)."""
    y = np.asarray(y, dtype=np.float64)
    m = np.max(np.abs(y)) + 1e-12
    if m > peak:
        y = y * (peak / m)
    return y.astype(np.float32)


def extract_one_second_segment(
    voice: np.ndarray, rng: np.random.Generator, seg_len: int
) -> tuple[np.ndarray, int]:
    """Random contiguous segment of length seg_len; pad with zeros if voice is shorter."""
    if len(voice) >= seg_len:
        start = int(rng.integers(0, len(voice) - seg_len + 1))
        seg = voice[start : start + seg_len].copy()
    else:
        start = 0
        seg = np.zeros(seg_len, dtype=np.float32)
        seg[: len(voice)] = voice.astype(np.float32)
    return seg, start


def run_train(
    *,
    rng: np.random.Generator,
    backgrounds: list[tuple[str, str]],
    vox_wavs: list[Path],
    dataset_root: Path,
    out_dir: Path,
    target_sr: int,
    max_mixes: int | None,
    peak_norm: bool,
) -> list[dict]:
    """N ~ Uniform{3,4,5} one-second segments, RMS-normalized, summed on RMS-normalized bg."""
    seg_len = target_sr  # 1 second
    out_sub = out_dir / "train"
    out_sub.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    bg_list = list(backgrounds)
    rng.shuffle(bg_list)
    n_mix = len(bg_list) if max_mixes is None else min(len(bg_list), max_mixes)

    for i in range(n_mix):
        split, fn = bg_list[i]
        bg_path = resolve_sonyc_wav(dataset_root, split, fn)
        if not bg_path.is_file():
            raise FileNotFoundError(f"Missing SONYC file: {bg_path}")

        bg, _ = load_mono_16k(bg_path, target_sr)
        bg = rms_normalize(bg)
        mix = bg.copy()
        n_seg = int(rng.choice([3, 4, 5]))

        segments: list[dict] = []
        for _ in range(n_seg):
            vw = vox_wavs[int(rng.integers(0, len(vox_wavs)))]
            voice, _ = load_mono_16k(vw, target_sr)
            voice = rms_normalize(voice)
            seg, src_start = extract_one_second_segment(voice, rng, seg_len)
            seg = rms_normalize(seg)

            max_pos = max(0, len(mix) - seg_len)
            insert = int(rng.integers(0, max_pos + 1))
            mix[insert : insert + seg_len] += seg

            segments.append(
                {
                    "vox_wav": str(vw.resolve()),
                    "source_start_sample": src_start,
                    "insert_sample": insert,
                }
            )

        if peak_norm:
            mix = peak_limit(mix)

        out_wav = out_sub / f"mix_{i:06d}.wav"
        sf.write(out_wav, mix, target_sr)

        rows.append(
            {
                "mode": "train",
                "mix_index": i,
                "mix_wav": str(out_wav.resolve()),
                "background_wav": str(bg_path.resolve()),
                "sonyc_split": split,
                "audio_filename": fn,
                "n_segments": n_seg,
                "segments_json": json.dumps(segments),
                "peak_normalized": peak_norm,
            }
        )
    return rows


def prepare_voice_eval_length(voice: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Trim (center) or pad voice to n_samples to match background."""
    if len(voice) >= n_samples:
        start = int(rng.integers(0, len(voice) - n_samples + 1))
        return voice[start : start + n_samples].copy()
    out = np.zeros(n_samples, dtype=np.float32)
    out[: len(voice)] = voice
    return out


def run_eval(
    *,
    rng: np.random.Generator,
    backgrounds: list[tuple[str, str]],
    vox_wavs: list[Path],
    dataset_root: Path,
    out_dir: Path,
    target_sr: int,
    max_mixes: int | None,
    snr: str,
    peak_norm: bool,
) -> list[dict]:
    """mix = alpha * voice + (1-alpha) * background; RMS before mixing."""
    if snr == "low":
        alpha_lo, alpha_hi = 0.1, 0.4
        sub = "eval_low_snr"
    elif snr == "high":
        alpha_lo, alpha_hi = 0.5, 0.7
        sub = "eval_high_snr"
    else:
        raise ValueError(snr)

    out_sub = out_dir / sub
    out_sub.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    bg_list = list(backgrounds)
    rng.shuffle(bg_list)
    n_bg = len(bg_list)
    n_mix = n_bg if max_mixes is None else min(n_bg, max_mixes)

    for i in range(n_mix):
        split, fn = bg_list[i]
        bg_path = resolve_sonyc_wav(dataset_root, split, fn)
        if not bg_path.is_file():
            raise FileNotFoundError(f"Missing SONYC file: {bg_path}")

        bg, _ = load_mono_16k(bg_path, target_sr)
        bg = rms_normalize(bg)
        n_samples = len(bg)

        vw = vox_wavs[int(rng.integers(0, len(vox_wavs)))]
        voice, _ = load_mono_16k(vw, target_sr)
        voice = rms_normalize(voice)
        voice = prepare_voice_eval_length(voice, n_samples, rng)

        alpha = float(rng.uniform(alpha_lo, alpha_hi))
        mix = alpha * voice.astype(np.float64) + (1.0 - alpha) * bg.astype(np.float64)
        mix = mix.astype(np.float32)
        if peak_norm:
            mix = peak_limit(mix)

        out_wav = out_sub / f"mix_{i:06d}.wav"
        sf.write(out_wav, mix, target_sr)

        rows.append(
            {
                "mode": "eval",
                "eval_snr": snr,
                "mix_index": i,
                "mix_wav": str(out_wav.resolve()),
                "background_wav": str(bg_path.resolve()),
                "vox_wav": str(vw.resolve()),
                "sonyc_split": split,
                "audio_filename": fn,
                "alpha": f"{alpha:.8f}",
                "peak_normalized": peak_norm,
            }
        )
    return rows


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main(argv: list[str] | None = None) -> int:
    repo = _repo_root()
    p = argparse.ArgumentParser(
        description="Mix SONYC-UST backgrounds with VoxCeleb voices (Cohen–Hadria 2019 §3.1.4)."
    )
    p.add_argument(
        "--mode",
        choices=("train", "eval"),
        required=True,
        help="train: N random 1s segments (§3.1.4); eval: alpha-weighted full-length mix",
    )
    p.add_argument(
        "--sonyc-annotations",
        type=Path,
        default=repo / "datasets/sonyc-v1-dataset/annotations.csv",
    )
    p.add_argument(
        "--sonyc-dataset-root",
        type=Path,
        default=repo / "datasets/sonyc-v1-dataset",
        help="Contains audio-dev/ and audio-eval/",
    )
    p.add_argument(
        "--vox-root",
        type=Path,
        default=repo / "datasets/voxceleb1-audio-wav-files-for-india-celebrity",
    )
    p.add_argument(
        "--vox-meta",
        type=Path,
        default=repo
        / "datasets/voxceleb1-audio-wav-files-for-india-celebrity/vox1_meta.csv",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for WAVs + manifest")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-sr", type=int, default=16000)
    p.add_argument("--max-mixes", type=int, default=None, help="Limit number of output mixes")
    p.add_argument(
        "--eval-snr",
        choices=("low", "high"),
        default="low",
        help="For --mode eval: Low alpha in [0.1,0.4], High in [0.5,0.7]",
    )
    p.add_argument(
        "--use-training-vox",
        action="store_true",
        help="In eval mode, use Vox meta Set=dev instead of Set=test.",
    )
    p.add_argument(
        "--no-peak-limit",
        action="store_true",
        help="Disable post-mix peak limiting (default: scale mix to max abs 0.99 if needed)",
    )
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    meta = parse_vox_meta(args.vox_meta)
    peak_norm = not args.no_peak_limit

    if args.mode == "train":
        backgrounds = load_unique_sonyc_files(args.sonyc_annotations, "train")
        if not backgrounds:
            raise SystemExit("No rows with split=train in annotations.")
        vox_wavs = collect_vox_wavs(args.vox_root, meta, "dev")
        rows = run_train(
            rng=rng,
            backgrounds=backgrounds,
            vox_wavs=vox_wavs,
            dataset_root=args.sonyc_dataset_root,
            out_dir=args.out_dir,
            target_sr=args.target_sr,
            max_mixes=args.max_mixes,
            peak_norm=peak_norm,
        )
        manifest_name = "manifest_train.csv"
        fields = [
            "mode",
            "mix_index",
            "mix_wav",
            "background_wav",
            "sonyc_split",
            "audio_filename",
            "n_segments",
            "segments_json",
            "peak_normalized",
        ]
    else:
        backgrounds = load_unique_sonyc_files(args.sonyc_annotations, "test")
        if not backgrounds:
            raise SystemExit("No rows with split=test in annotations.")
        eval_vox_set = "dev" if args.use_training_vox else "test"
        vox_wavs = collect_vox_wavs(args.vox_root, meta, eval_vox_set)
        rows = run_eval(
            rng=rng,
            backgrounds=backgrounds,
            vox_wavs=vox_wavs,
            dataset_root=args.sonyc_dataset_root,
            out_dir=args.out_dir,
            target_sr=args.target_sr,
            max_mixes=args.max_mixes,
            snr=args.eval_snr,
            peak_norm=peak_norm,
        )
        manifest_name = f"manifest_eval_{args.eval_snr}.csv"
        fields = [
            "mode",
            "eval_snr",
            "mix_index",
            "mix_wav",
            "background_wav",
            "vox_wav",
            "sonyc_split",
            "audio_filename",
            "alpha",
            "peak_normalized",
        ]

    for row in rows:
        row["seed"] = args.seed
    fields = ["seed"] + fields

    write_manifest(args.out_dir / manifest_name, fields, rows)
    print(f"Wrote {len(rows)} mixes under {args.out_dir}")
    print(f"Manifest: {args.out_dir / manifest_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
