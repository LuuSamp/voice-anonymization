#!/usr/bin/env python3
"""Generate synthetic training/eval mixes with Scaper."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data_preparation.scaper_mixes import (
    MANIFEST_FIELDS,
    PRESETS,
    SNR_TIERS,
    generate_batch,
    resolve_snr,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Scaper soundscapes from a built soundbank."
    )
    p.add_argument(
        "--soundbank",
        type=Path,
        required=True,
        help="Soundbank root (contains foreground/ and background/)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for mixes, jams, and manifest CSV",
    )
    p.add_argument(
        "--preset",
        choices=tuple(PRESETS.keys()),
        default="train",
        help="Event layout preset (SNR controlled separately)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-sr", type=int, default=16000)
    p.add_argument("--max-mixes", type=int, default=None)
    p.add_argument(
        "--snr-min",
        type=float,
        default=None,
        help="Minimum foreground SNR in dB (overrides --snr-tier)",
    )
    p.add_argument(
        "--snr-max",
        type=float,
        default=None,
        help="Maximum foreground SNR in dB (overrides --snr-tier)",
    )
    p.add_argument(
        "--snr-tier",
        choices=tuple(SNR_TIERS.keys()),
        default=None,
        help="Convenience SNR range: low 0-6, medium 6-15, high 15-30 dB",
    )
    p.add_argument(
        "--out-subdir",
        type=str,
        default=None,
        help="Subdirectory under --out-dir for WAVs (default: preset name)",
    )
    p.add_argument(
        "--manifest-name",
        type=str,
        default=None,
        help="Manifest CSV filename (default: manifest_{preset}.csv or manifest_eval_{tier}.csv)",
    )
    p.add_argument(
        "--no-peak-limit",
        action="store_true",
        help="Disable post-mix peak limiting to 0.99",
    )
    return p.parse_args(argv)


def _default_manifest_name(preset: str, snr_tier: str | None, snr: tuple[float, float]) -> str:
    if preset == "eval" and snr_tier:
        return f"manifest_eval_{snr_tier}.csv"
    if preset == "eval":
        lo, hi = snr
        return f"manifest_eval_snr_{lo:g}_{hi:g}.csv"
    return f"manifest_{preset}.csv"


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.soundbank.is_dir():
        raise SystemExit(f"Soundbank not found: {args.soundbank}")

    snr = resolve_snr(args.snr_min, args.snr_max, args.snr_tier, args.preset)
    rng = np.random.default_rng(args.seed)
    peak_norm = not args.no_peak_limit

    rows = generate_batch(
        soundbank_root=args.soundbank,
        preset_name=args.preset,
        out_dir=args.out_dir,
        snr=snr,
        rng=rng,
        target_sr=args.target_sr,
        max_mixes=args.max_mixes,
        peak_norm=peak_norm,
        out_subdir=args.out_subdir,
        seed=args.seed,
    )

    manifest_name = args.manifest_name or _default_manifest_name(
        args.preset, args.snr_tier, (snr.min_db, snr.max_db)
    )
    manifest_path = args.out_dir / manifest_name
    write_manifest(manifest_path, rows)

    print(f"Wrote {len(rows)} mixes under {args.out_dir}")
    print(f"SNR range: {snr.min_db:.1f} – {snr.max_db:.1f} dB")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
