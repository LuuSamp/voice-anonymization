#!/usr/bin/env python3
"""Build a Scaper-compatible soundbank from one or more dataset adapters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data_preparation.soundbank.build import AdapterSpec, build_soundbank


def _repo_root() -> Path:
    return _REPO


def _spec_from_cli(args: argparse.Namespace) -> AdapterSpec:
    adapter = args.adapter
    if adapter == "sonyc":
        return AdapterSpec(
            adapter="sonyc",
            role=args.role,
            label=args.label,
            dataset_root=args.sonyc_dataset_root or _repo_root() / "datasets/sonyc-v1-dataset",
            annotations_path=args.sonyc_annotations
            or _repo_root() / "datasets/sonyc-v1-dataset/annotations.csv",
            split=args.split or "train",
        )
    if adapter == "voxceleb":
        default_vox = _repo_root() / "datasets/voxceleb1-audio-wav-files-for-india-celebrity"
        return AdapterSpec(
            adapter="voxceleb",
            role=args.role,
            label=args.label,
            vox_root=args.vox_root or default_vox,
            meta_path=args.vox_meta or default_vox / "vox1_meta.csv",
            vox_set=args.set or "dev",
        )
    if adapter in {"audioset_raw", "audioset"}:
        return AdapterSpec(
            adapter="audioset_raw",
            role=args.role,
            label=args.label,
            audio_root=args.audio_root or _repo_root() / "datasets/audioset/audio",
            labels=args.labels.split(",") if args.labels else None,
        )
    if adapter == "generic":
        if args.source_root is None:
            raise SystemExit("--source-root is required for generic adapter")
        return AdapterSpec(
            adapter="generic",
            role=args.role,
            label=args.label,
            source_root=args.source_root,
            labels=[args.label],
        )
    raise SystemExit(f"Unknown adapter: {adapter!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build soundbank/{foreground,background}/{label}/ for Scaper."
    )
    p.add_argument("--out", type=Path, required=True, help="Soundbank output root")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON file: list of adapter spec objects (for multiple adapters)",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy WAVs instead of symlinking (default: symlink)",
    )

    # Single-adapter CLI (ignored when --config is set)
    p.add_argument("--adapter", choices=("sonyc", "voxceleb", "audioset_raw", "audioset", "generic"))
    p.add_argument("--role", choices=("foreground", "background"))
    p.add_argument("--label", type=str, default="", help="Target label folder name")
    p.add_argument("--split", type=str, default="train", help="SONYC split filter")
    p.add_argument("--set", type=str, default="dev", help="VoxCeleb meta Set (dev/test)")
    p.add_argument("--sonyc-dataset-root", type=Path, default=None)
    p.add_argument("--sonyc-annotations", type=Path, default=None)
    p.add_argument("--vox-root", type=Path, default=None)
    p.add_argument("--vox-meta", type=Path, default=None)
    p.add_argument("--audio-root", type=Path, default=None, help="AudioSet raw audio root")
    p.add_argument("--source-root", type=Path, default=None, help="Generic adapter source root")
    p.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated source labels (audioset_raw); use label='*' to import all",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = _repo_root()
    use_symlinks = not args.copy

    if args.config is not None:
        raw = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise SystemExit("--config must contain a JSON list of adapter specs")
        specs = [AdapterSpec.from_dict(entry, repo) for entry in raw]
    else:
        if not args.adapter or not args.role or not args.label:
            raise SystemExit(
                "Provide --config with multiple adapters, or --adapter, --role, and --label."
            )
        specs = [_spec_from_cli(args)]

    results = build_soundbank(args.out, specs, repo_root=repo, use_symlinks=use_symlinks)
    total = sum(r.n_files for r in results)
    print(f"Built soundbank at {args.out} ({total} files across {len(results)} adapter run(s))")
    for r in results:
        print(f"  {r.role}/{r.label}: {r.n_files} files -> {r.label_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
