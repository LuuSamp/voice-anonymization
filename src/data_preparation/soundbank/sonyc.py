"""SONYC-UST → Scaper background soundbank adapter."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.data_preparation.soundbank.base import BuildResult, Role, populate_label_dir, role_dir


def resolve_sonyc_wav(dataset_root: Path, split: str, audio_filename: str) -> Path:
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


@dataclass
class SonycAdapter:
    dataset_root: Path
    annotations_path: Path
    split: str = "train"

    def build(
        self,
        out_root: Path,
        role: Role,
        label: str,
        *,
        use_symlinks: bool = True,
    ) -> BuildResult:
        if role != "background":
            raise ValueError("SONYC adapter only supports role='background'")

        rows = load_unique_sonyc_files(self.annotations_path, self.split)
        wavs: list[Path] = []
        for split, fn in rows:
            path = resolve_sonyc_wav(self.dataset_root, split, fn)
            if path.is_file():
                wavs.append(path)

        if not wavs:
            raise FileNotFoundError(
                f"No SONYC WAVs found for split={self.split!r} under {self.dataset_root}"
            )

        label_dir = role_dir(out_root, role, label)
        sources = populate_label_dir(wavs, label_dir, use_symlinks=use_symlinks)
        return BuildResult(
            role=role,
            label=label,
            label_dir=label_dir,
            n_files=len(sources),
            source_paths=sources,
        )
