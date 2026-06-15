"""VoxCeleb → Scaper foreground soundbank adapter."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.data_preparation.soundbank.base import BuildResult, Role, populate_label_dir, role_dir


def parse_vox_meta(meta_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with meta_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            vid = row["VoxCeleb1 ID"].strip()
            st = row["Set"].strip().lower()
            out[vid] = st
    return out


def collect_vox_wavs(vox_root: Path, meta: dict[str, str], wanted_set: str) -> list[Path]:
    allowed = {vid for vid, st in meta.items() if st == wanted_set}
    if not allowed:
        raise ValueError(f"No speaker IDs with Set={wanted_set!r} in vox meta.")

    paths: list[Path] = []
    for p in vox_root.rglob("*.wav"):
        if set(p.parts) & allowed:
            paths.append(p)
    paths.sort()
    if not paths:
        raise FileNotFoundError(
            f"No WAV files found under {vox_root} for Set={wanted_set!r}."
        )
    return paths


@dataclass
class VoxCelebAdapter:
    vox_root: Path
    meta_path: Path
    vox_set: str = "dev"

    def build(
        self,
        out_root: Path,
        role: Role,
        label: str,
        *,
        use_symlinks: bool = True,
    ) -> BuildResult:
        if role != "foreground":
            raise ValueError("VoxCeleb adapter only supports role='foreground'")

        meta = parse_vox_meta(self.meta_path)
        wavs = collect_vox_wavs(self.vox_root, meta, self.vox_set)
        label_dir = role_dir(out_root, role, label)
        sources = populate_label_dir(wavs, label_dir, use_symlinks=use_symlinks)
        return BuildResult(
            role=role,
            label=label,
            label_dir=label_dir,
            n_files=len(sources),
            source_paths=sources,
        )
