"""Soundbank adapter protocol for Scaper-compatible folder layouts."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

Role = Literal["foreground", "background"]
MIN_WAV_BYTES = 1000


@dataclass(frozen=True)
class BuildResult:
    role: Role
    label: str
    label_dir: Path
    n_files: int
    source_paths: list[str] = field(default_factory=list)


class SoundbankAdapter(Protocol):
    """Build one label folder under ``out_root/{role}/{label}/``."""

    def build(
        self,
        out_root: Path,
        role: Role,
        label: str,
        *,
        use_symlinks: bool = True,
    ) -> BuildResult: ...


def role_dir(out_root: Path, role: Role, label: str) -> Path:
    return out_root / role / _sanitize_label(label)


def _sanitize_label(label: str) -> str:
    cleaned = label.strip().replace(os.sep, "-")
    return cleaned or "unknown"


def link_or_copy(src: Path, dst: Path, *, use_symlinks: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlinks:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def collect_wavs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    wavs = [
        p
        for p in root.rglob("*.wav")
        if p.is_file() and p.stat().st_size > MIN_WAV_BYTES
    ]
    return sorted(wavs)


def populate_label_dir(
    sources: list[Path],
    label_dir: Path,
    *,
    use_symlinks: bool,
    flat_names: bool = True,
) -> list[str]:
    """Copy or symlink sources into ``label_dir``; return resolved source paths."""
    label_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for i, src in enumerate(sources):
        if flat_names:
            name = f"{src.stem}_{i:04d}{src.suffix}" if len(sources) > 1 else src.name
        else:
            name = src.name
        dst = label_dir / name
        link_or_copy(src, dst, use_symlinks=use_symlinks)
        written.append(str(src.resolve()))
    return written
