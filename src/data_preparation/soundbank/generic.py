"""Generic label-folder tree → Scaper soundbank adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data_preparation.soundbank.base import BuildResult, Role, collect_wavs, populate_label_dir, role_dir


@dataclass
class GenericAdapter:
    """Copy or symlink ``source_root/{label}/*.wav`` into the soundbank."""

    source_root: Path
    labels: list[str] | None = None

    def build(
        self,
        out_root: Path,
        role: Role,
        label: str,
        *,
        use_symlinks: bool = True,
    ) -> BuildResult:
        if self.labels:
            if len(self.labels) != 1 or self.labels[0] != label:
                raise ValueError(
                    f"GenericAdapter label mismatch: config labels={self.labels!r}, "
                    f"build label={label!r}"
                )
            src_label_dir = self.source_root / label
        else:
            src_label_dir = self.source_root

        if not src_label_dir.is_dir():
            raise FileNotFoundError(f"Source label directory not found: {src_label_dir}")

        wavs = collect_wavs(src_label_dir)
        if not wavs:
            raise FileNotFoundError(f"No WAV files in {src_label_dir}")

        label_dir = role_dir(out_root, role, label)
        sources = populate_label_dir(wavs, label_dir, use_symlinks=use_symlinks)
        return BuildResult(
            role=role,
            label=label,
            label_dir=label_dir,
            n_files=len(sources),
            source_paths=sources,
        )
