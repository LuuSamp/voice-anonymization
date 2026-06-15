"""Raw downloaded AudioSet clips → Scaper soundbank adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data_preparation.soundbank.base import BuildResult, Role, collect_wavs, role_dir
from src.data_preparation.soundbank.generic import GenericAdapter


@dataclass
class AudioSetRawAdapter:
    """Map ``audio_root/{sub_label}/*.wav`` into one or all label folders."""

    audio_root: Path
    labels: list[str] | None = None

    def build(
        self,
        out_root: Path,
        role: Role,
        label: str,
        *,
        use_symlinks: bool = True,
    ) -> BuildResult:
        if not self.audio_root.is_dir():
            raise FileNotFoundError(f"AudioSet root not found: {self.audio_root}")

        if self.labels:
            return GenericAdapter(source_root=self.audio_root, labels=self.labels).build(
                out_root, role, label, use_symlinks=use_symlinks
            )

        # No explicit label: import every subfolder preserving label names.
        label_dirs = sorted(p for p in self.audio_root.iterdir() if p.is_dir())
        if not label:
            raise ValueError(
                "AudioSet raw adapter requires label='*' to import all sub-labels, "
                "or pass labels=[...] in adapter config."
            )
        if label != "*":
            sub = self.audio_root / label
            if not sub.is_dir():
                raise FileNotFoundError(f"AudioSet label folder not found: {sub}")
            return GenericAdapter(source_root=self.audio_root, labels=[label]).build(
                out_root, role, label, use_symlinks=use_symlinks
            )

        total_files = 0
        all_sources: list[str] = []
        last_dir = role_dir(out_root, role, "placeholder")
        for sub in label_dirs:
            wavs = collect_wavs(sub)
            if not wavs:
                continue
            result = GenericAdapter(source_root=self.audio_root, labels=[sub.name]).build(
                out_root, role, sub.name, use_symlinks=use_symlinks
            )
            total_files += result.n_files
            all_sources.extend(result.source_paths)
            last_dir = result.label_dir

        if total_files == 0:
            raise FileNotFoundError(f"No WAV files under {self.audio_root}")

        return BuildResult(
            role=role,
            label="*",
            label_dir=last_dir,
            n_files=total_files,
            source_paths=all_sources,
        )
