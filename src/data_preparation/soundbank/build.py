"""Orchestrate soundbank adapters and write a manifest."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data_preparation.soundbank.audioset_raw import AudioSetRawAdapter
from src.data_preparation.soundbank.base import BuildResult, Role
from src.data_preparation.soundbank.generic import GenericAdapter
from src.data_preparation.soundbank.sonyc import SonycAdapter
from src.data_preparation.soundbank.voxceleb import VoxCelebAdapter

MANIFEST_NAME = "manifest.json"


@dataclass
class AdapterSpec:
    adapter: str
    role: Role
    label: str
    # adapter-specific fields
    dataset_root: Path | None = None
    annotations_path: Path | None = None
    split: str | None = None
    vox_root: Path | None = None
    meta_path: Path | None = None
    vox_set: str | None = None
    audio_root: Path | None = None
    source_root: Path | None = None
    labels: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], repo_root: Path) -> AdapterSpec:
        def path_or_none(key: str) -> Path | None:
            val = data.get(key)
            if val is None:
                return None
            p = Path(val)
            return p if p.is_absolute() else repo_root / p

        return cls(
            adapter=str(data["adapter"]),
            role=data["role"],  # type: ignore[arg-type]
            label=str(data.get("label", "")),
            dataset_root=path_or_none("dataset_root"),
            annotations_path=path_or_none("annotations_path"),
            split=data.get("split"),
            vox_root=path_or_none("vox_root"),
            meta_path=path_or_none("meta_path"),
            vox_set=data.get("vox_set") or data.get("set"),
            audio_root=path_or_none("audio_root"),
            source_root=path_or_none("source_root"),
            labels=data.get("labels"),
        )


def create_adapter(spec: AdapterSpec, repo_root: Path):
    name = spec.adapter.lower()
    if name == "sonyc":
        if spec.dataset_root is None or spec.annotations_path is None:
            raise ValueError("sonyc adapter requires dataset_root and annotations_path")
        return SonycAdapter(
            dataset_root=spec.dataset_root,
            annotations_path=spec.annotations_path,
            split=spec.split or "train",
        )
    if name == "voxceleb":
        if spec.vox_root is None or spec.meta_path is None:
            raise ValueError("voxceleb adapter requires vox_root and meta_path")
        return VoxCelebAdapter(
            vox_root=spec.vox_root,
            meta_path=spec.meta_path,
            vox_set=spec.vox_set or "dev",
        )
    if name in {"audioset_raw", "audioset"}:
        if spec.audio_root is None:
            raise ValueError("audioset_raw adapter requires audio_root")
        return AudioSetRawAdapter(audio_root=spec.audio_root, labels=spec.labels)
    if name == "generic":
        if spec.source_root is None:
            raise ValueError("generic adapter requires source_root")
        return GenericAdapter(source_root=spec.source_root, labels=spec.labels or [spec.label])
    raise ValueError(f"Unknown adapter: {spec.adapter!r}")


def build_soundbank(
    out_root: Path,
    specs: list[AdapterSpec],
    *,
    repo_root: Path,
    use_symlinks: bool = True,
) -> list[BuildResult]:
    out_root.mkdir(parents=True, exist_ok=True)
    results: list[BuildResult] = []
    manifest_entries: list[dict[str, Any]] = []

    for spec in specs:
        adapter = create_adapter(spec, repo_root)
        result = adapter.build(out_root, spec.role, spec.label, use_symlinks=use_symlinks)
        results.append(result)
        manifest_entries.append(
            {
                "adapter": spec.adapter,
                "role": spec.role,
                "label": spec.label,
                "label_dir": str(result.label_dir.resolve()),
                "n_files": result.n_files,
                "spec": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in asdict(spec).items()
                    if v is not None
                },
            }
        )

    manifest_path = out_root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest_entries, indent=2), encoding="utf-8")
    return results


def list_background_wavs(soundbank_root: Path, bg_label: str | None = None) -> list[tuple[str, Path]]:
    """Return (label, wav_path) for all background files in a soundbank."""
    bg_root = soundbank_root / "background"
    if not bg_root.is_dir():
        return []

    found: list[tuple[str, Path]] = []
    label_dirs = [bg_root / bg_label] if bg_label else sorted(bg_root.iterdir())
    for label_dir in label_dirs:
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for wav in sorted(label_dir.glob("*.wav")):
            if wav.is_file() and wav.stat().st_size > 1000:
                found.append((label, wav))
    return found


def list_foreground_labels(soundbank_root: Path) -> list[str]:
    fg_root = soundbank_root / "foreground"
    if not fg_root.is_dir():
        return []
    return sorted(p.name for p in fg_root.iterdir() if p.is_dir())
