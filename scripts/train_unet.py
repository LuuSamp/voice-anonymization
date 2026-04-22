#!/usr/bin/env python3
"""
Train the voice-separation U-Net (Cohen–Hadria 2019 §2.1, §3.6).

Expects a CSV manifest with columns ``mix_wav`` and ``voice_wav`` (paths to 16 kHz–compatible
mono WAVs; files are resampled to 16 kHz). Does **not** depend on ``voice_blurring`` or
``prepare_sonyc_vox_mixes`` — build the manifest however you like.

Example::

  python scripts/train_unet.py --manifest data/pairs.csv --checkpoint-dir checkpoints/run1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from source_separation.losses import masked_l1_loss
from source_separation.stft import STFTConfig, waveform_to_magnitude
from source_separation.unet import VoiceSeparationUNet


def _load_mono_16k(path: Path, target_sr: int) -> np.ndarray:
    y, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32)


def _sync_mag_crops(
    mix: torch.Tensor,
    voice: torch.Tensor,
    config: STFTConfig,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same time crop on magnitude specs (mix, voice), shape ``(1, F, T_patch)``."""
    mag_m = waveform_to_magnitude(mix, config)
    mag_v = waveform_to_magnitude(voice, config)
    if mag_m.dim() == 2:
        mag_m = mag_m.unsqueeze(0)
        mag_v = mag_v.unsqueeze(0)
    T = min(mag_m.shape[-1], mag_v.shape[-1])
    mag_m = mag_m[..., :T]
    mag_v = mag_v[..., :T]
    if T < config.n_frames:
        pad = config.n_frames - T
        mag_m = torch.nn.functional.pad(mag_m, (0, pad))
        mag_v = torch.nn.functional.pad(mag_v, (0, pad))
        T = config.n_frames
    start_max = T - config.n_frames
    start = int(rng.integers(0, start_max + 1)) if start_max > 0 else 0
    mag_m = mag_m[..., start : start + config.n_frames]
    mag_v = mag_v[..., start : start + config.n_frames]
    return mag_m.unsqueeze(1), mag_v.unsqueeze(1)


class PairManifestDataset(Dataset):
    def __init__(
        self,
        rows: list[tuple[str, str]],
        config: STFTConfig,
        seed: int,
    ) -> None:
        self.rows = rows
        self.config = config
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        mix_path, voice_path = self.rows[idx]
        sr = self.config.sample_rate
        mix = torch.from_numpy(_load_mono_16k(Path(mix_path), sr))
        voice = torch.from_numpy(_load_mono_16k(Path(voice_path), sr))
        n = min(mix.numel(), voice.numel())
        mix = mix[:n]
        voice = voice[:n]
        rng = np.random.default_rng(self.seed + idx * 100_003)
        mag_m, mag_v = _sync_mag_crops(mix, voice, self.config, rng)
        return mag_m.squeeze(0), mag_v.squeeze(0)


def _collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    mix = torch.stack([b[0] for b in batch], dim=0)
    voice = torch.stack([b[1] for b in batch], dim=0)
    return mix, voice


def load_manifest(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "mix_wav" not in reader.fieldnames or "voice_wav" not in reader.fieldnames:
            raise SystemExit("Manifest must have columns: mix_wav, voice_wav")
        for row in reader:
            rows.append((row["mix_wav"].strip(), row["voice_wav"].strip()))
    if not rows:
        raise SystemExit("Manifest is empty.")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Train VoiceSeparationUNet (Cohen–Hadria 2019).")
    p.add_argument("--manifest", type=Path, required=True, help="CSV with mix_wav,voice_wav")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu", "mps"),
        help="Training device. 'auto' prefers cuda, then mps, then cpu.",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = STFTConfig()
    rows = load_manifest(args.manifest)
    ds = PairManifestDataset(rows, config, seed=args.seed)
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "mps":
        device = torch.device("mps")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    use_pin_memory = device.type == "cuda"
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
        drop_last=False,
        pin_memory=use_pin_memory,
    )

    model = VoiceSeparationUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Using device: {device}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n_batches = 0
        for mix_mag, voice_mag in loader:
            mix_mag = mix_mag.to(device, non_blocking=use_pin_memory)
            voice_mag = voice_mag.to(device, non_blocking=use_pin_memory)
            opt.zero_grad(set_to_none=True)
            mask = model(mix_mag)
            loss = masked_l1_loss(mix_mag, mask, voice_mag)
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1
        avg = total / max(n_batches, 1)
        print(f"epoch {epoch}/{args.epochs}  loss={avg:.6f}")

    ckpt_path = args.checkpoint_dir / "unet_voice_sep.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stft_config": {
                "sample_rate": config.sample_rate,
                "n_fft": config.n_fft,
                "hop_length": config.hop_length,
                "n_freq_bins": config.n_freq_bins,
                "n_frames": config.n_frames,
                "center": config.center,
            },
            "epochs": args.epochs,
            "seed": args.seed,
        },
        ckpt_path,
    )
    print(f"Saved {ckpt_path}")


if __name__ == "__main__":
    main()
