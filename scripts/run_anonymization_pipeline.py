#!/usr/bin/env python3
"""Run full anonymization pipeline: separation -> blur -> remix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from anonymization_pipeline import anonymize_audio
from source_separation import load_unet_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="Run full voice anonymization pipeline.")
    p.add_argument("--input-wav", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    p.add_argument("--blur-mode", choices=("cascade", "low_pass", "mfcc"), default="cascade")
    p.add_argument("--down-sr", type=float, default=500.0)
    p.add_argument("--out-sr", type=int, default=16000)
    p.add_argument("--mfcc-n-mfcc", type=int, default=5)
    p.add_argument("--mfcc-n-mels", type=int, default=128)
    p.add_argument("--mfcc-n-fft", type=int, default=2048)
    p.add_argument("--mfcc-hop-length", type=int, default=512)
    p.add_argument("--mfcc-n-iter", type=int, default=32)
    args = p.parse_args()

    y, sr = librosa.load(str(args.input_wav), sr=None, mono=True)
    y = np.asarray(y, dtype=np.float32)
    model, config, device = load_unet_checkpoint(args.checkpoint, device=args.device)

    result = anonymize_audio(
        y,
        sr,
        model=model,
        config=config,
        device=device,
        blur_mode=args.blur_mode,
        low_pass_kwargs={"down_sr": args.down_sr, "out_sr": args.out_sr},
        mfcc_kwargs={
            "n_mfcc": args.mfcc_n_mfcc,
            "n_mels": args.mfcc_n_mels,
            "n_fft": args.mfcc_n_fft,
            "hop_length": args.mfcc_hop_length,
            "n_iter": args.mfcc_n_iter,
        },
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_dir / "voice_est.wav", result.voice_est, result.sr)
    sf.write(args.output_dir / "background_est.wav", result.background_est, result.sr)
    sf.write(args.output_dir / "blurred_voice.wav", result.blurred_voice, result.sr)
    sf.write(args.output_dir / "anonymized_mix.wav", result.anonymized_mix, result.sr)
    print(f"Device: {device}")
    print(f"Blur mode: {result.blur_mode}")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

