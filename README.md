# Voice Anonymization Research Repository

This repository contains exploratory research code for voice anonymization and related audio processing tasks.
It is organized for iterative experimentation rather than production-style packaging.

## How this repo is meant to be used

- Start from reproducible scripts to prepare data and train models.
- Use notebooks for analysis/inspection after data and checkpoints are generated.
- Expect structure and files to evolve as research questions change.
- Prefer reproducible checkpoints and notes over strict software release workflows.

## Repository structure

```text
.
|-- src/
|   |-- data_preparation/   # Scaper soundbank + mix generation
|   |-- source_separation/  # U-Net separation
|   |-- voice_blurring/     # MFCC / low-pass blurring
|   `-- pipeline/           # End-to-end orchestrator
|-- data/
|   |-- raw/                # Raw downloaded audio
|   |-- synthetic/          # Scaper-generated mixes
|   `-- processed/          # U-Net pairs, stems, etc.
|-- configs/                # Soundbank adapter configs
|-- notebooks/              # Analysis and experiment inspection
|-- scripts/                # Data prep and training entrypoints
`-- requirements.txt        # Minimal Python dependencies
```

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:
  ```bash
   pip install -r requirements.txt
  ```
3. Install system dependencies for Scaper (Linux):
  ```bash
   # FFmpeg (required) and libsox (for soxbindings)
   sudo pacman -S ffmpeg sox   # Manjaro/Arch
  ```

## Preparing data

Synthetic mixes are generated with [Scaper](https://github.com/justinsalamon/scaper) in three steps: build a soundbank, generate soundscapes at chosen SNR tiers, then build U-Net training pairs.

### 1. Build a soundbank

Soundbanks have Scaper layout: `foreground/{label}/*.wav` and `background/{label}/*.wav`.

Cohen–Hadria-style SONYC + VoxCeleb (config file):

```bash
python scripts/build_soundbank.py \
  --out datasets/soundbanks/cohen_hadria \
  --config configs/cohen_hadria_soundbank.json
```

Single adapter example:

```bash
python scripts/build_soundbank.py --out datasets/soundbanks/audioset_fg \
  --adapter audioset_raw --role foreground --label '*' \
  --audio-root datasets/audioset/audio
```

Download AudioSet clips first (`scripts/download_audio_set.py`). Use `--mode speech` for foreground speech labels, `--mode background` for ambient backgrounds (`datasets/audioset/audio_bg/`), or `--mode both`.

### 2. Generate soundscapes (SNR-controlled)

Presets (`train`, `eval`, `overlay`) control event layout only. SNR is set with `--snr-min`/`--snr-max` or `--snr-tier low|medium|high`.

```bash
# Training mixes at high SNR
python scripts/generate_soundscapes.py --preset train --snr-min 20 --snr-max 30 \
  --soundbank datasets/soundbanks/cohen_hadria --out-dir data/synthetic/mixes_train

# Eval at different SNR tiers
python scripts/generate_soundscapes.py --preset eval --snr-tier low \
  --soundbank datasets/soundbanks/cohen_hadria_eval --out-dir data/synthetic/mixes_eval_low

python scripts/generate_soundscapes.py --preset eval --snr-min 3 --snr-max 8 \
  --soundbank datasets/soundbanks/audioset_full --out-dir data/synthetic/mixes_eval_custom
```

SNR tiers: **low** 0–6 dB, **medium** 6–15 dB, **high** 15–30 dB.

### 3. Build U-Net pair manifest

```bash
python scripts/build_unet_pairs.py \
  --train-manifest data/synthetic/mixes_train/manifest_train.csv \
  --pairs-manifest-out data/processed/unet/train_pairs.csv \
  --stems-dir data/processed/unet/voice_stems
```

Voice stems are reconstructed from Scaper JAMS annotations (`jams_path` column in the manifest).

## Training U-Net

Train with `scripts/train_unet.py`. The script expects a CSV manifest with:

- `mix_wav`: path to a mixture WAV
- `voice_wav`: path to the corresponding target voice WAV

Example training command:

```bash
python scripts/train_unet.py --manifest data/processed/pairs.csv --checkpoint-dir checkpoints/run1
```

Useful options:

```bash
python scripts/train_unet.py \
  --manifest data/processed/pairs.csv \
  --checkpoint-dir checkpoints/run1 \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-3 \
  --device auto \
  --num-workers 4 \
  --seed 0
```

Output:

- A model checkpoint at `checkpoints/<run>/unet_voice_sep.pt`

## References

- Voice anonymization reference listed in:
  - [https://markcartwright.com/files/cohen-hadria2019voiceanonymization.pdf](https://markcartwright.com/files/cohen-hadria2019voiceanonymization.pdf)
- Source separation architecture attribution:
  - A. Jansson, E. J. Humphrey, N. Montecchio, R. M. Bittner, A. Kumar, and T. Weyde, "Singing voice separation with deep U-Net convolutional networks" [(https://archives.ismir.net/ismir2017/paper/000171.pdf](https://archives.ismir.net/ismir2017/paper/000171.pdf))

## Datasets used

- [https://www.kaggle.com/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity](https://www.kaggle.com/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity)
- [https://zenodo.org/records/3692954](https://zenodo.org/records/3692954)
- [https://research.google.com/audioset/ontology/human_voice_1.html](https://research.google.com/audioset/ontology/human_voice_1.html)

