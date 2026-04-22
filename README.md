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
|-- anonymization_pipeline/   # Core anonymization pipeline components
|-- notebooks/                # Analysis and experiment inspection
|-- scripts/                  # Data prep and training entrypoints
|-- source_separation/        # Source separation experiments and related code
|-- voice_blurring/           # Voice blurring methods and prototypes
`-- requirements.txt          # Minimal Python dependencies
```

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:
  ```bash
   pip install -r requirements.txt
  ```

## Preparing data

Use `scripts/prepare_sonyc_vox_mixes.py` to generate SONYC + VoxCeleb mixtures.

- Default inputs:
  - SONYC annotations: `datasets/sonyc-v1-dataset/annotations.csv`
  - SONYC audio root: `datasets/sonyc-v1-dataset/`
  - Vox root: `datasets/voxceleb1-audio-wav-files-for-india-celebrity/`
  - Vox metadata: `datasets/voxceleb1-audio-wav-files-for-india-celebrity/vox1_meta.csv`
- Output is written to the path passed with `--out-dir`, plus a manifest CSV.

Examples:

```bash
python scripts/prepare_sonyc_vox_mixes.py --mode train --out-dir data/mixes_train
python scripts/prepare_sonyc_vox_mixes.py --mode eval --out-dir data/mixes_eval --eval-snr high
python scripts/prepare_sonyc_vox_mixes.py --mode eval --out-dir data/mixes_eval --use-training-vox
```

## Training U-Net

Train with `scripts/train_unet.py`. The script expects a CSV manifest with:

- `mix_wav`: path to a mixture WAV
- `voice_wav`: path to the corresponding target voice WAV

Example training command:

```bash
python scripts/train_unet.py --manifest data/pairs.csv --checkpoint-dir checkpoints/run1
```

Useful options:

```bash
python scripts/train_unet.py \
  --manifest data/pairs.csv \
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

- Voice anonymization reference listed in `pdf-material/fonts.txt`:
  - [https://markcartwright.com/files/cohen-hadria2019voiceanonymization.pdf](https://markcartwright.com/files/cohen-hadria2019voiceanonymization.pdf)

## Datasets used

- [https://www.kaggle.com/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity](https://www.kaggle.com/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity)
- [https://zenodo.org/records/3692954](https://zenodo.org/records/3692954)

