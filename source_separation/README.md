# Source Separation

This directory contains the voice/source separation components used in this repository's anonymization experiments.

## Attribution

The U-Net design here is adapted from:

- A. Jansson, E. J. Humphrey, N. Montecchio, R. M. Bittner, A. Kumar, and T. Weyde, "Singing voice separation with deep U-Net convolutional networks" ([arXiv:1706.09588](https://archives.ismir.net/ismir2017/paper/000171.pdf)).

This attribution is also discussed in the voice anonymization reference:

- M. Cohen-Hadria et al., "Voice anonymization in urban sound recordings" ([paper](https://markcartwright.com/files/cohen-hadria2019voiceanonymization.pdf)).

## Contents

- `unet.py`: U-Net mask prediction network.
- `stft.py`: STFT/ISTFT helpers and configuration.
- `infer.py`: model loading and waveform inference utilities.
- `losses.py`: loss functions used for training experiments.

## Notes

This code is research-oriented and may differ from the exact implementation details in the original paper.
