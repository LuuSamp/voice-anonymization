"""Source separation (U-Net mask) for Cohen–Hadria et al. (2019)."""

from source_separation.losses import masked_l1_loss
from source_separation.infer import (
    compute_voice_activity,
    crop_windows_from_activity,
    crop_windows_from_activity_with_config,
    load_unet_checkpoint,
    resolve_device,
    separate_voice,
    splice_anonymized_segments,
)
from source_separation.stft import STFTConfig, waveform_batch_to_model_input, waveform_to_magnitude
from source_separation.unet import VoiceSeparationUNet

__all__ = [
    "STFTConfig",
    "VoiceSeparationUNet",
    "masked_l1_loss",
    "load_unet_checkpoint",
    "separate_voice",
    "compute_voice_activity",
    "crop_windows_from_activity",
    "crop_windows_from_activity_with_config",
    "splice_anonymized_segments",
    "resolve_device",
    "waveform_to_magnitude",
    "waveform_batch_to_model_input",
]
