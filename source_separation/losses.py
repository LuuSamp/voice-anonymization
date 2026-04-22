"""Losses for Cohen–Hadria et al. (2019) §2.1 — masked L1 on magnitude spectra."""

from __future__ import annotations

import torch


def masked_l1_loss(
    mix_mag: torch.Tensor,
    mask: torch.Tensor,
    target_mag: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute error between ``mix_mag * mask`` and ``target_mag``.

    Parameters
    ----------
    mix_mag, target_mag
        Shape ``(N, 1, F, T)`` (same as U-Net I/O).
    mask
        Shape ``(N, 1, F, T)``, values in ``[0, 1]``.
    """
    est = mix_mag * mask
    return torch.mean(torch.abs(est - target_mag))
