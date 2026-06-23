"""Tiny waveform denoising baseline for target-noise suppression."""

from __future__ import annotations

import torch
from torch import nn


class TinyWaveformDenoiser(nn.Module):
    """Small CPU-friendly 1D convolutional waveform denoiser."""

    def __init__(self, channels: int = 16, kernel_size: int = 9) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(channels, 1, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Denoise waveform shaped [batch, 1, samples]."""
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("TinyWaveformDenoiser expects input shape [batch, 1, samples].")
        return self.net(waveform)
