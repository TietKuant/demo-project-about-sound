"""Noisereduce engine adapter."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from src.engine.base import DenoiseEngine


@dataclass(slots=True)
class NoisereduceEngine(DenoiseEngine):
    """Simple spectral-gating denoise engine backed by noisereduce."""

    loaded: bool = False

    @property
    def name(self) -> str:
        return "noisereduce"

    def load(self) -> None:
        """Validate required Python packages are importable."""
        importlib.import_module("noisereduce")
        importlib.import_module("soundfile")
        self.loaded = True

    def denoise(self, input_audio_path: str | Path, output_audio_path: str | Path) -> Path:
        """Read prepared WAV audio, reduce noise, and write the requested output path."""
        if not self.loaded:
            raise RuntimeError("Engine must be loaded before denoise() is called.")

        noisereduce = importlib.import_module("noisereduce")
        soundfile = importlib.import_module("soundfile")

        source_audio = Path(input_audio_path).resolve()
        denoised_audio_path = Path(output_audio_path).resolve()
        denoised_audio_path.parent.mkdir(parents=True, exist_ok=True)

        data, sample_rate = soundfile.read(source_audio)
        reduced_audio = noisereduce.reduce_noise(y=data, sr=sample_rate)
        soundfile.write(denoised_audio_path, reduced_audio, sample_rate)
        return denoised_audio_path
