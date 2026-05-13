"""Base contract for denoise engine adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class DenoiseEngine(ABC):
    """Minimal interface implemented by denoise engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a stable engine name for manifests and logs."""

    @abstractmethod
    def load(self) -> None:
        """Load any model resources needed for inference."""

    @abstractmethod
    def denoise(self, input_audio_path: str | Path, output_audio_path: str | Path) -> Path:
        """Plan or produce a cleaned audio artifact."""

