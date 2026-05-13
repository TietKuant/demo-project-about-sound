"""Placeholder DeepFilterNet engine adapter for dry-run scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.engine.base import DenoiseEngine


@dataclass(slots=True)
class DeepFilterNetEngine(DenoiseEngine):
    """Minimal adapter shape for the default MVP denoise engine.

    No real model loading or inference happens here yet. The class exists so the
    pipeline can be wired against a concrete engine boundary from day one.
    """

    model_name: str = "DeepFilterNet"
    model_variant: str = "default"
    weights_path: Path | None = None
    loaded: bool = False

    @property
    def name(self) -> str:
        return self.model_name

    def load(self) -> None:
        """Mark the engine as loaded for dry-run planning."""
        self.loaded = True

    def denoise(self, input_audio_path: str | Path, output_audio_path: str | Path) -> Path:
        """Return the planned cleaned-audio path without performing inference."""
        if not self.loaded:
            raise RuntimeError("Engine must be loaded before denoise() is called.")
        _ = Path(input_audio_path).resolve()
        return Path(output_audio_path).resolve()

