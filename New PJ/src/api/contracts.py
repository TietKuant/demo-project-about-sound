"""Boundary contracts for the CLI-first denoise pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DenoiseRequest:
    """Input contract for a single local denoise request."""

    input_path: Path
    output_dir: Path
    output_mode: str
    keep_intermediates: bool = False
    engine_name: str = "ffmpeg-arnndn"


@dataclass(slots=True)
class DenoiseResult:
    """Result contract returned by the baseline pipeline."""

    status: str
    final_output_path: Path | None
    intermediate_audio_path: Path | None
    engine_name: str
    run_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the result."""
        data = asdict(self)
        for key in ("final_output_path", "intermediate_audio_path"):
            value = data[key]
            data[key] = None if value is None else str(value)
        return data
