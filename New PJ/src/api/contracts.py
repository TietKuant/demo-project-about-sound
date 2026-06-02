"""Boundary contracts for the CLI-first denoise pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class OutputArtifact:
    """A user-visible file produced by a processing engine."""

    label: str
    path: Path
    media_type: str
    role: str = "primary"


@dataclass(slots=True)
class ProcessingResult:
    """Task-oriented result contract for single-output and multi-output engines."""

    task_name: str
    engine_name: str
    status: str
    runtime_sec: float | None
    outputs: list[OutputArtifact]
    error: str = ""

    @property
    def primary_output_path(self) -> Path | None:
        """Return the path of the first primary artifact, if one exists."""
        primary = next((artifact for artifact in self.outputs if artifact.role == "primary"), None)
        return None if primary is None else primary.path

    def get_output(self, label: str) -> OutputArtifact | None:
        """Return the first artifact with the requested label."""
        return next((artifact for artifact in self.outputs if artifact.label == label), None)

    @property
    def audio_outputs(self) -> list[OutputArtifact]:
        """Return audio artifacts only."""
        return [artifact for artifact in self.outputs if artifact.media_type == "audio"]

    @property
    def video_outputs(self) -> list[OutputArtifact]:
        """Return video artifacts only."""
        return [artifact for artifact in self.outputs if artifact.media_type == "video"]


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
