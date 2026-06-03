"""Lightweight audio analysis contract for recommendation routing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AudioAnalysis:
    """Analyzer output contract for future ML or manual/stub classifiers."""

    speech_score: float
    music_score: float
    noise_score: float
    event_labels: list[str] = field(default_factory=list)
    duration_sec: float | None = None
    source: str = "manual_or_stub"
