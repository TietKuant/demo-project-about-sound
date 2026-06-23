"""Rule-based task recommendation router for audio analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.analyzer.audio_analysis import AudioAnalysis
from src.router.task_registry import CLEAN_VOICE, EXTRACT_VOCALS


UNSUPPORTED_OR_UNCERTAIN_INPUT = "unsupported_or_uncertain_input"
TARGET_SOUND_REMOVAL_NOT_SUPPORTED = "target_sound_removal_not_supported_in_mvp"
TARGET_SOUND_EVENT_KEYWORDS = {"dog", "bark", "keyboard", "alarm", "siren"}


@dataclass(slots=True)
class Recommendation:
    """Recommended task and routing rationale."""

    recommended_task: str | None
    confidence: float
    reason: str
    warnings: list[str] = field(default_factory=list)


def recommend_task(analysis: AudioAnalysis) -> Recommendation:
    """Recommend a supported MVP task from analyzer scores."""
    warnings = _event_warnings(analysis.event_labels)

    if analysis.music_score >= 0.65 and analysis.music_score > analysis.speech_score:
        return Recommendation(
            recommended_task=EXTRACT_VOCALS,
            confidence=_clamp_score(analysis.music_score),
            reason="music_score_is_dominant",
            warnings=warnings,
        )

    if analysis.speech_score >= 0.55 and analysis.speech_score >= analysis.music_score:
        return Recommendation(
            recommended_task=CLEAN_VOICE,
            confidence=_clamp_score(analysis.speech_score),
            reason="speech_score_is_dominant",
            warnings=warnings,
        )

    warnings.append(UNSUPPORTED_OR_UNCERTAIN_INPUT)
    return Recommendation(
        recommended_task=None,
        confidence=min(0.5, max(analysis.speech_score, analysis.music_score, analysis.noise_score, 0.0)),
        reason="no_supported_task_confidently_matched",
        warnings=warnings,
    )


def _event_warnings(event_labels: list[str]) -> list[str]:
    normalized_labels = [label.casefold() for label in event_labels]
    for label in normalized_labels:
        if any(keyword in label for keyword in TARGET_SOUND_EVENT_KEYWORDS):
            return [TARGET_SOUND_REMOVAL_NOT_SUPPORTED]
    return []


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))
