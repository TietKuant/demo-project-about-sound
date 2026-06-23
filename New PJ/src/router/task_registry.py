"""Lightweight task registry for supported audio processing workflows."""

from __future__ import annotations

from dataclasses import dataclass


CLEAN_VOICE = "clean_voice"
TARGET_NOISE_SUPPRESSION = "target_noise_suppression"
EXTRACT_VOCALS = "extract_vocals"
REMOVE_VOCALS = "remove_vocals"
VOICE_PITCH_HIGH = "voice_pitch_high"
VOICE_PITCH_LOW = "voice_pitch_low"


@dataclass(slots=True)
class TaskSpec:
    """Metadata for one supported user-facing task."""

    name: str
    display_name: str
    description: str
    engine: str
    output_labels: list[str]
    input_kind: str


_TASK_DEFINITIONS: tuple[dict[str, object], ...] = (
    {
        "name": CLEAN_VOICE,
        "display_name": "Clean Voice / Reduce Speech Noise",
        "description": "Enhance speech and reduce background noise in audio or video input.",
        "engine": "deepfilternet",
        "output_labels": ["restored"],
        "input_kind": "audio_or_video",
    },
    {
        "name": TARGET_NOISE_SUPPRESSION,
        "display_name": "Target Noise Suppression (Experimental)",
        "description": "Run a custom-trained baseline for speech with selected target noise classes.",
        "engine": "target_noise_suppressor",
        "output_labels": ["enhanced"],
        "input_kind": "audio_only",
    },
    {
        "name": EXTRACT_VOCALS,
        "display_name": "Extract Vocal",
        "description": "Separate vocals from music into vocal and non-vocal stems.",
        "engine": "demucs",
        "output_labels": ["vocals", "no_vocals"],
        "input_kind": "audio_or_video_with_music",
    },
    {
        "name": REMOVE_VOCALS,
        "display_name": "Remove Vocal",
        "description": "Separate music into non-vocal and vocal stems, with accompaniment as the primary artifact.",
        "engine": "demucs",
        "output_labels": ["no_vocals", "vocals"],
        "input_kind": "audio_or_video_with_music",
    },
    {
        "name": VOICE_PITCH_HIGH,
        "display_name": "High pitch voice",
        "description": "Manual voice effects: raise voice pitch while preserving approximate duration.",
        "engine": "ffmpeg",
        "output_labels": ["High pitch voice"],
        "input_kind": "audio_or_video",
    },
    {
        "name": VOICE_PITCH_LOW,
        "display_name": "Low pitch voice",
        "description": "Manual voice effects: lower voice pitch while preserving approximate duration.",
        "engine": "ffmpeg",
        "output_labels": ["Low pitch voice"],
        "input_kind": "audio_or_video",
    },
)
_TASK_DEFINITIONS_BY_NAME = {str(task["name"]): task for task in _TASK_DEFINITIONS}


def _build_task_spec(definition: dict[str, object]) -> TaskSpec:
    return TaskSpec(
        name=str(definition["name"]),
        display_name=str(definition["display_name"]),
        description=str(definition["description"]),
        engine=str(definition["engine"]),
        output_labels=list(definition["output_labels"]),
        input_kind=str(definition["input_kind"]),
    )


def list_supported_tasks() -> list[TaskSpec]:
    """Return supported task metadata in stable display order."""
    return [_build_task_spec(task) for task in _TASK_DEFINITIONS]


def get_task_spec(task_name: str) -> TaskSpec:
    """Return metadata for one supported task."""
    try:
        definition = _TASK_DEFINITIONS_BY_NAME[task_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported task: {task_name}") from exc
    return _build_task_spec(definition)


def is_supported_task(task_name: str) -> bool:
    """Return whether the task is supported."""
    return task_name in _TASK_DEFINITIONS_BY_NAME
