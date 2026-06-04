"""Minimal teacher-facing Gradio demo for unified MVP audio tasks."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_audio_features import extract_audio_features
from scripts.run_audio_task import run_audio_task
from src.router.task_registry import (
    CLEAN_VOICE,
    EXTRACT_VOCALS,
    REMOVE_VOCALS,
    TARGET_NOISE_SUPPRESSION,
    list_supported_tasks,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/demo-runs")
SUMMARY_FIELDS = ["task", "engine", "status", "runtime_sec", "input_type", "primary_output_path", "error"]
FEATURE_FIELDS = [
    "duration_sec",
    "rms_energy",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
]
SPEECH_INTENT = "Speech / noisy speech"
MUSIC_INTENT = "Music or music video with vocals"
UNKNOWN_INTENT = "Unknown / not sure"
CONTENT_INTENTS = [SPEECH_INTENT, MUSIC_INTENT, UNKNOWN_INTENT]
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
MUSIC_TASKS = {EXTRACT_VOCALS, REMOVE_VOCALS}
ROUTER_CHECKPOINT_ENV_VAR = "AUDIO_ROUTER_CHECKPOINT"
ROUTER_CONFIDENCE_THRESHOLD = 0.60


def _read_summary(run_dir: Path) -> dict[str, str]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Task summary not found: {summary_path}")
    with summary_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"Task summary is empty: {summary_path}")
    return rows[0]


def _summary_table(row: dict[str, str]) -> list[list[str]]:
    return [[field, row.get(field, "")] for field in SUMMARY_FIELDS]


def _summary_markdown(row: dict[str, str], run_dir: Path) -> str:
    status = row.get("status", "")
    task = row.get("task", "")
    engine = row.get("engine", "")
    error = row.get("error", "")
    lines = [
        f"### Task run: `{status}`",
        f"- **Task:** `{task}`",
        f"- **Engine:** `{engine}`",
        f"- **Run directory:** `{run_dir}`",
    ]
    if error:
        lines.append(f"- **Error:** `{error}`")
    return "\n".join(lines)


def _input_type_for_path(path: Path) -> str:
    return "video" if path.suffix.lower() in VIDEO_SUFFIXES else "audio"


def _write_feature_manifest(input_path: Path, manifest_path: Path) -> None:
    fieldnames = [
        "sample_id",
        "category",
        "input_path",
        "input_type",
        "language",
        "expected_task",
        "has_clean_reference",
        "notes",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": input_path.stem,
                "category": "demo_upload",
                "input_path": str(input_path),
                "input_type": _input_type_for_path(input_path),
                "language": "unknown",
                "expected_task": "",
                "has_clean_reference": "false",
                "notes": "Temporary demo preflight analysis.",
            }
        )


def _extract_feature_row(input_path: Path) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="audio-task-demo-features-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        manifest_path = temp_dir / "manifest.csv"
        features_path = temp_dir / "features.csv"
        _write_feature_manifest(input_path, manifest_path)
        extract_audio_features(manifest_path=manifest_path, output_path=features_path)
        with features_path.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError("Audio feature extraction produced no rows.")
    return rows[0]


def _feature_table(row: dict[str, str]) -> list[list[str]]:
    return [[field, row.get(field, "")] for field in FEATURE_FIELDS]


def _parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _profile_notes(row: dict[str, str]) -> list[str]:
    notes: list[str] = []
    zcr = _parse_float(row.get("zero_crossing_rate"))
    centroid = _parse_float(row.get("spectral_centroid_hz"))
    rms = _parse_float(row.get("rms_energy"))

    if (centroid is not None and centroid >= 2200.0) or (zcr is not None and zcr >= 0.028):
        notes.append("High-frequency/noisy profile flag: elevated ZCR or spectral centroid.")
    else:
        notes.append("No high-frequency/noisy profile flag from the current thresholds.")
    if rms is not None:
        notes.append(f"RMS energy is `{rms:.6f}`, a simple loudness proxy.")
    return notes


def _recommend_for_intent(content_intent: str) -> tuple[str | None, str]:
    if content_intent == SPEECH_INTENT:
        return CLEAN_VOICE, "The declared content intent is speech/noisy speech."
    if content_intent == MUSIC_INTENT:
        return EXTRACT_VOCALS, "The declared content intent is music or a music video with vocals."
    return None, "The content intent is unknown, so the demo does not auto-select a task."


def _optional_router_result(input_path: Path) -> dict[str, object]:
    checkpoint_value = os.environ.get(ROUTER_CHECKPOINT_ENV_VAR, "").strip()
    if not checkpoint_value:
        return {
            "router_status": "disabled",
            "predicted_label": "",
            "confidence": None,
            "probabilities": {},
            "error": f"{ROUTER_CHECKPOINT_ENV_VAR} is not set.",
        }

    checkpoint_path = Path(checkpoint_value).expanduser()
    if not checkpoint_path.exists():
        return {
            "router_status": "disabled",
            "predicted_label": "",
            "confidence": None,
            "probabilities": {},
            "error": f"Router checkpoint not found: {checkpoint_path}",
        }

    try:
        from scripts.run_audio_router import run_audio_router

        summary = run_audio_router(checkpoint_path=checkpoint_path, input_path=input_path)
        return {
            "router_status": "enabled",
            "predicted_label": summary.get("predicted_label", ""),
            "confidence": summary.get("confidence"),
            "probabilities": summary.get("probabilities", {}),
            "error": "",
        }
    except Exception as exc:
        return {
            "router_status": "failed",
            "predicted_label": "",
            "confidence": None,
            "probabilities": {},
            "error": str(exc),
        }


def _probability_summary(probabilities: object) -> str:
    if not isinstance(probabilities, dict) or not probabilities:
        return "n/a"
    parts = []
    for label, value in sorted(probabilities.items()):
        try:
            parts.append(f"{label}={float(value):.3f}")
        except (TypeError, ValueError):
            parts.append(f"{label}={value}")
    return ", ".join(parts)


def _recommend_with_router(content_intent: str, router_result: dict[str, object]) -> tuple[str | None, str]:
    router_status = router_result.get("router_status")
    if router_status != "enabled":
        return _recommend_for_intent(content_intent)

    predicted_label = str(router_result.get("predicted_label", ""))
    confidence_raw = router_result.get("confidence")
    confidence = confidence_raw if isinstance(confidence_raw, (int, float)) else 0.0

    if confidence < ROUTER_CONFIDENCE_THRESHOLD:
        fallback_task, fallback_reason = _recommend_for_intent(content_intent)
        return fallback_task, (
            f"Router confidence `{confidence:.3f}` is below `{ROUTER_CONFIDENCE_THRESHOLD:.2f}`, "
            f"so the demo falls back to content intent. {fallback_reason}"
        )

    if predicted_label == "music":
        return EXTRACT_VOCALS, "Router predicts music with reasonable confidence; user intent should still be checked."

    if predicted_label == "speech_noise":
        if content_intent == SPEECH_INTENT:
            return CLEAN_VOICE, (
                "Router predicts noisy speech and the declared intent is speech/noisy speech. "
                "`clean_voice` remains the default recommendation; `target_noise_suppression` is experimental."
            )
        if content_intent == UNKNOWN_INTENT:
            return CLEAN_VOICE, (
                "Router predicts noisy speech, but content intent is unknown. "
                "`clean_voice` is suggested with caution."
            )
        return None, (
            "Router predicts noisy speech but the declared intent is music. "
            "No automatic task is selected because router output is content type, not final user intent."
        )

    if predicted_label == "environment_noise":
        return None, (
            "Router predicts environment noise. The MVP does not auto-select a restoration task for standalone "
            "environment noise."
        )

    return _recommend_for_intent(content_intent)


def analyze_demo_input(
    file_path: str | Path | None,
    content_intent: str = UNKNOWN_INTENT,
) -> tuple[str, list[list[str]], str | None]:
    """Run lightweight preflight feature analysis and return Gradio-ready outputs."""
    if file_path is None:
        return "### Error\nSelect an audio or video file before analysis.", [], None

    source = Path(file_path).expanduser()
    if not source.exists():
        return f"### Error\nInput file not found: `{source}`", [], None

    try:
        row = _extract_feature_row(source)
    except Exception as exc:
        return f"### Error\nAudio feature extraction failed: `{exc}`", [], None

    router_result = _optional_router_result(source)
    recommended_task, reason = _recommend_with_router(content_intent, router_result)
    profile_notes = _profile_notes(row)
    status = row.get("status", "")
    error = row.get("error", "")
    recommendation_text = f"`{recommended_task}`" if recommended_task else "No automatic recommendation"
    router_confidence = router_result.get("confidence")
    confidence_text = f"{float(router_confidence):.3f}" if isinstance(router_confidence, (int, float)) else "n/a"

    lines = [
        "### Preflight analysis",
        f"- **Feature extraction status:** `{status}`",
        f"- **Router status:** `{router_result.get('router_status', 'disabled')}`",
        f"- **Router predicted label:** `{router_result.get('predicted_label', '') or 'n/a'}`",
        f"- **Router confidence:** `{confidence_text}`",
        f"- **Router probabilities:** `{_probability_summary(router_result.get('probabilities'))}`",
        f"- **Content intent:** `{content_intent}`",
        f"- **Profile notes:** {' '.join(profile_notes)}",
        f"- **Recommended task:** {recommendation_text}",
        f"- **Reason:** {reason}",
        "- **Limitation:** routing is a baseline aid. If a router checkpoint is configured, the demo uses ML content-type prediction; otherwise it falls back to feature/intent rules. It does not determine final user intent automatically.",
    ]
    router_error = str(router_result.get("error", ""))
    if router_error:
        lines.append(f"- **Router note:** `{router_error}`")
    if error:
        lines.append(f"- **Feature extraction error:** `{error}`")
    return "\n".join(lines), _feature_table(row), recommended_task


def _intent_run_warning(content_intent: str, task: str) -> tuple[bool, str]:
    if content_intent == SPEECH_INTENT and task in MUSIC_TASKS:
        return True, (
            "### Blocked\n"
            "Speech/noisy speech intent is not compatible with vocal separation in the MVP demo. "
            "Choose `clean_voice` or change the content intent."
        )
    if content_intent == MUSIC_INTENT and task == CLEAN_VOICE:
        return False, (
            "### Caution\n"
            "Content intent is music/video with vocals. `clean_voice` can run, but it is intended for speech cleanup."
        )
    if content_intent == MUSIC_INTENT and task == TARGET_NOISE_SUPPRESSION:
        return True, (
            "### Blocked\n"
            "`target_noise_suppression` is an experimental speech/noise baseline and is not intended for music input."
        )
    if content_intent == UNKNOWN_INTENT:
        return False, "### Caution\nContent intent is unknown. The selected task will run without automatic validation."
    return False, ""


def run_demo_task(
    file_path: str | Path | None,
    task: str,
    content_intent: str = UNKNOWN_INTENT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[str, list[list[str]], str | None]:
    """Run one task through the unified runner and return Gradio-ready outputs."""
    if file_path is None:
        return "### Error\nSelect an audio or video file.", [], None

    source = Path(file_path).expanduser()
    if not source.exists():
        return f"### Error\nInput file not found: `{source}`", [], None

    should_block, warning = _intent_run_warning(content_intent, task)
    if should_block:
        return warning, [], None

    try:
        run_dir = run_audio_task(task=task, input_path=source, output_root=output_root)
        summary = _read_summary(run_dir)
        primary_output_path = summary.get("primary_output_path", "")
        downloadable_output = primary_output_path if primary_output_path and Path(primary_output_path).exists() else None
        markdown = _summary_markdown(summary, run_dir)
        if warning:
            markdown = f"{warning}\n\n{markdown}"
        return markdown, _summary_table(summary), downloadable_output
    except Exception as exc:
        return f"### Error\n{exc}", [], None


def create_demo() -> object:
    """Build the minimal queued Gradio demo."""
    import gradio as gr

    task_names = [task.name for task in list_supported_tasks()]
    with gr.Blocks(title="ML Audio Task Demo") as demo:
        gr.Markdown("# ML Audio Task Demo")
        with gr.Row():
            upload = gr.File(label="Audio or video input", type="filepath", file_types=["audio", "video"])
            intent_dropdown = gr.Dropdown(label="Content intent", choices=CONTENT_INTENTS, value=SPEECH_INTENT)
            task_dropdown = gr.Dropdown(label="Task", choices=task_names, value="clean_voice")
        analyze_button = gr.Button("Analyze Input")
        analysis_markdown = gr.Markdown()
        feature_table = gr.Dataframe(headers=["Feature", "Value"], label="Preflight features", interactive=False)
        recommended_task = gr.Textbox(label="Recommended task", interactive=False)
        run_button = gr.Button("Run Task", variant="primary")
        status_markdown = gr.Markdown()
        summary_table = gr.Dataframe(headers=["Field", "Value"], label="Summary", interactive=False)
        primary_output = gr.File(label="Primary output")

        analyze_button.click(
            fn=analyze_demo_input,
            inputs=[upload, intent_dropdown],
            outputs=[analysis_markdown, feature_table, recommended_task],
            concurrency_limit=1,
            concurrency_id="audio-task-demo",
        )
        run_button.click(
            fn=run_demo_task,
            inputs=[upload, task_dropdown, intent_dropdown],
            outputs=[status_markdown, summary_table, primary_output],
            concurrency_limit=1,
            concurrency_id="audio-task-demo",
        )

    return demo.queue(default_concurrency_limit=1, max_size=8)


if __name__ == "__main__":
    create_demo().launch()
