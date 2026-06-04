"""Minimal teacher-facing Gradio demo for unified MVP audio tasks."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_audio_features import extract_audio_features
from scripts.run_audio_task import run_audio_task
from src.router.task_registry import CLEAN_VOICE, EXTRACT_VOCALS, REMOVE_VOCALS, list_supported_tasks


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

    recommended_task, reason = _recommend_for_intent(content_intent)
    profile_notes = _profile_notes(row)
    status = row.get("status", "")
    error = row.get("error", "")
    recommendation_text = f"`{recommended_task}`" if recommended_task else "No automatic recommendation"

    lines = [
        "### Preflight analysis",
        f"- **Feature extraction status:** `{status}`",
        f"- **Content intent:** `{content_intent}`",
        f"- **Profile notes:** {' '.join(profile_notes)}",
        f"- **Recommended task:** {recommendation_text}",
        f"- **Reason:** {reason}",
        "- **Limitation:** this is a feature-based recommendation aid, not a deployed ML speech/music classifier.",
    ]
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
