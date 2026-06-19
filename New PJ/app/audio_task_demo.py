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
from src.planner.processing_planner import (
    ACTION_RUN_TASK,
    ANALYZE_ONLY,
    AUTO,
    BLOCK_TARGET_NOISE_SUPPRESSION_MANUAL_ONLY,
    EXTRACT_VOCALS_GOAL,
    IMPROVE_SPEECH_CLARITY,
    REDUCE_TARGET_NOISE,
    REMOVE_VOCALS_GOAL,
    ProcessingCapabilities,
    ProcessingFacts,
    ProcessingPlan,
    RouterEvidence,
    plan_processing,
)
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
TARGET_NOISE_INTENT = "Target noise reduction (experimental)"
AUTO_INTENT = "Auto detect / not sure"
UNKNOWN_INTENT = "Unknown / not sure"
CONTENT_INTENTS = [AUTO_INTENT, SPEECH_INTENT, MUSIC_INTENT, TARGET_NOISE_INTENT, UNKNOWN_INTENT]
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
ROUTER_CHECKPOINT_ENV_VAR = "AUDIO_ROUTER_CHECKPOINT"
ROUTER_CONFIDENCE_THRESHOLD = 0.90
DEMO_CSS = """
.gradio-container {
  max-width: 1080px !important;
  margin: 0 auto !important;
}
.demo-hero h1 {
  margin-bottom: 0.25rem;
}
.demo-subtitle {
  color: #475569;
  font-size: 0.98rem;
  margin-top: 0;
}
.demo-section {
  padding: 6px 0 12px 0;
}
.demo-section h2,
.demo-section h3 {
  margin-top: 0;
  margin-bottom: 0.5rem;
}
.compact-markdown p,
.compact-markdown li {
  margin-top: 0.2rem;
  margin-bottom: 0.2rem;
}
"""


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


def _planner_goal_for_intent(content_intent: str) -> str:
    if content_intent == SPEECH_INTENT:
        return IMPROVE_SPEECH_CLARITY
    if content_intent == MUSIC_INTENT:
        return EXTRACT_VOCALS_GOAL
    if content_intent == TARGET_NOISE_INTENT:
        return REDUCE_TARGET_NOISE
    if content_intent == AUTO_INTENT:
        return AUTO
    return ANALYZE_ONLY


def _planner_goal_for_task(task: str) -> str | None:
    return {
        CLEAN_VOICE: IMPROVE_SPEECH_CLARITY,
        EXTRACT_VOCALS: EXTRACT_VOCALS_GOAL,
        REMOVE_VOCALS: REMOVE_VOCALS_GOAL,
        TARGET_NOISE_SUPPRESSION: REDUCE_TARGET_NOISE,
    }.get(task)


def _processing_facts(input_path: Path, feature_row: dict[str, str]) -> ProcessingFacts:
    return ProcessingFacts(
        input_type=_input_type_for_path(input_path),
        duration_sec=_parse_float(feature_row.get("duration_sec")),
        rms_energy=_parse_float(feature_row.get("rms_energy")),
        is_valid_media=True,
    )


def _router_evidence(router_result: dict[str, object]) -> RouterEvidence:
    predicted_label = str(router_result.get("predicted_label") or "")
    normalized_label = {
        "music": "music_with_vocals",
        "environment_noise": "environment_only",
    }.get(predicted_label, predicted_label)
    warnings = router_result.get("warnings")
    return RouterEvidence(
        router_status=str(router_result.get("router_status") or "disabled"),
        predicted_label=normalized_label or None,
        confidence=(
            float(router_result["confidence"])
            if isinstance(router_result.get("confidence"), (int, float))
            else None
        ),
        accepted=router_result.get("accepted") is True,
        route_target=str(router_result.get("route_target") or "") or None,
        recommended_task=str(router_result.get("recommended_task") or "") or None,
        decision_reason=str(router_result.get("decision_reason") or "") or None,
        warnings=[str(warning) for warning in warnings] if isinstance(warnings, list) else [],
    )


def _demo_capabilities() -> ProcessingCapabilities:
    return ProcessingCapabilities(
        clean_voice_available=True,
        extract_vocals_available=True,
        remove_vocals_available=True,
        target_noise_suppression_available=False,
    )


def _display_values(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "none"


def _processing_plan_markdown(plan: ProcessingPlan) -> list[str]:
    recommended_task = f"`{plan.recommended_task}`" if plan.recommended_task else "none"
    algorithm = f"`{plan.algorithm}`" if plan.algorithm else "none"
    return [
        "",
        "#### Detailed processing plan",
        f"- **Mode:** `{plan.mode}`",
        f"- **Action:** `{plan.action}`",
        f"- **Recommended task:** {recommended_task}",
        f"- **Algorithm:** {algorithm}",
        f"- **Warnings:** {_display_values(plan.warnings)}",
        f"- **Blocked reasons:** {_display_values(plan.blocked_reasons)}",
        f"- **Alternatives:** {_display_values(plan.alternatives)}",
        f"- **Explanation:** {plan.explanation}",
    ]


def _decision_badge(action: str) -> str:
    return {
        "run_task": "RUN TASK",
        "manual_required": "MANUAL REQUIRED",
        "analyze_only": "ANALYZE ONLY",
        "no_process": "NO PROCESS",
    }.get(action, action.replace("_", " ").upper())


def _controller_decision_markdown(plan: ProcessingPlan) -> list[str]:
    recommended_task = f"`{plan.recommended_task}`" if plan.recommended_task else "none"
    engine_algorithm = " / ".join(
        value for value in (plan.engine_family, plan.algorithm) if value
    ) or "none"
    safety_items = [*plan.warnings, *plan.blocked_reasons]
    if plan.alternatives:
        safety_items.append(f"alternatives: {', '.join(plan.alternatives)}")
    lines = [
        "### Controller Decision",
        f"- **Decision:** **{_decision_badge(plan.action)}**",
        f"- **Recommended next step:** {recommended_task}",
        f"- **Engine/algorithm:** `{engine_algorithm}`",
        f"- **Why:** {plan.explanation}",
        f"- **Safety notes:** {_display_values(safety_items)}",
    ]
    if BLOCK_TARGET_NOISE_SUPPRESSION_MANUAL_ONLY in plan.blocked_reasons:
        fallback = "`clean_voice`" if CLEAN_VOICE in plan.alternatives else "manual review"
        lines.append(
            "- **Target-aware policy:** Target suppressor is experimental/manual-only. "
            f"Recommended automatic fallback: {fallback} when appropriate."
        )
    return lines


def _optional_router_result(input_path: Path) -> dict[str, object]:
    checkpoint_value = os.environ.get(ROUTER_CHECKPOINT_ENV_VAR, "").strip()
    if not checkpoint_value:
        return {
            "router_status": "disabled",
            "predicted_label": "",
            "confidence": None,
            "accepted": False,
            "confidence_threshold": ROUTER_CONFIDENCE_THRESHOLD,
            "route_target": "manual_required",
            "engine_target": "none",
            "recommended_task": None,
            "decision_reason": "router_checkpoint_not_configured",
            "warnings": [],
            "probabilities": {},
            "error": f"{ROUTER_CHECKPOINT_ENV_VAR} is not set.",
        }

    checkpoint_path = Path(checkpoint_value).expanduser()
    if not checkpoint_path.exists():
        return {
            "router_status": "disabled",
            "predicted_label": "",
            "confidence": None,
            "accepted": False,
            "confidence_threshold": ROUTER_CONFIDENCE_THRESHOLD,
            "route_target": "manual_required",
            "engine_target": "none",
            "recommended_task": None,
            "decision_reason": "router_checkpoint_missing",
            "warnings": [],
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
            "accepted": summary.get("accepted"),
            "confidence_threshold": summary.get("confidence_threshold"),
            "route_target": summary.get("route_target"),
            "engine_target": summary.get("engine_target"),
            "recommended_task": summary.get("recommended_task"),
            "decision_reason": summary.get("decision_reason"),
            "warnings": summary.get("warnings", []),
            "probabilities": summary.get("probabilities", {}),
            "error": "",
        }
    except Exception as exc:
        return {
            "router_status": "failed",
            "predicted_label": "",
            "confidence": None,
            "accepted": False,
            "confidence_threshold": ROUTER_CONFIDENCE_THRESHOLD,
            "route_target": "manual_required",
            "engine_target": "none",
            "recommended_task": None,
            "decision_reason": "router_failed",
            "warnings": [],
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


def _feature_value_summary(row: dict[str, str]) -> str:
    return ", ".join(f"{field}={row.get(field, '') or 'n/a'}" for field in FEATURE_FIELDS)


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
    plan = plan_processing(
        _planner_goal_for_intent(content_intent),
        _processing_facts(source, row),
        _router_evidence(router_result),
        _demo_capabilities(),
    )
    recommended_task = plan.recommended_task
    profile_notes = _profile_notes(row)
    status = row.get("status", "")
    error = row.get("error", "")
    router_confidence = router_result.get("confidence")
    confidence_text = f"{float(router_confidence):.3f}" if isinstance(router_confidence, (int, float)) else "n/a"
    router_warnings = router_result.get("warnings")
    warnings_text = ", ".join(str(warning) for warning in router_warnings) if isinstance(router_warnings, list) else "n/a"
    if not warnings_text:
        warnings_text = "n/a"

    lines = [
        *_controller_decision_markdown(plan),
        "",
        "### Detailed evidence",
        f"- **Feature extraction status:** `{status}`",
        f"- **Feature values:** `{_feature_value_summary(row)}`",
        f"- **Profile notes:** {' '.join(profile_notes)}",
        f"- **Processing goal:** `{content_intent}`",
        f"- **Router status:** `{router_result.get('router_status', 'disabled')}`",
        f"- **Router predicted label:** `{router_result.get('predicted_label', '') or 'n/a'}`",
        f"- **Router confidence:** `{confidence_text}`",
        f"- **Router accepted:** `{str(router_result.get('accepted', 'n/a')).lower()}`",
        f"- **Router threshold:** `{router_result.get('confidence_threshold', 'n/a')}`",
        f"- **Router route target:** `{router_result.get('route_target', 'n/a')}`",
        f"- **Router engine target:** `{router_result.get('engine_target', 'n/a')}`",
        f"- **Router decision reason:** `{router_result.get('decision_reason', 'n/a')}`",
        f"- **Router warnings:** `{warnings_text}`",
        f"- **Router probabilities:** `{_probability_summary(router_result.get('probabilities'))}`",
    ]
    lines.extend(_processing_plan_markdown(plan))
    router_error = str(router_result.get("error", ""))
    if router_error:
        lines.append(f"- **Router note:** `{router_error}`")
    if error:
        lines.append(f"- **Feature extraction error:** `{error}`")
    return "\n".join(lines), _feature_table(row), recommended_task


def analyze_demo_input_for_ui(
    file_path: str | Path | None,
    content_intent: str = AUTO_INTENT,
    current_task: str = CLEAN_VOICE,
) -> tuple[str, list[list[str]], str, str]:
    """Analyze input and return outputs that can update the recommendation and task dropdown."""
    markdown, table, recommended_task = analyze_demo_input(file_path, content_intent)
    task_names = {task.name for task in list_supported_tasks()}
    if recommended_task and recommended_task in task_names:
        return markdown, table, recommended_task, recommended_task
    fallback_task = current_task if current_task in task_names else CLEAN_VOICE
    return markdown, table, "No automatic recommendation", fallback_task


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

    goal = _planner_goal_for_task(task)
    if goal is None:
        return f"### Blocked\nUnsupported task: `{task}`", [], None

    try:
        feature_row = _extract_feature_row(source)
        facts = _processing_facts(source, feature_row)
    except Exception:
        facts = ProcessingFacts(input_type=_input_type_for_path(source))
    router_result = _optional_router_result(source)
    plan = plan_processing(
        goal,
        facts,
        _router_evidence(router_result),
        _demo_capabilities(),
    )
    if plan.action != ACTION_RUN_TASK or plan.blocked_reasons:
        lines = ["### Blocked", *_processing_plan_markdown(plan)]
        return "\n".join(lines), [], None

    try:
        run_dir = run_audio_task(task=task, input_path=source, output_root=output_root)
        summary = _read_summary(run_dir)
        primary_output_path = summary.get("primary_output_path", "")
        downloadable_output = primary_output_path if primary_output_path and Path(primary_output_path).exists() else None
        markdown = _summary_markdown(summary, run_dir)
        if plan.warnings:
            markdown = (
                "### Processing warnings\n"
                f"{_display_values(plan.warnings)}\n\n"
                f"{markdown}"
            )
        return markdown, _summary_table(summary), downloadable_output
    except Exception as exc:
        return f"### Error\n{exc}", [], None


def create_demo() -> object:
    """Build the minimal queued Gradio demo."""
    import gradio as gr

    task_names = [task.name for task in list_supported_tasks()]
    with gr.Blocks(title="Target-Aware Audio Processing Controller", css=DEMO_CSS) as demo:
        gr.Markdown(
            "# Target-Aware Audio Processing Controller\n"
            "<p class='demo-subtitle'>Upload audio/video, choose a goal, and let the controller recommend "
            "a safe processing path.</p>",
            elem_classes=["demo-hero"],
        )

        with gr.Row():
            with gr.Column(scale=5, elem_classes=["demo-section"]):
                gr.Markdown("## Input")
                upload = gr.File(label="Audio or video input", type="filepath", file_types=["audio", "video"])
                intent_dropdown = gr.Dropdown(label="Processing goal", choices=CONTENT_INTENTS, value=AUTO_INTENT)
                task_dropdown = gr.Dropdown(label="Manual task override", choices=task_names, value=CLEAN_VOICE)
                with gr.Row():
                    analyze_button = gr.Button("Analyze Input")
                    run_button = gr.Button("Run Task", variant="primary")

            with gr.Column(scale=7, elem_classes=["demo-section"]):
                gr.Markdown("## Controller")
                recommended_task = gr.Textbox(label="Controller recommendation", interactive=False)
                analysis_markdown = gr.Markdown(elem_classes=["compact-markdown"])
                feature_table = gr.Dataframe(headers=["Feature", "Value"], label="Audio feature values", interactive=False)
                gr.Markdown("## Processing Result")
                status_markdown = gr.Markdown(elem_classes=["compact-markdown"])
                summary_table = gr.Dataframe(headers=["Field", "Value"], label="Run summary", interactive=False)
                primary_output = gr.File(label="Primary output")

        analyze_button.click(
            fn=analyze_demo_input_for_ui,
            inputs=[upload, intent_dropdown, task_dropdown],
            outputs=[analysis_markdown, feature_table, recommended_task, task_dropdown],
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
