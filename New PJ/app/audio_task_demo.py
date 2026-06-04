"""Minimal teacher-facing Gradio demo for unified MVP audio tasks."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_audio_task import run_audio_task
from src.router.task_registry import list_supported_tasks


DEFAULT_OUTPUT_ROOT = Path("outputs/demo-runs")
SUMMARY_FIELDS = ["task", "engine", "status", "runtime_sec", "input_type", "primary_output_path", "error"]


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


def run_demo_task(
    file_path: str | Path | None,
    task: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[str, list[list[str]], str | None]:
    """Run one task through the unified runner and return Gradio-ready outputs."""
    if file_path is None:
        return "### Error\nSelect an audio or video file.", [], None

    source = Path(file_path).expanduser()
    if not source.exists():
        return f"### Error\nInput file not found: `{source}`", [], None

    try:
        run_dir = run_audio_task(task=task, input_path=source, output_root=output_root)
        summary = _read_summary(run_dir)
        primary_output_path = summary.get("primary_output_path", "")
        downloadable_output = primary_output_path if primary_output_path and Path(primary_output_path).exists() else None
        return _summary_markdown(summary, run_dir), _summary_table(summary), downloadable_output
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
            task_dropdown = gr.Dropdown(label="Task", choices=task_names, value="clean_voice")
        run_button = gr.Button("Run Task", variant="primary")
        status_markdown = gr.Markdown()
        summary_table = gr.Dataframe(headers=["Field", "Value"], label="Summary", interactive=False)
        primary_output = gr.File(label="Primary output")

        run_button.click(
            fn=run_demo_task,
            inputs=[upload, task_dropdown],
            outputs=[status_markdown, summary_table, primary_output],
            concurrency_limit=1,
            concurrency_id="audio-task-demo",
        )

    return demo.queue(default_concurrency_limit=1, max_size=8)


if __name__ == "__main__":
    create_demo().launch()
