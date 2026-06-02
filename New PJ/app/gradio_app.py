"""Minimal local Gradio Restore tab for Fast Mode."""

from __future__ import annotations

import time
import uuid
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.contracts import DenoiseRequest
from src.eval.metrics import load_mono_audio
from src.eval.plots import generate_restoration_plots
from src.io.paths import derive_output_mode, infer_input_type, validate_input_path
from src.pipeline.run_pipeline import run_pipeline
from src.reporting.report_writer import write_restore_report


DEFAULT_OUTPUT_ROOT = Path("outputs/ui-runs")


def _audio_duration_sec(*paths: Path | None) -> float | None:
    for path in paths:
        if path is None:
            continue
        try:
            audio, sample_rate = load_mono_audio(path)
        except Exception:
            continue
        return len(audio) / sample_rate
    return None


def _audio_preview_path(path: Path) -> str | None:
    try:
        return str(path) if infer_input_type(path) == "audio" else None
    except ValueError:
        return None


def restore_fast_mode(
    input_path: str | Path | None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[str, str | None, str | None, str | None, list[str], str | None, str | None, str, str, str, str]:
    """Run Fast Mode through the pipeline and return UI-ready values."""
    if input_path is None:
        return "", None, None, None, [], None, None, "", "", "", "Select an audio or video file."

    started_at = time.perf_counter()
    original_output = str(input_path)
    try:
        original_path = validate_input_path(input_path)
        original_output = str(original_path)
        output_mode = derive_output_mode(infer_input_type(original_path))
        run_dir = Path(output_root).resolve() / f"{original_path.stem}-{uuid.uuid4().hex[:8]}"
        result = run_pipeline(
            DenoiseRequest(
                input_path=original_path,
                output_dir=run_dir,
                output_mode=output_mode,
                engine_name="deepfilternet",
            )
        )
        runtime_sec = time.perf_counter() - started_at
        if result.final_output_path is None:
            raise RuntimeError("Pipeline did not return a final output path.")

        duration_sec = _audio_duration_sec(original_path, result.intermediate_audio_path)
        rtf = None if duration_sec is None else runtime_sec / duration_sec
        plot_paths: list[str] = []
        warning = ""
        original_preview = _audio_preview_path(original_path)
        restored_preview = _audio_preview_path(result.final_output_path)
        before_plot_path = (
            result.intermediate_audio_path
            if result.intermediate_audio_path is not None and result.intermediate_audio_path.exists()
            else original_path
        )
        if restored_preview is not None:
            try:
                plots = generate_restoration_plots(before_plot_path, result.final_output_path, run_dir / "report")
                plot_paths = [str(path) for path in plots.values()]
            except Exception as exc:
                warning = f"Plot generation failed: {exc}"
        else:
            warning = "Plot generation skipped: restored output is not an audio file."
        report_json_path: str | None = None
        report_markdown_path: str | None = None
        try:
            report_paths = write_restore_report(
                report_dir=run_dir / "report",
                input_path=original_path,
                output_path=result.final_output_path,
                engine_name=result.engine_name,
                runtime_sec=runtime_sec,
                audio_duration_sec=duration_sec,
                rtf=rtf,
                plot_paths=plot_paths,
                warning=warning,
            )
            report_json_path, report_markdown_path = (str(path) for path in report_paths)
        except Exception as exc:
            report_warning = f"Report writing failed: {exc}"
            warning = f"{warning} {report_warning}".strip()
        return (
            str(original_path),
            str(result.final_output_path),
            original_preview,
            restored_preview,
            plot_paths,
            report_json_path,
            report_markdown_path,
            f"{runtime_sec:.6f}",
            "" if duration_sec is None else f"{duration_sec:.6f}",
            "" if rtf is None else f"{rtf:.6f}",
            warning,
        )
    except Exception as exc:
        runtime_sec = time.perf_counter() - started_at
        return original_output, None, None, None, [], None, None, f"{runtime_sec:.6f}", "", "", str(exc)


def create_app() -> object:
    """Build the queued Gradio app."""
    import gradio as gr

    with gr.Blocks(title="Speech Restoration Studio") as app:
        with gr.Tab("Restore"):
            upload = gr.File(label="Audio or video input", type="filepath", file_types=["audio", "video"])
            restore_button = gr.Button("Run Fast Mode", variant="primary")
            with gr.Row():
                original_output = gr.File(label="Original")
                restored_output = gr.File(label="Restored output")
            with gr.Row():
                original_preview = gr.Audio(label="Original audio preview")
                restored_preview = gr.Audio(label="Restored audio preview")
            plot_gallery = gr.Gallery(label="Waveform and spectrogram report", columns=2)
            with gr.Row():
                report_json_output = gr.File(label="Report JSON")
                report_markdown_output = gr.File(label="Report Markdown")
            with gr.Row():
                runtime_output = gr.Textbox(label="Runtime (sec)", interactive=False)
                duration_output = gr.Textbox(label="Audio duration (sec)", interactive=False)
                rtf_output = gr.Textbox(label="RTF", interactive=False)
            error_output = gr.Textbox(label="Error", interactive=False)

            restore_button.click(
                fn=restore_fast_mode,
                inputs=upload,
                outputs=[
                    original_output,
                    restored_output,
                    original_preview,
                    restored_preview,
                    plot_gallery,
                    report_json_output,
                    report_markdown_output,
                    runtime_output,
                    duration_output,
                    rtf_output,
                    error_output,
                ],
                concurrency_limit=1,
                concurrency_id="fast-mode",
            )

    return app.queue(default_concurrency_limit=1, max_size=8)


if __name__ == "__main__":
    create_app().launch()
