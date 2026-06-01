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
from src.io.paths import derive_output_mode, infer_input_type, validate_input_path
from src.pipeline.run_pipeline import run_pipeline


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


def restore_fast_mode(input_path: str | Path | None, output_root: Path = DEFAULT_OUTPUT_ROOT) -> tuple[str, str | None, str, str, str, str]:
    """Run Fast Mode through the pipeline and return UI-ready values."""
    if input_path is None:
        return "", None, "", "", "", "Select an audio or video file."

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
        return (
            str(original_path),
            str(result.final_output_path),
            f"{runtime_sec:.6f}",
            "" if duration_sec is None else f"{duration_sec:.6f}",
            "" if rtf is None else f"{rtf:.6f}",
            "",
        )
    except Exception as exc:
        runtime_sec = time.perf_counter() - started_at
        return original_output, None, f"{runtime_sec:.6f}", "", "", str(exc)


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
