"""Unified CLI runner for MVP audio processing tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_music_separation import run_music_separation
from src.api.contracts import DenoiseRequest
from src.io.paths import derive_output_mode, infer_input_type
from src.media.ffmpeg_wrapper import FFmpegWrapper
from src.pipeline.run_pipeline import run_pipeline
from src.router.task_registry import (
    CLEAN_VOICE,
    EXTRACT_VOCALS,
    REMOVE_VOCALS,
    TARGET_NOISE_SUPPRESSION,
    VOICE_PITCH_HIGH,
    VOICE_PITCH_LOW,
    get_task_spec,
    list_supported_tasks,
)


FIELDNAMES = [
    "run_id",
    "task",
    "engine",
    "input_path",
    "input_type",
    "status",
    "runtime_sec",
    "primary_output_path",
    "error",
]
MUSIC_TASKS = {EXTRACT_VOCALS, REMOVE_VOCALS}
VOICE_PITCH_FACTORS = {
    VOICE_PITCH_HIGH: 1.25,
    VOICE_PITCH_LOW: 0.75,
}


def _run_id_for_input(input_path: Path) -> str:
    path_hash = hashlib.sha1(str(input_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{input_path.stem}-{path_hash}"


def _read_music_summary(run_dir: Path) -> dict[str, str]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return {}
    with summary_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return rows[0] if rows else {}


def _resolve_target_noise_checkpoint(target_noise_checkpoint: Path | None) -> Path:
    checkpoint = target_noise_checkpoint
    if checkpoint is None:
        env_value = os.getenv("TARGET_NOISE_SUPPRESSOR_CHECKPOINT")
        checkpoint = Path(env_value) if env_value else None
    if checkpoint is None:
        raise ValueError(
            "target_noise_suppression requires --target-noise-checkpoint "
            "or TARGET_NOISE_SUPPRESSOR_CHECKPOINT."
        )
    return Path(checkpoint).expanduser()


def _run_target_noise_suppression_inference(
    *,
    checkpoint_path: Path,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    device: str,
) -> dict[str, Path]:
    from scripts.run_target_noise_suppressor import run_target_noise_suppressor

    return run_target_noise_suppressor(
        checkpoint_path=checkpoint_path,
        input_path=input_path,
        output_path=output_path,
        summary_path=summary_path,
        device=device,
    )


def _summary_row(
    *,
    run_id: str,
    task: str,
    engine: str,
    input_path: Path,
    input_type: str,
    status: str,
    runtime_sec: float,
    primary_output_path: str,
    error: str,
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "task": task,
        "engine": engine,
        "input_path": str(input_path),
        "input_type": input_type,
        "status": status,
        "runtime_sec": f"{runtime_sec:.6f}",
        "primary_output_path": primary_output_path,
        "error": error,
    }


def run_audio_task(
    *,
    task: str,
    input_path: Path,
    output_root: Path = Path("outputs/audio-task-runs"),
    demucs_python: Path = Path(".venv-demucs/bin/python"),
    engine: str = "deepfilternet",
    model: str = "htdemucs",
    device: str = "cpu",
    jobs: int = 1,
    target_noise_checkpoint: Path | None = None,
) -> Path:
    """Run one supported MVP task and write summary artifacts."""
    task_spec = get_task_spec(task)
    source = Path(input_path).expanduser().resolve()
    input_type = infer_input_type(source)
    run_id = _run_id_for_input(source)
    run_dir = Path(output_root).resolve() / task / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    status = "failed"
    primary_output_path = ""
    error = ""
    summary_engine = engine if task == CLEAN_VOICE else task_spec.engine

    try:
        if task == CLEAN_VOICE:
            result = run_pipeline(
                DenoiseRequest(
                    input_path=source,
                    output_dir=run_dir,
                    output_mode=derive_output_mode(input_type),
                    engine_name=engine,
                )
            )
            if result.final_output_path is None:
                error = "Pipeline did not return a final output path."
            else:
                status = "success"
                primary_output_path = str(result.final_output_path)
        elif task in MUSIC_TASKS:
            music_run_dir = run_music_separation(
                input_path=source,
                output_root=run_dir,
                demucs_python=demucs_python,
                model=model,
                device=device,
                jobs=jobs,
                task_name=task,
            )
            music_summary = _read_music_summary(music_run_dir)
            status = music_summary.get("status", "failed")
            primary_output_path = music_summary.get("primary_output_path", "")
            error = music_summary.get("error", "")
        elif task == TARGET_NOISE_SUPPRESSION:
            if input_type == "video":
                raise ValueError("target_noise_suppression currently supports audio input only.")
            checkpoint = _resolve_target_noise_checkpoint(target_noise_checkpoint)
            enhanced_path = run_dir / f"{source.stem}.target_noise_suppressed.wav"
            inference_summary_path = run_dir / "target_noise_suppressor_summary.json"
            inference_paths = _run_target_noise_suppression_inference(
                checkpoint_path=checkpoint,
                input_path=source,
                output_path=enhanced_path,
                summary_path=inference_summary_path,
                device=device,
            )
            status = "success"
            primary_output_path = str(inference_paths["output"])
        elif task in VOICE_PITCH_FACTORS:
            ffmpeg = FFmpegWrapper()
            ffmpeg.probe_input(source)
            prepared_path = ffmpeg.prepare_audio(source, run_dir)
            effect_name = "high_pitch" if task == VOICE_PITCH_HIGH else "low_pitch"
            output_path = run_dir / f"{source.stem}.{effect_name}.wav"
            ffmpeg.apply_pitch_effect(
                prepared_path,
                output_path,
                pitch_factor=VOICE_PITCH_FACTORS[task],
            )
            status = "success"
            primary_output_path = str(output_path)
        else:
            error = f"Unsupported task: {task}"
    except Exception as exc:
        error = str(exc)

    runtime_sec = time.perf_counter() - started_at
    row = _summary_row(
        run_id=run_id,
        task=task,
        engine=summary_engine,
        input_path=source,
        input_type=input_type,
        status=status,
        runtime_sec=runtime_sec,
        primary_output_path=primary_output_path,
        error=error,
    )
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)
    (run_dir / "summary.json").write_text(json.dumps([row], indent=2), encoding="utf-8")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the unified audio task CLI parser."""
    task_choices = tuple(task.name for task in list_supported_tasks())
    parser = argparse.ArgumentParser(description="Run one supported MVP audio task.")
    parser.add_argument("--task", required=True, choices=task_choices)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("outputs/audio-task-runs"), type=Path)
    parser.add_argument("--demucs-python", default=Path(".venv-demucs/bin/python"), type=Path)
    parser.add_argument("--engine", default="deepfilternet")
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", default=1, type=int)
    parser.add_argument("--target-noise-checkpoint", default=None, type=Path)
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        run_dir = run_audio_task(
            task=args.task,
            input_path=args.input,
            output_root=args.output_root,
            demucs_python=args.demucs_python,
            engine=args.engine,
            model=args.model,
            device=args.device,
            jobs=args.jobs,
            target_noise_checkpoint=args.target_noise_checkpoint,
        )
        print(f"Wrote audio task run: {run_dir}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
