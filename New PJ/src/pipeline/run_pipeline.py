"""Manifest-first pipeline entrypoint with real arnndn denoise."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.api.contracts import DenoiseRequest, DenoiseResult
from src.eval.checks import evaluate_output_artifact
from src.engine.base import DenoiseEngine
from src.engine.deepfilternet_cli_engine import DeepFilterNetCliEngine
from src.engine.ffmpeg_arnndn_engine import FFmpegArnndnEngine
from src.engine.noisereduce_engine import NoisereduceEngine
from src.io.paths import (
    derive_manifest_path,
    derive_output_mode,
    derive_planned_output_path,
    infer_input_type,
    validate_input_path,
    validate_output_dir,
)
from src.media.ffmpeg_wrapper import FFmpegWrapper
from src.storage.manifest import build_run_manifest, write_manifest


def _log(message: str) -> None:
    print(message)


def _build_engine(engine_name: str, ffmpeg_wrapper: FFmpegWrapper) -> DenoiseEngine:
    """Return the selected denoise engine."""
    if engine_name == "ffmpeg-arnndn":
        return FFmpegArnndnEngine(ffmpeg_wrapper=ffmpeg_wrapper)
    if engine_name == "noisereduce":
        return NoisereduceEngine()
    if engine_name == "deepfilternet":
        return DeepFilterNetCliEngine()
    raise ValueError(f"Unsupported denoise engine: {engine_name}")


def run_pipeline(request: DenoiseRequest, ffmpeg_wrapper: FFmpegWrapper | None = None) -> DenoiseResult:
    """Run the denoise pipeline and write a manifest artifact."""
    stage_statuses: dict[str, str] = {}

    input_path = validate_input_path(request.input_path)
    output_dir = validate_output_dir(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_statuses["validate"] = "completed_real"
    _log(f"[validate] input={input_path}")

    ffmpeg = ffmpeg_wrapper or FFmpegWrapper()
    probe = ffmpeg.probe_input(input_path)
    stage_statuses["probe_input"] = "completed_real"
    _log(f"[probe_input] completed_real inferred_input_type={probe.input_type}")

    prepared_audio_path = ffmpeg.prepare_audio(input_path, output_dir)
    stage_statuses["prepare_audio"] = "completed_real"
    _log(f"[prepare_audio] completed_real path={prepared_audio_path}")

    engine = _build_engine(request.engine_name, ffmpeg)
    engine.load()
    denoised_audio_path = engine.denoise(
        prepared_audio_path,
        derive_planned_output_path(input_path, output_dir, output_mode="audio"),
    )
    stage_statuses["denoise"] = "completed_real"
    _log(f"[denoise] completed_real path={denoised_audio_path}")

    if request.output_mode == "video":
        final_media_path = ffmpeg.remux_video(input_path, denoised_audio_path, output_dir)
        stage_statuses["remux_video"] = "completed_real"
        _log(f"[remux_video] completed_real path={final_media_path}")
    else:
        final_media_path = ffmpeg.export_audio(denoised_audio_path, input_path, output_dir)
        stage_statuses["export_audio"] = "completed_real"
        _log(f"[export_audio] completed_real path={final_media_path}")

    manifest_path = derive_manifest_path(input_path, output_dir)
    manifest_payload = build_run_manifest(
        input_path=str(input_path),
        inferred_input_type=probe.input_type,
        planned_output_paths={
            "prepared_audio_path": str(prepared_audio_path),
            "clean_audio_path": str(denoised_audio_path),
            "final_media_path": str(final_media_path),
        },
        selected_engine=engine.name,
        stage_statuses=stage_statuses,
        dry_run=False,
        final_status="completed_real",
    )
    manifest_payload["manifest_path"] = str(manifest_path)
    write_manifest(manifest_path, manifest_payload)
    stage_statuses["write_manifest"] = "completed_real"
    _log(f"[write_manifest] path={manifest_path}")

    eval_summary = evaluate_output_artifact(final_media_path)
    stage_statuses["eval"] = "completed_real" if eval_summary["output_exists"] else "failed"
    _log(f"[eval] output_exists={eval_summary['output_exists']}")

    manifest_payload["stage_statuses"] = stage_statuses
    manifest_payload["eval"] = eval_summary
    write_manifest(manifest_path, manifest_payload)

    return DenoiseResult(
        status="completed_real",
        final_output_path=final_media_path,
        intermediate_audio_path=prepared_audio_path,
        engine_name=engine.name,
        run_summary=manifest_payload,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the arnndn pipeline entrypoint."""
    parser = argparse.ArgumentParser(description="Offline denoise demo pipeline using ffmpeg arnndn.")
    parser.add_argument("input_path", help="Path to a local audio or video file.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where the manifest artifact will be written.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep the prepared intermediate audio path in the returned result metadata.",
    )
    parser.add_argument(
        "--arnndn-mix",
        type=float,
        default=None,
        help="Optional arnndn mix value passed to ffmpeg.",
    )
    parser.add_argument(
        "--engine",
        choices=("ffmpeg-arnndn", "noisereduce", "deepfilternet"),
        default="ffmpeg-arnndn",
        help="Denoise engine to use.",
    )
    return parser


def main() -> int:
    """CLI entrypoint for the arnndn scaffold."""
    args = build_arg_parser().parse_args()
    try:
        input_path = validate_input_path(args.input_path)
        output_dir = validate_output_dir(args.output_dir)
        output_mode = derive_output_mode(infer_input_type(input_path))

        request = DenoiseRequest(
            input_path=input_path,
            output_dir=output_dir,
            output_mode=output_mode,
            keep_intermediates=args.keep_intermediates,
            engine_name=args.engine,
        )
        result = run_pipeline(request, ffmpeg_wrapper=FFmpegWrapper(arnndn_mix=args.arnndn_mix))
        _log(f"[result] status={result.status} output={result.final_output_path}")
        return 0
    except Exception as exc:
        _log(f"[error] {exc}")
        return 1


run_dry_run_pipeline = run_pipeline


if __name__ == "__main__":
    raise SystemExit(main())
