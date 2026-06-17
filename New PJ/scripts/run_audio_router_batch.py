"""Batch validation harness for Audio Router inference."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_audio_router import DEFAULT_CONFIDENCE_THRESHOLD, run_audio_router


COMMON_MEDIA_SUFFIXES = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
}
BATCH_COLUMNS = [
    "input_path",
    "status",
    "predicted_label",
    "confidence",
    "confidence_threshold",
    "accepted",
    "route_target",
    "engine_target",
    "recommended_task",
    "decision_reason",
    "warnings",
    "prob_environment_only",
    "prob_music_with_vocals",
    "prob_speech_clean",
    "prob_speech_target_noise",
    "error",
]


def _safe_summary_name(path: Path, input_dir: Path) -> str:
    try:
        relative_path = Path(path).resolve().relative_to(Path(input_dir).resolve())
    except ValueError:
        relative_path = Path(path).name
    stem_with_parent = Path(relative_path).with_suffix("").as_posix()
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem_with_parent).strip("_") or "input"
    return f"{safe_stem}.router_summary.json"


def _scan_media_files(input_dir: Path, patterns: list[str] | None = None) -> list[Path]:
    root = Path(input_dir)
    if patterns:
        files: set[Path] = set()
        for pattern in patterns:
            files.update(path for path in root.rglob(pattern) if path.is_file())
        return sorted(files)
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in COMMON_MEDIA_SUFFIXES)


def run_audio_router_batch(
    *,
    checkpoint_path: Path,
    input_dir: Path,
    output_csv: Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    device: str = "cpu",
    patterns: list[str] | None = None,
) -> Path:
    """Run router inference over a directory of audio/video files."""
    input_root = Path(input_dir)
    if not input_root.exists():
        raise FileNotFoundError(f"Audio Router batch input_dir not found: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Audio Router batch input_dir is not a directory: {input_root}")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_dir = output_path.parent / "router_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    media_files = _scan_media_files(input_root, patterns)
    if not media_files:
        supported_suffixes = ", ".join(sorted(COMMON_MEDIA_SUFFIXES))
        raise ValueError(
            f"Audio Router batch found no supported media files in {input_root}. "
            f"Supported suffixes: {supported_suffixes}"
        )

    rows: list[dict[str, str]] = []
    for input_path in media_files:
        summary_path = summary_dir / _safe_summary_name(input_path, input_root)
        try:
            summary = run_audio_router(
                checkpoint_path=checkpoint_path,
                input_path=input_path,
                output_summary=summary_path,
                confidence_threshold=confidence_threshold,
                device=device,
            )
            probabilities = summary.get("probabilities", {})
            rows.append(
                {
                    "input_path": str(input_path),
                    "status": str(summary["status"]),
                    "predicted_label": str(summary["predicted_label"]),
                    "confidence": f"{float(summary['confidence']):.10f}",
                    "confidence_threshold": f"{float(summary['confidence_threshold']):.10f}",
                    "accepted": str(summary["accepted"]).lower(),
                    "route_target": str(summary["route_target"]),
                    "engine_target": str(summary["engine_target"]),
                    "recommended_task": str(summary["recommended_task"] or ""),
                    "decision_reason": str(summary["decision_reason"]),
                    "warnings": ";".join(str(warning) for warning in summary.get("warnings", [])),
                    "prob_environment_only": f"{float(probabilities.get('environment_only', 0.0)):.10f}",
                    "prob_music_with_vocals": f"{float(probabilities.get('music_with_vocals', 0.0)):.10f}",
                    "prob_speech_clean": f"{float(probabilities.get('speech_clean', 0.0)):.10f}",
                    "prob_speech_target_noise": f"{float(probabilities.get('speech_target_noise', 0.0)):.10f}",
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "input_path": str(input_path),
                    "status": "failed",
                    "predicted_label": "",
                    "confidence": "",
                    "confidence_threshold": f"{confidence_threshold:.10f}",
                    "accepted": "false",
                    "route_target": "",
                    "engine_target": "",
                    "recommended_task": "",
                    "decision_reason": "",
                    "warnings": "",
                    "prob_environment_only": "",
                    "prob_music_with_vocals": "",
                    "prob_speech_clean": "",
                    "prob_speech_target_noise": "",
                    "error": str(exc),
                }
            )

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=BATCH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path.resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Audio Router inference over a directory.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--glob", action="append", default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        output_path = run_audio_router_batch(
            checkpoint_path=args.checkpoint,
            input_dir=args.input_dir,
            output_csv=args.output_csv,
            confidence_threshold=args.confidence_threshold,
            device=args.device,
            patterns=args.glob,
        )
        print(f"Wrote Audio Router batch results: {output_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
