"""Run supported workflows over a Vietnamese real-sample manifest."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_music_separation import run_music_separation
from src.api.contracts import DenoiseRequest
from src.io.paths import derive_output_mode, infer_input_type
from src.pipeline.run_pipeline import run_pipeline
from src.router.task_registry import CLEAN_VOICE, EXTRACT_VOCALS, REMOVE_VOCALS, get_task_spec


REQUIRED_COLUMNS = {
    "sample_id",
    "category",
    "input_path",
    "input_type",
    "language",
    "expected_task",
    "has_clean_reference",
    "notes",
}
SUPPORTED_TASKS = {CLEAN_VOICE, EXTRACT_VOCALS, REMOVE_VOCALS}
FIELDNAMES = [
    "sample_id",
    "category",
    "input_path",
    "input_type",
    "language",
    "expected_task",
    "status",
    "output_root",
    "runtime_sec",
    "audio_duration_sec",
    "rtf",
    "primary_output_path",
    "error",
    "notes",
]


def _safe_component(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def _load_manifest(manifest_path: Path, limit: int | None) -> list[dict[str, str]]:
    with Path(manifest_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Vietnamese real sample manifest is missing required columns: {missing}")
        rows = list(reader)
    return rows if limit is None else rows[:limit]


def _resolve_input_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path)


def _audio_duration_sec(input_path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(input_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        duration_sec = float(result.stdout.strip())
    except ValueError:
        return None
    if duration_sec <= 0:
        return None
    return duration_sec


def _format_duration(duration_sec: float | None) -> str:
    return "" if duration_sec is None else f"{duration_sec:.6f}"


def _format_rtf(runtime_sec: float, duration_sec: float | None) -> str:
    if duration_sec is None or duration_sec <= 0:
        return ""
    return f"{runtime_sec / duration_sec:.6f}"


def _empty_row(manifest_row: dict[str, str], input_path: Path) -> dict[str, str]:
    return {
        "sample_id": manifest_row["sample_id"],
        "category": manifest_row["category"],
        "input_path": str(input_path),
        "input_type": manifest_row["input_type"],
        "language": manifest_row["language"],
        "expected_task": manifest_row["expected_task"],
        "status": "failed",
        "output_root": "",
        "runtime_sec": "",
        "audio_duration_sec": "",
        "rtf": "",
        "primary_output_path": "",
        "error": "",
        "notes": manifest_row["notes"],
    }


def _read_music_summary(run_dir: Path) -> dict[str, str]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return {}
    with summary_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        return {}
    return rows[0]


def _run_clean_voice(input_path: Path, output_dir: Path) -> Path | None:
    output_mode = derive_output_mode(infer_input_type(input_path))
    result = run_pipeline(
        DenoiseRequest(
            input_path=input_path,
            output_dir=output_dir,
            output_mode=output_mode,
            engine_name="deepfilternet",
        )
    )
    return result.final_output_path


def run_vietnamese_real_sample_suite(
    *,
    manifest_path: Path,
    output_root: Path = Path("outputs/vietnamese-real-runs"),
    limit: int | None = None,
    demucs_python: Path = Path(".venv-demucs/bin/python"),
    skip_missing: bool = False,
) -> Path:
    """Run supported workflows for rows in a Vietnamese real-sample manifest."""
    rows = _load_manifest(manifest_path, limit)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []

    for manifest_row in rows:
        input_path = _resolve_input_path(manifest_row["input_path"])
        expected_task = manifest_row["expected_task"]
        sample_output_root = root / _safe_component(manifest_row["sample_id"]) / _safe_component(expected_task)
        summary_row = _empty_row(manifest_row, input_path)

        if expected_task not in SUPPORTED_TASKS:
            summary_row["error"] = f"Unsupported expected_task: {expected_task}"
            summary_rows.append(summary_row)
            continue

        get_task_spec(expected_task)
        if not input_path.exists():
            summary_row["status"] = "skipped" if skip_missing else "failed"
            summary_row["error"] = f"Input file not found: {input_path}"
            summary_rows.append(summary_row)
            continue

        duration_sec = _audio_duration_sec(input_path)
        summary_row["audio_duration_sec"] = _format_duration(duration_sec)
        started_at = time.perf_counter()
        try:
            if expected_task == CLEAN_VOICE:
                sample_output_root.mkdir(parents=True, exist_ok=True)
                primary_output_path = _run_clean_voice(input_path, sample_output_root)
                runtime_sec = time.perf_counter() - started_at
                summary_row.update(
                    {
                        "status": "success" if primary_output_path is not None else "failed",
                        "output_root": str(sample_output_root),
                        "runtime_sec": f"{runtime_sec:.6f}",
                        "rtf": _format_rtf(runtime_sec, duration_sec),
                        "primary_output_path": "" if primary_output_path is None else str(primary_output_path),
                        "error": "" if primary_output_path is not None else "Pipeline did not return a final output path.",
                    }
                )
            else:
                task_run_dir = run_music_separation(
                    input_path=input_path,
                    output_root=sample_output_root,
                    demucs_python=demucs_python,
                    task_name=expected_task,
                )
                runtime_sec = time.perf_counter() - started_at
                music_summary = _read_music_summary(task_run_dir)
                summary_row.update(
                    {
                        "status": music_summary.get("status", "failed"),
                        "output_root": str(task_run_dir),
                        "runtime_sec": f"{runtime_sec:.6f}",
                        "rtf": _format_rtf(runtime_sec, duration_sec),
                        "primary_output_path": music_summary.get("primary_output_path", ""),
                        "error": music_summary.get("error", ""),
                    }
                )
        except Exception as exc:
            runtime_sec = time.perf_counter() - started_at
            summary_row.update(
                {
                    "status": "failed",
                    "output_root": str(sample_output_root),
                    "runtime_sec": f"{runtime_sec:.6f}",
                    "rtf": _format_rtf(runtime_sec, duration_sec),
                    "error": str(exc),
                }
            )
        summary_rows.append(summary_row)

    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)
    (root / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    return root


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the Vietnamese real-sample suite CLI parser."""
    parser = argparse.ArgumentParser(description="Run Vietnamese real-sample workflow checks.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("outputs/vietnamese-real-runs"), type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--demucs-python", default=Path(".venv-demucs/bin/python"), type=Path)
    parser.add_argument("--skip-missing", action="store_true")
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        output_root = run_vietnamese_real_sample_suite(
            manifest_path=args.manifest,
            output_root=args.output_root,
            limit=args.limit,
            demucs_python=args.demucs_python,
            skip_missing=args.skip_missing,
        )
        print(f"Wrote Vietnamese real sample suite summaries: {output_root}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
