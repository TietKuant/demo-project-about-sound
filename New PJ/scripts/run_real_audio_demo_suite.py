"""Run real-audio demo cases through existing processing tasks and router evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_audio_task import run_audio_task
from src.router.task_registry import (
    CLEAN_VOICE,
    EXTRACT_VOCALS,
    REMOVE_VOCALS,
    TARGET_NOISE_SUPPRESSION,
)


MANIFEST_COLUMNS = ["input_path", "case_id", "expected_task", "description", "notes"]
REPORT_COLUMNS = [
    "case_id",
    "input_path",
    "expected_task",
    "description",
    "notes",
    "input_duration_sec",
    "input_sample_rate_hz",
    "input_channels",
    "processing_time_sec",
    "human_rating",
    "human_notes",
    "status",
    "selected_task",
    "router_label",
    "router_confidence",
    "router_accepted",
    "route_target",
    "engine_target",
    "output_files",
    "error",
]
MANUAL_TASKS = {
    CLEAN_VOICE,
    EXTRACT_VOCALS,
    REMOVE_VOCALS,
    TARGET_NOISE_SUPPRESSION,
}
ALLOWED_EXPECTED_TASKS = MANUAL_TASKS | {"auto"}
MEDIA_SUFFIXES = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
}
ROUTER_CHECKPOINT_ENV = "AUDIO_ROUTER_CHECKPOINT"


def _safe_case_id(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("_") or "case"


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Real-audio demo manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Real-audio demo manifest is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Real-audio demo manifest is empty.")

    seen_case_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        case_id = row.get("case_id", "").strip()
        expected_task = row.get("expected_task", "").strip()
        if not case_id:
            raise ValueError(f"Real-audio demo manifest row {index} has an empty case_id.")
        if case_id in seen_case_ids:
            raise ValueError(f"Real-audio demo manifest has duplicate case_id: {case_id}")
        seen_case_ids.add(case_id)
        if expected_task not in ALLOWED_EXPECTED_TASKS:
            allowed = ", ".join(sorted(ALLOWED_EXPECTED_TASKS))
            raise ValueError(f"Unsupported expected_task '{expected_task}' on row {index}. Allowed: {allowed}")
    return rows


def _resolve_input_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (Path(manifest_path).resolve().parent / path).resolve()


def _input_metadata(input_path: Path) -> dict[str, str]:
    """Return best-effort input media metadata without failing the case."""
    metadata = {
        "input_duration_sec": "",
        "input_sample_rate_hz": "",
        "input_channels": "",
        "metadata_warning": "",
    }
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type,sample_rate,channels",
                "-of",
                "json",
                str(input_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            payload = json.loads(result.stdout)
            duration = payload.get("format", {}).get("duration", "")
            audio_stream = next(
                (
                    stream
                    for stream in payload.get("streams", [])
                    if stream.get("codec_type") == "audio"
                ),
                {},
            )
            metadata["input_duration_sec"] = "" if duration in (None, "") else f"{float(duration):.6f}"
            metadata["input_sample_rate_hz"] = str(audio_stream.get("sample_rate", "") or "")
            metadata["input_channels"] = str(audio_stream.get("channels", "") or "")
            if any(metadata[field] for field in ("input_duration_sec", "input_sample_rate_hz", "input_channels")):
                return metadata
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError, OSError):
        pass

    if input_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(input_path), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                frames = wav_file.getnframes()
                metadata["input_sample_rate_hz"] = str(sample_rate)
                metadata["input_channels"] = str(wav_file.getnchannels())
                metadata["input_duration_sec"] = f"{frames / sample_rate:.6f}" if sample_rate > 0 else ""
                return metadata
        except (OSError, wave.Error):
            pass

    metadata["metadata_warning"] = "Input metadata unavailable."
    return metadata


def _read_task_summary(run_dir: Path) -> dict[str, str]:
    summary_path = Path(run_dir) / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Task summary not found: {summary_path}")
    with summary_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"Task summary is empty: {summary_path}")
    return rows[0]


def _output_files(run_dir: Path, primary_output_path: str = "") -> list[str]:
    files: set[Path] = set()
    primary = Path(primary_output_path) if primary_output_path else None
    if primary is not None and primary.exists():
        files.add(primary.resolve())
    root = Path(run_dir)
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES:
                files.add(path.resolve())
    return [str(path) for path in sorted(files)]


def _router_checkpoint(explicit_checkpoint: Path | None) -> Path | None:
    if explicit_checkpoint is not None:
        return Path(explicit_checkpoint).expanduser()
    value = os.getenv(ROUTER_CHECKPOINT_ENV, "").strip()
    return Path(value).expanduser() if value else None


def _run_router(
    *,
    checkpoint_path: Path,
    input_path: Path,
    output_summary: Path,
    device: str,
) -> dict[str, Any]:
    from scripts.run_audio_router import run_audio_router

    return run_audio_router(
        checkpoint_path=checkpoint_path,
        input_path=input_path,
        output_summary=output_summary,
        device=device,
    )


def _base_case_summary(row: dict[str, str], input_path: Path) -> dict[str, Any]:
    return {
        "case_id": row["case_id"].strip(),
        "input_path": str(input_path),
        "expected_task": row["expected_task"].strip(),
        "description": row.get("description", ""),
        "notes": row.get("notes", ""),
        "input_duration_sec": "",
        "input_sample_rate_hz": "",
        "input_channels": "",
        "processing_time_sec": "",
        "human_rating": row.get("human_rating", ""),
        "human_notes": row.get("human_notes", ""),
        "metadata_warning": "",
        "status": "failed",
        "selected_task": "",
        "router_label": "",
        "router_confidence": "",
        "router_accepted": "",
        "route_target": "",
        "engine_target": "",
        "output_files": [],
        "error": "",
    }


def _apply_router_fields(summary: dict[str, Any], router_result: dict[str, Any]) -> None:
    summary["router_label"] = str(router_result.get("predicted_label", ""))
    confidence = router_result.get("confidence")
    summary["router_confidence"] = "" if confidence in (None, "") else float(confidence)
    accepted = router_result.get("accepted")
    summary["router_accepted"] = "" if accepted is None else bool(accepted)
    summary["route_target"] = str(router_result.get("route_target", ""))
    summary["engine_target"] = str(router_result.get("engine_target", ""))


def _run_selected_task(
    *,
    summary: dict[str, Any],
    task: str,
    input_path: Path,
    case_dir: Path,
    demucs_python: Path,
    clean_voice_engine: str,
    model: str,
    device: str,
    jobs: int,
    target_noise_checkpoint: Path | None,
) -> None:
    run_dir = run_audio_task(
        task=task,
        input_path=input_path,
        output_root=case_dir,
        demucs_python=demucs_python,
        engine=clean_voice_engine,
        model=model,
        device=device,
        jobs=jobs,
        target_noise_checkpoint=target_noise_checkpoint,
    )
    task_summary = _read_task_summary(run_dir)
    summary["selected_task"] = task
    summary["status"] = task_summary.get("status", "failed")
    summary["error"] = task_summary.get("error", "")
    summary["output_files"] = _output_files(run_dir, task_summary.get("primary_output_path", ""))


def _run_auto_case(
    *,
    summary: dict[str, Any],
    input_path: Path,
    case_dir: Path,
    router_checkpoint: Path | None,
    demucs_python: Path,
    clean_voice_engine: str,
    model: str,
    device: str,
    jobs: int,
    target_noise_checkpoint: Path | None,
) -> None:
    if router_checkpoint is None:
        summary["status"] = "abstained"
        summary["route_target"] = "manual_required"
        summary["engine_target"] = "none"
        summary["router_accepted"] = False
        summary["error"] = f"Auto router unavailable: provide a checkpoint or set {ROUTER_CHECKPOINT_ENV}."
        return
    if not router_checkpoint.exists():
        summary["status"] = "abstained"
        summary["route_target"] = "manual_required"
        summary["engine_target"] = "none"
        summary["router_accepted"] = False
        summary["error"] = f"Auto router checkpoint not found: {router_checkpoint}"
        return

    router_result = _run_router(
        checkpoint_path=router_checkpoint,
        input_path=input_path,
        output_summary=case_dir / "router_summary.json",
        device=device,
    )
    _apply_router_fields(summary, router_result)
    route_target = summary["route_target"]
    recommended_task = router_result.get("recommended_task")
    if router_result.get("accepted") is not True or route_target in {"manual_required", "abstain"}:
        summary["status"] = "abstained"
        summary["error"] = str(router_result.get("decision_reason", "manual_required"))
        return
    if route_target == TARGET_NOISE_SUPPRESSION or recommended_task == TARGET_NOISE_SUPPRESSION:
        summary["status"] = "abstained"
        summary["selected_task"] = ""
        summary["route_target"] = "manual_required"
        summary["engine_target"] = "none"
        summary["error"] = "Experimental auto target_noise_suppression is disabled; select the task manually."
        return
    if route_target in {"no_process", "out_of_scope"}:
        summary["status"] = route_target
        summary["selected_task"] = route_target
        return

    selected_task = str(recommended_task or route_target)
    if selected_task not in MANUAL_TASKS:
        summary["status"] = "abstained"
        summary["route_target"] = "manual_required"
        summary["engine_target"] = "none"
        summary["error"] = f"Router did not provide a supported safe task: {selected_task}"
        return
    _run_selected_task(
        summary=summary,
        task=selected_task,
        input_path=input_path,
        case_dir=case_dir,
        demucs_python=demucs_python,
        clean_voice_engine=clean_voice_engine,
        model=model,
        device=device,
        jobs=jobs,
        target_noise_checkpoint=target_noise_checkpoint,
    )


def _write_case_summary(case_dir: Path, summary: dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _csv_row(summary: dict[str, Any]) -> dict[str, str]:
    return {
        field: (
            ";".join(str(path) for path in summary["output_files"])
            if field == "output_files"
            else str(summary.get(field, "")).lower()
            if field == "router_accepted" and summary.get(field, "") != ""
            else str(summary.get(field, ""))
        )
        for field in REPORT_COLUMNS
    }


def _write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    status_counts = Counter(str(summary["status"]) for summary in summaries)
    task_counts = Counter(str(summary["selected_task"] or summary["expected_task"]) for summary in summaries)
    abstained_auto = [
        summary
        for summary in summaries
        if summary["expected_task"] == "auto" and summary["status"] == "abstained"
    ]
    lines = [
        "# Real Audio Demo Suite Report",
        "",
        f"- Total cases: {len(summaries)}",
        f"- Success cases: {status_counts.get('success', 0)}",
        f"- Failed cases: {status_counts.get('failed', 0)}",
        f"- Abstained auto cases: {len(abstained_auto)}",
        "",
        "## Status Counts",
        *[f"- {status}: {count}" for status, count in sorted(status_counts.items())],
        "",
        "## Task Counts",
        *[f"- {task}: {count}" for task, count in sorted(task_counts.items())],
        "",
        "## Cases",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"### {summary['case_id']}",
                f"- Expected task: `{summary['expected_task']}`",
                f"- Selected task: `{summary['selected_task'] or 'none'}`",
                f"- Status: `{summary['status']}`",
                f"- Processing time: `{summary['processing_time_sec'] or 'n/a'}` seconds",
                (
                    f"- Input metadata: duration=`{summary['input_duration_sec'] or 'n/a'}`, "
                    f"sample_rate=`{summary['input_sample_rate_hz'] or 'n/a'}`, "
                    f"channels=`{summary['input_channels'] or 'n/a'}`"
                ),
                f"- Human rating: `{summary['human_rating'] or 'n/a'}`",
                f"- Human notes: {summary['human_notes'] or 'n/a'}",
                f"- Router route: `{summary['route_target'] or 'n/a'}`",
                "- Output files:",
            ]
        )
        if summary["output_files"]:
            lines.extend(f"  - `{output}`" for output in summary["output_files"])
        else:
            lines.append("  - none")
        if summary["error"]:
            lines.append(f"- Error: `{summary['error']}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_real_audio_demo_suite(
    *,
    manifest_path: Path,
    output_root: Path = Path("outputs/real-audio-demo-suite"),
    router_checkpoint: Path | None = None,
    demucs_python: Path = Path(".venv-demucs/bin/python"),
    clean_voice_engine: str = "deepfilternet",
    model: str = "htdemucs",
    device: str = "cpu",
    jobs: int = 1,
    target_noise_checkpoint: Path | None = None,
) -> Path:
    """Run all real-audio demo manifest cases and write CSV/Markdown reports."""
    manifest = Path(manifest_path)
    rows = _read_manifest(manifest)
    output_dir = Path(output_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_router_checkpoint = _router_checkpoint(router_checkpoint)
    summaries: list[dict[str, Any]] = []

    for row in rows:
        input_path = _resolve_input_path(row["input_path"].strip(), manifest)
        case_dir = output_dir / _safe_case_id(row["case_id"].strip())
        summary = _base_case_summary(row, input_path)
        started_at = time.perf_counter()
        try:
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            summary.update(_input_metadata(input_path))
            expected_task = row["expected_task"].strip()
            if expected_task == "auto":
                _run_auto_case(
                    summary=summary,
                    input_path=input_path,
                    case_dir=case_dir,
                    router_checkpoint=resolved_router_checkpoint,
                    demucs_python=demucs_python,
                    clean_voice_engine=clean_voice_engine,
                    model=model,
                    device=device,
                    jobs=jobs,
                    target_noise_checkpoint=target_noise_checkpoint,
                )
            else:
                _run_selected_task(
                    summary=summary,
                    task=expected_task,
                    input_path=input_path,
                    case_dir=case_dir,
                    demucs_python=demucs_python,
                    clean_voice_engine=clean_voice_engine,
                    model=model,
                    device=device,
                    jobs=jobs,
                    target_noise_checkpoint=target_noise_checkpoint,
                )
        except Exception as exc:
            summary["status"] = "failed"
            summary["error"] = str(exc)
        summary["processing_time_sec"] = f"{time.perf_counter() - started_at:.6f}"
        _write_case_summary(case_dir, summary)
        summaries.append(summary)

    report_csv = output_dir / "report.csv"
    with report_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(_csv_row(summary) for summary in summaries)
    _write_markdown(output_dir / "report.md", summaries)
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-audio demo cases through existing processing tasks.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("outputs/real-audio-demo-suite"), type=Path)
    parser.add_argument("--router-checkpoint", default=None, type=Path)
    parser.add_argument("--demucs-python", default=Path(".venv-demucs/bin/python"), type=Path)
    parser.add_argument("--clean-voice-engine", default="deepfilternet")
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", default=1, type=int)
    parser.add_argument("--target-noise-checkpoint", default=None, type=Path)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        output_dir = run_real_audio_demo_suite(
            manifest_path=args.manifest,
            output_root=args.output_root,
            router_checkpoint=args.router_checkpoint,
            demucs_python=args.demucs_python,
            clean_voice_engine=args.clean_voice_engine,
            model=args.model,
            device=args.device,
            jobs=args.jobs,
            target_noise_checkpoint=args.target_noise_checkpoint,
        )
        print(f"Wrote real-audio demo suite: {output_dir}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
