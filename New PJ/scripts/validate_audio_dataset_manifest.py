"""Validate audio dataset manifest paths, labels, and media metadata."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


AUDIT_COLUMNS = [
    "row_index",
    "path",
    "resolved_path",
    "label",
    "exists",
    "readable",
    "duration_sec",
    "sample_rate",
    "channels",
    "status",
    "error",
]


def _load_manifest(manifest_path: Path, path_column: str) -> tuple[list[dict[str, str]], set[str]]:
    with Path(manifest_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        if path_column not in fieldnames:
            raise ValueError(f"Manifest is missing required path column: {path_column}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows, fieldnames


def _resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(manifest_path).resolve().parent / path).resolve()


def _probe_audio_metadata(path: Path) -> dict[str, str]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "format=duration:stream=sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip() or "ffprobe failed"
        raise RuntimeError(details)

    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    first_stream = streams[0] if streams else {}
    return {
        "duration_sec": str((payload.get("format") or {}).get("duration", "")),
        "sample_rate": str(first_stream.get("sample_rate", "")),
        "channels": str(first_stream.get("channels", "")),
    }


def _validate_duration(value: str, min_duration_sec: float, max_duration_sec: float) -> str | None:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return "duration could not be probed"
    if duration < min_duration_sec:
        return f"duration below minimum {min_duration_sec:g}s"
    if duration > max_duration_sec:
        return f"duration above maximum {max_duration_sec:g}s"
    return None


def _audit_row(
    *,
    row_index: int,
    row: dict[str, str],
    manifest_path: Path,
    path_column: str,
    label_column: str | None,
    allowed_labels: set[str] | None,
    seen_paths: set[str],
    min_duration_sec: float,
    max_duration_sec: float,
) -> dict[str, str]:
    path_value = row.get(path_column, "")
    label = row.get(label_column, "") if label_column else ""
    resolved_path = _resolve_path(path_value, manifest_path) if path_value else Path("")
    resolved_key = str(resolved_path)
    errors: list[str] = []
    metadata = {"duration_sec": "", "sample_rate": "", "channels": ""}
    exists = bool(path_value) and resolved_path.exists()
    readable = False

    if not path_value:
        errors.append("empty path")
    if resolved_key in seen_paths:
        errors.append("duplicate path")
    seen_paths.add(resolved_key)
    if label_column and allowed_labels is not None and label not in allowed_labels:
        errors.append(f"label not allowed: {label}")
    if not exists:
        errors.append("file does not exist")
    else:
        try:
            metadata = _probe_audio_metadata(resolved_path)
            readable = True
            duration_error = _validate_duration(metadata["duration_sec"], min_duration_sec, max_duration_sec)
            if duration_error:
                errors.append(duration_error)
            if not metadata["sample_rate"]:
                errors.append("sample_rate missing")
            if not metadata["channels"]:
                errors.append("channels missing")
        except Exception as exc:
            errors.append(f"metadata probe failed: {exc}")

    status = "valid" if not errors else "invalid"
    return {
        "row_index": str(row_index),
        "path": path_value,
        "resolved_path": str(resolved_path),
        "label": label,
        "exists": str(exists).lower(),
        "readable": str(readable).lower(),
        "duration_sec": metadata["duration_sec"],
        "sample_rate": metadata["sample_rate"],
        "channels": metadata["channels"],
        "status": status,
        "error": "; ".join(errors),
    }


def validate_audio_dataset_manifest(
    *,
    manifest_path: Path,
    path_column: str,
    output_root: Path,
    label_column: str | None = None,
    allowed_labels: list[str] | None = None,
    min_duration_sec: float = 0.25,
    max_duration_sec: float = 60.0,
) -> tuple[Path, Path]:
    """Validate one audio manifest and write row-level audit artifacts."""
    rows, fieldnames = _load_manifest(manifest_path, path_column)
    allowed_label_set = set(allowed_labels) if allowed_labels is not None else None
    if label_column and label_column not in fieldnames:
        raise ValueError(f"Manifest is missing label column: {label_column}")

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    seen_paths: set[str] = set()
    audit_rows = [
        _audit_row(
            row_index=index,
            row=row,
            manifest_path=manifest_path,
            path_column=path_column,
            label_column=label_column,
            allowed_labels=allowed_label_set,
            seen_paths=seen_paths,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
        )
        for index, row in enumerate(rows, start=1)
    ]

    audit_path = output / "manifest_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(audit_rows)

    duplicate_path_count = sum(1 for row in audit_rows if "duplicate path" in row["error"])
    counts_by_label = Counter(row["label"] for row in audit_rows if row["label"])
    counts_by_status = Counter(row["status"] for row in audit_rows)
    summary = {
        "total_rows": len(audit_rows),
        "valid_rows": counts_by_status.get("valid", 0),
        "invalid_rows": counts_by_status.get("invalid", 0),
        "duplicate_path_count": duplicate_path_count,
        "counts_by_label": dict(sorted(counts_by_label.items())),
        "counts_by_status": dict(sorted(counts_by_status.items())),
    }
    summary_path = output / "manifest_audit_summary.json"
    with summary_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
        json_file.write("\n")

    return audit_path.resolve(), summary_path.resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the dataset manifest validator parser."""
    parser = argparse.ArgumentParser(description="Validate an audio dataset manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--path-column", required=True)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--allowed-labels", nargs="+", default=None)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--min-duration-sec", type=float, default=0.25)
    parser.add_argument("--max-duration-sec", type=float, default=60.0)
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        audit_path, summary_path = validate_audio_dataset_manifest(
            manifest_path=args.manifest,
            path_column=args.path_column,
            label_column=args.label_column,
            allowed_labels=args.allowed_labels,
            output_root=args.output_root,
            min_duration_sec=args.min_duration_sec,
            max_duration_sec=args.max_duration_sec,
        )
        print(f"Wrote manifest audit CSV: {audit_path}")
        print(f"Wrote manifest audit summary: {summary_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
