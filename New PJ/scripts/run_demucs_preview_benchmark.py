"""Run a bounded Demucs benchmark over a MUSDB18 preview manifest."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
import uuid
from pathlib import Path


REQUIRED_COLUMNS = {"track_id", "split", "input_path", "dataset", "source"}
FIELDNAMES = [
    "track_id",
    "split",
    "dataset",
    "source",
    "input_path",
    "status",
    "runtime_sec",
    "vocals_path",
    "no_vocals_path",
    "returncode",
    "error",
]


def _load_manifest(manifest_path: Path, limit: int | None) -> list[dict[str, str]]:
    with Path(manifest_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"MUSDB preview manifest is missing required columns: {missing}")
        rows = list(reader)
    return rows if limit is None else rows[:limit]


def _safe_component(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def _find_output(run_output_dir: Path, expected_path: Path, filename: str) -> Path | None:
    if expected_path.exists():
        return expected_path.resolve()
    matches = sorted(run_output_dir.rglob(filename), key=lambda path: path.as_posix())
    return matches[0].resolve() if matches else None


def _error_message(result: subprocess.CompletedProcess[str], outputs_exist: bool) -> str:
    if result.stderr and result.stderr.strip():
        return result.stderr.strip()[-2000:]
    if result.stdout and result.stdout.strip():
        return result.stdout.strip()[-2000:]
    if result.returncode == 0 and not outputs_exist:
        return "missing expected outputs"
    return f"demucs exited with returncode {result.returncode}"


def run_demucs_preview_benchmark(
    *,
    manifest_path: Path,
    output_root: Path,
    demucs_python: Path = Path(".venv-demucs/bin/python"),
    limit: int | None = None,
    device: str = "cpu",
    jobs: int = 1,
    model: str = "htdemucs",
) -> Path:
    """Run Demucs for each preview manifest row and write CSV and JSON summaries."""
    rows = _load_manifest(manifest_path, limit)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    benchmark_run_dir = root / "runs" / uuid.uuid4().hex[:8]
    summary_rows: list[dict[str, str]] = []

    for row in rows:
        track_id = row["track_id"]
        input_path = Path(row["input_path"]).expanduser().resolve()
        run_output_dir = benchmark_run_dir / _safe_component(track_id)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(demucs_python),
            "-m",
            "demucs",
            "--two-stems",
            "vocals",
            "--device",
            device,
            "-j",
            str(jobs),
            "-n",
            model,
            "-o",
            str(run_output_dir),
            str(input_path),
        ]

        started_at = time.perf_counter()
        try:
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            runtime_sec = time.perf_counter() - started_at
            expected_dir = run_output_dir / model / f"{track_id}.stem"
            vocals_path = _find_output(run_output_dir, expected_dir / "vocals.wav", "vocals.wav")
            no_vocals_path = _find_output(run_output_dir, expected_dir / "no_vocals.wav", "no_vocals.wav")
            outputs_exist = vocals_path is not None and no_vocals_path is not None
            success = result.returncode == 0 and outputs_exist
            error = "" if success else _error_message(result, outputs_exist)
            returncode = str(result.returncode)
        except OSError as exc:
            runtime_sec = time.perf_counter() - started_at
            vocals_path = None
            no_vocals_path = None
            success = False
            error = str(exc)
            returncode = ""

        summary_rows.append(
            {
                "track_id": track_id,
                "split": row["split"],
                "dataset": row["dataset"],
                "source": row["source"],
                "input_path": str(input_path),
                "status": "success" if success else "failed",
                "runtime_sec": f"{runtime_sec:.6f}",
                "vocals_path": "" if vocals_path is None else str(vocals_path),
                "no_vocals_path": "" if no_vocals_path is None else str(no_vocals_path),
                "returncode": returncode,
                "error": error,
            }
        )

    csv_path = root / "summary.csv"
    json_path = root / "summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)
    json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    return root


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the Demucs preview benchmark CLI parser."""
    parser = argparse.ArgumentParser(description="Run Demucs over a MUSDB18 preview manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("outputs/benchmarks/demucs-preview"), type=Path)
    parser.add_argument("--demucs-python", default=Path(".venv-demucs/bin/python"), type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--model", default="htdemucs")
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    output_root = run_demucs_preview_benchmark(
        manifest_path=args.manifest,
        output_root=args.output_root,
        demucs_python=args.demucs_python,
        limit=args.limit,
        device=args.device,
        jobs=args.jobs,
        model=args.model,
    )
    print(f"Wrote Demucs preview benchmark summaries: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
