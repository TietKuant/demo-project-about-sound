"""Run restoration modes for one input and write comparison summaries."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.contracts import DenoiseRequest
from src.eval.metrics import load_mono_audio
from src.io.paths import derive_output_mode, infer_input_type, validate_input_path
from src.pipeline.run_pipeline import run_pipeline


DEFAULT_ENGINES = "noisy_input,deepfilternet"
FIELDNAMES = [
    "run_id",
    "input_path",
    "engine",
    "status",
    "runtime_sec",
    "audio_duration_sec",
    "rtf",
    "output_path",
    "error",
]


def _parse_engines(value: str) -> list[str]:
    engines = [engine.strip() for engine in value.split(",") if engine.strip()]
    if not engines:
        raise ValueError("At least one engine must be provided.")
    return engines


def _audio_duration_sec(input_path: Path) -> float | None:
    try:
        audio, sample_rate = load_mono_audio(input_path)
    except Exception:
        return None
    return len(audio) / sample_rate


def _engine_output_dir(run_dir: Path, engine: str) -> Path:
    engine_dir = (run_dir / engine).resolve()
    if not engine_dir.is_relative_to(run_dir):
        raise ValueError(f"Engine name must not escape the compare run directory: {engine}")
    return engine_dir


def _empty_summary_row(run_id: str, input_path: Path, engine: str, duration_sec: float | None) -> dict[str, str]:
    return {
        "run_id": run_id,
        "input_path": str(input_path),
        "engine": engine,
        "status": "failed",
        "runtime_sec": "",
        "audio_duration_sec": "" if duration_sec is None else f"{duration_sec:.6f}",
        "rtf": "",
        "output_path": "",
        "error": "",
    }


def compare_restoration_modes(*, input_path: Path, output_root: Path, engines: list[str]) -> Path:
    """Run comparison modes for one input and return the scoped run directory."""
    source = validate_input_path(input_path)
    output_mode = derive_output_mode(infer_input_type(source))
    run_id = f"{source.stem}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    duration_sec = _audio_duration_sec(source)

    rows: list[dict[str, str]] = []
    for engine in engines:
        engine_dir = _engine_output_dir(run_dir, engine)
        engine_dir.mkdir(parents=True, exist_ok=True)
        row = _empty_summary_row(run_id, source, engine, duration_sec)

        if engine == "noisy_input":
            output_path = engine_dir / f"{source.stem}.noisy_input{source.suffix}"
            shutil.copy2(source, output_path)
            row.update(
                {
                    "status": "success",
                    "runtime_sec": "0.000000",
                    "rtf": "0.000000" if duration_sec is not None else "",
                    "output_path": str(output_path),
                }
            )
            rows.append(row)
            continue

        started_at = time.perf_counter()
        try:
            result = run_pipeline(
                DenoiseRequest(
                    input_path=source,
                    output_dir=engine_dir,
                    output_mode=output_mode,
                    engine_name=engine,
                )
            )
            runtime_sec = time.perf_counter() - started_at
            if result.final_output_path is None:
                raise RuntimeError("Pipeline did not return a final output path.")
            row.update(
                {
                    "status": "success",
                    "runtime_sec": f"{runtime_sec:.6f}",
                    "rtf": "" if duration_sec is None else f"{runtime_sec / duration_sec:.6f}",
                    "output_path": str(result.final_output_path),
                }
            )
        except Exception as exc:
            runtime_sec = time.perf_counter() - started_at
            row["runtime_sec"] = f"{runtime_sec:.6f}"
            row["rtf"] = "" if duration_sec is None else f"{runtime_sec / duration_sec:.6f}"
            row["error"] = str(exc)
        rows.append(row)

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the comparison CLI parser."""
    parser = argparse.ArgumentParser(description="Compare restoration modes for one audio or video file.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-root", default=Path("outputs/compare-runs"), type=Path)
    parser.add_argument("--engines", default=DEFAULT_ENGINES)
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        run_dir = compare_restoration_modes(
            input_path=args.input_path,
            output_root=args.output_root,
            engines=_parse_engines(args.engines),
        )
        print(f"Wrote restoration comparison: {run_dir}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
