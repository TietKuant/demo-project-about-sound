"""Run music/vocal separation for one local input using Demucs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.contracts import ProcessingResult
from src.engine.demucs_cli_engine import DemucsCliEngine


FIELDNAMES = [
    "run_id",
    "input_path",
    "task",
    "engine",
    "status",
    "runtime_sec",
    "vocals_path",
    "no_vocals_path",
    "error",
]


def _run_id_for_input(input_path: Path) -> str:
    path_hash = hashlib.sha1(str(input_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{input_path.stem}-{path_hash}"


def _summary_row(run_id: str, input_path: Path, result: ProcessingResult) -> dict[str, str]:
    vocals = result.get_output("vocals") if result.status == "success" else None
    no_vocals = result.get_output("no_vocals") if result.status == "success" else None
    return {
        "run_id": run_id,
        "input_path": str(input_path),
        "task": result.task_name,
        "engine": result.engine_name,
        "status": result.status,
        "runtime_sec": "" if result.runtime_sec is None else f"{result.runtime_sec:.6f}",
        "vocals_path": "" if vocals is None else str(vocals.path),
        "no_vocals_path": "" if no_vocals is None else str(no_vocals.path),
        "error": result.error,
    }


def run_music_separation(
    *,
    input_path: Path,
    output_root: Path,
    demucs_python: Path = Path(".venv-demucs/bin/python"),
    model: str = "htdemucs",
    device: str = "cpu",
    jobs: int = 1,
) -> Path:
    """Run Demucs separation for one input and write summary artifacts."""
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")

    run_id = _run_id_for_input(source)
    run_dir = Path(output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    engine = DemucsCliEngine(
        demucs_python=demucs_python,
        model=model,
        device=device,
        jobs=jobs,
    )
    result = engine.separate(source, run_dir)
    row = _summary_row(run_id, source, result)

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)
    (run_dir / "summary.json").write_text(json.dumps([row], indent=2), encoding="utf-8")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the music separation CLI parser."""
    parser = argparse.ArgumentParser(description="Run Demucs music/vocal separation for one input file.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("outputs/music-separation-runs"), type=Path)
    parser.add_argument("--demucs-python", default=Path(".venv-demucs/bin/python"), type=Path)
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", default=1, type=int)
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        run_dir = run_music_separation(
            input_path=args.input,
            output_root=args.output_root,
            demucs_python=args.demucs_python,
            model=args.model,
            device=args.device,
            jobs=args.jobs,
        )
        print(f"Wrote music separation run: {run_dir}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
