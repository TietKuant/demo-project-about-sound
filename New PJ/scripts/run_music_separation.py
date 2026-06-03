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

from src.api.contracts import OutputArtifact, ProcessingResult
from src.engine.demucs_cli_engine import DemucsCliEngine
from src.router.task_registry import EXTRACT_VOCALS, REMOVE_VOCALS, TaskSpec, get_task_spec


FIELDNAMES = [
    "run_id",
    "input_path",
    "task",
    "engine",
    "status",
    "runtime_sec",
    "primary_output_label",
    "primary_output_path",
    "vocals_path",
    "no_vocals_path",
    "error",
]


def _run_id_for_input(input_path: Path) -> str:
    path_hash = hashlib.sha1(str(input_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{input_path.stem}-{path_hash}"


def _get_music_task_spec(task_name: str) -> TaskSpec:
    task_spec = get_task_spec(task_name)
    if task_spec.name not in {EXTRACT_VOCALS, REMOVE_VOCALS}:
        raise ValueError(f"Unsupported music separation task: {task_name}")
    return task_spec


def _adapt_result_for_task(result: ProcessingResult, task_spec: TaskSpec) -> ProcessingResult:
    if result.status != "success":
        return ProcessingResult(
            task_name=task_spec.name,
            engine_name=result.engine_name,
            status="failed",
            runtime_sec=result.runtime_sec,
            outputs=[],
            error=result.error,
        )

    outputs: list[OutputArtifact] = []
    for index, label in enumerate(task_spec.output_labels):
        artifact = result.get_output(label)
        if artifact is None:
            return ProcessingResult(
                task_name=task_spec.name,
                engine_name=result.engine_name,
                status="failed",
                runtime_sec=result.runtime_sec,
                outputs=[],
                error=f"Demucs result missing expected output: {label}",
            )
        outputs.append(
            OutputArtifact(
                label=artifact.label,
                path=artifact.path,
                media_type=artifact.media_type,
                role="primary" if index == 0 else "secondary",
            )
        )

    return ProcessingResult(
        task_name=task_spec.name,
        engine_name=result.engine_name,
        status="success",
        runtime_sec=result.runtime_sec,
        outputs=outputs,
    )


def _summary_row(run_id: str, input_path: Path, result: ProcessingResult) -> dict[str, str]:
    vocals = result.get_output("vocals") if result.status == "success" else None
    no_vocals = result.get_output("no_vocals") if result.status == "success" else None
    primary = result.outputs[0] if result.status == "success" and result.outputs else None
    return {
        "run_id": run_id,
        "input_path": str(input_path),
        "task": result.task_name,
        "engine": result.engine_name,
        "status": result.status,
        "runtime_sec": "" if result.runtime_sec is None else f"{result.runtime_sec:.6f}",
        "primary_output_label": "" if primary is None else primary.label,
        "primary_output_path": "" if primary is None else str(primary.path),
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
    task_name: str = EXTRACT_VOCALS,
) -> Path:
    """Run Demucs separation for one input and write summary artifacts."""
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")
    task_spec = _get_music_task_spec(task_name)

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
    task_result = _adapt_result_for_task(result, task_spec)
    row = _summary_row(run_id, source, task_result)

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
    parser.add_argument("--task", choices=(EXTRACT_VOCALS, REMOVE_VOCALS), default=EXTRACT_VOCALS)
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
            task_name=args.task,
        )
        print(f"Wrote music separation run: {run_dir}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
