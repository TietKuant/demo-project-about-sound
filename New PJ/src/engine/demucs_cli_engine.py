"""Demucs CLI adapter for music/vocal separation."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from src.api.contracts import OutputArtifact, ProcessingResult


@dataclass(slots=True)
class DemucsCliEngine:
    """Run Demucs from an isolated Python environment and return output artifacts."""

    demucs_python: Path = Path(".venv-demucs/bin/python")
    model: str = "htdemucs"
    device: str = "cpu"
    jobs: int = 1

    @property
    def name(self) -> str:
        return "demucs"

    def separate(self, input_path: Path, output_dir: Path) -> ProcessingResult:
        """Separate vocals from one input file using the Demucs CLI."""
        source = Path(input_path).resolve()
        run_output_dir = Path(output_dir).resolve()
        run_output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            str(self.demucs_python),
            "-m",
            "demucs",
            "--two-stems",
            "vocals",
            "--device",
            self.device,
            "-j",
            str(self.jobs),
            "-n",
            self.model,
            "-o",
            str(run_output_dir),
            str(source),
        ]

        started_at = time.perf_counter()
        try:
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            runtime_sec = time.perf_counter() - started_at
        except OSError as exc:
            runtime_sec = time.perf_counter() - started_at
            return self._failed_result(runtime_sec, str(exc))

        expected_dir = run_output_dir / self.model / source.stem
        vocals_path = _find_output(run_output_dir, expected_dir / "vocals.wav", "vocals.wav")
        no_vocals_path = _find_output(run_output_dir, expected_dir / "no_vocals.wav", "no_vocals.wav")
        outputs_exist = vocals_path is not None and no_vocals_path is not None

        if result.returncode != 0 or not outputs_exist:
            return self._failed_result(runtime_sec, _error_message(result, outputs_exist))

        return ProcessingResult(
            task_name="extract_vocals",
            engine_name=self.name,
            status="success",
            runtime_sec=runtime_sec,
            outputs=[
                OutputArtifact(label="vocals", path=vocals_path, media_type="audio", role="primary"),
                OutputArtifact(label="no_vocals", path=no_vocals_path, media_type="audio", role="secondary"),
            ],
        )

    def _failed_result(self, runtime_sec: float, error: str) -> ProcessingResult:
        return ProcessingResult(
            task_name="extract_vocals",
            engine_name=self.name,
            status="failed",
            runtime_sec=runtime_sec,
            outputs=[],
            error=error,
        )


def _find_output(output_dir: Path, expected_path: Path, filename: str) -> Path | None:
    if expected_path.exists():
        return expected_path.resolve()
    matches = sorted(output_dir.rglob(filename), key=lambda path: path.as_posix())
    return matches[0].resolve() if matches else None


def _error_message(result: subprocess.CompletedProcess[str], outputs_exist: bool) -> str:
    if result.returncode == 0 and not outputs_exist:
        return "missing expected outputs"
    if result.stderr and result.stderr.strip():
        return result.stderr.strip()[-2000:]
    if result.stdout and result.stdout.strip():
        return result.stdout.strip()[-2000:]
    return f"demucs exited with returncode {result.returncode}"
