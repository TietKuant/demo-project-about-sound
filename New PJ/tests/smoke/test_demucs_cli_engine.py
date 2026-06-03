"""Smoke tests for the Demucs CLI engine adapter."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from src.engine.demucs_cli_engine import DemucsCliEngine


def _write_fake_outputs(command: list[str]) -> tuple[Path, Path]:
    output_dir = Path(command[command.index("-o") + 1])
    model = command[command.index("-n") + 1]
    input_path = Path(command[-1])
    track_dir = output_dir / model / input_path.stem
    track_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = track_dir / "vocals.wav"
    no_vocals_path = track_dir / "no_vocals.wav"
    vocals_path.write_bytes(b"vocals")
    no_vocals_path.write_bytes(b"no-vocals")
    return vocals_path, no_vocals_path


def test_demucs_cli_engine_returns_multi_output_processing_result(tmp_path: Path) -> None:
    input_path = tmp_path / "Song.stem.mp4"
    output_dir = tmp_path / "outputs"
    input_path.write_bytes(b"music")
    commands: list[list[str]] = []

    def mock_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert kwargs == {"text": True, "capture_output": True, "check": False}
        _write_fake_outputs(command)
        return subprocess.CompletedProcess(command, 0, "completed", "")

    engine = DemucsCliEngine(demucs_python=Path("/isolated/bin/python"))
    with patch("src.engine.demucs_cli_engine.subprocess.run", side_effect=mock_run):
        result = engine.separate(input_path, output_dir)

    command = commands[0]
    assert command[:3] == ["/isolated/bin/python", "-m", "demucs"]
    assert command[command.index("--two-stems") + 1] == "vocals"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("-j") + 1] == "1"
    assert command[command.index("-n") + 1] == "htdemucs"

    vocals = result.get_output("vocals")
    no_vocals = result.get_output("no_vocals")
    assert result.task_name == "extract_vocals"
    assert result.engine_name == "demucs"
    assert result.status == "success"
    assert result.runtime_sec is not None
    assert vocals is not None
    assert no_vocals is not None
    assert vocals.media_type == "audio"
    assert vocals.role == "primary"
    assert no_vocals.media_type == "audio"
    assert no_vocals.role == "secondary"
    assert result.primary_output_path == vocals.path
    assert vocals.path.exists()
    assert no_vocals.path.exists()


def test_demucs_cli_engine_fails_when_returncode_is_nonzero(tmp_path: Path) -> None:
    input_path = tmp_path / "Song.stem.mp4"
    output_dir = tmp_path / "outputs"
    input_path.write_bytes(b"music")
    engine = DemucsCliEngine()

    with patch(
        "src.engine.demucs_cli_engine.subprocess.run",
        return_value=subprocess.CompletedProcess([], 7, "", "demucs failed"),
    ):
        result = engine.separate(input_path, output_dir)

    assert result.status == "failed"
    assert result.outputs == []
    assert result.runtime_sec is not None
    assert "demucs failed" in result.error


def test_demucs_cli_engine_fails_when_outputs_are_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "Song.stem.mp4"
    output_dir = tmp_path / "outputs"
    input_path.write_bytes(b"music")
    engine = DemucsCliEngine()

    with patch(
        "src.engine.demucs_cli_engine.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, "", ""),
    ):
        result = engine.separate(input_path, output_dir)

    assert result.status == "failed"
    assert result.outputs == []
    assert result.runtime_sec is not None
    assert result.error == "missing expected outputs"
