"""Smoke tests for the minimal unified audio task Gradio demo callback."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

from app.audio_task_demo import run_demo_task
from src.router.task_registry import CLEAN_VOICE, EXTRACT_VOCALS, REMOVE_VOCALS


def _write_summary(run_dir: Path, *, task: str, primary_output_path: Path | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "task",
        "engine",
        "input_path",
        "input_type",
        "status",
        "runtime_sec",
        "primary_output_path",
        "error",
    ]
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "mock-run",
                "task": task,
                "engine": "deepfilternet" if task == CLEAN_VOICE else "demucs",
                "input_path": "",
                "input_type": "audio",
                "status": "success",
                "runtime_sec": "1.250000",
                "primary_output_path": "" if primary_output_path is None else str(primary_output_path),
                "error": "",
            }
        )


def test_demo_clean_voice_passes_through_and_returns_primary_output(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    output_root = tmp_path / "demo-runs"
    primary_output = tmp_path / "speech.denoised.wav"
    input_path.write_bytes(b"audio")
    primary_output.write_bytes(b"restored")
    calls: list[dict[str, object]] = []

    def mock_run_audio_task(**kwargs: object) -> Path:
        calls.append(kwargs)
        run_dir = output_root / "clean_voice" / "mock-run"
        _write_summary(run_dir, task=CLEAN_VOICE, primary_output_path=primary_output)
        return run_dir

    with patch("app.audio_task_demo.run_audio_task", side_effect=mock_run_audio_task):
        markdown, table, downloadable = run_demo_task(input_path, CLEAN_VOICE, output_root=output_root)

    assert calls[0]["task"] == CLEAN_VOICE
    assert calls[0]["input_path"] == input_path
    assert calls[0]["output_root"] == output_root
    assert "success" in markdown
    assert ["task", CLEAN_VOICE] in table
    assert ["primary_output_path", str(primary_output)] in table
    assert downloadable == str(primary_output)


def test_demo_extract_vocals_passes_through(tmp_path: Path) -> None:
    input_path = tmp_path / "song.mp4"
    output_root = tmp_path / "demo-runs"
    input_path.write_bytes(b"video")
    calls: list[dict[str, object]] = []

    def mock_run_audio_task(**kwargs: object) -> Path:
        calls.append(kwargs)
        run_dir = output_root / "extract_vocals" / "mock-run"
        primary_output = run_dir / "vocals.wav"
        primary_output.parent.mkdir(parents=True, exist_ok=True)
        primary_output.write_bytes(b"vocals")
        _write_summary(run_dir, task=EXTRACT_VOCALS, primary_output_path=primary_output)
        return run_dir

    with patch("app.audio_task_demo.run_audio_task", side_effect=mock_run_audio_task):
        _, table, downloadable = run_demo_task(input_path, EXTRACT_VOCALS, output_root=output_root)

    assert calls[0]["task"] == EXTRACT_VOCALS
    assert ["task", EXTRACT_VOCALS] in table
    assert downloadable.endswith("vocals.wav")


def test_demo_remove_vocals_passes_through(tmp_path: Path) -> None:
    input_path = tmp_path / "song.mp4"
    output_root = tmp_path / "demo-runs"
    input_path.write_bytes(b"video")
    calls: list[dict[str, object]] = []

    def mock_run_audio_task(**kwargs: object) -> Path:
        calls.append(kwargs)
        run_dir = output_root / "remove_vocals" / "mock-run"
        primary_output = run_dir / "no_vocals.wav"
        primary_output.parent.mkdir(parents=True, exist_ok=True)
        primary_output.write_bytes(b"no-vocals")
        _write_summary(run_dir, task=REMOVE_VOCALS, primary_output_path=primary_output)
        return run_dir

    with patch("app.audio_task_demo.run_audio_task", side_effect=mock_run_audio_task):
        _, table, downloadable = run_demo_task(input_path, REMOVE_VOCALS, output_root=output_root)

    assert calls[0]["task"] == REMOVE_VOCALS
    assert ["task", REMOVE_VOCALS] in table
    assert downloadable.endswith("no_vocals.wav")


def test_demo_missing_file_input_returns_clear_error(tmp_path: Path) -> None:
    missing_input = tmp_path / "missing.wav"

    with patch("app.audio_task_demo.run_audio_task") as run_mock:
        markdown, table, downloadable = run_demo_task(missing_input, CLEAN_VOICE, output_root=tmp_path / "runs")

    run_mock.assert_not_called()
    assert "Input file not found" in markdown
    assert table == []
    assert downloadable is None


def test_demo_missing_uploaded_file_returns_clear_error() -> None:
    markdown, table, downloadable = run_demo_task(None, CLEAN_VOICE)

    assert "Select an audio or video file" in markdown
    assert table == []
    assert downloadable is None
