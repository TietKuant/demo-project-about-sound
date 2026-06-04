"""Smoke tests for the unified audio task runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

from scripts.run_audio_task import run_audio_task
from src.api.contracts import DenoiseResult
from src.router.task_registry import CLEAN_VOICE, EXTRACT_VOCALS, REMOVE_VOCALS


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_music_summary(run_dir: Path, task_name: str, primary_output_path: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "child-run",
                "input_path": "",
                "task": task_name,
                "engine": "demucs",
                "status": "success",
                "runtime_sec": "1.000000",
                "primary_output_label": primary_output_path.stem,
                "primary_output_path": str(primary_output_path),
                "vocals_path": str(run_dir / "vocals.wav"),
                "no_vocals_path": str(run_dir / "no_vocals.wav"),
                "error": "",
            }
        )


def test_clean_voice_calls_run_pipeline_and_writes_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    output_root = tmp_path / "outputs"
    input_path.write_bytes(b"audio")
    calls: list[object] = []

    def mock_run_pipeline(request: object) -> DenoiseResult:
        calls.append(request)
        output_path = Path(getattr(request, "output_dir")) / "speech.denoised.wav"
        return DenoiseResult(
            status="completed_real",
            final_output_path=output_path,
            intermediate_audio_path=None,
            engine_name="deepfilternet",
        )

    with patch("scripts.run_audio_task.run_pipeline", side_effect=mock_run_pipeline):
        with patch("scripts.run_audio_task.run_music_separation") as music_mock:
            run_dir = run_audio_task(task=CLEAN_VOICE, input_path=input_path, output_root=output_root)

    music_mock.assert_not_called()
    assert len(calls) == 1
    assert getattr(calls[0], "engine_name") == "deepfilternet"
    assert getattr(calls[0], "output_mode") == "audio"
    rows = _read_csv_rows(run_dir / "summary.csv")
    json_rows = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert rows == json_rows
    assert rows[0]["task"] == CLEAN_VOICE
    assert rows[0]["engine"] == "deepfilternet"
    assert rows[0]["status"] == "success"
    assert rows[0]["primary_output_path"].endswith("speech.denoised.wav")


def test_extract_vocals_calls_music_separation_with_task_name(tmp_path: Path) -> None:
    input_path = tmp_path / "song.mp4"
    output_root = tmp_path / "outputs"
    input_path.write_bytes(b"video")
    calls: list[dict[str, object]] = []

    def mock_music_separation(**kwargs: object) -> Path:
        calls.append(kwargs)
        run_dir = Path(kwargs["output_root"]) / "child-run"
        _write_music_summary(run_dir, EXTRACT_VOCALS, run_dir / "vocals.wav")
        return run_dir

    with patch("scripts.run_audio_task.run_pipeline") as pipeline_mock:
        with patch("scripts.run_audio_task.run_music_separation", side_effect=mock_music_separation):
            run_dir = run_audio_task(task=EXTRACT_VOCALS, input_path=input_path, output_root=output_root)

    pipeline_mock.assert_not_called()
    assert len(calls) == 1
    assert calls[0]["task_name"] == EXTRACT_VOCALS
    rows = _read_csv_rows(run_dir / "summary.csv")
    assert rows[0]["task"] == EXTRACT_VOCALS
    assert rows[0]["engine"] == "demucs"
    assert rows[0]["status"] == "success"
    assert rows[0]["primary_output_path"].endswith("vocals.wav")


def test_remove_vocals_calls_music_separation_with_task_name(tmp_path: Path) -> None:
    input_path = tmp_path / "song.mp4"
    output_root = tmp_path / "outputs"
    input_path.write_bytes(b"video")
    calls: list[dict[str, object]] = []

    def mock_music_separation(**kwargs: object) -> Path:
        calls.append(kwargs)
        run_dir = Path(kwargs["output_root"]) / "child-run"
        _write_music_summary(run_dir, REMOVE_VOCALS, run_dir / "no_vocals.wav")
        return run_dir

    with patch("scripts.run_audio_task.run_pipeline") as pipeline_mock:
        with patch("scripts.run_audio_task.run_music_separation", side_effect=mock_music_separation):
            run_dir = run_audio_task(task=REMOVE_VOCALS, input_path=input_path, output_root=output_root)

    pipeline_mock.assert_not_called()
    assert len(calls) == 1
    assert calls[0]["task_name"] == REMOVE_VOCALS
    rows = _read_csv_rows(run_dir / "summary.csv")
    json_rows = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert rows == json_rows
    assert rows[0]["task"] == REMOVE_VOCALS
    assert rows[0]["engine"] == "demucs"
    assert rows[0]["status"] == "success"
    assert rows[0]["primary_output_path"].endswith("no_vocals.wav")
