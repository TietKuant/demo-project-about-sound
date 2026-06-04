"""Smoke tests for the unified audio task runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

from scripts.run_audio_task import run_audio_task
from src.api.contracts import DenoiseResult
from src.router.task_registry import CLEAN_VOICE, EXTRACT_VOCALS, REMOVE_VOCALS, TARGET_NOISE_SUPPRESSION


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


def test_target_noise_suppression_calls_inference_and_writes_success_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    output_root = tmp_path / "outputs"
    checkpoint_path = tmp_path / "checkpoint.pt"
    input_path.write_bytes(b"audio")
    checkpoint_path.write_bytes(b"checkpoint")
    calls: list[dict[str, object]] = []

    def mock_inference(**kwargs: object) -> dict[str, Path]:
        calls.append(kwargs)
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"enhanced")
        return {"output": output_path, "summary": Path(kwargs["summary_path"])}

    with patch("scripts.run_audio_task.run_pipeline") as pipeline_mock:
        with patch("scripts.run_audio_task.run_music_separation") as music_mock:
            with patch("scripts.run_audio_task._run_target_noise_suppression_inference", side_effect=mock_inference):
                run_dir = run_audio_task(
                    task=TARGET_NOISE_SUPPRESSION,
                    input_path=input_path,
                    output_root=output_root,
                    target_noise_checkpoint=checkpoint_path,
                )

    pipeline_mock.assert_not_called()
    music_mock.assert_not_called()
    assert len(calls) == 1
    assert calls[0]["checkpoint_path"] == checkpoint_path
    assert calls[0]["input_path"] == input_path.resolve()
    assert Path(calls[0]["output_path"]).name == "speech.target_noise_suppressed.wav"
    assert Path(calls[0]["summary_path"]).name == "target_noise_suppressor_summary.json"
    rows = _read_csv_rows(run_dir / "summary.csv")
    assert rows[0]["task"] == TARGET_NOISE_SUPPRESSION
    assert rows[0]["engine"] == "target_noise_suppressor"
    assert rows[0]["status"] == "success"
    assert rows[0]["primary_output_path"].endswith("speech.target_noise_suppressed.wav")


def test_target_noise_suppression_missing_checkpoint_writes_failed_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    input_path.write_bytes(b"audio")

    with patch("scripts.run_audio_task._run_target_noise_suppression_inference") as inference_mock:
        with patch.dict("scripts.run_audio_task.os.environ", {}, clear=True):
            run_dir = run_audio_task(
                task=TARGET_NOISE_SUPPRESSION,
                input_path=input_path,
                output_root=tmp_path / "outputs",
            )

    inference_mock.assert_not_called()
    rows = _read_csv_rows(run_dir / "summary.csv")
    assert rows[0]["status"] == "failed"
    assert rows[0]["engine"] == "target_noise_suppressor"
    assert "--target-noise-checkpoint" in rows[0]["error"]
    assert "TARGET_NOISE_SUPPRESSOR_CHECKPOINT" in rows[0]["error"]
    assert rows[0]["primary_output_path"] == ""


def test_target_noise_suppression_uses_checkpoint_from_environment(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    checkpoint_path = tmp_path / "checkpoint.pt"
    input_path.write_bytes(b"audio")
    checkpoint_path.write_bytes(b"checkpoint")
    calls: list[dict[str, object]] = []

    def mock_inference(**kwargs: object) -> dict[str, Path]:
        calls.append(kwargs)
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"enhanced")
        return {"output": output_path, "summary": Path(kwargs["summary_path"])}

    with patch("scripts.run_audio_task._run_target_noise_suppression_inference", side_effect=mock_inference):
        with patch.dict(
            "scripts.run_audio_task.os.environ",
            {"TARGET_NOISE_SUPPRESSOR_CHECKPOINT": str(checkpoint_path)},
            clear=True,
        ):
            run_dir = run_audio_task(
                task=TARGET_NOISE_SUPPRESSION,
                input_path=input_path,
                output_root=tmp_path / "outputs",
            )

    assert calls[0]["checkpoint_path"] == checkpoint_path
    rows = _read_csv_rows(run_dir / "summary.csv")
    assert rows[0]["status"] == "success"
    assert rows[0]["engine"] == "target_noise_suppressor"


def test_target_noise_suppression_rejects_video_without_inference(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.mov"
    checkpoint_path = tmp_path / "checkpoint.pt"
    input_path.write_bytes(b"video")
    checkpoint_path.write_bytes(b"checkpoint")

    with patch("scripts.run_audio_task._run_target_noise_suppression_inference") as inference_mock:
        run_dir = run_audio_task(
            task=TARGET_NOISE_SUPPRESSION,
            input_path=input_path,
            output_root=tmp_path / "outputs",
            target_noise_checkpoint=checkpoint_path,
        )

    inference_mock.assert_not_called()
    rows = _read_csv_rows(run_dir / "summary.csv")
    assert rows[0]["status"] == "failed"
    assert rows[0]["engine"] == "target_noise_suppressor"
    assert rows[0]["error"] == "target_noise_suppression currently supports audio input only."
