"""Smoke tests for the Vietnamese real-sample suite runner."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from scripts.run_vietnamese_real_sample_suite import run_vietnamese_real_sample_suite
from src.api.contracts import DenoiseResult
from src.router.task_registry import CLEAN_VOICE, EXTRACT_VOCALS, REMOVE_VOCALS


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "sample_id",
        "category",
        "input_path",
        "input_type",
        "language",
        "expected_task",
        "has_clean_reference",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _manifest_row(sample_id: str, input_path: Path, expected_task: str) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "category": "vietnamese_music_or_music_video_optional",
        "input_path": str(input_path),
        "input_type": "audio",
        "language": "vi",
        "expected_task": expected_task,
        "has_clean_reference": "false",
        "notes": f"notes for {sample_id}",
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _mock_ffprobe_duration(duration: str = "2.000000") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["ffprobe"], 0, duration, "")


def _write_music_summary(run_dir: Path, task_name: str, primary_path: Path) -> None:
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
                "run_id": "mock-run",
                "input_path": "",
                "task": task_name,
                "engine": "demucs",
                "status": "success",
                "runtime_sec": "1.000000",
                "primary_output_label": primary_path.stem,
                "primary_output_path": str(primary_path),
                "vocals_path": str(run_dir / "vocals.wav"),
                "no_vocals_path": str(run_dir / "no_vocals.wav"),
                "error": "",
            }
        )


def test_clean_voice_row_calls_run_pipeline_and_writes_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    input_path.write_bytes(b"audio")
    manifest_path = tmp_path / "manifest.csv"
    output_root = tmp_path / "outputs"
    _write_manifest(manifest_path, [_manifest_row("vi_speech_001", input_path, CLEAN_VOICE)])
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

    with patch("scripts.run_vietnamese_real_sample_suite.subprocess.run", return_value=_mock_ffprobe_duration("2.0")):
        with patch("scripts.run_vietnamese_real_sample_suite.run_pipeline", side_effect=mock_run_pipeline):
            with patch("scripts.run_vietnamese_real_sample_suite.run_music_separation") as music_mock:
                result_root = run_vietnamese_real_sample_suite(manifest_path=manifest_path, output_root=output_root)

    csv_rows = _read_csv_rows(result_root / "summary.csv")
    json_rows = json.loads((result_root / "summary.json").read_text(encoding="utf-8"))
    assert csv_rows == json_rows
    assert len(calls) == 1
    assert getattr(calls[0], "engine_name") == "deepfilternet"
    assert getattr(calls[0], "output_mode") == "audio"
    music_mock.assert_not_called()
    assert csv_rows[0]["sample_id"] == "vi_speech_001"
    assert csv_rows[0]["expected_task"] == CLEAN_VOICE
    assert csv_rows[0]["status"] == "success"
    assert csv_rows[0]["audio_duration_sec"] == "2.000000"
    assert csv_rows[0]["rtf"]
    assert csv_rows[0]["primary_output_path"].endswith("speech.denoised.wav")


def test_vocal_rows_call_music_separation_with_expected_task_names(tmp_path: Path) -> None:
    extract_input = tmp_path / "music_extract.mp4"
    remove_input = tmp_path / "music_remove.mp4"
    extract_input.write_bytes(b"video")
    remove_input.write_bytes(b"video")
    manifest_path = tmp_path / "manifest.csv"
    output_root = tmp_path / "outputs"
    _write_manifest(
        manifest_path,
        [
            _manifest_row("vi_music_extract", extract_input, EXTRACT_VOCALS),
            _manifest_row("vi_music_remove", remove_input, REMOVE_VOCALS),
        ],
    )
    task_calls: list[str] = []

    def mock_music_separation(**kwargs: object) -> Path:
        task_name = str(kwargs["task_name"])
        task_calls.append(task_name)
        run_dir = Path(kwargs["output_root"]) / "mock-run"
        primary_name = "vocals.wav" if task_name == EXTRACT_VOCALS else "no_vocals.wav"
        _write_music_summary(run_dir, task_name, run_dir / primary_name)
        return run_dir

    with patch("scripts.run_vietnamese_real_sample_suite.subprocess.run", return_value=_mock_ffprobe_duration("4.0")):
        with patch("scripts.run_vietnamese_real_sample_suite.run_pipeline") as pipeline_mock:
            with patch("scripts.run_vietnamese_real_sample_suite.run_music_separation", side_effect=mock_music_separation):
                result_root = run_vietnamese_real_sample_suite(
                    manifest_path=manifest_path,
                    output_root=output_root,
                    demucs_python=Path("/isolated/bin/python"),
                )

    pipeline_mock.assert_not_called()
    assert task_calls == [EXTRACT_VOCALS, REMOVE_VOCALS]
    rows = _read_csv_rows(result_root / "summary.csv")
    assert [row["expected_task"] for row in rows] == [EXTRACT_VOCALS, REMOVE_VOCALS]
    assert [row["audio_duration_sec"] for row in rows] == ["4.000000", "4.000000"]
    assert rows[0]["rtf"]
    assert rows[1]["rtf"]
    assert rows[0]["primary_output_path"].endswith("vocals.wav")
    assert rows[1]["primary_output_path"].endswith("no_vocals.wav")


def test_missing_file_with_skip_missing_records_skipped(tmp_path: Path) -> None:
    missing_input = tmp_path / "missing.wav"
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [_manifest_row("vi_missing", missing_input, CLEAN_VOICE)])

    with patch("scripts.run_vietnamese_real_sample_suite.run_pipeline") as pipeline_mock:
        result_root = run_vietnamese_real_sample_suite(
            manifest_path=manifest_path,
            output_root=tmp_path / "outputs",
            skip_missing=True,
        )

    pipeline_mock.assert_not_called()
    row = _read_csv_rows(result_root / "summary.csv")[0]
    assert row["status"] == "skipped"
    assert row["audio_duration_sec"] == ""
    assert row["rtf"] == ""
    assert "Input file not found" in row["error"]


def test_missing_file_without_skip_missing_records_failed(tmp_path: Path) -> None:
    missing_input = tmp_path / "missing.wav"
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [_manifest_row("vi_missing", missing_input, CLEAN_VOICE)])

    with patch("scripts.run_vietnamese_real_sample_suite.run_pipeline") as pipeline_mock:
        result_root = run_vietnamese_real_sample_suite(
            manifest_path=manifest_path,
            output_root=tmp_path / "outputs",
            skip_missing=False,
        )

    pipeline_mock.assert_not_called()
    row = _read_csv_rows(result_root / "summary.csv")[0]
    assert row["status"] == "failed"
    assert row["audio_duration_sec"] == ""
    assert row["rtf"] == ""
    assert "Input file not found" in row["error"]


def test_duration_failure_does_not_fail_processing(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    input_path.write_bytes(b"audio")
    manifest_path = tmp_path / "manifest.csv"
    output_root = tmp_path / "outputs"
    _write_manifest(manifest_path, [_manifest_row("vi_speech_002", input_path, CLEAN_VOICE)])

    def mock_run_pipeline(request: object) -> DenoiseResult:
        output_path = Path(getattr(request, "output_dir")) / "speech.denoised.wav"
        return DenoiseResult(
            status="completed_real",
            final_output_path=output_path,
            intermediate_audio_path=None,
            engine_name="deepfilternet",
        )

    with patch(
        "scripts.run_vietnamese_real_sample_suite.subprocess.run",
        return_value=subprocess.CompletedProcess(["ffprobe"], 1, "", "ffprobe failed"),
    ):
        with patch("scripts.run_vietnamese_real_sample_suite.run_pipeline", side_effect=mock_run_pipeline):
            result_root = run_vietnamese_real_sample_suite(manifest_path=manifest_path, output_root=output_root)

    row = _read_csv_rows(result_root / "summary.csv")[0]
    assert row["status"] == "success"
    assert row["audio_duration_sec"] == ""
    assert row["rtf"] == ""
    assert row["primary_output_path"].endswith("speech.denoised.wav")


def test_ffprobe_duration_uses_safe_subprocess_command(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.m4a"
    input_path.write_bytes(b"audio")
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [_manifest_row("vi_speech_003", input_path, CLEAN_VOICE)])
    captured_kwargs: list[dict[str, object]] = []
    captured_commands: list[list[str]] = []

    def mock_ffprobe(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured_commands.append(command)
        captured_kwargs.append(kwargs)
        return subprocess.CompletedProcess(command, 0, "3.5\n", "")

    def mock_run_pipeline(request: object) -> DenoiseResult:
        return DenoiseResult(
            status="completed_real",
            final_output_path=Path(getattr(request, "output_dir")) / "speech.denoised.wav",
            intermediate_audio_path=None,
            engine_name="deepfilternet",
        )

    with patch("scripts.run_vietnamese_real_sample_suite.subprocess.run", side_effect=mock_ffprobe):
        with patch("scripts.run_vietnamese_real_sample_suite.run_pipeline", side_effect=mock_run_pipeline):
            result_root = run_vietnamese_real_sample_suite(
                manifest_path=manifest_path,
                output_root=tmp_path / "outputs",
            )

    command = captured_commands[0]
    assert command[:6] == ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of"]
    assert command[6] == "default=noprint_wrappers=1:nokey=1"
    assert command[7] == str(input_path)
    assert captured_kwargs[0] == {"capture_output": True, "text": True, "check": False}
    row = _read_csv_rows(result_root / "summary.csv")[0]
    assert row["audio_duration_sec"] == "3.500000"
    assert row["rtf"]
