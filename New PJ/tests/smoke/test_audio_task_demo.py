"""Smoke tests for the minimal unified audio task Gradio demo callback."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

from app.audio_task_demo import (
    MUSIC_INTENT,
    SPEECH_INTENT,
    UNKNOWN_INTENT,
    analyze_demo_input,
    run_demo_task,
)
from src.router.task_registry import CLEAN_VOICE, EXTRACT_VOCALS, REMOVE_VOCALS, TARGET_NOISE_SUPPRESSION


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
                "engine": "deepfilternet"
                if task == CLEAN_VOICE
                else "target_noise_suppressor"
                if task == TARGET_NOISE_SUPPRESSION
                else "demucs",
                "input_path": "",
                "input_type": "audio",
                "status": "success",
                "runtime_sec": "1.250000",
                "primary_output_path": "" if primary_output_path is None else str(primary_output_path),
                "error": "",
            }
        )


def _feature_row(
    *,
    status: str = "success",
    duration_sec: str = "9.000000",
    rms_energy: str = "0.0500000000",
    zero_crossing_rate: str = "0.0200000000",
    spectral_centroid_hz: str = "1800.000000",
    spectral_bandwidth_hz: str = "2500.000000",
    error: str = "",
) -> dict[str, str]:
    return {
        "status": status,
        "duration_sec": duration_sec,
        "rms_energy": rms_energy,
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_centroid_hz": spectral_centroid_hz,
        "spectral_bandwidth_hz": spectral_bandwidth_hz,
        "error": error,
    }


def test_analyze_demo_input_recommends_clean_voice_for_speech_intent(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    input_path.write_bytes(b"audio")

    with patch("app.audio_task_demo._extract_feature_row", return_value=_feature_row()):
        markdown, table, recommended_task = analyze_demo_input(input_path, SPEECH_INTENT)

    assert recommended_task == CLEAN_VOICE
    assert ["duration_sec", "9.000000"] in table
    assert ["rms_energy", "0.0500000000"] in table
    assert "feature-based recommendation aid" in markdown


def test_analyze_demo_input_recommends_extract_vocals_for_music_intent(tmp_path: Path) -> None:
    input_path = tmp_path / "song.mp4"
    input_path.write_bytes(b"video")

    with patch(
        "app.audio_task_demo._extract_feature_row",
        return_value=_feature_row(zero_crossing_rate="0.0300000000", spectral_centroid_hz="2300.000000"),
    ):
        markdown, table, recommended_task = analyze_demo_input(input_path, MUSIC_INTENT)

    assert recommended_task == EXTRACT_VOCALS
    assert ["spectral_centroid_hz", "2300.000000"] in table
    assert "High-frequency/noisy profile" in markdown


def test_analyze_demo_input_unknown_intent_returns_no_recommendation(tmp_path: Path) -> None:
    input_path = tmp_path / "unknown.wav"
    input_path.write_bytes(b"audio")

    with patch("app.audio_task_demo._extract_feature_row", return_value=_feature_row()):
        markdown, table, recommended_task = analyze_demo_input(input_path, UNKNOWN_INTENT)

    assert recommended_task is None
    assert ["zero_crossing_rate", "0.0200000000"] in table
    assert "No automatic recommendation" in markdown
    assert "unknown" in markdown.lower()


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
        markdown, table, downloadable = run_demo_task(
            input_path,
            CLEAN_VOICE,
            content_intent=SPEECH_INTENT,
            output_root=output_root,
        )

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
        _, table, downloadable = run_demo_task(
            input_path,
            EXTRACT_VOCALS,
            content_intent=MUSIC_INTENT,
            output_root=output_root,
        )

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
        _, table, downloadable = run_demo_task(
            input_path,
            REMOVE_VOCALS,
            content_intent=MUSIC_INTENT,
            output_root=output_root,
        )

    assert calls[0]["task"] == REMOVE_VOCALS
    assert ["task", REMOVE_VOCALS] in table
    assert downloadable.endswith("no_vocals.wav")


def test_demo_target_noise_suppression_allows_speech_intent(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    output_root = tmp_path / "demo-runs"
    input_path.write_bytes(b"audio")
    calls: list[dict[str, object]] = []

    def mock_run_audio_task(**kwargs: object) -> Path:
        calls.append(kwargs)
        run_dir = output_root / "target_noise_suppression" / "mock-run"
        primary_output = run_dir / "speech.target_noise_suppressed.wav"
        primary_output.parent.mkdir(parents=True, exist_ok=True)
        primary_output.write_bytes(b"enhanced")
        _write_summary(run_dir, task=TARGET_NOISE_SUPPRESSION, primary_output_path=primary_output)
        return run_dir

    with patch("app.audio_task_demo.run_audio_task", side_effect=mock_run_audio_task):
        markdown, table, downloadable = run_demo_task(
            input_path,
            TARGET_NOISE_SUPPRESSION,
            content_intent=SPEECH_INTENT,
            output_root=output_root,
        )

    assert calls[0]["task"] == TARGET_NOISE_SUPPRESSION
    assert "success" in markdown
    assert ["task", TARGET_NOISE_SUPPRESSION] in table
    assert downloadable.endswith("speech.target_noise_suppressed.wav")


def test_demo_blocks_speech_intent_with_extract_vocals(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    input_path.write_bytes(b"audio")

    with patch("app.audio_task_demo.run_audio_task") as run_mock:
        markdown, table, downloadable = run_demo_task(
            input_path,
            EXTRACT_VOCALS,
            content_intent=SPEECH_INTENT,
            output_root=tmp_path / "runs",
        )

    run_mock.assert_not_called()
    assert "Blocked" in markdown
    assert "Speech/noisy speech" in markdown
    assert table == []
    assert downloadable is None


def test_demo_blocks_music_intent_with_target_noise_suppression(tmp_path: Path) -> None:
    input_path = tmp_path / "music.wav"
    input_path.write_bytes(b"audio")

    with patch("app.audio_task_demo.run_audio_task") as run_mock:
        markdown, table, downloadable = run_demo_task(
            input_path,
            TARGET_NOISE_SUPPRESSION,
            content_intent=MUSIC_INTENT,
            output_root=tmp_path / "runs",
        )

    run_mock.assert_not_called()
    assert "Blocked" in markdown
    assert "not intended for music input" in markdown
    assert table == []
    assert downloadable is None


def test_demo_unknown_intent_allows_target_noise_suppression_with_caution(tmp_path: Path) -> None:
    input_path = tmp_path / "unknown.wav"
    output_root = tmp_path / "demo-runs"
    input_path.write_bytes(b"audio")

    def mock_run_audio_task(**kwargs: object) -> Path:
        run_dir = output_root / "target_noise_suppression" / "mock-run"
        primary_output = run_dir / "unknown.target_noise_suppressed.wav"
        primary_output.parent.mkdir(parents=True, exist_ok=True)
        primary_output.write_bytes(b"enhanced")
        _write_summary(run_dir, task=TARGET_NOISE_SUPPRESSION, primary_output_path=primary_output)
        return run_dir

    with patch("app.audio_task_demo.run_audio_task", side_effect=mock_run_audio_task):
        markdown, table, downloadable = run_demo_task(
            input_path,
            TARGET_NOISE_SUPPRESSION,
            content_intent=UNKNOWN_INTENT,
            output_root=output_root,
        )

    assert "Caution" in markdown
    assert ["task", TARGET_NOISE_SUPPRESSION] in table
    assert downloadable.endswith("unknown.target_noise_suppressed.wav")


def test_demo_blocks_speech_intent_with_remove_vocals(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    input_path.write_bytes(b"audio")

    with patch("app.audio_task_demo.run_audio_task") as run_mock:
        markdown, table, downloadable = run_demo_task(
            input_path,
            REMOVE_VOCALS,
            content_intent=SPEECH_INTENT,
            output_root=tmp_path / "runs",
        )

    run_mock.assert_not_called()
    assert "Blocked" in markdown
    assert "Speech/noisy speech" in markdown
    assert table == []
    assert downloadable is None


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
