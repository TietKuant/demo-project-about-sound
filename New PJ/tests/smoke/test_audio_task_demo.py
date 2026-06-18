"""Smoke tests for the minimal unified audio task Gradio demo callback."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from app.audio_task_demo import (
    AUTO_INTENT,
    MUSIC_INTENT,
    ROUTER_CHECKPOINT_ENV_VAR,
    SPEECH_INTENT,
    TARGET_NOISE_INTENT,
    UNKNOWN_INTENT,
    analyze_demo_input,
    analyze_demo_input_for_ui,
    run_demo_task,
)
from src.router.task_registry import CLEAN_VOICE, EXTRACT_VOCALS, REMOVE_VOCALS, TARGET_NOISE_SUPPRESSION


@pytest.fixture(autouse=True)
def _disable_router_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ROUTER_CHECKPOINT_ENV_VAR, raising=False)


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
    assert "Router status:** `disabled`" in markdown
    assert "### Controller decision" in markdown
    assert "### Processing plan" in markdown
    assert "Recommended task:** `clean_voice`" in markdown


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
    assert "Recommended task:** none" in markdown
    assert "Action:** `analyze_only`" in markdown


def test_target_noise_intent_analysis_shows_manual_only_policy(tmp_path: Path) -> None:
    input_path = tmp_path / "speech-plus-siren.wav"
    input_path.write_bytes(b"audio")

    with patch("app.audio_task_demo._extract_feature_row", return_value=_feature_row()):
        markdown, _table, recommended_task = analyze_demo_input(input_path, TARGET_NOISE_INTENT)

    assert recommended_task is None
    assert "### Controller decision" in markdown
    assert "Action:** `manual_required`" in markdown
    assert "target_noise_suppression_manual_only" in markdown
    assert "target_suppressor_experimental" in markdown
    assert "Alternatives:** `clean_voice`" in markdown
    assert "Target suppressor is experimental/manual-only" in markdown


def test_target_noise_intent_analysis_keeps_current_task_dropdown(tmp_path: Path) -> None:
    input_path = tmp_path / "speech-plus-siren.wav"
    input_path.write_bytes(b"audio")

    with patch("app.audio_task_demo._extract_feature_row", return_value=_feature_row()):
        markdown, _table, recommended_text, dropdown_value = analyze_demo_input_for_ui(
            input_path,
            TARGET_NOISE_INTENT,
            current_task=REMOVE_VOCALS,
        )

    assert recommended_text == "No automatic recommendation"
    assert dropdown_value == REMOVE_VOCALS
    assert "target_noise_suppression_manual_only" in markdown


def test_auto_mode_with_router_music_keeps_current_task_dropdown(tmp_path: Path) -> None:
    input_path = tmp_path / "song.wav"
    input_path.write_bytes(b"audio")
    router_result = {
        "router_status": "enabled",
        "predicted_label": "music",
        "confidence": 0.58,
        "probabilities": {"speech_noise": 0.20, "music": 0.58, "environment_noise": 0.22},
        "error": "",
    }

    with (
        patch("app.audio_task_demo._extract_feature_row", return_value=_feature_row()),
        patch("app.audio_task_demo._optional_router_result", return_value=router_result),
    ):
        markdown, table, recommended_text, dropdown_value = analyze_demo_input_for_ui(
            input_path,
            AUTO_INTENT,
            current_task=CLEAN_VOICE,
        )

    assert recommended_text == "No automatic recommendation"
    assert dropdown_value == CLEAN_VOICE
    assert ["duration_sec", "9.000000"] in table
    assert "Router predicted label:** `music`" in markdown
    assert "Blocked reasons:** `router_not_trusted`" in markdown


def test_analyze_demo_input_for_ui_keeps_current_task_when_no_recommendation(tmp_path: Path) -> None:
    input_path = tmp_path / "environment.wav"
    input_path.write_bytes(b"audio")
    router_result = {
        "router_status": "enabled",
        "predicted_label": "environment_noise",
        "confidence": 0.90,
        "probabilities": {"speech_noise": 0.02, "music": 0.08, "environment_noise": 0.90},
        "error": "",
    }

    with (
        patch("app.audio_task_demo._extract_feature_row", return_value=_feature_row()),
        patch("app.audio_task_demo._optional_router_result", return_value=router_result),
    ):
        _markdown, _table, recommended_text, dropdown_value = analyze_demo_input_for_ui(
            input_path,
            AUTO_INTENT,
            current_task=REMOVE_VOCALS,
        )

    assert recommended_text == "No automatic recommendation"
    assert dropdown_value == REMOVE_VOCALS


def test_analyze_demo_input_with_router_speech_noise_recommends_clean_voice(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    input_path.write_bytes(b"audio")
    router_result = {
        "router_status": "enabled",
        "predicted_label": "speech_noise",
        "confidence": 0.92,
        "accepted": True,
        "confidence_threshold": 0.90,
        "route_target": "clean_voice",
        "engine_target": "deepfilternet",
        "recommended_task": CLEAN_VOICE,
        "decision_reason": "accepted_router_prediction",
        "warnings": [],
        "probabilities": {"speech_noise": 0.92, "music": 0.05, "environment_noise": 0.03},
        "error": "",
    }

    with (
        patch("app.audio_task_demo._extract_feature_row", return_value=_feature_row()),
        patch("app.audio_task_demo._optional_router_result", return_value=router_result),
    ):
        markdown, _table, recommended_task = analyze_demo_input(input_path, SPEECH_INTENT)

    assert recommended_task == CLEAN_VOICE
    assert "Router status:** `enabled`" in markdown
    assert "Router predicted label:** `speech_noise`" in markdown
    assert "0.920" in markdown
    assert "Recommended task:** `clean_voice`" in markdown


def test_auto_accepted_noisy_speech_recommends_clean_voice_through_planner(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    input_path.write_bytes(b"audio")
    router_result = {
        "router_status": "enabled",
        "predicted_label": "speech_noisy_general",
        "confidence": 0.95,
        "accepted": True,
        "route_target": CLEAN_VOICE,
        "recommended_task": CLEAN_VOICE,
        "decision_reason": "accepted_router_prediction",
        "warnings": [],
        "probabilities": {"speech_noisy_general": 0.95},
        "error": "",
    }

    with (
        patch("app.audio_task_demo._extract_feature_row", return_value=_feature_row()),
        patch("app.audio_task_demo._optional_router_result", return_value=router_result),
    ):
        markdown, _table, recommended_task = analyze_demo_input(input_path, AUTO_INTENT)

    assert recommended_task == CLEAN_VOICE
    assert "Action:** `run_task`" in markdown
    assert "Algorithm:** `DeepFilterNet`" in markdown


def test_auto_accepted_target_noise_uses_clean_voice_fallback(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    input_path.write_bytes(b"audio")
    router_result = {
        "router_status": "enabled",
        "predicted_label": "speech_target_noise",
        "confidence": 0.96,
        "accepted": True,
        "route_target": "manual_required",
        "recommended_task": None,
        "decision_reason": "target_noise_suppression is experimental",
        "warnings": ["target_noise_suppression_is_experimental"],
        "probabilities": {"speech_target_noise": 0.96},
        "error": "",
    }

    with (
        patch("app.audio_task_demo._extract_feature_row", return_value=_feature_row()),
        patch("app.audio_task_demo._optional_router_result", return_value=router_result),
    ):
        markdown, _table, recommended_task = analyze_demo_input(input_path, AUTO_INTENT)

    assert recommended_task == CLEAN_VOICE
    assert "Decision:** `run_task`" in markdown
    assert "Recommended next step:** `clean_voice`" in markdown
    assert "Algorithm:** `DeepFilterNet`" in markdown
    assert "target_suppressor_experimental" in markdown
    assert "Recommended task:** `target_noise_suppression`" not in markdown


def test_very_short_input_is_analysis_only(tmp_path: Path) -> None:
    input_path = tmp_path / "short.wav"
    input_path.write_bytes(b"audio")

    with patch(
        "app.audio_task_demo._extract_feature_row",
        return_value=_feature_row(duration_sec="0.200000"),
    ):
        markdown, _table, recommended_task = analyze_demo_input(input_path, SPEECH_INTENT)

    assert recommended_task is None
    assert "Decision:** `analyze_only`" in markdown
    assert "input_too_short" in markdown


def test_extract_vocals_target_noise_router_keeps_goal_with_mismatch_warning(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    input_path.write_bytes(b"audio")
    router_result = {
        "router_status": "enabled",
        "predicted_label": "speech_target_noise",
        "confidence": 0.96,
        "accepted": True,
        "route_target": "manual_required",
        "recommended_task": None,
        "decision_reason": "target_noise_suppression is experimental",
        "warnings": [],
        "probabilities": {"speech_target_noise": 0.96},
        "error": "",
    }

    with (
        patch("app.audio_task_demo._extract_feature_row", return_value=_feature_row()),
        patch("app.audio_task_demo._optional_router_result", return_value=router_result),
    ):
        markdown, _table, recommended_task = analyze_demo_input(input_path, MUSIC_INTENT)

    assert recommended_task == EXTRACT_VOCALS
    assert "router_goal_mismatch" in markdown


def test_analyze_demo_input_with_router_music_and_manual_music_recommends_extract_vocals(tmp_path: Path) -> None:
    input_path = tmp_path / "song.wav"
    input_path.write_bytes(b"audio")
    router_result = {
        "router_status": "enabled",
        "predicted_label": "music",
        "confidence": 0.93,
        "accepted": True,
        "confidence_threshold": 0.90,
        "route_target": "manual_required",
        "engine_target": "demucs",
        "recommended_task": EXTRACT_VOCALS,
        "decision_reason": "accepted_router_prediction",
        "warnings": ["manual_music_task_selection_required"],
        "probabilities": {"speech_noise": 0.04, "music": 0.93, "environment_noise": 0.03},
        "error": "",
    }

    with (
        patch("app.audio_task_demo._extract_feature_row", return_value=_feature_row()),
        patch("app.audio_task_demo._optional_router_result", return_value=router_result),
    ):
        markdown, _table, recommended_task = analyze_demo_input(input_path, MUSIC_INTENT)

    assert recommended_task == EXTRACT_VOCALS
    assert "Router predicted label:** `music`" in markdown
    assert "music=0.930" in markdown
    assert "Recommended task:** `extract_vocals`" in markdown


def test_analyze_demo_input_rejects_unaccepted_router_target_noise_prediction(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    input_path.write_bytes(b"audio")
    router_result = {
        "router_status": "enabled",
        "predicted_label": "speech_target_noise",
        "confidence": 0.75,
        "accepted": False,
        "confidence_threshold": 0.90,
        "route_target": "manual_required",
        "engine_target": "none",
        "recommended_task": None,
        "decision_reason": "low_confidence",
        "warnings": ["low_confidence_router_prediction"],
        "probabilities": {
            "speech_target_noise": 0.75,
            "music_with_vocals": 0.18,
            "environment_only": 0.07,
        },
        "error": "",
    }

    with (
        patch("app.audio_task_demo._extract_feature_row", return_value=_feature_row()),
        patch("app.audio_task_demo._optional_router_result", return_value=router_result),
    ):
        markdown, _table, recommended_task = analyze_demo_input(input_path, SPEECH_INTENT)

    assert recommended_task == CLEAN_VOICE
    assert "Router accepted:** `false`" in markdown
    assert "Recommended task:** `clean_voice`" in markdown


def test_auto_mode_with_router_environment_noise_returns_no_recommendation(tmp_path: Path) -> None:
    input_path = tmp_path / "environment.wav"
    input_path.write_bytes(b"audio")
    router_result = {
        "router_status": "enabled",
        "predicted_label": "environment_noise",
        "confidence": 0.91,
        "probabilities": {"speech_noise": 0.03, "music": 0.06, "environment_noise": 0.91},
        "error": "",
    }

    with (
        patch("app.audio_task_demo._extract_feature_row", return_value=_feature_row()),
        patch("app.audio_task_demo._optional_router_result", return_value=router_result),
    ):
        markdown, _table, recommended_task = analyze_demo_input(input_path, AUTO_INTENT)

    assert recommended_task is None
    assert "Router predicted label:** `environment_noise`" in markdown
    assert "Blocked reasons:** `router_not_trusted`" in markdown


def test_analyze_demo_input_router_failure_falls_back_safely(tmp_path: Path) -> None:
    input_path = tmp_path / "song.wav"
    input_path.write_bytes(b"audio")
    router_result = {
        "router_status": "failed",
        "predicted_label": "",
        "confidence": None,
        "probabilities": {},
        "error": "mock router failure",
    }

    with (
        patch("app.audio_task_demo._extract_feature_row", return_value=_feature_row()),
        patch("app.audio_task_demo._optional_router_result", return_value=router_result),
    ):
        markdown, _table, recommended_task = analyze_demo_input(input_path, MUSIC_INTENT)

    assert recommended_task == EXTRACT_VOCALS
    assert "Router status:** `failed`" in markdown
    assert "mock router failure" in markdown


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


def test_demo_router_music_does_not_block_remove_vocals(tmp_path: Path) -> None:
    input_path = tmp_path / "song.wav"
    output_root = tmp_path / "demo-runs"
    input_path.write_bytes(b"audio")
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
        markdown, table, downloadable = run_demo_task(
            input_path,
            REMOVE_VOCALS,
            content_intent=MUSIC_INTENT,
            output_root=output_root,
        )

    assert calls[0]["task"] == REMOVE_VOCALS
    assert "Blocked" not in markdown
    assert ["task", REMOVE_VOCALS] in table
    assert downloadable.endswith("no_vocals.wav")


def test_demo_target_noise_suppression_is_blocked_by_planner(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    output_root = tmp_path / "demo-runs"
    input_path.write_bytes(b"audio")
    with patch("app.audio_task_demo.run_audio_task") as run_mock:
        markdown, table, downloadable = run_demo_task(
            input_path,
            TARGET_NOISE_SUPPRESSION,
            content_intent=SPEECH_INTENT,
            output_root=output_root,
        )

    run_mock.assert_not_called()
    assert "Blocked" in markdown
    assert "target_noise_suppression_manual_only" in markdown
    assert table == []
    assert downloadable is None


def test_demo_speech_intent_extract_vocals_runs_with_planner_warning(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    input_path.write_bytes(b"audio")

    def mock_run_audio_task(**kwargs: object) -> Path:
        run_dir = tmp_path / "runs" / "extract_vocals" / "mock-run"
        primary_output = run_dir / "vocals.wav"
        primary_output.parent.mkdir(parents=True, exist_ok=True)
        primary_output.write_bytes(b"vocals")
        _write_summary(run_dir, task=EXTRACT_VOCALS, primary_output_path=primary_output)
        return run_dir

    router_result = {
        "router_status": "enabled",
        "predicted_label": "speech_target_noise",
        "confidence": 0.95,
        "accepted": True,
        "warnings": [],
    }
    with (
        patch("app.audio_task_demo.run_audio_task", side_effect=mock_run_audio_task) as run_mock,
        patch("app.audio_task_demo._optional_router_result", return_value=router_result),
    ):
        markdown, table, downloadable = run_demo_task(
            input_path,
            EXTRACT_VOCALS,
            content_intent=SPEECH_INTENT,
            output_root=tmp_path / "runs",
        )

    run_mock.assert_called_once()
    assert "router_goal_mismatch" in markdown
    assert ["task", EXTRACT_VOCALS] in table
    assert downloadable is not None


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
    assert "target_noise_suppression_manual_only" in markdown
    assert table == []
    assert downloadable is None


def test_demo_unknown_intent_blocks_target_noise_suppression(tmp_path: Path) -> None:
    input_path = tmp_path / "unknown.wav"
    output_root = tmp_path / "demo-runs"
    input_path.write_bytes(b"audio")

    with patch("app.audio_task_demo.run_audio_task") as run_mock:
        markdown, table, downloadable = run_demo_task(
            input_path,
            TARGET_NOISE_SUPPRESSION,
            content_intent=UNKNOWN_INTENT,
            output_root=output_root,
        )

    run_mock.assert_not_called()
    assert "target_noise_suppression_manual_only" in markdown
    assert table == []
    assert downloadable is None


def test_demo_auto_intent_allows_run_with_caution(tmp_path: Path) -> None:
    input_path = tmp_path / "auto.wav"
    output_root = tmp_path / "demo-runs"
    input_path.write_bytes(b"audio")

    def mock_run_audio_task(**kwargs: object) -> Path:
        run_dir = output_root / "clean_voice" / "mock-run"
        primary_output = run_dir / "auto.denoised.wav"
        primary_output.parent.mkdir(parents=True, exist_ok=True)
        primary_output.write_bytes(b"enhanced")
        _write_summary(run_dir, task=CLEAN_VOICE, primary_output_path=primary_output)
        return run_dir

    with patch("app.audio_task_demo.run_audio_task", side_effect=mock_run_audio_task):
        markdown, table, downloadable = run_demo_task(
            input_path,
            CLEAN_VOICE,
            content_intent=AUTO_INTENT,
            output_root=output_root,
        )

    assert ["task", CLEAN_VOICE] in table
    assert downloadable.endswith("auto.denoised.wav")


def test_demo_speech_intent_remove_vocals_still_runs(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    input_path.write_bytes(b"audio")

    def mock_run_audio_task(**kwargs: object) -> Path:
        run_dir = tmp_path / "runs" / "remove_vocals" / "mock-run"
        primary_output = run_dir / "no_vocals.wav"
        primary_output.parent.mkdir(parents=True, exist_ok=True)
        primary_output.write_bytes(b"no-vocals")
        _write_summary(run_dir, task=REMOVE_VOCALS, primary_output_path=primary_output)
        return run_dir

    with patch("app.audio_task_demo.run_audio_task", side_effect=mock_run_audio_task) as run_mock:
        markdown, table, downloadable = run_demo_task(
            input_path,
            REMOVE_VOCALS,
            content_intent=SPEECH_INTENT,
            output_root=tmp_path / "runs",
        )

    run_mock.assert_called_once()
    assert ["task", REMOVE_VOCALS] in table
    assert downloadable is not None


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
