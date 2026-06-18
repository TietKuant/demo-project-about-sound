"""Smoke tests for the real-audio demo suite runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.run_real_audio_demo_suite import REPORT_COLUMNS, _input_metadata, run_real_audio_demo_suite
from src.router.task_registry import CLEAN_VOICE, TARGET_NOISE_SUPPRESSION


def _write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    columns = fieldnames or ["input_path", "case_id", "expected_task", "description", "notes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _fake_task_run(**kwargs: object) -> Path:
    task = str(kwargs["task"])
    output_root = Path(kwargs["output_root"])
    run_dir = output_root / task / "mock-run"
    output_path = run_dir / f"{task}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"processed")
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run_id",
                "task",
                "engine",
                "input_path",
                "input_type",
                "status",
                "runtime_sec",
                "primary_output_path",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "mock-run",
                "task": task,
                "engine": "mock-engine",
                "input_path": kwargs["input_path"],
                "input_type": "audio",
                "status": "success",
                "runtime_sec": "0.1",
                "primary_output_path": output_path,
                "error": "",
            }
        )
    return run_dir


def test_real_audio_demo_suite_validates_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        run_real_audio_demo_suite(manifest_path=tmp_path / "missing.csv")

    invalid_manifest = tmp_path / "invalid.csv"
    _write_manifest(invalid_manifest, [{"input_path": "audio.wav"}], fieldnames=["input_path"])
    with pytest.raises(ValueError, match="missing columns"):
        run_real_audio_demo_suite(manifest_path=invalid_manifest)


def test_real_audio_demo_suite_runs_manual_task_and_writes_reports(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    input_path.write_bytes(b"audio")
    manifest = tmp_path / "manifest.csv"
    output_root = tmp_path / "outputs"
    _write_manifest(
        manifest,
        [
            {
                "input_path": str(input_path),
                "case_id": "speech-case",
                "expected_task": CLEAN_VOICE,
                "description": "Real speech cleanup demo",
                "notes": "manual task",
            }
        ],
    )

    metadata = {
        "input_duration_sec": "3.500000",
        "input_sample_rate_hz": "16000",
        "input_channels": "1",
        "metadata_warning": "",
    }
    with (
        patch("scripts.run_real_audio_demo_suite._input_metadata", return_value=metadata),
        patch("scripts.run_real_audio_demo_suite.run_audio_task", side_effect=_fake_task_run) as task_mock,
    ):
        result = run_real_audio_demo_suite(manifest_path=manifest, output_root=output_root)

    task_mock.assert_called_once()
    assert task_mock.call_args.kwargs["task"] == CLEAN_VOICE
    rows = _read_rows(result / "report.csv")
    assert list(rows[0].keys()) == REPORT_COLUMNS
    assert rows[0]["status"] == "success"
    assert rows[0]["selected_task"] == CLEAN_VOICE
    assert rows[0]["output_files"].endswith("clean_voice.wav")
    assert rows[0]["description"] == "Real speech cleanup demo"
    assert rows[0]["notes"] == "manual task"
    assert rows[0]["input_duration_sec"] == "3.500000"
    assert rows[0]["input_sample_rate_hz"] == "16000"
    assert rows[0]["input_channels"] == "1"
    assert rows[0]["processing_time_sec"]
    assert rows[0]["auto_expectation_result"] == "not_applicable"
    case_summary = json.loads((result / "speech-case" / "case_summary.json").read_text(encoding="utf-8"))
    assert case_summary["status"] == "success"
    assert case_summary["output_files"]
    markdown = (result / "report.md").read_text(encoding="utf-8")
    assert "Total cases: 1" in markdown
    assert "clean_voice.wav" in markdown


def test_real_audio_demo_suite_runs_manual_target_noise_with_checkpoint(tmp_path: Path) -> None:
    input_path = tmp_path / "speech-with-siren.wav"
    checkpoint = tmp_path / "target-noise.pt"
    input_path.write_bytes(b"audio")
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "input_path": str(input_path),
                "case_id": "manual-target-noise",
                "expected_task": TARGET_NOISE_SUPPRESSION,
                "description": "Experimental manual candidate",
                "notes": "",
            }
        ],
    )

    with (
        patch("scripts.run_real_audio_demo_suite._input_metadata", return_value={}),
        patch("scripts.run_real_audio_demo_suite.run_audio_task", side_effect=_fake_task_run) as task_mock,
    ):
        result = run_real_audio_demo_suite(
            manifest_path=manifest,
            output_root=tmp_path / "outputs",
            target_noise_checkpoint=checkpoint,
        )

    task_mock.assert_called_once()
    assert task_mock.call_args.kwargs["task"] == TARGET_NOISE_SUPPRESSION
    assert task_mock.call_args.kwargs["target_noise_checkpoint"] == checkpoint
    row = _read_rows(result / "report.csv")[0]
    assert row["status"] == "success"
    assert row["selected_task"] == TARGET_NOISE_SUPPRESSION
    assert row["auto_expectation_result"] == "not_applicable"


def test_real_audio_demo_suite_records_missing_input_and_continues(tmp_path: Path) -> None:
    existing_input = tmp_path / "existing.wav"
    existing_input.write_bytes(b"audio")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "input_path": str(tmp_path / "missing.wav"),
                "case_id": "missing-case",
                "expected_task": CLEAN_VOICE,
                "description": "",
                "notes": "",
            },
            {
                "input_path": str(existing_input),
                "case_id": "success-case",
                "expected_task": CLEAN_VOICE,
                "description": "",
                "notes": "",
            },
        ],
    )

    with (
        patch(
            "scripts.run_real_audio_demo_suite._input_metadata",
            return_value={
                "input_duration_sec": "",
                "input_sample_rate_hz": "",
                "input_channels": "",
                "metadata_warning": "Input metadata unavailable.",
            },
        ),
        patch("scripts.run_real_audio_demo_suite.run_audio_task", side_effect=_fake_task_run) as task_mock,
    ):
        result = run_real_audio_demo_suite(manifest_path=manifest, output_root=tmp_path / "outputs")

    assert task_mock.call_count == 1
    rows = {row["case_id"]: row for row in _read_rows(result / "report.csv")}
    assert rows["missing-case"]["status"] == "failed"
    assert "Input file not found" in rows["missing-case"]["error"]
    assert rows["missing-case"]["processing_time_sec"]
    assert rows["success-case"]["status"] == "success"
    assert rows["success-case"]["processing_time_sec"]


def test_real_audio_demo_suite_auto_abstain_records_manual_required(tmp_path: Path) -> None:
    input_path = tmp_path / "ambiguous.wav"
    checkpoint = tmp_path / "router.pt"
    input_path.write_bytes(b"audio")
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "input_path": str(input_path),
                "case_id": "auto-abstain",
                "expected_task": "auto",
                "description": "Ambiguous input",
                "notes": "",
            }
        ],
    )
    router_result = {
        "predicted_label": "music_with_vocals",
        "confidence": 0.70,
        "accepted": False,
        "route_target": "manual_required",
        "engine_target": "none",
        "recommended_task": None,
        "decision_reason": "low_confidence",
    }

    with (
        patch(
            "scripts.run_real_audio_demo_suite._input_metadata",
            return_value={
                "input_duration_sec": "",
                "input_sample_rate_hz": "",
                "input_channels": "",
                "metadata_warning": "",
            },
        ),
        patch("scripts.run_real_audio_demo_suite._run_router", return_value=router_result),
        patch("scripts.run_real_audio_demo_suite.run_audio_task") as task_mock,
    ):
        result = run_real_audio_demo_suite(
            manifest_path=manifest,
            output_root=tmp_path / "outputs",
            router_checkpoint=checkpoint,
        )

    task_mock.assert_not_called()
    row = _read_rows(result / "report.csv")[0]
    assert row["status"] == "abstained"
    assert row["route_target"] == "manual_required"
    assert row["router_accepted"] == "false"
    assert row["selected_task"] == ""
    assert "low_confidence" in row["error"]
    assert row["processing_time_sec"]
    assert row["auto_expectation_result"] == "unspecified"


def test_input_metadata_failure_is_non_fatal(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp3"
    input_path.write_bytes(b"not-real-audio")

    with patch("scripts.run_real_audio_demo_suite.subprocess.run", side_effect=FileNotFoundError):
        metadata = _input_metadata(input_path)

    assert metadata["input_duration_sec"] == ""
    assert metadata["input_sample_rate_hz"] == ""
    assert metadata["input_channels"] == ""
    assert metadata["metadata_warning"] == "Input metadata unavailable."


def test_real_audio_demo_suite_carries_human_listening_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "speech.wav"
    input_path.write_bytes(b"audio")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "input_path": str(input_path),
                "case_id": "rated-case",
                "expected_task": CLEAN_VOICE,
                "description": "Listening case",
                "notes": "source note",
                "human_rating": "4",
                "human_notes": "Speech is clearer.",
                "expected_auto_behavior": "manual clean voice processing",
                "acceptable_routes": "",
                "dangerous_routes": "",
                "objective_metric_name": "example_metric",
                "objective_score_before": "0.42",
                "objective_score_after": "0.61",
            }
        ],
        fieldnames=[
            "input_path",
            "case_id",
            "expected_task",
            "description",
            "notes",
            "human_rating",
            "human_notes",
            "expected_auto_behavior",
            "acceptable_routes",
            "dangerous_routes",
            "objective_metric_name",
            "objective_score_before",
            "objective_score_after",
        ],
    )

    with (
        patch(
            "scripts.run_real_audio_demo_suite._input_metadata",
            return_value={
                "input_duration_sec": "2.000000",
                "input_sample_rate_hz": "48000",
                "input_channels": "2",
                "metadata_warning": "",
            },
        ),
        patch("scripts.run_real_audio_demo_suite.run_audio_task", side_effect=_fake_task_run),
    ):
        result = run_real_audio_demo_suite(manifest_path=manifest, output_root=tmp_path / "outputs")

    row = _read_rows(result / "report.csv")[0]
    assert row["human_rating"] == "4"
    assert row["human_notes"] == "Speech is clearer."
    assert row["expected_auto_behavior"] == "manual clean voice processing"
    assert row["objective_metric_name"] == "example_metric"
    assert row["objective_score_before"] == "0.42"
    assert row["objective_score_after"] == "0.61"
    assert row["auto_expectation_result"] == "not_applicable"
    case_summary = json.loads((result / "rated-case" / "case_summary.json").read_text(encoding="utf-8"))
    assert case_summary["human_rating"] == "4"
    assert case_summary["human_notes"] == "Speech is clearer."
    assert case_summary["expected_auto_behavior"] == "manual clean voice processing"
    assert case_summary["objective_metric_name"] == "example_metric"


def test_real_audio_demo_suite_auto_target_noise_abstains(tmp_path: Path) -> None:
    input_path = tmp_path / "target-noise.wav"
    checkpoint = tmp_path / "router.pt"
    input_path.write_bytes(b"audio")
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "input_path": str(input_path),
                "case_id": "auto-target-noise",
                "expected_task": "auto",
                "description": "",
                "notes": "",
                "expected_auto_behavior": "Do not auto-run experimental target suppression.",
                "acceptable_routes": "manual_required",
                "dangerous_routes": "target_noise_suppression",
            }
        ],
        fieldnames=[
            "input_path",
            "case_id",
            "expected_task",
            "description",
            "notes",
            "expected_auto_behavior",
            "acceptable_routes",
            "dangerous_routes",
        ],
    )
    router_result = {
        "predicted_label": "speech_target_noise",
        "confidence": 0.96,
        "accepted": True,
        "route_target": TARGET_NOISE_SUPPRESSION,
        "engine_target": "target_noise_suppressor",
        "recommended_task": TARGET_NOISE_SUPPRESSION,
        "decision_reason": "accepted_router_prediction",
    }

    with (
        patch("scripts.run_real_audio_demo_suite._input_metadata", return_value={}),
        patch("scripts.run_real_audio_demo_suite._run_router", return_value=router_result),
        patch("scripts.run_real_audio_demo_suite.run_audio_task") as task_mock,
    ):
        result = run_real_audio_demo_suite(
            manifest_path=manifest,
            output_root=tmp_path / "outputs",
            router_checkpoint=checkpoint,
        )

    task_mock.assert_not_called()
    row = _read_rows(result / "report.csv")[0]
    assert row["status"] == "abstained"
    assert row["route_target"] == "manual_required"
    assert row["selected_task"] == ""
    assert "Blocked dangerous auto route before execution" in row["error"]
    assert TARGET_NOISE_SUPPRESSION in row["error"]
    assert row["processing_time_sec"]
    assert row["auto_expectation_result"] == "dangerous_failure"


def test_real_audio_demo_suite_auto_target_noise_remains_manual_only(tmp_path: Path) -> None:
    input_path = tmp_path / "target-noise.wav"
    checkpoint = tmp_path / "router.pt"
    input_path.write_bytes(b"audio")
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "input_path": str(input_path),
                "case_id": "auto-target-noise-safe-manifest",
                "expected_task": "auto",
                "description": "",
                "notes": "",
                "expected_auto_behavior": "Require manual target suppression.",
                "acceptable_routes": "manual_required",
                "dangerous_routes": "",
            }
        ],
        fieldnames=[
            "input_path",
            "case_id",
            "expected_task",
            "description",
            "notes",
            "expected_auto_behavior",
            "acceptable_routes",
            "dangerous_routes",
        ],
    )
    router_result = {
        "predicted_label": "speech_target_noise",
        "confidence": 0.96,
        "accepted": True,
        "route_target": TARGET_NOISE_SUPPRESSION,
        "engine_target": "target_noise_suppressor",
        "recommended_task": TARGET_NOISE_SUPPRESSION,
        "decision_reason": "accepted_router_prediction",
    }

    with (
        patch("scripts.run_real_audio_demo_suite._input_metadata", return_value={}),
        patch("scripts.run_real_audio_demo_suite._run_router", return_value=router_result),
        patch("scripts.run_real_audio_demo_suite.run_audio_task") as task_mock,
    ):
        result = run_real_audio_demo_suite(
            manifest_path=manifest,
            output_root=tmp_path / "outputs",
            router_checkpoint=checkpoint,
        )

    task_mock.assert_not_called()
    row = _read_rows(result / "report.csv")[0]
    assert row["status"] == "abstained"
    assert row["route_target"] == "manual_required"
    assert row["selected_task"] == ""
    assert "experimental auto target_noise_suppression is disabled" in row["error"].lower()
    assert row["auto_expectation_result"] == "pass"


def test_real_audio_demo_suite_auto_no_process_does_not_run_engine(tmp_path: Path) -> None:
    input_path = tmp_path / "clean.wav"
    checkpoint = tmp_path / "router.pt"
    input_path.write_bytes(b"audio")
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "input_path": str(input_path),
                "case_id": "auto-no-process",
                "expected_task": "auto",
                "description": "",
                "notes": "",
                "expected_auto_behavior": "No processing for clean speech.",
                "acceptable_routes": "no_process;manual_required",
                "dangerous_routes": "clean_voice;target_noise_suppression",
            }
        ],
        fieldnames=[
            "input_path",
            "case_id",
            "expected_task",
            "description",
            "notes",
            "expected_auto_behavior",
            "acceptable_routes",
            "dangerous_routes",
        ],
    )
    router_result = {
        "predicted_label": "speech_clean",
        "confidence": 0.97,
        "accepted": True,
        "route_target": "no_process",
        "engine_target": "none",
        "recommended_task": None,
        "decision_reason": "accepted_router_prediction",
    }

    with (
        patch("scripts.run_real_audio_demo_suite._input_metadata", return_value={}),
        patch("scripts.run_real_audio_demo_suite._run_router", return_value=router_result),
        patch("scripts.run_real_audio_demo_suite.run_audio_task") as task_mock,
    ):
        result = run_real_audio_demo_suite(
            manifest_path=manifest,
            output_root=tmp_path / "outputs",
            router_checkpoint=checkpoint,
        )

    task_mock.assert_not_called()
    row = _read_rows(result / "report.csv")[0]
    assert row["status"] == "no_process"
    assert row["selected_task"] == "no_process"
    assert row["output_files"] == ""
    assert row["processing_time_sec"]
    assert row["auto_expectation_result"] == "pass"
    case_summary = json.loads((result / "auto-no-process" / "case_summary.json").read_text(encoding="utf-8"))
    assert case_summary["acceptable_routes"] == "no_process;manual_required"
    assert case_summary["dangerous_routes"] == "clean_voice;target_noise_suppression"


def test_real_audio_demo_suite_auto_dangerous_route_is_flagged(tmp_path: Path) -> None:
    input_path = tmp_path / "environment.wav"
    checkpoint = tmp_path / "router.pt"
    input_path.write_bytes(b"audio")
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "input_path": str(input_path),
                "case_id": "auto-dangerous",
                "expected_task": "auto",
                "description": "Environment-only red-team case",
                "notes": "",
                "expected_auto_behavior": "Abstain or classify out of scope.",
                "acceptable_routes": "manual_required;out_of_scope",
                "dangerous_routes": "clean_voice;target_noise_suppression",
            }
        ],
        fieldnames=[
            "input_path",
            "case_id",
            "expected_task",
            "description",
            "notes",
            "expected_auto_behavior",
            "acceptable_routes",
            "dangerous_routes",
        ],
    )
    router_result = {
        "predicted_label": "speech_noisy_general",
        "confidence": 0.96,
        "accepted": True,
        "route_target": CLEAN_VOICE,
        "engine_target": "deepfilternet",
        "recommended_task": CLEAN_VOICE,
        "decision_reason": "accepted_router_prediction",
    }

    with (
        patch("scripts.run_real_audio_demo_suite._input_metadata", return_value={}),
        patch("scripts.run_real_audio_demo_suite._run_router", return_value=router_result),
        patch("scripts.run_real_audio_demo_suite.run_audio_task") as task_mock,
    ):
        result = run_real_audio_demo_suite(
            manifest_path=manifest,
            output_root=tmp_path / "outputs",
            router_checkpoint=checkpoint,
        )

    task_mock.assert_not_called()
    row = _read_rows(result / "report.csv")[0]
    assert row["status"] == "abstained"
    assert row["selected_task"] == ""
    assert row["route_target"] == "manual_required"
    assert row["engine_target"] == "none"
    assert row["auto_expectation_result"] == "dangerous_failure"
    assert "Blocked dangerous auto route before execution" in row["error"]
    assert CLEAN_VOICE in row["error"]
