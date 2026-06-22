"""Smoke tests for the FastAPI audio controller surface."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app.api_server as api_server
from src.router.task_registry import CLEAN_VOICE, EXTRACT_VOCALS, TARGET_NOISE_SUPPRESSION


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api_server, "UPLOAD_ROOT", tmp_path / "uploads")
    monkeypatch.setattr(api_server, "RUN_ROOT", tmp_path / "runs")
    return TestClient(api_server.api)


def _feature_row() -> dict[str, str]:
    return {
        "status": "success",
        "duration_sec": "2.000000",
        "rms_energy": "0.0500000000",
        "zero_crossing_rate": "0.0200000000",
        "spectral_centroid_hz": "1800.000000",
        "spectral_bandwidth_hz": "2500.000000",
        "error": "",
    }


def _router_result() -> dict[str, object]:
    return {
        "router_status": "disabled",
        "predicted_label": "",
        "confidence": None,
        "accepted": False,
        "route_target": "manual_required",
        "recommended_task": None,
        "decision_reason": "router_checkpoint_not_configured",
        "warnings": [],
    }


def _analyze(
    client: TestClient,
    goal: str = "improve_speech_clarity",
    router_result: dict[str, object] | None = None,
) -> dict[str, object]:
    with (
        patch("app.api_server._extract_feature_row", return_value=_feature_row()),
        patch("app.api_server._optional_router_result", return_value=router_result or _router_result()),
    ):
        response = client.post(
            "/api/analyze",
            data={"goal": goal},
            files={"file": ("speech.wav", b"audio", "audio/wav")},
        )
    assert response.status_code == 200
    return response.json()


def test_api_health(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}


def test_api_analyze_returns_controller_router_and_features(client: TestClient) -> None:
    payload = _analyze(client)

    assert payload["filename"] == "speech.wav"
    assert payload["controller"]["decision"] == "run_task"
    assert payload["controller"]["recommended_task"] == CLEAN_VOICE
    assert payload["controller"]["workflow_kind"] == "speech_cleanup"
    assert payload["controller"]["recommended_tasks"] == [CLEAN_VOICE]
    assert payload["router"]["status"] == "disabled"
    assert payload["features"]["duration_sec"] == "2.000000"


def test_api_run_blocks_target_noise_suppression(client: TestClient) -> None:
    analyzed = _analyze(client)

    response = client.post(
        "/api/run",
        json={"file_id": analyzed["file_id"], "task": TARGET_NOISE_SUPPRESSION},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "blocked"
    assert "target_noise_suppression_manual_only" in payload["controller"]["blocked_reasons"]


def test_api_run_clean_voice_and_serve_output(client: TestClient) -> None:
    analyzed = _analyze(client)

    def mock_run_audio_task(**kwargs: object) -> Path:
        run_dir = Path(kwargs["output_root"]) / CLEAN_VOICE / "mock-run"
        output_path = run_dir / "speech.restored.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"restored")
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
                    "task": CLEAN_VOICE,
                    "engine": "deepfilternet",
                    "input_path": kwargs["input_path"],
                    "input_type": "audio",
                    "status": "success",
                    "runtime_sec": "0.1",
                    "primary_output_path": output_path,
                    "error": "",
                }
            )
        return run_dir

    with patch("app.api_server.run_audio_task", side_effect=mock_run_audio_task):
        response = client.post(
            "/api/run",
            json={"file_id": analyzed["file_id"], "task": CLEAN_VOICE},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["download_url"].endswith("?kind=output")
    output_response = client.get(payload["download_url"])
    assert output_response.status_code == 200
    assert output_response.content == b"restored"


def test_api_run_plan_speech_cleanup_returns_labeled_output(client: TestClient) -> None:
    analyzed = _analyze(client)

    def mock_run_audio_task(**kwargs: object) -> Path:
        run_dir = Path(kwargs["output_root"]) / CLEAN_VOICE / "plan-run"
        output_path = run_dir / "speech.restored.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"enhanced")
        _write_task_summary(run_dir, CLEAN_VOICE, output_path)
        return run_dir

    with patch("app.api_server.run_audio_task", side_effect=mock_run_audio_task) as run_mock:
        response = client.post("/api/run-plan", json={"file_id": analyzed["file_id"]})

    run_mock.assert_called_once()
    assert run_mock.call_args.kwargs["task"] == CLEAN_VOICE
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["controller"]["workflow_kind"] == "speech_cleanup"
    assert [output["label"] for output in payload["outputs"]] == ["enhanced_speech"]
    served = client.get(payload["outputs"][0]["download_url"])
    assert served.status_code == 200
    assert served.content == b"enhanced"


def test_api_run_plan_music_package_returns_vocals_and_instrumental(client: TestClient) -> None:
    router_result = {
        "router_status": "enabled",
        "predicted_label": "music_with_vocals",
        "confidence": 0.97,
        "accepted": True,
        "route_target": "manual_required",
        "recommended_task": EXTRACT_VOCALS,
        "decision_reason": "accepted_router_prediction",
        "warnings": [],
    }
    analyzed = _analyze(client, goal="auto", router_result=router_result)

    def mock_run_audio_task(**kwargs: object) -> Path:
        run_dir = Path(kwargs["output_root"]) / EXTRACT_VOCALS / "plan-run"
        vocals_path = run_dir / "nested" / "vocals.wav"
        no_vocals_path = run_dir / "nested" / "no_vocals.wav"
        vocals_path.parent.mkdir(parents=True, exist_ok=True)
        vocals_path.write_bytes(b"vocals")
        no_vocals_path.write_bytes(b"instrumental")
        _write_task_summary(run_dir, EXTRACT_VOCALS, vocals_path)
        with (vocals_path.parent / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
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
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "run_id": "nested",
                    "input_path": kwargs["input_path"],
                    "task": EXTRACT_VOCALS,
                    "engine": "demucs",
                    "status": "success",
                    "runtime_sec": "0.1",
                    "primary_output_label": "vocals",
                    "primary_output_path": vocals_path,
                    "vocals_path": vocals_path,
                    "no_vocals_path": no_vocals_path,
                    "error": "",
                }
            )
        return run_dir

    with patch("app.api_server.run_audio_task", side_effect=mock_run_audio_task) as run_mock:
        response = client.post("/api/run-plan", json={"file_id": analyzed["file_id"]})

    run_mock.assert_called_once()
    assert run_mock.call_args.kwargs["task"] == EXTRACT_VOCALS
    payload = response.json()
    assert payload["controller"]["workflow_kind"] == "music_separation_package"
    assert [output["label"] for output in payload["outputs"]] == ["vocals", "no_vocals"]
    assert client.get(payload["outputs"][0]["download_url"]).content == b"vocals"
    assert client.get(payload["outputs"][1]["download_url"]).content == b"instrumental"


def test_api_run_plan_target_noise_goal_remains_blocked(client: TestClient) -> None:
    analyzed = _analyze(client, goal="reduce_target_noise")

    with patch("app.api_server.run_audio_task") as run_mock:
        response = client.post("/api/run-plan", json={"file_id": analyzed["file_id"]})

    run_mock.assert_not_called()
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["controller"]["workflow_kind"] == "target_noise_guard"
    assert "target_noise_suppression_manual_only" in payload["controller"]["blocked_reasons"]


def test_api_run_plan_blocks_guarded_auto_abstention(client: TestClient) -> None:
    router_result = {
        "router_status": "enabled",
        "predicted_label": "speech_target_noise",
        "confidence": 0.96,
        "accepted": False,
        "route_target": "manual_required",
        "recommended_task": None,
        "decision_reason": "speech_cleanup_blocked_by_speech_gate",
        "warnings": ["guarded_router_safe_abstain"],
        "guard_applied": True,
        "final_workflow": "safe_abstain",
        "final_accepted": False,
        "speech_gate_label": "non_speech",
        "speech_gate_confidence": 0.94,
    }
    analyzed = _analyze(client, goal="auto", router_result=router_result)

    with patch("app.api_server.run_audio_task") as run_mock:
        response = client.post(
            "/api/run-plan",
            json={"file_id": analyzed["file_id"]},
        )

    run_mock.assert_not_called()
    assert response.json()["status"] == "blocked"
    assert analyzed["router"]["guard_applied"] is True
    assert analyzed["router"]["final_workflow"] == "safe_abstain"


def test_api_run_plan_allows_guarded_accepted_noisy_speech(
    client: TestClient,
) -> None:
    router_result = {
        "router_status": "enabled",
        "predicted_label": "speech_noisy_general",
        "confidence": 0.95,
        "accepted": True,
        "route_target": CLEAN_VOICE,
        "recommended_task": CLEAN_VOICE,
        "decision_reason": "accepted",
        "warnings": [],
        "guard_applied": False,
        "final_workflow": "speech_cleanup",
        "final_accepted": True,
        "speech_gate_label": "speech_present",
        "speech_gate_confidence": 0.93,
    }
    analyzed = _analyze(client, goal="auto", router_result=router_result)

    def mock_run_audio_task(**kwargs: object) -> Path:
        run_dir = Path(kwargs["output_root"]) / CLEAN_VOICE / "guarded-run"
        output_path = run_dir / "speech.restored.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"enhanced")
        _write_task_summary(run_dir, CLEAN_VOICE, output_path)
        return run_dir

    with patch(
        "app.api_server.run_audio_task",
        side_effect=mock_run_audio_task,
    ) as run_mock:
        response = client.post(
            "/api/run-plan",
            json={"file_id": analyzed["file_id"]},
        )

    run_mock.assert_called_once()
    assert response.json()["status"] == "success"
    assert response.json()["task"] == CLEAN_VOICE


def _write_task_summary(run_dir: Path, task: str, output_path: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
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
                "run_id": "plan-run",
                "task": task,
                "engine": "deepfilternet" if task == CLEAN_VOICE else "demucs",
                "input_path": "",
                "input_type": "audio",
                "status": "success",
                "runtime_sec": "0.1",
                "primary_output_path": output_path,
                "error": "",
            }
        )
