"""Smoke tests for the FastAPI audio controller surface."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app.api_server as api_server
from src.router.task_registry import CLEAN_VOICE, TARGET_NOISE_SUPPRESSION


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


def _analyze(client: TestClient, goal: str = "improve_speech_clarity") -> dict[str, object]:
    with (
        patch("app.api_server._extract_feature_row", return_value=_feature_row()),
        patch("app.api_server._optional_router_result", return_value=_router_result()),
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
