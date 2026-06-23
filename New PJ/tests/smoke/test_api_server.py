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
    for env_var in (
        api_server.DETECTION_FUSION_HEAD_CHECKPOINT_ENV_VAR,
        api_server.DETECTION_FUSION_ROUTER_CHECKPOINT_ENV_VAR,
        api_server.DETECTION_FUSION_SPEECH_GATE_CHECKPOINT_ENV_VAR,
        api_server.DETECTION_FUSION_ROUTE_MODE_ENV_VAR,
    ):
        monkeypatch.delenv(env_var, raising=False)
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
    assert payload["experimental_detection_fusion"] == {
        "enabled": False,
        "reason": "detection_fusion_checkpoint_not_configured",
        "missing_configuration": [
            api_server.DETECTION_FUSION_HEAD_CHECKPOINT_ENV_VAR,
            api_server.DETECTION_FUSION_ROUTER_CHECKPOINT_ENV_VAR,
            api_server.DETECTION_FUSION_SPEECH_GATE_CHECKPOINT_ENV_VAR,
        ],
    }


def test_api_analyze_exposes_experimental_detection_fusion(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoints = {
        api_server.DETECTION_FUSION_HEAD_CHECKPOINT_ENV_VAR: (
            tmp_path / "heads.pt"
        ),
        api_server.DETECTION_FUSION_ROUTER_CHECKPOINT_ENV_VAR: (
            tmp_path / "router.pt"
        ),
        api_server.DETECTION_FUSION_SPEECH_GATE_CHECKPOINT_ENV_VAR: (
            tmp_path / "gate.pt"
        ),
    }
    for env_var, checkpoint in checkpoints.items():
        checkpoint.write_bytes(b"fixture")
        monkeypatch.setenv(env_var, str(checkpoint))

    fusion_result = {
        "fusion_label": "speech_target_noise",
        "recommended_workflow": "speech_target_noise_cleanup",
        "detection_scores": {
            "speech_present": 0.91,
            "music_present": 0.04,
            "target_event_present": 0.88,
        },
        "review_recommended": True,
        "review_reasons": ["router_fusion_conflict"],
        "abstain": False,
        "abstain_reason": "",
    }
    with patch(
        "app.api_server.run_detection_fusion",
        return_value=fusion_result,
    ) as fusion_mock:
        payload = _analyze(client)

    fusion_mock.assert_called_once()
    assert payload["controller"]["recommended_task"] == CLEAN_VOICE
    assert payload["router"]["status"] == "disabled"
    assert payload["experimental_detection_fusion"] == {
        "enabled": True,
        **fusion_result,
    }


def _fusion_payload(
    fusion_label: str,
    *,
    speech: float,
    music: float,
    target: float,
    workflow: str,
) -> dict[str, object]:
    return {
        "enabled": True,
        "fusion_label": fusion_label,
        "recommended_workflow": workflow,
        "detection_scores": {
            "speech_present": speech,
            "music_present": music,
            "target_event_present": target,
        },
        "review_recommended": False,
        "review_reasons": [],
        "abstain": False,
        "abstain_reason": "",
    }


def _assert_fusion_summary(
    payload: dict[str, object],
    *,
    decision_source: str,
    fusion_label: str,
    workflow: str,
    speech: float,
    music: float,
    target: float,
    active: bool,
) -> None:
    assert payload["detection_fusion_summary"] == {
        "controller": {"decision_source": decision_source},
        "fusion_label": fusion_label,
        "recommended_workflow": workflow,
        "scores": {
            "speech_present": speech,
            "music_present": music,
            "target_event_present": target,
        },
        "review_recommended": False,
        "review_reasons": [],
        "route_mode": "active" if active else "off",
        "active_route_mode": active,
    }


def test_detection_fusion_route_mode_off_preserves_primary_abstain(
    client: TestClient,
) -> None:
    fusion = _fusion_payload(
        "speech_present_general",
        speech=0.95,
        music=0.05,
        target=0.10,
        workflow="speech_cleanup",
    )
    with patch(
        "app.api_server._experimental_detection_fusion",
        return_value=fusion,
    ):
        payload = _analyze(client, goal="auto")

    assert payload["controller"]["decision"] == "manual_required"
    assert payload["controller"]["decision_source"] == "primary_router"
    _assert_fusion_summary(
        payload,
        decision_source="primary_router",
        fusion_label="speech_present_general",
        workflow="speech_cleanup",
        speech=0.95,
        music=0.05,
        target=0.10,
        active=False,
    )


@pytest.mark.parametrize(
    (
        "fusion",
        "expected_workflow",
        "expected_task",
    ),
    [
        (
            _fusion_payload(
                "speech_target_noise",
                speech=0.91,
                music=0.05,
                target=0.95,
                workflow="speech_target_noise_cleanup",
            ),
            "speech_target_noise_cleanup",
            CLEAN_VOICE,
        ),
        (
            _fusion_payload(
                "speech_present_general",
                speech=0.92,
                music=0.10,
                target=0.20,
                workflow="speech_cleanup",
            ),
            "speech_cleanup",
            CLEAN_VOICE,
        ),
        (
            _fusion_payload(
                "music_with_vocals",
                speech=0.20,
                music=0.96,
                target=0.10,
                workflow="music_separation_package",
            ),
            "music_separation_package",
            EXTRACT_VOCALS,
        ),
    ],
)
def test_active_detection_fusion_promotes_strong_abstained_evidence(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fusion: dict[str, object],
    expected_workflow: str,
    expected_task: str,
) -> None:
    monkeypatch.setenv(api_server.DETECTION_FUSION_ROUTE_MODE_ENV_VAR, "active")
    with patch(
        "app.api_server._experimental_detection_fusion",
        return_value=fusion,
    ):
        payload = _analyze(client, goal="auto")

    assert payload["controller"]["decision"] == "run_task"
    assert payload["controller"]["workflow_kind"] == expected_workflow
    assert payload["controller"]["recommended_task"] == expected_task
    assert payload["controller"]["decision_source"] == "detection_fusion_active"
    assert "detection_fusion_active" in payload["controller"]["warnings"]
    metadata = api_server._read_metadata(str(payload["file_id"]))
    replayed_plan = api_server._plan_from_metadata(
        metadata,
        Path(str(metadata["input_path"])),
    )
    assert replayed_plan.action == "run_task"
    assert replayed_plan.workflow_kind == expected_workflow
    assert replayed_plan.recommended_task == expected_task


def test_active_detection_fusion_keeps_accepted_primary_router_plan(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(api_server.DETECTION_FUSION_ROUTE_MODE_ENV_VAR, "active")
    accepted_router = {
        "router_status": "enabled",
        "predicted_label": "speech_noisy_general",
        "confidence": 0.95,
        "accepted": True,
        "route_target": CLEAN_VOICE,
        "recommended_task": CLEAN_VOICE,
        "decision_reason": "accepted",
        "warnings": [],
    }
    conflicting_fusion = _fusion_payload(
        "music_with_vocals",
        speech=0.10,
        music=0.99,
        target=0.10,
        workflow="music_separation_package",
    )
    with patch(
        "app.api_server._experimental_detection_fusion",
        return_value=conflicting_fusion,
    ):
        payload = _analyze(
            client,
            goal="auto",
            router_result=accepted_router,
        )

    assert payload["controller"]["decision"] == "run_task"
    assert payload["controller"]["workflow_kind"] == "speech_cleanup"
    assert payload["controller"]["recommended_task"] == CLEAN_VOICE
    assert payload["controller"]["decision_source"] == "primary_router"


@pytest.mark.parametrize(
    "fusion",
    [
        _fusion_payload(
            "environment_only",
            speech=0.10,
            music=0.10,
            target=0.95,
            workflow="environment_event_analysis",
        ),
        _fusion_payload(
            "speech_target_noise",
            speech=0.79,
            music=0.05,
            target=0.89,
            workflow="speech_target_noise_cleanup",
        ),
    ],
)
def test_active_detection_fusion_does_not_promote_environment_or_weak_scores(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fusion: dict[str, object],
) -> None:
    monkeypatch.setenv(api_server.DETECTION_FUSION_ROUTE_MODE_ENV_VAR, "active")
    with patch(
        "app.api_server._experimental_detection_fusion",
        return_value=fusion,
    ):
        payload = _analyze(client, goal="auto")

    assert payload["controller"]["decision"] == "manual_required"
    assert payload["controller"]["workflow_kind"] == "safe_abstain"
    assert payload["controller"]["decision_source"] == "primary_router"


def test_active_detection_fusion_demo_smoke(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(api_server.DETECTION_FUSION_ROUTE_MODE_ENV_VAR, "active")
    speech_fusion = _fusion_payload(
        "speech_present_general",
        speech=0.94,
        music=0.10,
        target=0.20,
        workflow="speech_cleanup",
    )
    with patch(
        "app.api_server._experimental_detection_fusion",
        return_value=speech_fusion,
    ):
        speech_analyzed = _analyze(client, goal="auto")

    assert speech_analyzed["controller"]["decision"] == "run_task"
    assert speech_analyzed["controller"]["recommended_task"] == CLEAN_VOICE
    _assert_fusion_summary(
        speech_analyzed,
        decision_source="detection_fusion_active",
        fusion_label="speech_present_general",
        workflow="speech_cleanup",
        speech=0.94,
        music=0.10,
        target=0.20,
        active=True,
    )

    target_fusion = _fusion_payload(
        "speech_target_noise",
        speech=0.93,
        music=0.05,
        target=0.96,
        workflow="speech_target_noise_cleanup",
    )
    with patch(
        "app.api_server._experimental_detection_fusion",
        return_value=target_fusion,
    ):
        target_analyzed = _analyze(client, goal="auto")

    assert target_analyzed["controller"]["workflow_kind"] == (
        "speech_target_noise_cleanup"
    )
    _assert_fusion_summary(
        target_analyzed,
        decision_source="detection_fusion_active",
        fusion_label="speech_target_noise",
        workflow="speech_target_noise_cleanup",
        speech=0.93,
        music=0.05,
        target=0.96,
        active=True,
    )

    def mock_run_audio_task(**kwargs: object) -> Path:
        run_dir = Path(kwargs["output_root"]) / CLEAN_VOICE / "demo-smoke-run"
        output_path = run_dir / "speech.restored.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"enhanced")
        _write_task_summary(run_dir, CLEAN_VOICE, output_path)
        return run_dir

    with patch(
        "app.api_server.run_audio_task",
        side_effect=mock_run_audio_task,
    ):
        run_response = client.post(
            "/api/run",
            json={
                "file_id": target_analyzed["file_id"],
                "task": CLEAN_VOICE,
            },
        )

    assert [artifact["label"] for artifact in run_response.json()["outputs"]] == [
        "enhanced_speech",
        "target_noise_report_json",
        "target_noise_report_csv",
    ]

    environment_fusion = _fusion_payload(
        "environment_only",
        speech=0.10,
        music=0.10,
        target=0.95,
        workflow="environment_event_analysis",
    )
    with patch(
        "app.api_server._experimental_detection_fusion",
        return_value=environment_fusion,
    ):
        environment_analyzed = _analyze(client, goal="auto")

    with patch("app.api_server.run_audio_task") as run_mock:
        environment_run = client.post(
            "/api/run-plan",
            json={"file_id": environment_analyzed["file_id"]},
        )

    run_mock.assert_not_called()
    assert environment_run.json()["status"] == "blocked"


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
    assert payload["controller"]["workflow_kind"] == "speech_cleanup"
    assert payload["controller"]["decision_source"] == "planner"
    assert payload["download_url"].endswith("?kind=output")
    output_response = client.get(payload["download_url"])
    assert output_response.status_code == 200
    assert output_response.content == b"restored"


def test_api_run_preserves_active_fusion_target_noise_cleanup(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(api_server.DETECTION_FUSION_ROUTE_MODE_ENV_VAR, "active")
    fusion = _fusion_payload(
        "speech_target_noise",
        speech=0.93,
        music=0.05,
        target=0.96,
        workflow="speech_target_noise_cleanup",
    )
    with patch(
        "app.api_server._experimental_detection_fusion",
        return_value=fusion,
    ):
        analyzed = _analyze(client, goal="auto")

    metadata = api_server._read_metadata(str(analyzed["file_id"]))
    assert metadata["controller"] == analyzed["controller"]
    assert metadata["experimental_detection_fusion"] == fusion

    def mock_run_audio_task(**kwargs: object) -> Path:
        run_dir = Path(kwargs["output_root"]) / CLEAN_VOICE / "fusion-target-run"
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
            "/api/run",
            json={"file_id": analyzed["file_id"], "task": CLEAN_VOICE},
        )

    run_mock.assert_called_once()
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["controller"]["decision_source"] == "detection_fusion_active"
    assert payload["controller"]["workflow_kind"] == (
        "speech_target_noise_cleanup"
    )
    assert payload["controller"]["recommended_task"] == CLEAN_VOICE
    assert payload["controller"]["expected_outputs"] == [
        "enhanced_speech",
        "target_noise_report",
    ]
    assert "target_noise_suppression_fallback" in payload["controller"]["warnings"]
    assert [artifact["label"] for artifact in payload["outputs"]] == [
        "enhanced_speech",
        "target_noise_report_json",
        "target_noise_report_csv",
    ]
    assert client.get(payload["outputs"][0]["download_url"]).content == b"enhanced"
    assert client.get(payload["outputs"][1]["download_url"]).status_code == 200
    assert client.get(payload["outputs"][2]["download_url"]).status_code == 200


def test_api_run_preserves_active_fusion_general_speech_cleanup(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(api_server.DETECTION_FUSION_ROUTE_MODE_ENV_VAR, "active")
    fusion = _fusion_payload(
        "speech_present_general",
        speech=0.94,
        music=0.10,
        target=0.20,
        workflow="speech_cleanup",
    )
    with patch(
        "app.api_server._experimental_detection_fusion",
        return_value=fusion,
    ):
        analyzed = _analyze(client, goal="auto")

    def mock_run_audio_task(**kwargs: object) -> Path:
        run_dir = Path(kwargs["output_root"]) / CLEAN_VOICE / "fusion-speech-run"
        output_path = run_dir / "speech.restored.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"enhanced")
        _write_task_summary(run_dir, CLEAN_VOICE, output_path)
        return run_dir

    with patch(
        "app.api_server.run_audio_task",
        side_effect=mock_run_audio_task,
    ):
        response = client.post(
            "/api/run",
            json={"file_id": analyzed["file_id"], "task": CLEAN_VOICE},
        )

    payload = response.json()
    assert payload["status"] == "success"
    assert payload["controller"]["decision_source"] == "detection_fusion_active"
    assert payload["controller"]["workflow_kind"] == "speech_cleanup"
    assert payload["controller"]["recommended_task"] == CLEAN_VOICE
    assert payload["outputs"] == []


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


def test_api_run_plan_writes_guarded_environment_event_report(
    client: TestClient,
) -> None:
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
    payload = response.json()
    assert payload["status"] == "analysis_complete"
    assert payload["controller"]["workflow_kind"] == (
        "environment_event_analysis"
    )
    assert [artifact["label"] for artifact in payload["outputs"]] == [
        "environment_event_report_json",
        "environment_event_report_csv",
    ]
    report_response = client.get(payload["outputs"][0]["download_url"])
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["router_predicted_label"] == "speech_target_noise"
    assert report["speech_gate_label"] == "non_speech"
    assert report["guard_applied"] is True
    assert report["target_event_detected"] is True
    assert "environment event analysis report" in report["explanation"]
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


def test_api_run_plan_target_noise_cleanup_writes_audio_and_reports(
    client: TestClient,
) -> None:
    router_result = {
        "router_status": "enabled",
        "predicted_label": "speech_target_noise",
        "confidence": 0.96,
        "accepted": True,
        "route_target": CLEAN_VOICE,
        "recommended_task": CLEAN_VOICE,
        "decision_reason": "accepted",
        "warnings": [],
        "guard_applied": False,
        "final_workflow": "speech_cleanup",
        "final_accepted": True,
        "speech_gate_label": "speech_present",
        "speech_gate_confidence": 0.94,
    }
    analyzed = _analyze(client, goal="auto", router_result=router_result)

    def mock_run_audio_task(**kwargs: object) -> Path:
        run_dir = Path(kwargs["output_root"]) / CLEAN_VOICE / "target-run"
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
    assert run_mock.call_args.kwargs["task"] == CLEAN_VOICE
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["controller"]["workflow_kind"] == (
        "speech_target_noise_cleanup"
    )
    assert [artifact["label"] for artifact in payload["outputs"]] == [
        "enhanced_speech",
        "target_noise_report_json",
        "target_noise_report_csv",
    ]
    report_response = client.get(payload["outputs"][1]["download_url"])
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["workflow_kind"] == "speech_target_noise_cleanup"
    assert report["recommended_task"] == CLEAN_VOICE
    assert report["router_predicted_label"] == "speech_target_noise"
    assert report["target_specific_suppressor_enabled"] is False
    assert report["fallback_engine"] == "DeepFilterNet"
    assert report["target_event_detected"] is True


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
