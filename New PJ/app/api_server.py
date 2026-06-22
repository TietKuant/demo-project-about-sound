"""FastAPI surface for the target-aware audio processing controller."""

from __future__ import annotations

import csv
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.audio_task_demo import (
    _demo_capabilities,
    _extract_feature_row,
    _input_type_for_path,
    _optional_router_result,
    _processing_facts,
    _router_evidence,
)
from scripts.run_detection_fusion import run_detection_fusion
from scripts.run_audio_task import run_audio_task
from src.planner.processing_planner import (
    ACTION_ANALYZE_ONLY,
    ACTION_MANUAL_REQUIRED,
    ACTION_NO_PROCESS,
    ACTION_RUN_TASK,
    ANALYZE_ONLY,
    AUTO,
    EXTRACT_VOCALS_GOAL,
    IMPROVE_SPEECH_CLARITY,
    REDUCE_TARGET_NOISE,
    REMOVE_VOCALS_GOAL,
    ProcessingPlan,
    WORKFLOW_ENVIRONMENT_EVENT_ANALYSIS,
    WORKFLOW_MUSIC_SEPARATION_PACKAGE,
    WORKFLOW_SPEECH_CLEANUP,
    WORKFLOW_SPEECH_TARGET_NOISE_CLEANUP,
    plan_processing,
)
from src.router.task_registry import (
    CLEAN_VOICE,
    EXTRACT_VOCALS,
    REMOVE_VOCALS,
    TARGET_NOISE_SUPPRESSION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = PROJECT_ROOT / "outputs" / "api-uploads"
RUN_ROOT = PROJECT_ROOT / "outputs" / "api-runs"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
DETECTION_FUSION_HEAD_CHECKPOINT_ENV_VAR = (
    "DETECTION_FUSION_HEAD_CHECKPOINT"
)
DETECTION_FUSION_ROUTER_CHECKPOINT_ENV_VAR = (
    "DETECTION_FUSION_ROUTER_CHECKPOINT"
)
DETECTION_FUSION_SPEECH_GATE_CHECKPOINT_ENV_VAR = (
    "DETECTION_FUSION_SPEECH_GATE_CHECKPOINT"
)
DETECTION_FUSION_SPEECH_THRESHOLD = 0.6
DETECTION_FUSION_MUSIC_THRESHOLD = 0.7
DETECTION_FUSION_TARGET_THRESHOLD = 0.8

SUPPORTED_GOALS = {
    ANALYZE_ONLY,
    AUTO,
    IMPROVE_SPEECH_CLARITY,
    EXTRACT_VOCALS_GOAL,
    REMOVE_VOCALS_GOAL,
    REDUCE_TARGET_NOISE,
}
TASK_GOALS = {
    CLEAN_VOICE: IMPROVE_SPEECH_CLARITY,
    EXTRACT_VOCALS: EXTRACT_VOCALS_GOAL,
    REMOVE_VOCALS: REMOVE_VOCALS_GOAL,
    TARGET_NOISE_SUPPRESSION: REDUCE_TARGET_NOISE,
}
FEATURE_FIELDS = [
    "duration_sec",
    "rms_energy",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
]


class RunRequest(BaseModel):
    file_id: str
    task: str


class RunPlanRequest(BaseModel):
    file_id: str


api = FastAPI(title="Target-Aware Audio Processing Controller API")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validated_file_id(file_id: str) -> str:
    try:
        return str(uuid.UUID(file_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Unknown file_id.") from exc


def _file_dir(file_id: str) -> Path:
    return UPLOAD_ROOT / _validated_file_id(file_id)


def _metadata_path(file_id: str) -> Path:
    return _file_dir(file_id) / "metadata.json"


def _read_metadata(file_id: str) -> dict[str, Any]:
    path = _metadata_path(file_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown file_id.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Upload metadata is unavailable.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Upload metadata is invalid.")
    return payload


def _write_metadata(file_id: str, metadata: dict[str, Any]) -> None:
    path = _metadata_path(file_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _plan_payload(plan: ProcessingPlan) -> dict[str, Any]:
    return {
        "decision": plan.action,
        "workflow_kind": plan.workflow_kind,
        "recommended_task": plan.recommended_task,
        "recommended_tasks": plan.recommended_tasks,
        "engine_family": plan.engine_family,
        "algorithm": plan.algorithm,
        "expected_outputs": plan.expected_outputs,
        "why": plan.explanation,
        "warnings": plan.warnings,
        "blocked_reasons": plan.blocked_reasons,
        "alternatives": plan.alternatives,
    }


def _router_payload(router_result: dict[str, object]) -> dict[str, Any]:
    confidence = router_result.get("confidence")
    speech_gate_confidence = router_result.get("speech_gate_confidence")
    return {
        "status": str(router_result.get("router_status") or "disabled"),
        "predicted_label": str(router_result.get("predicted_label") or ""),
        "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
        "accepted": router_result.get("accepted") is True,
        "decision_reason": str(router_result.get("decision_reason") or ""),
        "guard_applied": router_result.get("guard_applied") is True,
        "final_workflow": str(router_result.get("final_workflow") or ""),
        "final_accepted": router_result.get("final_accepted") is True,
        "speech_gate_label": str(router_result.get("speech_gate_label") or ""),
        "speech_gate_confidence": (
            float(speech_gate_confidence)
            if isinstance(speech_gate_confidence, (int, float))
            else None
        ),
    }


def _feature_payload(feature_row: dict[str, str]) -> dict[str, str]:
    return {field: str(feature_row.get(field, "")) for field in FEATURE_FIELDS}


def _experimental_detection_fusion(input_path: Path) -> dict[str, Any]:
    checkpoint_values = {
        "head_checkpoint": os.environ.get(
            DETECTION_FUSION_HEAD_CHECKPOINT_ENV_VAR, ""
        ).strip(),
        "router_checkpoint": os.environ.get(
            DETECTION_FUSION_ROUTER_CHECKPOINT_ENV_VAR, ""
        ).strip(),
        "speech_gate_checkpoint": os.environ.get(
            DETECTION_FUSION_SPEECH_GATE_CHECKPOINT_ENV_VAR, ""
        ).strip(),
    }
    missing_config = [
        env_var
        for env_var, value in (
            (
                DETECTION_FUSION_HEAD_CHECKPOINT_ENV_VAR,
                checkpoint_values["head_checkpoint"],
            ),
            (
                DETECTION_FUSION_ROUTER_CHECKPOINT_ENV_VAR,
                checkpoint_values["router_checkpoint"],
            ),
            (
                DETECTION_FUSION_SPEECH_GATE_CHECKPOINT_ENV_VAR,
                checkpoint_values["speech_gate_checkpoint"],
            ),
        )
        if not value
    ]
    if missing_config:
        return {
            "enabled": False,
            "reason": "detection_fusion_checkpoint_not_configured",
            "missing_configuration": missing_config,
        }

    missing_files = [
        str(Path(value).expanduser())
        for value in checkpoint_values.values()
        if not Path(value).expanduser().is_file()
    ]
    if missing_files:
        return {
            "enabled": False,
            "reason": "detection_fusion_checkpoint_missing",
            "missing_checkpoints": missing_files,
        }

    try:
        result = run_detection_fusion(
            input_path=input_path,
            head_checkpoint=checkpoint_values["head_checkpoint"],
            router_checkpoint=checkpoint_values["router_checkpoint"],
            speech_gate_checkpoint=checkpoint_values["speech_gate_checkpoint"],
            speech_threshold=DETECTION_FUSION_SPEECH_THRESHOLD,
            music_threshold=DETECTION_FUSION_MUSIC_THRESHOLD,
            target_threshold=DETECTION_FUSION_TARGET_THRESHOLD,
        )
    except Exception as exc:
        return {
            "enabled": False,
            "reason": "detection_fusion_failed",
            "error": str(exc),
        }
    return {
        "enabled": True,
        **result,
    }


def _read_task_summary(run_dir: Path) -> dict[str, str]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.is_file():
        raise HTTPException(status_code=500, detail=f"Task summary not found: {summary_path}")
    with summary_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise HTTPException(status_code=500, detail="Task summary is empty.")
    return rows[0]


def _registered_path(metadata: dict[str, Any], kind: str, label: str | None = None) -> Path:
    if kind == "input":
        value = str(metadata.get("input_path") or "")
    elif label:
        value = str(dict(metadata.get("outputs_by_label") or {}).get(label) or "")
    else:
        value = str(metadata.get("primary_output_path") or "")
    if not value:
        detail = f"No registered {kind} file" + (f" for label: {label}" if label else "")
        raise HTTPException(status_code=404, detail=f"{detail}.")
    path = Path(value).resolve()
    allowed_root = UPLOAD_ROOT.resolve() if kind == "input" else RUN_ROOT.resolve()
    if not path.is_relative_to(allowed_root) or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Registered {kind} file is unavailable.")
    return path


def _plan_from_metadata(metadata: dict[str, Any], input_path: Path) -> ProcessingPlan:
    feature_row = {
        str(key): str(value)
        for key, value in dict(metadata.get("features") or {}).items()
    }
    router_result = dict(metadata.get("router") or {})
    return plan_processing(
        str(metadata.get("goal") or AUTO),
        _processing_facts(input_path, feature_row),
        _router_evidence(router_result),
        _demo_capabilities(),
    )


def _output_artifact(
    file_id: str,
    label: str,
    path: Path,
    media_type: str = "audio",
) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(RUN_ROOT.resolve()) or not resolved.is_file():
        raise HTTPException(status_code=500, detail=f"Output artifact is unavailable: {label}")
    return {
        "label": label,
        "path": str(resolved),
        "download_url": f"/api/files/{file_id}?kind=output&label={label}",
        "media_type": media_type,
    }


def _write_environment_event_report(
    *,
    file_id: str,
    metadata: dict[str, Any],
    input_path: Path,
    plan: ProcessingPlan,
) -> tuple[Path, list[dict[str, str]]]:
    run_dir = (
        RUN_ROOT
        / file_id
        / WORKFLOW_ENVIRONMENT_EVENT_ANALYSIS
        / "report"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    features = dict(metadata.get("features") or {})
    router = dict(metadata.get("router") or {})
    report = {
        "input_filename": str(metadata.get("filename") or input_path.name),
        "input_path": str(input_path.resolve()),
        "duration_sec": features.get("duration_sec", ""),
        "rms_energy": features.get("rms_energy", ""),
        "spectral_centroid_hz": features.get("spectral_centroid_hz", ""),
        "spectral_bandwidth_hz": features.get(
            "spectral_bandwidth_hz", ""
        ),
        "router_predicted_label": router.get("predicted_label", ""),
        "router_confidence": router.get("confidence"),
        "speech_gate_label": router.get("speech_gate_label", ""),
        "speech_gate_confidence": router.get("speech_gate_confidence"),
        "guard_applied": router.get("guard_applied") is True,
        "final_workflow": router.get("final_workflow", ""),
        "decision_reason": router.get("decision_reason", ""),
        "explanation": plan.explanation,
        "target_event_detected": (
            router.get("predicted_label") == "speech_target_noise"
        ),
    }
    json_path = run_dir / "report.json"
    csv_path = run_dir / "report.csv"
    json_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report))
        writer.writeheader()
        writer.writerow(report)
    return run_dir, [
        _output_artifact(
            file_id,
            "environment_event_report_json",
            json_path,
            media_type="application/json",
        ),
        _output_artifact(
            file_id,
            "environment_event_report_csv",
            csv_path,
            media_type="text/csv",
        ),
    ]


def _write_target_noise_report(
    *,
    file_id: str,
    metadata: dict[str, Any],
    input_path: Path,
    plan: ProcessingPlan,
    run_dir: Path,
) -> list[dict[str, str]]:
    router = dict(metadata.get("router") or {})
    report = {
        "input_filename": str(metadata.get("filename") or input_path.name),
        "input_path": str(input_path.resolve()),
        "workflow_kind": plan.workflow_kind,
        "recommended_task": plan.recommended_task,
        "router_predicted_label": router.get("predicted_label", ""),
        "router_confidence": router.get("confidence"),
        "speech_gate_label": router.get("speech_gate_label", ""),
        "speech_gate_confidence": router.get("speech_gate_confidence"),
        "guard_applied": router.get("guard_applied") is True,
        "final_workflow": router.get("final_workflow", ""),
        "final_accepted": router.get("final_accepted") is True,
        "explanation": plan.explanation,
        "target_specific_suppressor_enabled": False,
        "fallback_engine": "DeepFilterNet",
        "target_event_detected": True,
    }
    json_path = run_dir / "target_noise_report.json"
    csv_path = run_dir / "target_noise_report.csv"
    json_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report))
        writer.writeheader()
        writer.writerow(report)
    return [
        _output_artifact(
            file_id,
            "target_noise_report_json",
            json_path,
            media_type="application/json",
        ),
        _output_artifact(
            file_id,
            "target_noise_report_csv",
            csv_path,
            media_type="text/csv",
        ),
    ]


def _read_music_artifacts(run_dir: Path, file_id: str) -> list[dict[str, str]]:
    top_summary = (run_dir / "summary.csv").resolve()
    for summary_path in sorted(run_dir.rglob("summary.csv")):
        if summary_path.resolve() == top_summary:
            continue
        with summary_path.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
        if not rows:
            continue
        row = rows[0]
        vocals_path = str(row.get("vocals_path") or "")
        no_vocals_path = str(row.get("no_vocals_path") or "")
        if vocals_path and no_vocals_path:
            return [
                _output_artifact(file_id, "vocals", Path(vocals_path)),
                _output_artifact(file_id, "no_vocals", Path(no_vocals_path)),
            ]
    raise HTTPException(status_code=500, detail="Nested music separation summary is unavailable.")


def _store_artifacts(metadata: dict[str, Any], artifacts: list[dict[str, str]], run_dir: Path) -> None:
    metadata["outputs_by_label"] = {
        artifact["label"]: artifact["path"]
        for artifact in artifacts
    }
    metadata["primary_output_path"] = artifacts[0]["path"] if artifacts else ""
    metadata["last_run_dir"] = str(run_dir.resolve())


@api.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@api.post("/api/analyze")
async def analyze(file: UploadFile = File(...), goal: str = Form(...)) -> dict[str, Any]:
    if goal not in SUPPORTED_GOALS:
        raise HTTPException(status_code=400, detail=f"Unsupported goal: {goal}")

    file_id = str(uuid.uuid4())
    upload_dir = _file_dir(file_id)
    upload_dir.mkdir(parents=True, exist_ok=False)
    filename = Path(file.filename or "upload.bin").name or "upload.bin"
    input_path = upload_dir / f"input{Path(filename).suffix.lower() or '.bin'}"
    try:
        with input_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)
    finally:
        await file.close()

    if input_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        feature_row = _extract_feature_row(input_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Audio feature extraction failed: {exc}") from exc

    router_result = _optional_router_result(input_path)
    experimental_detection_fusion = _experimental_detection_fusion(input_path)
    facts = _processing_facts(input_path, feature_row)
    plan = plan_processing(
        goal,
        facts,
        _router_evidence(router_result),
        _demo_capabilities(),
    )
    metadata = {
        "file_id": file_id,
        "filename": filename,
        "input_path": str(input_path.resolve()),
        "input_type": _input_type_for_path(input_path),
        "goal": goal,
        "features": feature_row,
        "router": router_result,
        "primary_output_path": "",
        "outputs_by_label": {},
    }
    _write_metadata(file_id, metadata)
    return {
        "file_id": file_id,
        "filename": filename,
        "controller": _plan_payload(plan),
        "router": _router_payload(router_result),
        "features": _feature_payload(feature_row),
        "experimental_detection_fusion": experimental_detection_fusion,
    }


@api.post("/api/run")
def run(request: RunRequest) -> dict[str, Any]:
    goal = TASK_GOALS.get(request.task)
    if goal is None:
        raise HTTPException(status_code=400, detail=f"Unsupported task: {request.task}")

    metadata = _read_metadata(request.file_id)
    input_path = _registered_path(metadata, "input")
    feature_row = {
        str(key): str(value)
        for key, value in dict(metadata.get("features") or {}).items()
    }
    router_result = dict(metadata.get("router") or {})
    plan = plan_processing(
        goal,
        _processing_facts(input_path, feature_row),
        _router_evidence(router_result),
        _demo_capabilities(),
    )
    if plan.action != ACTION_RUN_TASK or plan.blocked_reasons:
        return {
            "status": "blocked",
            "task": request.task,
            "controller": _plan_payload(plan),
            "run_dir": "",
            "primary_output_path": "",
            "download_url": None,
        }

    run_dir = run_audio_task(
        task=request.task,
        input_path=input_path,
        output_root=RUN_ROOT / request.file_id,
    )
    summary = _read_task_summary(run_dir)
    primary_output_path = str(summary.get("primary_output_path") or "")
    metadata["primary_output_path"] = primary_output_path
    metadata["last_run_dir"] = str(Path(run_dir).resolve())
    metadata["last_task"] = request.task
    _write_metadata(request.file_id, metadata)
    return {
        "status": summary.get("status", "failed"),
        "task": request.task,
        "controller": _plan_payload(plan),
        "run_dir": str(Path(run_dir).resolve()),
        "primary_output_path": primary_output_path,
        "download_url": (
            f"/api/files/{request.file_id}?kind=output"
            if primary_output_path and Path(primary_output_path).is_file()
            else None
        ),
        "error": summary.get("error", ""),
    }


@api.post("/api/run-plan")
def run_plan(request: RunPlanRequest) -> dict[str, Any]:
    metadata = _read_metadata(request.file_id)
    input_path = _registered_path(metadata, "input")
    plan = _plan_from_metadata(metadata, input_path)

    if plan.workflow_kind == WORKFLOW_ENVIRONMENT_EVENT_ANALYSIS:
        run_dir, artifacts = _write_environment_event_report(
            file_id=request.file_id,
            metadata=metadata,
            input_path=input_path,
            plan=plan,
        )
        _store_artifacts(metadata, artifacts, run_dir)
        metadata["last_task"] = WORKFLOW_ENVIRONMENT_EVENT_ANALYSIS
        _write_metadata(request.file_id, metadata)
        return {
            "status": "analysis_complete",
            "controller": _plan_payload(plan),
            "run_dir": str(run_dir.resolve()),
            "outputs": artifacts,
            "primary_output_path": artifacts[0]["path"],
            "download_url": artifacts[0]["download_url"],
            "error": "",
        }
    if plan.action == ACTION_NO_PROCESS:
        return {
            "status": "no_process",
            "controller": _plan_payload(plan),
            "run_dir": "",
            "outputs": [],
        }
    if plan.action in {ACTION_MANUAL_REQUIRED, ACTION_ANALYZE_ONLY} or plan.blocked_reasons:
        return {
            "status": "blocked",
            "controller": _plan_payload(plan),
            "run_dir": "",
            "outputs": [],
        }
    if plan.action != ACTION_RUN_TASK or not plan.recommended_task:
        return {
            "status": "blocked",
            "controller": _plan_payload(plan),
            "run_dir": "",
            "outputs": [],
        }

    task = plan.recommended_task
    run_dir = run_audio_task(
        task=task,
        input_path=input_path,
        output_root=RUN_ROOT / request.file_id,
    )
    summary = _read_task_summary(run_dir)
    if summary.get("status") != "success":
        return {
            "status": summary.get("status", "failed"),
            "controller": _plan_payload(plan),
            "run_dir": str(run_dir.resolve()),
            "outputs": [],
            "error": summary.get("error", ""),
        }

    if plan.workflow_kind in {
        WORKFLOW_SPEECH_CLEANUP,
        WORKFLOW_SPEECH_TARGET_NOISE_CLEANUP,
    }:
        primary_path = Path(str(summary.get("primary_output_path") or ""))
        artifacts = [_output_artifact(request.file_id, "enhanced_speech", primary_path)]
        if plan.workflow_kind == WORKFLOW_SPEECH_TARGET_NOISE_CLEANUP:
            artifacts.extend(
                _write_target_noise_report(
                    file_id=request.file_id,
                    metadata=metadata,
                    input_path=input_path,
                    plan=plan,
                    run_dir=run_dir,
                )
            )
    elif plan.workflow_kind == WORKFLOW_MUSIC_SEPARATION_PACKAGE or task in {EXTRACT_VOCALS, REMOVE_VOCALS}:
        artifacts = _read_music_artifacts(run_dir, request.file_id)
    else:
        primary_path = Path(str(summary.get("primary_output_path") or ""))
        label = plan.output_labels[0] if plan.output_labels else task
        artifacts = [_output_artifact(request.file_id, label, primary_path)]

    _store_artifacts(metadata, artifacts, run_dir)
    metadata["last_task"] = task
    _write_metadata(request.file_id, metadata)
    return {
        "status": "success",
        "task": task,
        "controller": _plan_payload(plan),
        "run_dir": str(run_dir.resolve()),
        "outputs": artifacts,
        "primary_output_path": artifacts[0]["path"],
        "download_url": artifacts[0]["download_url"],
        "error": "",
    }


@api.get("/api/files/{file_id}")
def files(
    file_id: str,
    kind: str = Query(default="input", pattern="^(input|output)$"),
    label: str | None = Query(default=None),
) -> FileResponse:
    metadata = _read_metadata(file_id)
    path = _registered_path(metadata, kind, label)
    return FileResponse(path)


if FRONTEND_DIST.is_dir():
    api.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


app = api
