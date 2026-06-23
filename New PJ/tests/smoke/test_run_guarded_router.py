from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import run_guarded_router as guarded


def _runtime_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    input_path = tmp_path / "input.wav"
    router_checkpoint = tmp_path / "router.pt"
    gate_checkpoint = tmp_path / "gate.pt"
    input_path.write_bytes(b"audio")
    router_checkpoint.write_bytes(b"router")
    gate_checkpoint.write_bytes(b"gate")
    return input_path, router_checkpoint, gate_checkpoint


def _run_with_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    baseline_label: str,
    baseline_confidence: float,
    gate_label: str,
    gate_confidence: float,
) -> dict[str, object]:
    input_path, router_checkpoint, gate_checkpoint = _runtime_files(tmp_path)
    router_state = {"kind": "router", "config": {"feature_columns": ["rms_energy"]}}
    gate_state = {"kind": "gate", "config": {"feature_columns": ["rms_energy"]}}

    def fake_load(path):
        return router_state if path == router_checkpoint.resolve() else gate_state

    def fake_predict(state, _features):
        if state["kind"] == "router":
            return (
                baseline_label,
                baseline_confidence,
                {baseline_label: baseline_confidence},
            )
        return gate_label, gate_confidence, {gate_label: gate_confidence}

    monkeypatch.setattr(guarded, "_load_checkpoint", fake_load)
    monkeypatch.setattr(
        guarded,
        "_extract_input_features",
        lambda _path, _columns: {"rms_energy": 0.1},
    )
    monkeypatch.setattr(guarded, "_predict_checkpoint", fake_predict)
    return guarded.run_guarded_router(
        input_path=input_path,
        router_checkpoint=router_checkpoint,
        speech_gate_checkpoint=gate_checkpoint,
        router_threshold=0.70,
        speech_threshold=0.70,
    )


def test_speech_cleanup_is_blocked_by_non_speech_gate(tmp_path, monkeypatch):
    result = _run_with_predictions(
        tmp_path,
        monkeypatch,
        baseline_label="speech_noisy_general",
        baseline_confidence=0.92,
        gate_label="non_speech",
        gate_confidence=0.95,
    )

    assert result["router_workflow"] == "speech_cleanup"
    assert result["final_workflow"] == "safe_abstain"
    assert result["guard_applied"] is True
    assert result["reason"] == "speech_cleanup_blocked_by_speech_gate"


def test_speech_cleanup_runs_when_gate_confirms_speech(tmp_path, monkeypatch):
    result = _run_with_predictions(
        tmp_path,
        monkeypatch,
        baseline_label="speech_target_noise",
        baseline_confidence=0.91,
        gate_label="speech_present",
        gate_confidence=0.89,
    )

    assert result["final_workflow"] == "speech_cleanup"
    assert result["guard_applied"] is False
    assert result["reason"] == "accepted"


def test_low_baseline_confidence_abstains_before_workflow_acceptance(
    tmp_path,
    monkeypatch,
):
    result = _run_with_predictions(
        tmp_path,
        monkeypatch,
        baseline_label="environment_only",
        baseline_confidence=0.69,
        gate_label="non_speech",
        gate_confidence=0.98,
    )

    assert result["router_workflow"] == "no_process"
    assert result["final_workflow"] == "safe_abstain"
    assert result["reason"] == "baseline_confidence_below_threshold"


def test_high_confidence_no_process_is_accepted(tmp_path, monkeypatch):
    result = _run_with_predictions(
        tmp_path,
        monkeypatch,
        baseline_label="speech_clean",
        baseline_confidence=0.94,
        gate_label="non_speech",
        gate_confidence=0.90,
    )

    assert result["final_workflow"] == "no_process"
    assert result["guard_applied"] is False
    assert result["reason"] == "accepted"


def test_json_output_shape(tmp_path, monkeypatch, capsys):
    input_path, router_checkpoint, gate_checkpoint = _runtime_files(tmp_path)
    result = {
        "input_path": str(input_path.resolve()),
        "router_label": "speech_clean",
        "router_workflow": "no_process",
        "router_confidence": 0.9,
        "router_accepted": True,
        "speech_gate_label": "speech_present",
        "speech_gate_confidence": 0.8,
        "speech_gate_accepted": True,
        "guard_applied": False,
        "final_workflow": "no_process",
        "final_accepted": True,
        "reason": "accepted",
        "router_probabilities": {"speech_clean": 0.9},
        "speech_gate_probabilities": {"speech_present": 0.8},
    }
    monkeypatch.setattr(guarded, "run_guarded_router", lambda **_kwargs: result)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_guarded_router.py",
            "--input",
            str(input_path),
            "--router-checkpoint",
            str(router_checkpoint),
            "--speech-gate-checkpoint",
            str(gate_checkpoint),
            "--json",
        ],
    )

    assert guarded.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "input_path",
        "router_label",
        "router_workflow",
        "router_confidence",
        "router_accepted",
        "speech_gate_label",
        "speech_gate_confidence",
        "speech_gate_accepted",
        "guard_applied",
        "final_workflow",
        "final_accepted",
        "reason",
        "router_probabilities",
        "speech_gate_probabilities",
    }


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("input", "input file not found"),
        ("router", "router checkpoint not found"),
        ("gate", "gate checkpoint not found"),
    ],
)
def test_missing_runtime_file_validation(tmp_path, missing, message):
    input_path, router_checkpoint, gate_checkpoint = _runtime_files(tmp_path)
    paths = {
        "input": input_path,
        "router": router_checkpoint,
        "gate": gate_checkpoint,
    }
    paths[missing].unlink()

    with pytest.raises(FileNotFoundError, match=message):
        guarded.run_guarded_router(
            input_path=input_path,
            router_checkpoint=router_checkpoint,
            speech_gate_checkpoint=gate_checkpoint,
        )


def test_invalid_checkpoint_schema_raises_clear_error(tmp_path, monkeypatch):
    checkpoint = tmp_path / "bad.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        guarded.torch,
        "load",
        lambda *_args, **_kwargs: {
            "model_state_dict": {},
            "config": {"feature_columns": ["rms_energy"]},
        },
    )

    with pytest.raises(ValueError, match="feature_stats"):
        guarded._load_checkpoint(checkpoint)
