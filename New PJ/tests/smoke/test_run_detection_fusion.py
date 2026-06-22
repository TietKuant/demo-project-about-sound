from __future__ import annotations

import json

import numpy as np
import pytest

import scripts.run_detection_fusion as runtime


def _scores(speech: float, music: float, target: float):
    return {
        "speech_present": speech,
        "music_present": music,
        "target_event_present": target,
        "siren_present": 0.1,
        "car_horn_present": 0.2,
        "dog_bark_present": 0.3,
    }


def test_feature_vector_uses_checkpoint_feature_column_order():
    vector = runtime._feature_vector(
        {"first": 5.0, "second": 12.0},
        ["second", "first"],
        {
            "first": {"mean": 1.0, "std": 2.0},
            "second": {"mean": 2.0, "std": 5.0},
        },
    )

    assert np.allclose(vector, [[2.0, 2.0]])


@pytest.mark.parametrize(
    ("scores", "expected_label", "expected_workflow"),
    [
        (_scores(0.9, 0.8, 0.9), "music_with_vocals", "music_separation_package"),
        (
            _scores(0.8, 0.1, 0.9),
            "speech_target_noise",
            "speech_target_noise_cleanup",
        ),
        (_scores(0.8, 0.1, 0.2), "speech_present_general", "speech_cleanup"),
        (
            _scores(0.2, 0.1, 0.9),
            "environment_only",
            "environment_event_analysis",
        ),
        (_scores(0.2, 0.1, 0.2), "environment_only", "no_process"),
    ],
)
def test_fusion_rule_maps_scores_to_labels_and_workflows(
    scores,
    expected_label,
    expected_workflow,
):
    label, workflow = runtime._fusion_decision(
        scores,
        speech_threshold=0.6,
        music_threshold=0.7,
        target_threshold=0.8,
    )

    assert label == expected_label
    assert workflow == expected_workflow


def test_run_detection_fusion_writes_required_json(monkeypatch, tmp_path):
    input_path = tmp_path / "input.wav"
    head_checkpoint = tmp_path / "heads.pt"
    router_checkpoint = tmp_path / "router.pt"
    gate_checkpoint = tmp_path / "gate.pt"
    for path in (
        input_path,
        head_checkpoint,
        router_checkpoint,
        gate_checkpoint,
    ):
        path.write_bytes(b"fixture")
    output_json = tmp_path / "result.json"
    monkeypatch.setattr(
        runtime,
        "_extract_signal_features",
        lambda path: {"duration_sec": 1.0},
    )
    monkeypatch.setattr(
        runtime,
        "run_guarded_router",
        lambda **kwargs: {
            "router_label": "speech_target_noise",
            "router_confidence": 0.85,
            "router_probabilities": {
                "environment_only": 0.02,
                "music_with_vocals": 0.03,
                "speech_clean": 0.05,
                "speech_noisy_general": 0.10,
                "speech_target_noise": 0.80,
            },
            "speech_gate_label": "speech_present",
            "speech_gate_confidence": 0.90,
            "speech_gate_probabilities": {
                "non_speech": 0.10,
                "speech_present": 0.90,
            },
        },
    )
    monkeypatch.setattr(
        runtime,
        "_load_head_checkpoint",
        lambda path, device: {"config": {}},
    )
    monkeypatch.setattr(
        runtime,
        "_predict_detection_heads",
        lambda checkpoint, features, device: _scores(0.8, 0.1, 0.9),
    )

    result = runtime.run_detection_fusion(
        input_path=input_path,
        head_checkpoint=head_checkpoint,
        router_checkpoint=router_checkpoint,
        speech_gate_checkpoint=gate_checkpoint,
        output_json=output_json,
    )

    assert {
        "input_path",
        "signal_features",
        "router",
        "speech_gate",
        "detection_scores",
        "thresholds",
        "fusion_label",
        "recommended_workflow",
        "abstain",
        "abstain_reason",
        "review_recommended",
        "review_reasons",
    } <= set(result)
    assert result["fusion_label"] == "speech_target_noise"
    assert result["recommended_workflow"] == "speech_target_noise_cleanup"
    assert json.loads(output_json.read_text(encoding="utf-8"))["fusion_label"] == (
        "speech_target_noise"
    )


def test_feature_vector_fails_clearly_for_missing_column():
    with pytest.raises(ValueError, match="missing required feature columns.*missing"):
        runtime._feature_vector(
            {"present": 1.0},
            ["present", "missing"],
            {
                "present": {"mean": 0.0, "std": 1.0},
                "missing": {"mean": 0.0, "std": 1.0},
            },
        )
