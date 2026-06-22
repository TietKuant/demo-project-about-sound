#!/usr/bin/env python3
"""Run the experimental human-labeled router with a speech-present guard."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import torch
import torch.nn.functional as F

from scripts.extract_audio_features import _audio_for_features, _features
from src.router.audio_router_model import AudioRouterMLP


WORKFLOW_BY_LABEL = {
    "speech_clean": "no_process",
    "speech_noisy_general": "speech_cleanup",
    "speech_target_noise": "speech_cleanup",
    "music_with_vocals": "music_separation_package",
    "environment_only": "no_process",
}
SPEECH_PRESENT = "speech_present"
SPEECH_CLEANUP = "speech_cleanup"
SAFE_ABSTAIN = "safe_abstain"


def _validate_runtime_paths(
    input_path: str | Path,
    router_checkpoint: str | Path,
    gate_checkpoint: str | Path,
) -> tuple[Path, Path, Path]:
    input_file = Path(input_path).expanduser().resolve(strict=False)
    router_file = Path(router_checkpoint).expanduser().resolve(strict=False)
    gate_file = Path(gate_checkpoint).expanduser().resolve(strict=False)
    if not input_file.is_file():
        raise FileNotFoundError(f"Guarded router input file not found: {input_file}")
    if not router_file.is_file():
        raise FileNotFoundError(
            f"Human-labeled router checkpoint not found: {router_file}"
        )
    if not gate_file.is_file():
        raise FileNotFoundError(
            f"Speech gate checkpoint not found: {gate_file}"
        )
    return input_file, router_file, gate_file


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a mapping: {checkpoint_path}")
    missing = {"model_state_dict", "config"} - set(checkpoint)
    if missing:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing keys: "
            + ", ".join(sorted(missing))
        )
    config = checkpoint["config"]
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint config is invalid: {checkpoint_path}")
    config_missing = {
        "feature_columns",
        "feature_stats",
        "label_to_index",
    } - set(config)
    if config_missing:
        raise ValueError(
            f"Checkpoint config {checkpoint_path} is missing keys: "
            + ", ".join(sorted(config_missing))
        )
    return checkpoint


def _extract_input_features(
    input_path: Path,
    feature_columns: Sequence[str],
) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="guarded-router-features-") as temp_name:
        audio, sample_rate = _audio_for_features(input_path, Path(temp_name))
    extracted = _features(audio, sample_rate)
    extracted["duration_sec"] = float(audio.size / sample_rate)
    missing = [column for column in feature_columns if column not in extracted]
    if missing:
        raise ValueError(
            "Extracted audio features are missing columns: "
            + ", ".join(sorted(missing))
        )
    return {column: float(extracted[column]) for column in feature_columns}


def _normalized_vector(
    features: Mapping[str, float],
    feature_columns: Sequence[str],
    feature_stats: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    values: list[float] = []
    for column in feature_columns:
        if column not in features:
            raise ValueError(f"Input features are missing column: {column}")
        if column not in feature_stats:
            raise ValueError(f"Checkpoint feature stats are missing column: {column}")
        mean = float(feature_stats[column]["mean"])
        std = float(feature_stats[column]["std"])
        if abs(std) < 1e-8:
            std = 1.0
        values.append((float(features[column]) - mean) / std)
    return np.asarray([values], dtype=np.float32)


def _predict_checkpoint(
    checkpoint: Mapping[str, Any],
    features: Mapping[str, float],
) -> tuple[str, float]:
    config = checkpoint["config"]
    feature_columns = list(config["feature_columns"])
    label_to_index = {
        str(label): int(index)
        for label, index in dict(config["label_to_index"]).items()
    }
    if not label_to_index:
        raise ValueError("Checkpoint label mapping is empty.")
    index_to_label = {index: label for label, index in label_to_index.items()}
    expected_indexes = set(range(len(label_to_index)))
    if set(index_to_label) != expected_indexes:
        raise ValueError("Checkpoint label indexes must be contiguous from zero.")
    normalized = _normalized_vector(
        features,
        feature_columns,
        config["feature_stats"],
    )
    model = AudioRouterMLP(
        input_dim=len(feature_columns),
        num_classes=len(label_to_index),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(normalized, dtype=torch.float32))
        probabilities = F.softmax(logits, dim=1)[0].cpu()
    predicted_index = int(torch.argmax(probabilities).item())
    return index_to_label[predicted_index], float(probabilities[predicted_index].item())


def _guard_decision(
    *,
    baseline_workflow: str,
    baseline_confidence: float,
    speech_gate_label: str,
    speech_gate_confidence: float,
    threshold: float,
) -> dict[str, object]:
    if baseline_confidence < threshold:
        return {
            "guard_applied": True,
            "final_workflow": SAFE_ABSTAIN,
            "reason": "baseline_confidence_below_threshold",
        }
    if (
        baseline_workflow == SPEECH_CLEANUP
        and (
            speech_gate_label != SPEECH_PRESENT
            or speech_gate_confidence < threshold
        )
    ):
        return {
            "guard_applied": True,
            "final_workflow": SAFE_ABSTAIN,
            "reason": "speech_cleanup_blocked_by_speech_gate",
        }
    return {
        "guard_applied": False,
        "final_workflow": baseline_workflow,
        "reason": "accepted",
    }


def run_guarded_router(
    *,
    input_path: str | Path,
    router_checkpoint: str | Path,
    gate_checkpoint: str | Path,
    threshold: float = 0.70,
) -> dict[str, object]:
    """Return a guarded workflow decision for one media input."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    input_file, router_file, gate_file = _validate_runtime_paths(
        input_path,
        router_checkpoint,
        gate_checkpoint,
    )
    router_state = _load_checkpoint(router_file)
    gate_state = _load_checkpoint(gate_file)
    feature_columns = list(
        dict.fromkeys(
            [
                *router_state["config"]["feature_columns"],
                *gate_state["config"]["feature_columns"],
            ]
        )
    )
    features = _extract_input_features(input_file, feature_columns)
    baseline_label, baseline_confidence = _predict_checkpoint(
        router_state,
        features,
    )
    if baseline_label not in WORKFLOW_BY_LABEL:
        raise ValueError(f"Unsupported baseline router label: {baseline_label}")
    baseline_workflow = WORKFLOW_BY_LABEL[baseline_label]
    speech_gate_label, speech_gate_confidence = _predict_checkpoint(
        gate_state,
        features,
    )
    if speech_gate_label not in {"speech_present", "non_speech"}:
        raise ValueError(f"Unsupported speech gate label: {speech_gate_label}")
    decision = _guard_decision(
        baseline_workflow=baseline_workflow,
        baseline_confidence=baseline_confidence,
        speech_gate_label=speech_gate_label,
        speech_gate_confidence=speech_gate_confidence,
        threshold=threshold,
    )
    return {
        "input": str(input_file),
        "baseline_label": baseline_label,
        "baseline_confidence": baseline_confidence,
        "baseline_workflow": baseline_workflow,
        "speech_gate_label": speech_gate_label,
        "speech_gate_confidence": speech_gate_confidence,
        "threshold": threshold,
        **decision,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the experimental guarded audio router."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--router-checkpoint", required=True, type=Path)
    parser.add_argument("--gate-checkpoint", required=True, type=Path)
    parser.add_argument("--threshold", default=0.70, type=float)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = run_guarded_router(
            input_path=args.input,
            router_checkpoint=args.router_checkpoint,
            gate_checkpoint=args.gate_checkpoint,
            threshold=args.threshold,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"{result['final_workflow']} "
            f"(baseline={result['baseline_label']} "
            f"{result['baseline_confidence']:.4f}; "
            f"gate={result['speech_gate_label']} "
            f"{result['speech_gate_confidence']:.4f}; "
            f"reason={result['reason']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
