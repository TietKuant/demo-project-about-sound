#!/usr/bin/env python3
"""Run experimental detection evidence heads and threshold fusion on one input."""

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

from scripts.build_detection_evidence_dataset import (
    ROUTER_LABELS,
    SIGNAL_FEATURE_COLUMNS,
    _rank_probabilities,
)
from scripts.extract_audio_features import _audio_for_features, _features
from scripts.run_guarded_router import run_guarded_router
from scripts.train_detection_evidence_heads import DetectionEvidenceHeads


REQUIRED_HEADS = (
    "speech_present",
    "music_present",
    "target_event_present",
    "siren_present",
    "car_horn_present",
    "dog_bark_present",
)
WORKFLOW_BY_FUSION_LABEL = {
    "music_with_vocals": "music_separation_package",
    "speech_target_noise": "speech_target_noise_cleanup",
    "speech_present_general": "speech_cleanup",
}
ROUTER_CONFIDENCE_THRESHOLD = 0.70
REVIEW_MARGIN = 0.05


def _validate_paths(
    input_path: str | Path,
    head_checkpoint: str | Path,
    router_checkpoint: str | Path,
    speech_gate_checkpoint: str | Path,
) -> tuple[Path, Path, Path, Path]:
    paths = tuple(
        Path(value).expanduser().resolve(strict=False)
        for value in (
            input_path,
            head_checkpoint,
            router_checkpoint,
            speech_gate_checkpoint,
        )
    )
    labels = (
        "Input file",
        "Detection head checkpoint",
        "Router checkpoint",
        "Speech gate checkpoint",
    )
    for label, path in zip(labels, paths):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    return paths


def _validate_threshold(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def _extract_signal_features(input_path: Path) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="detection-fusion-") as temp_name:
        audio, sample_rate = _audio_for_features(input_path, Path(temp_name))
    extracted = _features(audio, sample_rate)
    signal_features = {"duration_sec": float(audio.size / sample_rate)}
    for column in SIGNAL_FEATURE_COLUMNS:
        if column == "duration_sec":
            continue
        if column not in extracted:
            raise ValueError(f"Extracted signal features are missing column: {column}")
        signal_features[column] = float(extracted[column])
    return signal_features


def _build_evidence_features(
    signal_features: Mapping[str, float],
    guarded_result: Mapping[str, object],
) -> dict[str, float]:
    evidence = {str(key): float(value) for key, value in signal_features.items()}
    router_probabilities = {
        str(label): float(score)
        for label, score in dict(
            guarded_result.get("router_probabilities") or {}
        ).items()
    }
    for label in ROUTER_LABELS:
        if label in router_probabilities:
            evidence[f"router_prob_{label}"] = router_probabilities[label]
    speech_gate_probabilities = {
        str(label): float(score)
        for label, score in dict(
            guarded_result.get("speech_gate_probabilities") or {}
        ).items()
    }
    for label in ("non_speech", "speech_present"):
        if label in speech_gate_probabilities:
            evidence[f"speech_gate_prob_{label}"] = speech_gate_probabilities[label]
    _, top_score, _, second_score, score_margin = _rank_probabilities(
        router_probabilities
    )
    if top_score:
        evidence["top_score"] = float(top_score)
    if second_score:
        evidence["second_score"] = float(second_score)
    if score_margin:
        evidence["score_margin"] = float(score_margin)
    return evidence


def _feature_vector(
    features: Mapping[str, float],
    feature_columns: Sequence[str],
    feature_stats: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    missing = [column for column in feature_columns if column not in features]
    if missing:
        raise ValueError(
            "Detection fusion evidence is missing required feature columns: "
            + ", ".join(sorted(missing))
        )
    missing_stats = [
        column for column in feature_columns if column not in feature_stats
    ]
    if missing_stats:
        raise ValueError(
            "Detection head checkpoint is missing feature stats for: "
            + ", ".join(sorted(missing_stats))
        )
    normalized: list[float] = []
    for column in feature_columns:
        stats = feature_stats[column]
        if "mean" not in stats or "std" not in stats:
            raise ValueError(
                f"Detection head feature stats are invalid for: {column}"
            )
        mean = float(stats["mean"])
        std = float(stats["std"])
        if abs(std) < 1e-8:
            std = 1.0
        normalized.append((float(features[column]) - mean) / std)
    return np.asarray([normalized], dtype=np.float32)


def _load_head_checkpoint(
    checkpoint_path: Path,
    device: str,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Detection head checkpoint must be a mapping.")
    missing = {"model_state_dict", "config"} - set(checkpoint)
    if missing:
        raise ValueError(
            "Detection head checkpoint is missing keys: "
            + ", ".join(sorted(missing))
        )
    config = checkpoint["config"]
    if not isinstance(config, dict):
        raise ValueError("Detection head checkpoint config must be a mapping.")
    config_missing = {"feature_columns", "feature_stats", "heads"} - set(config)
    if config_missing:
        raise ValueError(
            "Detection head checkpoint config is missing keys: "
            + ", ".join(sorted(config_missing))
        )
    heads = [str(head) for head in config["heads"]]
    missing_heads = set(REQUIRED_HEADS) - set(heads)
    if missing_heads:
        raise ValueError(
            "Detection head checkpoint is missing trained heads: "
            + ", ".join(sorted(missing_heads))
        )
    return checkpoint


def _predict_detection_heads(
    checkpoint: Mapping[str, Any],
    evidence_features: Mapping[str, float],
    device: str,
) -> dict[str, float]:
    config = checkpoint["config"]
    feature_columns = [str(column) for column in config["feature_columns"]]
    feature_vector = _feature_vector(
        evidence_features,
        feature_columns,
        config["feature_stats"],
    )
    model = DetectionEvidenceHeads(len(feature_columns))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(feature_vector, dtype=torch.float32, device=device))
    return {
        head: float(torch.sigmoid(logits[head])[0].cpu().item())
        for head in config["heads"]
    }


def _fusion_decision(
    scores: Mapping[str, float],
    *,
    speech_threshold: float,
    music_threshold: float,
    target_threshold: float,
) -> tuple[str, str]:
    speech_score = float(scores["speech_present"])
    music_score = float(scores["music_present"])
    target_score = float(scores["target_event_present"])
    if music_score >= music_threshold:
        fusion_label = "music_with_vocals"
    elif speech_score >= speech_threshold and target_score >= target_threshold:
        fusion_label = "speech_target_noise"
    elif speech_score >= speech_threshold:
        fusion_label = "speech_present_general"
    else:
        fusion_label = "environment_only"

    if fusion_label in WORKFLOW_BY_FUSION_LABEL:
        workflow = WORKFLOW_BY_FUSION_LABEL[fusion_label]
    elif target_score >= target_threshold:
        workflow = "environment_event_analysis"
    else:
        workflow = "no_process"
    return fusion_label, workflow


def _router_conflicts(router_label: str, fusion_label: str) -> bool:
    if not router_label:
        return False
    if router_label == "music_with_vocals":
        return fusion_label != "music_with_vocals"
    if router_label in {
        "speech_clean",
        "speech_noisy_general",
        "speech_target_noise",
    }:
        return fusion_label not in {
            "speech_present_general",
            "speech_target_noise",
        }
    if router_label == "environment_only":
        return fusion_label != "environment_only"
    return True


def _safety_flags(
    *,
    scores: Mapping[str, float],
    router_label: str,
    router_confidence: float,
    fusion_label: str,
    speech_threshold: float,
    music_threshold: float,
    target_threshold: float,
) -> dict[str, object]:
    threshold_pairs = (
        ("speech_present", speech_threshold),
        ("music_present", music_threshold),
        ("target_event_present", target_threshold),
    )
    all_below_threshold = all(
        float(scores[head]) < threshold for head, threshold in threshold_pairs
    )
    abstain = (
        all_below_threshold
        and router_confidence < ROUTER_CONFIDENCE_THRESHOLD
    )
    near_threshold = any(
        abs(float(scores[head]) - threshold) < REVIEW_MARGIN
        for head, threshold in threshold_pairs
    )
    router_conflict = _router_conflicts(router_label, fusion_label)
    review_reasons: list[str] = []
    if near_threshold:
        review_reasons.append("score_near_fusion_threshold")
    if router_conflict:
        review_reasons.append("router_fusion_conflict")
    if abstain:
        abstain_reason = "all_detection_scores_below_threshold_and_router_untrusted"
    else:
        abstain_reason = ""
    return {
        "abstain": abstain,
        "abstain_reason": abstain_reason,
        "review_recommended": bool(review_reasons),
        "review_reasons": review_reasons,
    }


def run_detection_fusion(
    *,
    input_path: str | Path,
    head_checkpoint: str | Path,
    router_checkpoint: str | Path,
    speech_gate_checkpoint: str | Path,
    output_json: str | Path | None = None,
    device: str = "cpu",
    speech_threshold: float = 0.6,
    music_threshold: float = 0.7,
    target_threshold: float = 0.8,
) -> dict[str, object]:
    """Run evidence extraction, detector heads, and research fusion."""
    for name, value in (
        ("--speech-threshold", speech_threshold),
        ("--music-threshold", music_threshold),
        ("--target-threshold", target_threshold),
    ):
        _validate_threshold(name, value)
    input_file, head_file, router_file, gate_file = _validate_paths(
        input_path,
        head_checkpoint,
        router_checkpoint,
        speech_gate_checkpoint,
    )
    signal_features = _extract_signal_features(input_file)
    guarded = run_guarded_router(
        input_path=input_file,
        router_checkpoint=router_file,
        speech_gate_checkpoint=gate_file,
    )
    evidence_features = _build_evidence_features(signal_features, guarded)
    checkpoint = _load_head_checkpoint(head_file, device)
    scores = _predict_detection_heads(checkpoint, evidence_features, device)
    fusion_label, recommended_workflow = _fusion_decision(
        scores,
        speech_threshold=speech_threshold,
        music_threshold=music_threshold,
        target_threshold=target_threshold,
    )
    router_label = str(guarded.get("router_label") or "")
    router_confidence = float(guarded.get("router_confidence") or 0.0)
    safety = _safety_flags(
        scores=scores,
        router_label=router_label,
        router_confidence=router_confidence,
        fusion_label=fusion_label,
        speech_threshold=speech_threshold,
        music_threshold=music_threshold,
        target_threshold=target_threshold,
    )
    if safety["abstain"]:
        recommended_workflow = "safe_abstain"
    result = {
        "input_path": str(input_file),
        "signal_features": signal_features,
        "router": {
            "label": router_label,
            "confidence": router_confidence,
            "probabilities": dict(guarded.get("router_probabilities") or {}),
        },
        "speech_gate": {
            "label": str(guarded.get("speech_gate_label") or ""),
            "confidence": float(
                guarded.get("speech_gate_confidence") or 0.0
            ),
            "probabilities": dict(
                guarded.get("speech_gate_probabilities") or {}
            ),
        },
        "detection_scores": scores,
        "thresholds": {
            "speech_threshold": speech_threshold,
            "music_threshold": music_threshold,
            "target_threshold": target_threshold,
            "router_confidence_threshold": ROUTER_CONFIDENCE_THRESHOLD,
            "review_margin": REVIEW_MARGIN,
        },
        "fusion_label": fusion_label,
        "recommended_workflow": recommended_workflow,
        **safety,
    }
    if output_json is not None:
        destination = Path(output_json).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run experimental detection evidence head fusion."
    )
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--head-checkpoint", required=True, type=Path)
    parser.add_argument("--router-checkpoint", required=True, type=Path)
    parser.add_argument("--speech-gate-checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--speech-threshold", default=0.6, type=float)
    parser.add_argument("--music-threshold", default=0.7, type=float)
    parser.add_argument("--target-threshold", default=0.8, type=float)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = run_detection_fusion(
            input_path=args.input_path,
            head_checkpoint=args.head_checkpoint,
            router_checkpoint=args.router_checkpoint,
            speech_gate_checkpoint=args.speech_gate_checkpoint,
            output_json=args.output_json,
            device=args.device,
            speech_threshold=args.speech_threshold,
            music_threshold=args.music_threshold,
            target_threshold=args.target_threshold,
        )
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
