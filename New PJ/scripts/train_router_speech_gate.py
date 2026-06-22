#!/usr/bin/env python3
"""Train and evaluate a binary speech-present safety gate."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import torch
import torch.nn.functional as F

from scripts.train_audio_router import TRAINING_FEATURE_COLUMNS, _classification_report
from scripts.train_human_labeled_router import (
    EVALUATION_THRESHOLDS,
    UNKNOWN_LABEL,
    _extract_feature_rows as _extract_human_feature_rows,
    _load_and_validate_manifest,
)
from src.router.audio_router_model import AudioRouterMLP


SPEECH_PRESENT_LABELS = {
    "speech_clean",
    "speech_noisy_general",
    "speech_target_noise",
}
NON_SPEECH_LABELS = {"music_with_vocals", "environment_only"}
GATE_LABELS = ["speech_present", "non_speech"]
GATE_BY_HUMAN_LABEL = {
    **{label: "speech_present" for label in SPEECH_PRESENT_LABELS},
    **{label: "non_speech" for label in NON_SPEECH_LABELS},
}
SPEECH_CLEANUP = "speech_cleanup"
SAFE_ABSTAIN = "safe_abstain"
DANGEROUS_TRUE_WORKFLOWS = {"no_process", "music_separation_package"}


def _prepare_split_rows(
    rows: Sequence[dict[str, str]],
    *,
    test_ratio: float,
    seed: int,
    excluded_sample_ids: set[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("--test-ratio must be between 0 and 1.")
    grouped: dict[str, list[dict[str, str]]] = {
        label: [] for label in GATE_LABELS
    }
    hard_unknown_rows: list[dict[str, str]] = []
    excluded_ids = excluded_sample_ids or set()
    for row in rows:
        if row["sample_id"] in excluded_ids:
            continue
        if row["human_label"] == UNKNOWN_LABEL:
            hard_unknown_rows.append(
                {**row, "gate_label": "hard_unknown", "split": "hard_unknown"}
            )
            continue
        gate_label = GATE_BY_HUMAN_LABEL[row["human_label"]]
        grouped[gate_label].append({**row, "gate_label": gate_label})

    rng = random.Random(seed)
    split_rows: list[dict[str, str]] = []
    for gate_label in GATE_LABELS:
        class_rows = grouped[gate_label]
        if len(class_rows) < 2:
            raise ValueError(
                f"Gate class {gate_label} has {len(class_rows)} row(s) remaining "
                "after excluding baseline sample IDs; at least 2 are required "
                "for train/test evaluation."
            )
        rng.shuffle(class_rows)
        test_count = round(len(class_rows) * test_ratio)
        test_count = max(1, min(len(class_rows) - 1, test_count))
        for index, row in enumerate(class_rows):
            row["split"] = "test" if index < test_count else "train"
            split_rows.append(row)
    return split_rows, hard_unknown_rows


def _extract_feature_rows(
    rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    return _extract_human_feature_rows(rows)


def _normalize(
    train_values: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    stats = {
        column: {"mean": float(mean[index]), "std": float(std[index])}
        for index, column in enumerate(TRAINING_FEATURE_COLUMNS)
    }
    return ((values - mean) / std).astype(np.float32), stats


def _fit_gate_model(
    *,
    train_rows: Sequence[dict[str, str]],
    test_rows: Sequence[dict[str, str]],
    hard_unknown_rows: Sequence[dict[str, str]],
    extra_rows: Sequence[dict[str, str]],
    max_epochs: int,
    seed: int,
    learning_rate: float,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    label_to_index = {label: index for index, label in enumerate(GATE_LABELS)}
    all_rows = [*train_rows, *test_rows, *hard_unknown_rows, *extra_rows]
    values = np.asarray(
        [
            [float(row[column]) for column in TRAINING_FEATURE_COLUMNS]
            for row in all_rows
        ],
        dtype=np.float32,
    )
    normalized, feature_stats = _normalize(
        values[: len(train_rows)],
        values,
    )
    x = torch.tensor(normalized, dtype=torch.float32)
    y = torch.tensor(
        [label_to_index[row["gate_label"]] for row in train_rows],
        dtype=torch.long,
    )
    model = AudioRouterMLP(
        input_dim=len(TRAINING_FEATURE_COLUMNS),
        num_classes=len(GATE_LABELS),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x[: len(train_rows)]), y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = F.softmax(model(x), dim=1).cpu().numpy().astype(float)
    test_start = len(train_rows)
    hard_start = test_start + len(test_rows)
    extra_start = hard_start + len(hard_unknown_rows)
    return {
        "model_state_dict": model.state_dict(),
        "feature_stats": feature_stats,
        "test_probabilities": probabilities[test_start:hard_start].tolist(),
        "hard_unknown_probabilities": probabilities[
            hard_start:extra_start
        ].tolist(),
        "extra_probabilities": probabilities[extra_start:].tolist(),
    }


def _prediction_rows(
    rows: Sequence[dict[str, str]],
    probabilities: Sequence[Sequence[float]],
    *,
    threshold: float,
    include_truth: bool,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row, values in zip(rows, probabilities):
        predicted_index = int(np.argmax(values))
        confidence = float(values[predicted_index])
        result = {
            "sample_id": row["sample_id"],
            "path": row["path"],
            "filename": row["filename"],
            "predicted_gate": GATE_LABELS[predicted_index],
            "confidence": f"{confidence:.10f}",
            "accepted": str(confidence >= threshold).lower(),
            "prob_speech_present": f"{float(values[0]):.10f}",
            "prob_non_speech": f"{float(values[1]):.10f}",
        }
        if include_truth:
            result["true_gate"] = row["gate_label"]
            result["human_label"] = row["human_label"]
            result["true_workflow"] = row["workflow_label"]
        output.append(result)
    return output


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _threshold_rows(
    rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for threshold in EVALUATION_THRESHOLDS:
        accepted = [
            row for row in rows if float(row["confidence"]) >= threshold
        ]
        accepted_correct = sum(
            row["true_gate"] == row["predicted_gate"] for row in accepted
        )
        output.append(
            {
                "threshold": f"{threshold:.2f}",
                "accepted": str(len(accepted)),
                "total": str(len(rows)),
                "coverage": f"{_safe_ratio(len(accepted), len(rows)):.10f}",
                "accepted_correct": str(accepted_correct),
                "accepted_accuracy": (
                    f"{_safe_ratio(accepted_correct, len(accepted)):.10f}"
                ),
            }
        )
    return output


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_baseline_predictions(router_output_dir: str | Path) -> list[dict[str, str]]:
    path = Path(router_output_dir).expanduser().resolve() / "test_predictions.csv"
    if not path.is_file():
        raise ValueError(f"Baseline router predictions not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "path",
            "filename",
            "true_workflow",
            "predicted_workflow",
            "confidence",
            "accepted",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Baseline router predictions are missing columns: "
                + ", ".join(sorted(missing))
            )
        return list(reader)


def _combined_guarded_evaluation(
    baseline_rows: Sequence[dict[str, str]],
    gate_rows: Sequence[dict[str, str]],
    *,
    threshold: float,
) -> tuple[list[dict[str, str]], dict[str, float | int], list[dict[str, str]]]:
    gate_by_id = {row["sample_id"]: row for row in gate_rows}
    combined: list[dict[str, str]] = []
    for baseline in baseline_rows:
        gate = gate_by_id.get(baseline["sample_id"])
        if gate is None:
            continue
        baseline_accepted = baseline["accepted"].lower() == "true"
        gate_passes = (
            gate["predicted_gate"] == "speech_present"
            and float(gate["confidence"]) >= threshold
        )
        guarded_workflow = baseline["predicted_workflow"]
        guarded_accepted = baseline_accepted
        guard_applied = False
        if guarded_workflow == SPEECH_CLEANUP and not gate_passes:
            guarded_workflow = SAFE_ABSTAIN
            guarded_accepted = False
            guard_applied = True
        combined.append(
            {
                "sample_id": baseline["sample_id"],
                "path": baseline["path"],
                "filename": baseline["filename"],
                "true_workflow": baseline["true_workflow"],
                "baseline_workflow": baseline["predicted_workflow"],
                "baseline_confidence": baseline["confidence"],
                "baseline_accepted": str(baseline_accepted).lower(),
                "gate_prediction": gate["predicted_gate"],
                "gate_confidence": gate["confidence"],
                "gate_accepted": gate["accepted"],
                "guard_applied": str(guard_applied).lower(),
                "guarded_workflow": guarded_workflow,
                "guarded_accepted": str(guarded_accepted).lower(),
            }
        )

    def dangerous(row: dict[str, str], workflow_field: str, accepted_field: str) -> bool:
        return (
            row["true_workflow"] in DANGEROUS_TRUE_WORKFLOWS
            and row[workflow_field] == SPEECH_CLEANUP
            and row[accepted_field] == "true"
        )

    baseline_dangerous = sum(
        dangerous(row, "baseline_workflow", "baseline_accepted")
        for row in combined
    )
    guarded_dangerous_rows = [
        row
        for row in combined
        if dangerous(row, "guarded_workflow", "guarded_accepted")
    ]
    baseline_accepted_rows = [
        row for row in combined if row["baseline_accepted"] == "true"
    ]
    guarded_accepted_rows = [
        row for row in combined if row["guarded_accepted"] == "true"
    ]
    metrics: dict[str, float | int] = {
        "evaluation_rows": len(combined),
        "baseline_rows": len(baseline_rows),
        "combined_rows": len(combined),
        "missing_gate_rows": len(baseline_rows) - len(combined),
        "baseline_dangerous_speech_invocations": baseline_dangerous,
        "guarded_dangerous_speech_invocations": len(guarded_dangerous_rows),
        "baseline_workflow_accepted_errors": sum(
            row["true_workflow"] != row["baseline_workflow"]
            for row in baseline_accepted_rows
        ),
        "guarded_workflow_accepted_errors": sum(
            row["true_workflow"] != row["guarded_workflow"]
            for row in guarded_accepted_rows
        ),
        "baseline_coverage_at_threshold": _safe_ratio(
            len(baseline_accepted_rows), len(combined)
        ),
        "guarded_coverage_at_threshold": _safe_ratio(
            len(guarded_accepted_rows), len(combined)
        ),
        "guarded_missed_speech_cleanup": sum(
            row["true_workflow"] == SPEECH_CLEANUP
            and row["guarded_workflow"] != SPEECH_CLEANUP
            for row in combined
        ),
        "guarded_safe_abstain_count": sum(
            row["guarded_workflow"] == SAFE_ABSTAIN for row in combined
        ),
    }
    return combined, metrics, guarded_dangerous_rows


def train_router_speech_gate(
    *,
    manifest_path: str | Path,
    router_output_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    test_ratio: float = 0.25,
    max_epochs: int = 80,
    threshold: float = 0.70,
    learning_rate: float = 0.01,
) -> Path:
    """Train the gate and measure its effect on accepted workflow errors."""
    if max_epochs <= 0:
        raise ValueError("--max-epochs must be greater than zero.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    rows = _load_and_validate_manifest(manifest_path)
    baseline_rows = _load_baseline_predictions(router_output_dir)
    baseline_sample_ids = {row["sample_id"] for row in baseline_rows}
    manifest_by_id = {row["sample_id"]: row for row in rows}
    missing_manifest_ids = [
        row["sample_id"]
        for row in baseline_rows
        if row["sample_id"] not in manifest_by_id
    ]
    if missing_manifest_ids:
        raise ValueError(
            "Baseline router sample_id is missing from the validated manifest: "
            + ", ".join(missing_manifest_ids[:5])
        )
    split_rows, hard_unknown_rows = _prepare_split_rows(
        rows,
        test_ratio=test_ratio,
        seed=seed,
        excluded_sample_ids=baseline_sample_ids,
    )
    combined_evaluation_rows = [
        {
            **manifest_by_id[baseline["sample_id"]],
            "gate_label": GATE_BY_HUMAN_LABEL.get(
                manifest_by_id[baseline["sample_id"]]["human_label"],
                "hard_unknown",
            ),
            "split": "combined_evaluation",
        }
        for baseline in baseline_rows
    ]
    feature_rows = _extract_feature_rows(
        [*split_rows, *hard_unknown_rows, *combined_evaluation_rows]
    )
    failed = [row for row in feature_rows if row["status"] != "success"]
    if failed:
        raise ValueError(
            f"Audio feature extraction failed for {len(failed)} row(s); "
            f"first failure {failed[0]['sample_id']}: {failed[0]['error']}"
        )
    train_rows = [row for row in feature_rows if row["split"] == "train"]
    test_rows = [row for row in feature_rows if row["split"] == "test"]
    hard_rows = [row for row in feature_rows if row["split"] == "hard_unknown"]
    combined_score_rows = [
        row for row in feature_rows if row["split"] == "combined_evaluation"
    ]
    fit = _fit_gate_model(
        train_rows=train_rows,
        test_rows=test_rows,
        hard_unknown_rows=hard_rows,
        extra_rows=combined_score_rows,
        max_epochs=max_epochs,
        seed=seed,
        learning_rate=learning_rate,
    )
    test_predictions = _prediction_rows(
        test_rows,
        fit["test_probabilities"],
        threshold=threshold,
        include_truth=True,
    )
    hard_predictions = _prediction_rows(
        hard_rows,
        fit["hard_unknown_probabilities"],
        threshold=threshold,
        include_truth=False,
    )
    combined_gate_predictions = _prediction_rows(
        combined_score_rows,
        fit["extra_probabilities"],
        threshold=threshold,
        include_truth=False,
    )
    truth = [GATE_LABELS.index(row["true_gate"]) for row in test_predictions]
    predicted = [
        GATE_LABELS.index(row["predicted_gate"]) for row in test_predictions
    ]
    report = _classification_report(
        truth=truth,
        predictions=predicted,
        index_to_label={index: label for index, label in enumerate(GATE_LABELS)},
    )
    accepted = [
        row for row in test_predictions if float(row["confidence"]) >= threshold
    ]
    accepted_correct = sum(
        row["true_gate"] == row["predicted_gate"] for row in accepted
    )
    threshold_rows = _threshold_rows(test_predictions)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": fit["model_state_dict"],
            "config": {
                "feature_columns": list(TRAINING_FEATURE_COLUMNS),
                "label_to_index": {
                    label: index for index, label in enumerate(GATE_LABELS)
                },
                "feature_stats": fit["feature_stats"],
                "threshold": threshold,
            },
        },
        output / "checkpoint.pt",
    )
    (output / "feature_stats.json").write_text(
        json.dumps(fit["feature_stats"], indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output / "split_manifest.csv",
        [
            "sample_id",
            "path",
            "filename",
            "human_label",
            "workflow_label",
            "gate_label",
            "split",
        ],
        [*split_rows, *hard_unknown_rows],
    )
    prediction_fields = [
        "sample_id",
        "path",
        "filename",
        "human_label",
        "true_workflow",
        "true_gate",
        "predicted_gate",
        "confidence",
        "accepted",
        "prob_speech_present",
        "prob_non_speech",
    ]
    _write_csv(
        output / "test_predictions.csv",
        prediction_fields,
        test_predictions,
    )
    _write_csv(
        output / "threshold_metrics.csv",
        [
            "threshold",
            "accepted",
            "total",
            "coverage",
            "accepted_correct",
            "accepted_accuracy",
        ],
        threshold_rows,
    )
    if hard_rows:
        _write_csv(
            output / "hard_unknown_predictions.csv",
            [
                "sample_id",
                "path",
                "filename",
                "predicted_gate",
                "confidence",
                "accepted",
                "prob_speech_present",
                "prob_non_speech",
            ],
            hard_predictions,
        )
    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1"],
        "speech_present_recall": report["per_class"]["speech_present"]["recall"],
        "non_speech_recall": report["per_class"]["non_speech"]["recall"],
        "false_speech_invocations": sum(
            row["true_gate"] == "non_speech"
            and row["predicted_gate"] == "speech_present"
            for row in test_predictions
        ),
        "missed_speech_present": sum(
            row["true_gate"] == "speech_present"
            and row["predicted_gate"] == "non_speech"
            for row in test_predictions
        ),
        "threshold": threshold,
        "accepted_at_threshold": len(accepted),
        "coverage_at_threshold": _safe_ratio(len(accepted), len(test_predictions)),
        "accepted_accuracy_at_threshold": _safe_ratio(
            accepted_correct, len(accepted)
        ),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "excluded_unknown_rows": len(hard_rows),
        "per_class": report["per_class"],
        "confusion_matrix": report["confusion_matrix"],
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    combined_rows, combined_metrics, dangerous_rows = (
        _combined_guarded_evaluation(
            baseline_rows,
            combined_gate_predictions,
            threshold=threshold,
        )
    )
    combined_metrics["excluded_from_gate_training_for_combined_eval"] = len(
        baseline_sample_ids
    )
    combined_fields = [
        "sample_id",
        "path",
        "filename",
        "true_workflow",
        "baseline_workflow",
        "baseline_confidence",
        "baseline_accepted",
        "gate_prediction",
        "gate_confidence",
        "gate_accepted",
        "guard_applied",
        "guarded_workflow",
        "guarded_accepted",
    ]
    _write_csv(
        output / "combined_guarded_predictions.csv",
        combined_fields,
        combined_rows,
    )
    _write_csv(
        output / "combined_accepted_dangerous_errors.csv",
        combined_fields,
        dangerous_rows,
    )
    (output / "combined_guarded_metrics.json").write_text(
        json.dumps(combined_metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Speech gate: accuracy={metrics['accuracy']:.4f}, "
        f"speech_recall={metrics['speech_present_recall']:.4f}, "
        f"non_speech_recall={metrics['non_speech_recall']:.4f}"
    )
    print(
        "Dangerous speech invocations: "
        f"baseline={combined_metrics['baseline_dangerous_speech_invocations']}, "
        f"guarded={combined_metrics['guarded_dangerous_speech_invocations']}; "
        f"guarded coverage={combined_metrics['guarded_coverage_at_threshold']:.4f}"
    )
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a binary speech-present router safety gate."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--router-output-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--test-ratio", default=0.25, type=float)
    parser.add_argument("--max-epochs", default=80, type=int)
    parser.add_argument("--threshold", default=0.70, type=float)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        train_router_speech_gate(
            manifest_path=args.manifest,
            router_output_dir=args.router_output_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            test_ratio=args.test_ratio,
            max_epochs=args.max_epochs,
            threshold=args.threshold,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
