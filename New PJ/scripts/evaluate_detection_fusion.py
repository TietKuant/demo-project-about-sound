#!/usr/bin/env python3
"""Evaluate evidence-head fusion thresholds without changing runtime routing."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SPEECH_GENERAL_LABEL = "speech_present_general"
TARGET_LABEL = "speech_target_noise"
EVALUATION_LABELS = (
    "environment_only",
    "music_with_vocals",
    "speech_clean",
    "speech_noisy_general",
    TARGET_LABEL,
)
OUTPUT_COLUMNS = [
    "input_path",
    "content_label",
    "speech_present_probability",
    "music_present_probability",
    "target_event_present_probability",
    "speech_threshold",
    "music_threshold",
    "target_threshold",
    "raw_fusion_label",
    "normalized_fusion_label",
]


def _parse_thresholds(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("Threshold lists must not be empty.")
        try:
            thresholds = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"Invalid threshold list: {value}") from exc
    else:
        thresholds = [float(item) for item in value]
    if not thresholds or any(item < 0.0 or item > 1.0 for item in thresholds):
        raise ValueError("Thresholds must contain values between 0 and 1.")
    return thresholds


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _float_value(row: Mapping[str, str], column: str) -> float:
    value = str(row.get(column) or "").strip()
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {column} for input_path={row.get('input_path')}: {value}"
        ) from exc


def _join_test_rows(
    evidence_rows: Sequence[dict[str, str]],
    prediction_rows: Sequence[dict[str, str]],
) -> list[dict[str, object]]:
    predictions_by_path: dict[str, dict[str, str]] = {}
    for row in prediction_rows:
        input_path = row["input_path"]
        if input_path in predictions_by_path:
            raise ValueError(f"Duplicate predictions input_path: {input_path}")
        predictions_by_path[input_path] = row

    joined: list[dict[str, object]] = []
    seen_evidence: set[str] = set()
    for evidence in evidence_rows:
        if evidence.get("split") != "test":
            continue
        input_path = evidence["input_path"]
        if input_path in seen_evidence:
            raise ValueError(f"Duplicate evidence input_path: {input_path}")
        seen_evidence.add(input_path)
        prediction = predictions_by_path.get(input_path)
        if prediction is None:
            raise ValueError(f"Missing predictions for test input_path: {input_path}")
        joined.append(
            {
                "input_path": input_path,
                "content_label": evidence["content_label"],
                "speech_probability": _float_value(
                    prediction, "speech_present_probability"
                ),
                "music_probability": _float_value(
                    prediction, "music_present_probability"
                ),
                "target_probability": _float_value(
                    prediction, "target_event_present_probability"
                ),
            }
        )
    if not joined:
        raise ValueError("Evidence CSV has no test rows.")
    return joined


def _fusion_label(
    row: Mapping[str, object],
    *,
    speech_threshold: float,
    music_threshold: float,
    target_threshold: float,
) -> str:
    if float(row["music_probability"]) >= music_threshold:
        return "music_with_vocals"
    if (
        float(row["speech_probability"]) >= speech_threshold
        and float(row["target_probability"]) >= target_threshold
    ):
        return TARGET_LABEL
    if float(row["speech_probability"]) >= speech_threshold:
        return SPEECH_GENERAL_LABEL
    return "environment_only"


def _normalized_label(raw_label: str, true_label: str) -> str:
    if raw_label == SPEECH_GENERAL_LABEL and true_label in {
        "speech_clean",
        "speech_noisy_general",
    }:
        return true_label
    return raw_label


def _evaluate_thresholds(
    rows: Sequence[dict[str, object]],
    *,
    speech_threshold: float,
    music_threshold: float,
    target_threshold: float,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    true_labels = sorted({str(row["content_label"]) for row in rows})
    predicted_labels = set(EVALUATION_LABELS) | {SPEECH_GENERAL_LABEL}
    confusion = {
        true_label: {predicted: 0 for predicted in sorted(predicted_labels)}
        for true_label in true_labels
    }
    output_rows: list[dict[str, str]] = []
    false_target_counts = Counter()

    for row in rows:
        true_label = str(row["content_label"])
        raw_label = _fusion_label(
            row,
            speech_threshold=speech_threshold,
            music_threshold=music_threshold,
            target_threshold=target_threshold,
        )
        normalized_label = _normalized_label(raw_label, true_label)
        confusion[true_label].setdefault(normalized_label, 0)
        confusion[true_label][normalized_label] += 1
        if normalized_label == TARGET_LABEL and true_label != TARGET_LABEL:
            false_target_counts[true_label] += 1
        output_rows.append(
            {
                "input_path": str(row["input_path"]),
                "content_label": true_label,
                "speech_present_probability": str(row["speech_probability"]),
                "music_present_probability": str(row["music_probability"]),
                "target_event_present_probability": str(row["target_probability"]),
                "speech_threshold": str(speech_threshold),
                "music_threshold": str(music_threshold),
                "target_threshold": str(target_threshold),
                "raw_fusion_label": raw_label,
                "normalized_fusion_label": normalized_label,
            }
        )

    recall_by_label = {}
    for label in true_labels:
        total = sum(confusion[label].values())
        recall_by_label[label] = (
            float(confusion[label].get(label, 0) / total) if total else 0.0
        )
    macro_recall = (
        float(sum(recall_by_label.values()) / len(recall_by_label))
        if recall_by_label
        else 0.0
    )
    false_env = false_target_counts["environment_only"]
    false_clean = false_target_counts["speech_clean"]
    false_noisy = false_target_counts["speech_noisy_general"]
    false_target_penalty = float(
        (false_env + false_clean + false_noisy) / len(rows)
    )
    result = {
        "speech_threshold": speech_threshold,
        "music_threshold": music_threshold,
        "target_threshold": target_threshold,
        "confusion": confusion,
        "recall_by_label": recall_by_label,
        "false_env_to_target": false_env,
        "false_clean_to_target": false_clean,
        "false_noisy_to_target": false_noisy,
        "macro_recall": macro_recall,
        "score_for_selection": macro_recall - false_target_penalty,
    }
    return result, output_rows


def _selection_key(result: Mapping[str, object], score_field: str) -> tuple[float, ...]:
    return (
        float(result[score_field]),
        float(result["macro_recall"]),
        float(result["speech_threshold"]),
        float(result["music_threshold"]),
        float(result["target_threshold"]),
    )


def _meets_research_constraints(result: Mapping[str, object]) -> bool:
    recall = result["recall_by_label"]
    return (
        float(recall.get("environment_only", 0.0)) >= 0.88
        and float(recall.get("music_with_vocals", 0.0)) >= 0.90
        and float(recall.get(TARGET_LABEL, 0.0)) >= 0.60
        and float(recall.get("speech_noisy_general", 0.0)) >= 0.90
        and int(result["false_env_to_target"]) <= 3
        and int(result["false_clean_to_target"]) <= 3
        and int(result["false_noisy_to_target"]) <= 5
    )


def evaluate_detection_fusion(
    *,
    evidence_csv: str | Path,
    predictions_csv: str | Path,
    output_json: str | Path,
    output_csv: str | Path | None = None,
    speech_thresholds: str | Sequence[float] = "0.6,0.7,0.8",
    music_thresholds: str | Sequence[float] = "0.5,0.6,0.7",
    target_thresholds: str | Sequence[float] = "0.6,0.7,0.8,0.9",
) -> dict[str, object]:
    """Evaluate a threshold grid and emit the selected research candidate."""
    evidence_rows = _read_csv(
        Path(evidence_csv).expanduser().resolve(strict=True),
        {"input_path", "content_label", "split"},
    )
    prediction_rows = _read_csv(
        Path(predictions_csv).expanduser().resolve(strict=True),
        {
            "input_path",
            "speech_present_probability",
            "music_present_probability",
            "target_event_present_probability",
        },
    )
    joined_rows = _join_test_rows(evidence_rows, prediction_rows)
    speech_values = _parse_thresholds(speech_thresholds)
    music_values = _parse_thresholds(music_thresholds)
    target_values = _parse_thresholds(target_thresholds)

    threshold_results = []
    output_by_thresholds: dict[tuple[float, float, float], list[dict[str, str]]] = {}
    for speech_threshold, music_threshold, target_threshold in product(
        speech_values, music_values, target_values
    ):
        result, fused_rows = _evaluate_thresholds(
            joined_rows,
            speech_threshold=speech_threshold,
            music_threshold=music_threshold,
            target_threshold=target_threshold,
        )
        threshold_results.append(result)
        output_by_thresholds[
            (speech_threshold, music_threshold, target_threshold)
        ] = fused_rows

    best_macro = max(
        threshold_results,
        key=lambda result: _selection_key(result, "macro_recall"),
    )
    best_balanced = max(
        threshold_results,
        key=lambda result: _selection_key(result, "score_for_selection"),
    )
    eligible = [
        result for result in threshold_results if _meets_research_constraints(result)
    ]
    recommended = (
        max(
            eligible,
            key=lambda result: _selection_key(result, "score_for_selection"),
        )
        if eligible
        else None
    )
    recommended_thresholds: dict[str, float] | None
    if recommended is None:
        recommended_thresholds = None
        recommendation_reason = (
            "No threshold combination satisfied research constraints."
        )
    else:
        recommended_thresholds = {
            "speech_threshold": float(recommended["speech_threshold"]),
            "music_threshold": float(recommended["music_threshold"]),
            "target_threshold": float(recommended["target_threshold"]),
        }
        recommendation_reason = (
            "Best balanced candidate satisfying research constraints."
        )

    report = {
        "row_count": len(joined_rows),
        "threshold_results": threshold_results,
        "best_by_macro_recall": best_macro,
        "best_balanced_candidate": best_balanced,
        "recommended_research_thresholds": recommended_thresholds,
        "recommended_research_result": recommended,
        "recommended_research_reason": recommendation_reason,
    }
    destination = Path(output_json).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if output_csv is not None:
        csv_destination = Path(output_csv).expanduser().resolve()
        csv_destination.parent.mkdir(parents=True, exist_ok=True)
        selected_rows: list[dict[str, str]] = []
        if recommended is not None:
            key = (
                float(recommended["speech_threshold"]),
                float(recommended["music_threshold"]),
                float(recommended["target_threshold"]),
            )
            selected_rows = output_by_thresholds[key]
        with csv_destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(selected_rows)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate evidence-head fusion threshold combinations."
    )
    parser.add_argument("--evidence-csv", required=True, type=Path)
    parser.add_argument("--predictions-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--speech-thresholds", default="0.6,0.7,0.8")
    parser.add_argument("--music-thresholds", default="0.5,0.6,0.7")
    parser.add_argument("--target-thresholds", default="0.6,0.7,0.8,0.9")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        report = evaluate_detection_fusion(
            evidence_csv=args.evidence_csv,
            predictions_csv=args.predictions_csv,
            output_json=args.output_json,
            output_csv=args.output_csv,
            speech_thresholds=args.speech_thresholds,
            music_thresholds=args.music_thresholds,
            target_thresholds=args.target_thresholds,
        )
        print(
            json.dumps(
                {
                    "recommended_research_thresholds": report[
                        "recommended_research_thresholds"
                    ],
                    "reason": report["recommended_research_reason"],
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
