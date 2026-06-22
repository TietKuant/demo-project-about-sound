#!/usr/bin/env python3
"""Train a measurable router baseline from a human-labeled manifest."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import torch
import torch.nn.functional as F

from scripts.extract_audio_features import _audio_for_features, _features
from scripts.train_audio_router import (
    FEATURE_COLUMNS,
    TRAINING_FEATURE_COLUMNS,
    _classification_report,
)
from src.router.audio_router_model import AudioRouterMLP


DEFAULT_LABELS = [
    "speech_clean",
    "speech_noisy_general",
    "speech_target_noise",
    "music_with_vocals",
    "environment_only",
]
UNKNOWN_LABEL = "unknown_mixed"
CONFIDENCE_THRESHOLD = 0.70
REQUIRED_COLUMNS = {
    "sample_id",
    "path",
    "human_label",
    "workflow_label",
}
FEATURE_ROW_COLUMNS = [
    "sample_id",
    "path",
    "filename",
    "human_label",
    "split",
    "status",
    *FEATURE_COLUMNS,
    "error",
]
SPLIT_COLUMNS = [
    "sample_id",
    "path",
    "filename",
    "human_label",
    "workflow_label",
    "split",
]


def _load_and_validate_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Human-labeled router manifest is missing columns: "
                + ", ".join(sorted(missing))
            )
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError(f"Human-labeled router manifest is empty: {manifest}")

    rows: list[dict[str, str]] = []
    seen_sample_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for row_number, row in enumerate(raw_rows, start=2):
        sample_id = (row.get("sample_id") or "").strip()
        human_label = (row.get("human_label") or "").strip()
        workflow_label = (row.get("workflow_label") or "").strip()
        raw_path = (row.get("path") or "").strip()
        if not sample_id:
            raise ValueError(f"Row {row_number} has blank sample_id.")
        if not human_label:
            raise ValueError(f"Row {row_number} has blank human_label.")
        if not workflow_label:
            raise ValueError(f"Row {row_number} has blank workflow_label.")
        if human_label not in {*DEFAULT_LABELS, UNKNOWN_LABEL}:
            raise ValueError(
                f"Row {row_number} has unsupported human_label: {human_label}"
            )
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)

        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = manifest.parent / path
        path = path.resolve(strict=False)
        if not path.is_file():
            raise ValueError(f"Row {row_number} path does not exist: {path}")
        if path in seen_paths:
            raise ValueError(f"Duplicate path: {path}")
        seen_paths.add(path)
        rows.append(
            {
                "sample_id": sample_id,
                "path": str(path),
                "filename": (row.get("filename") or path.name).strip(),
                "human_label": human_label,
                "workflow_label": workflow_label,
            }
        )
    return rows


def _stratified_split(
    rows: Sequence[dict[str, str]],
    *,
    labels: Sequence[str],
    test_ratio: float,
    seed: int,
    allow_small_classes: bool,
) -> list[dict[str, str]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("--test-ratio must be between 0 and 1.")
    rng = random.Random(seed)
    split_rows: list[dict[str, str]] = []
    for label in labels:
        class_rows = [dict(row) for row in rows if row["human_label"] == label]
        if not class_rows:
            raise ValueError(f"No labeled rows found for trainable class: {label}")
        if len(class_rows) < 2 and not allow_small_classes:
            raise ValueError(
                f"Class {label} has {len(class_rows)} row; at least 2 are required "
                "for stratified train/test splitting. Use --allow-small-classes "
                "to keep singleton classes in training only."
            )
        rng.shuffle(class_rows)
        if len(class_rows) == 1:
            test_count = 0
        else:
            test_count = round(len(class_rows) * test_ratio)
            test_count = max(1, min(len(class_rows) - 1, test_count))
        for index, row in enumerate(class_rows):
            row["split"] = "test" if index < test_count else "train"
            split_rows.append(row)
    if not any(row["split"] == "test" for row in split_rows):
        raise ValueError("The stratified split produced no test rows.")
    return split_rows


def _prepare_split_rows(
    rows: Sequence[dict[str, str]],
    *,
    include_unknown_class: bool,
    test_ratio: float,
    seed: int,
    allow_small_classes: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    labels = list(DEFAULT_LABELS)
    if include_unknown_class:
        labels.append(UNKNOWN_LABEL)
    trainable_rows = [row for row in rows if row["human_label"] in labels]
    split_rows = _stratified_split(
        trainable_rows,
        labels=labels,
        test_ratio=test_ratio,
        seed=seed,
        allow_small_classes=allow_small_classes,
    )
    hard_unknown_rows: list[dict[str, str]] = []
    if not include_unknown_class:
        hard_unknown_rows = [
            {**row, "split": "hard_unknown"}
            for row in rows
            if row["human_label"] == UNKNOWN_LABEL
        ]
    return split_rows, hard_unknown_rows, labels


def _extract_feature_rows(
    rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    feature_rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="human-router-features-") as temp_name:
        temp_dir = Path(temp_name)
        for manifest_row in rows:
            path = Path(manifest_row["path"])
            row = {
                **manifest_row,
                "status": "failed",
                "error": "",
                **{column: "" for column in FEATURE_COLUMNS},
            }
            try:
                audio, sample_rate = _audio_for_features(path, temp_dir)
                features = _features(audio, sample_rate)
                row["status"] = "success"
                row["duration_sec"] = f"{audio.size / sample_rate:.6f}"
                for column in TRAINING_FEATURE_COLUMNS:
                    row[column] = f"{features[column]:.10f}"
            except Exception as exc:
                row["error"] = str(exc)
            feature_rows.append(row)
    return feature_rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _normalize_features(
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


def _fit_model(
    *,
    train_rows: Sequence[dict[str, str]],
    test_rows: Sequence[dict[str, str]],
    hard_unknown_rows: Sequence[dict[str, str]],
    labels: Sequence[str],
    max_epochs: int,
    seed: int,
    learning_rate: float,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    label_to_index = {label: index for index, label in enumerate(labels)}
    all_rows = [*train_rows, *test_rows, *hard_unknown_rows]
    values = np.asarray(
        [
            [float(row[column]) for column in TRAINING_FEATURE_COLUMNS]
            for row in all_rows
        ],
        dtype=np.float32,
    )
    normalized, feature_stats = _normalize_features(
        values[: len(train_rows)],
        values,
    )
    x = torch.tensor(normalized, dtype=torch.float32)
    train_targets = torch.tensor(
        [label_to_index[row["human_label"]] for row in train_rows],
        dtype=torch.long,
    )
    model = AudioRouterMLP(
        input_dim=len(TRAINING_FEATURE_COLUMNS),
        num_classes=len(labels),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x[: len(train_rows)])
        loss = F.cross_entropy(logits, train_targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probabilities = F.softmax(model(x), dim=1).cpu().numpy().astype(float)
    test_start = len(train_rows)
    hard_start = test_start + len(test_rows)
    return {
        "model_state_dict": model.state_dict(),
        "feature_stats": feature_stats,
        "test_probabilities": probabilities[test_start:hard_start].tolist(),
        "hard_unknown_probabilities": probabilities[hard_start:].tolist(),
    }


def _prediction_rows(
    rows: Sequence[dict[str, str]],
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    include_truth: bool,
) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for row, row_probabilities in zip(rows, probabilities):
        predicted_index = int(np.argmax(row_probabilities))
        confidence = float(row_probabilities[predicted_index])
        output_row = {
            "sample_id": row["sample_id"],
            "path": row["path"],
            "filename": row["filename"],
            "predicted_label": labels[predicted_index],
            "confidence": f"{confidence:.10f}",
            "accepted": str(confidence >= CONFIDENCE_THRESHOLD).lower(),
        }
        if include_truth:
            output_row["true_label"] = row["human_label"]
        for index, label in enumerate(labels):
            output_row[f"prob_{label}"] = f"{float(row_probabilities[index]):.10f}"
        output_rows.append(output_row)
    return output_rows


def _confusion_csv_rows(
    matrix: dict[str, dict[str, int]],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {"true_label": label, **matrix[label]}
        for label in labels
    ]


def _worst_confusions(
    matrix: dict[str, dict[str, int]],
) -> list[tuple[str, str, int]]:
    pairs = [
        (true_label, predicted_label, count)
        for true_label, predictions in matrix.items()
        for predicted_label, count in predictions.items()
        if true_label != predicted_label and count
    ]
    return sorted(pairs, key=lambda item: (-item[2], item[0], item[1]))


def train_human_labeled_router(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    test_ratio: float = 0.25,
    max_epochs: int = 80,
    include_unknown_class: bool = False,
    allow_small_classes: bool = False,
    learning_rate: float = 0.01,
) -> Path:
    """Train and evaluate the existing MLP on confirmed human labels."""
    if max_epochs <= 0:
        raise ValueError("--max-epochs must be greater than zero.")
    if learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than zero.")
    manifest_rows = _load_and_validate_manifest(manifest_path)
    split_rows, hard_unknown_rows, labels = _prepare_split_rows(
        manifest_rows,
        include_unknown_class=include_unknown_class,
        test_ratio=test_ratio,
        seed=seed,
        allow_small_classes=allow_small_classes,
    )
    all_split_rows = [*split_rows, *hard_unknown_rows]
    feature_rows = _extract_feature_rows(all_split_rows)
    failed_rows = [row for row in feature_rows if row["status"] != "success"]
    if failed_rows:
        first = failed_rows[0]
        raise ValueError(
            f"Audio feature extraction failed for {len(failed_rows)} row(s); "
            f"first failure {first['sample_id']}: {first['error']}"
        )
    train_rows = [row for row in feature_rows if row["split"] == "train"]
    test_rows = [row for row in feature_rows if row["split"] == "test"]
    hard_feature_rows = [
        row for row in feature_rows if row["split"] == "hard_unknown"
    ]
    fit = _fit_model(
        train_rows=train_rows,
        test_rows=test_rows,
        hard_unknown_rows=hard_feature_rows,
        labels=labels,
        max_epochs=max_epochs,
        seed=seed,
        learning_rate=learning_rate,
    )
    test_prediction_rows = _prediction_rows(
        test_rows,
        fit["test_probabilities"],
        labels,
        include_truth=True,
    )
    truth = [labels.index(row["human_label"]) for row in test_rows]
    predictions = [
        labels.index(row["predicted_label"]) for row in test_prediction_rows
    ]
    report = _classification_report(
        truth=truth,
        predictions=predictions,
        index_to_label={index: label for index, label in enumerate(labels)},
    )
    hard_prediction_rows = _prediction_rows(
        hard_feature_rows,
        fit["hard_unknown_probabilities"],
        labels,
        include_truth=False,
    )

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    label_to_index = {label: index for index, label in enumerate(labels)}
    torch.save(
        {
            "model_state_dict": fit["model_state_dict"],
            "config": {
                "feature_columns": list(TRAINING_FEATURE_COLUMNS),
                "label_to_index": label_to_index,
                "feature_stats": fit["feature_stats"],
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "training_source": "human_labeled_router_manifest",
            },
        },
        output / "checkpoint.pt",
    )
    (output / "label_mapping.json").write_text(
        json.dumps(
            {
                "label_to_index": label_to_index,
                "index_to_label": {
                    str(index): label for index, label in enumerate(labels)
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "feature_stats.json").write_text(
        json.dumps(fit["feature_stats"], indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "feature_rows.csv", FEATURE_ROW_COLUMNS, feature_rows)
    _write_csv(output / "split_manifest.csv", SPLIT_COLUMNS, all_split_rows)
    prediction_fields = [
        "sample_id",
        "path",
        "filename",
        "true_label",
        "predicted_label",
        "confidence",
        "accepted",
        *[f"prob_{label}" for label in labels],
    ]
    _write_csv(
        output / "test_predictions.csv",
        prediction_fields,
        test_prediction_rows,
    )
    confusion_matrix = report["confusion_matrix"]
    _write_csv(
        output / "confusion_matrix.csv",
        ["true_label", *labels],
        _confusion_csv_rows(confusion_matrix, labels),
    )
    if hard_feature_rows:
        hard_fields = [
            "sample_id",
            "path",
            "filename",
            "predicted_label",
            "confidence",
            "accepted",
            *[f"prob_{label}" for label in labels],
        ]
        _write_csv(
            output / "hard_unknown_predictions.csv",
            hard_fields,
            hard_prediction_rows,
        )
    metrics = {
        "total_rows": len(manifest_rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "excluded_unknown_rows": len(hard_feature_rows),
        "labels": labels,
        "accuracy": report["accuracy"],
        "macro_precision": report["macro_precision"],
        "macro_recall": report["macro_recall"],
        "macro_f1": report["macro_f1"],
        "per_class": report["per_class"],
        "confusion_matrix": confusion_matrix,
        "counts_by_label": dict(
            sorted(Counter(row["human_label"] for row in manifest_rows).items())
        ),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Rows used: {len(train_rows) + len(test_rows)} "
        f"(train={len(train_rows)}, test={len(test_rows)})"
    )
    print(f"Rows excluded unknown: {len(hard_feature_rows)}")
    for label in labels:
        print(
            f"{label}: "
            f"{sum(row['human_label'] == label for row in split_rows)}"
        )
    print(
        f"Accuracy: {report['accuracy']:.4f}; "
        f"macro_f1: {report['macro_f1']:.4f}"
    )
    worst = _worst_confusions(confusion_matrix)
    if worst:
        print(
            "Worst confusions: "
            + ", ".join(
                f"{true_label}->{predicted_label}={count}"
                for true_label, predicted_label, count in worst[:5]
            )
        )
    else:
        print("Worst confusions: none")
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a baseline router from a human-labeled CSV."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--test-ratio", default=0.25, type=float)
    parser.add_argument("--max-epochs", default=80, type=int)
    parser.add_argument("--include-unknown-class", action="store_true")
    parser.add_argument("--allow-small-classes", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        train_human_labeled_router(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            seed=args.seed,
            test_ratio=args.test_ratio,
            max_epochs=args.max_epochs,
            include_unknown_class=args.include_unknown_class,
            allow_small_classes=args.allow_small_classes,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
