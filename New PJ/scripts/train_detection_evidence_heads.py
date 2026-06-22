#!/usr/bin/env python3
"""Train three binary detector heads from Detection Evidence Dataset v1."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import torch
from torch import nn
import torch.nn.functional as F

from scripts.build_detection_evidence_dataset import (
    ROUTER_LABELS,
    SIGNAL_FEATURE_COLUMNS,
)


HEADS = ("speech_present", "music_present", "target_event_present")
FEATURE_COLUMNS = [
    *SIGNAL_FEATURE_COLUMNS,
    *[f"router_prob_{label}" for label in ROUTER_LABELS],
    "speech_gate_prob_non_speech",
    "speech_gate_prob_speech_present",
    "top_score",
    "second_score",
    "score_margin",
]
CLASS_WEIGHTING_CHOICES = {"none", "balanced"}
PREDICTION_THRESHOLD = 0.5


class DetectionEvidenceHeads(nn.Module):
    """Small shared representation with three binary logits."""

    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict(
            {head: nn.Linear(hidden_dim, 1) for head in HEADS}
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(features)
        return {
            head: layer(shared).squeeze(1)
            for head, layer in self.heads.items()
        }


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "input_path",
            "split",
            "content_label",
            "contains_speech",
            "contains_music",
            "contains_target_noise",
            "target_noise_label",
            *FEATURE_COLUMNS,
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Detection evidence CSV is missing training columns: "
                + ", ".join(sorted(missing))
            )
        rows = [
            row
            for row in reader
            if row.get("status", "success") == "success"
            and row.get("split") in {"train", "test"}
        ]
    if not rows:
        raise ValueError("Detection evidence CSV has no successful train/test rows.")
    return rows


def _targets(row: Mapping[str, str]) -> dict[str, int]:
    return {
        "speech_present": int(_truthy(row.get("contains_speech"))),
        "music_present": int(
            _truthy(row.get("contains_music"))
            or row.get("content_label") == "music_with_vocals"
        ),
        "target_event_present": int(
            _truthy(row.get("contains_target_noise"))
            or bool(str(row.get("target_noise_label") or "").strip())
        ),
    }


def _feature_value(row: Mapping[str, str], column: str) -> float:
    text = str(row.get(column) or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"Sample {row.get('sample_id')} has non-numeric {column}: {text}"
        ) from exc


def _validate_coverage(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for head in HEADS:
        counts[head] = {}
        for split in ("train", "test"):
            split_counts = Counter(
                _targets(row)[head] for row in rows if row["split"] == split
            )
            counts[head][split] = {
                "negative": split_counts.get(0, 0),
                "positive": split_counts.get(1, 0),
            }
            if not split_counts.get(0) or not split_counts.get(1):
                raise ValueError(
                    f"Head {head} lacks positive or negative examples in {split}: "
                    f"positive={split_counts.get(1, 0)}, "
                    f"negative={split_counts.get(0, 0)}"
                )
    return counts


def _binary_metrics(truth: Sequence[int], predictions: Sequence[int]) -> dict[str, object]:
    true_positive = sum(t == 1 and p == 1 for t, p in zip(truth, predictions))
    true_negative = sum(t == 0 and p == 0 for t, p in zip(truth, predictions))
    false_positive = sum(t == 0 and p == 1 for t, p in zip(truth, predictions))
    false_negative = sum(t == 1 and p == 0 for t, p in zip(truth, predictions))
    total = len(truth)

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, true_positive + false_negative)
    return {
        "accuracy": ratio(true_positive + true_negative, total),
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * precision * recall, precision + recall),
        "confusion": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
    }


def train_detection_evidence_heads(
    *,
    evidence_csv: str | Path,
    output_dir: str | Path,
    epochs: int = 80,
    learning_rate: float = 0.01,
    seed: int = 42,
    class_weighting: str = "balanced",
) -> Path:
    """Train shared-input binary detector heads and write test metrics."""
    if epochs <= 0:
        raise ValueError("--epochs must be greater than zero.")
    if learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than zero.")
    if class_weighting not in CLASS_WEIGHTING_CHOICES:
        raise ValueError(
            "--class-weighting must be one of: "
            + ", ".join(sorted(CLASS_WEIGHTING_CHOICES))
        )
    rows = _load_rows(Path(evidence_csv).expanduser().resolve(strict=True))
    split_counts = _validate_coverage(rows)
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    ordered_rows = [*train_rows, *test_rows]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    all_values = np.asarray(
        [
            [_feature_value(row, column) for column in FEATURE_COLUMNS]
            for row in ordered_rows
        ],
        dtype=np.float32,
    )
    train_values = all_values[: len(train_rows)]
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = ((all_values - mean) / std).astype(np.float32)
    x = torch.tensor(normalized, dtype=torch.float32)
    train_size = len(train_rows)
    targets = {
        head: torch.tensor(
            [_targets(row)[head] for row in ordered_rows],
            dtype=torch.float32,
        )
        for head in HEADS
    }
    model = DetectionEvidenceHeads(len(FEATURE_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positive_weights: dict[str, float] = {}
    for head in HEADS:
        positive = split_counts[head]["train"]["positive"]
        negative = split_counts[head]["train"]["negative"]
        positive_weights[head] = (
            float(negative / positive) if class_weighting == "balanced" else 1.0
        )
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x[:train_size])
        losses = [
            F.binary_cross_entropy_with_logits(
                logits[head],
                targets[head][:train_size],
                pos_weight=torch.tensor(positive_weights[head]),
            )
            for head in HEADS
        ]
        loss = sum(losses)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(x[train_size:])
        test_probabilities = {
            head: torch.sigmoid(test_logits[head]).cpu().numpy().astype(float)
            for head in HEADS
        }
    metrics: dict[str, object] = {
        "threshold": PREDICTION_THRESHOLD,
        "feature_columns": FEATURE_COLUMNS,
        "class_weighting": class_weighting,
        "positive_weights": positive_weights,
        "counts_by_head_and_split": split_counts,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "heads": {},
    }
    for head in HEADS:
        truth = [_targets(row)[head] for row in test_rows]
        predictions = [
            int(probability >= PREDICTION_THRESHOLD)
            for probability in test_probabilities[head]
        ]
        metrics["heads"][head] = _binary_metrics(truth, predictions)

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_stats = {
        column: {"mean": float(mean[index]), "std": float(std[index])}
        for index, column in enumerate(FEATURE_COLUMNS)
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "feature_columns": FEATURE_COLUMNS,
                "feature_stats": feature_stats,
                "heads": list(HEADS),
                "threshold": PREDICTION_THRESHOLD,
            },
        },
        output / "checkpoint.pt",
    )
    (output / "feature_columns.json").write_text(
        json.dumps(
            {
                "feature_columns": FEATURE_COLUMNS,
                "feature_stats": feature_stats,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    prediction_fields = ["sample_id", "input_path"]
    for head in HEADS:
        prediction_fields.extend(
            [f"{head}_true", f"{head}_probability", f"{head}_predicted"]
        )
    with (output / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for index, row in enumerate(test_rows):
            output_row = {
                "sample_id": row["sample_id"],
                "input_path": row["input_path"],
            }
            for head in HEADS:
                probability = float(test_probabilities[head][index])
                output_row[f"{head}_true"] = str(_targets(row)[head])
                output_row[f"{head}_probability"] = f"{probability:.10f}"
                output_row[f"{head}_predicted"] = str(
                    int(probability >= PREDICTION_THRESHOLD)
                )
            writer.writerow(output_row)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train binary heads from Detection Evidence Dataset v1."
    )
    parser.add_argument("--evidence-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--class-weighting",
        choices=sorted(CLASS_WEIGHTING_CHOICES),
        default="balanced",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        output = train_detection_evidence_heads(
            evidence_csv=args.evidence_csv,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            class_weighting=args.class_weighting,
        )
        print(f"Wrote detection head artifacts: {output}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
