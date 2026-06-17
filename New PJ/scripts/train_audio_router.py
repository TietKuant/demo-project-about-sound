"""Train a baseline audio router classifier from lightweight audio features."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - exercised only when runtime lacks torch.
    raise RuntimeError("Audio router training requires torch to be installed.") from exc

from scripts.extract_audio_features import _audio_for_features, _features
from src.router.audio_router_model import AudioRouterMLP


V3_REQUIRED_COLUMNS = {"sample_id", "input_path", "router_label", "source", "split", "notes"}
V4_REQUIRED_COLUMNS = {"sample_id", "input_path", "content_label", "source_dataset", "split", "notes"}
MANIFEST_SCHEMAS = [
    {
        "manifest_schema": "audio_router_v3",
        "required_columns": V3_REQUIRED_COLUMNS,
        "label_column": "router_label",
        "source_column": "source",
    },
    {
        "manifest_schema": "audio_router_v4",
        "required_columns": V4_REQUIRED_COLUMNS,
        "label_column": "content_label",
        "source_column": "source_dataset",
    },
]
FEATURE_COLUMNS = [
    "duration_sec",
    "rms_energy",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff_hz",
    "spectral_flatness",
    "low_band_energy_ratio",
    "mid_band_energy_ratio",
    "high_band_energy_ratio",
    "rms_std",
    "zcr_std",
    "silence_ratio",
]
TRAINING_FEATURE_COLUMNS = [column for column in FEATURE_COLUMNS if column != "duration_sec"]
EXCLUDED_FEATURE_COLUMNS = [column for column in FEATURE_COLUMNS if column not in TRAINING_FEATURE_COLUMNS]
FEATURE_POLICY_NOTES = (
    "duration_sec is retained in feature_rows.csv for quality control, but excluded "
    "from default training features to reduce dataset/source fingerprint leakage."
)
FEATURE_ROW_COLUMNS = [
    "sample_id",
    "input_path",
    "router_label",
    "source",
    "split",
    "status",
    *FEATURE_COLUMNS,
    "notes",
    "error",
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _detect_manifest_schema(fieldnames: list[str] | None) -> dict[str, object]:
    columns = set(fieldnames or [])
    for schema in MANIFEST_SCHEMAS:
        required_columns = schema["required_columns"]
        if isinstance(required_columns, set) and required_columns <= columns:
            return schema

    missing_descriptions = []
    for schema in MANIFEST_SCHEMAS:
        required_columns = schema["required_columns"]
        if isinstance(required_columns, set):
            missing = ", ".join(sorted(required_columns - columns))
            missing_descriptions.append(f"{schema['manifest_schema']} missing: {missing or 'none'}")
    raise ValueError(
        "Audio router manifest must match either the V3 schema "
        "(sample_id,input_path,router_label,source,split,notes) or the V4 schema "
        "(sample_id,input_path,content_label,source_dataset,split,notes). "
        + "; ".join(missing_descriptions)
    )


def _normalize_manifest_row(row: dict[str, str], schema: dict[str, object]) -> dict[str, str]:
    label_column = str(schema["label_column"])
    source_column = str(schema["source_column"])
    return {
        "sample_id": row["sample_id"],
        "input_path": row["input_path"],
        "router_label": row[label_column],
        "source": row[source_column],
        "split": row["split"],
        "notes": row["notes"],
    }


def _load_manifest(manifest_path: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    with Path(manifest_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        schema = _detect_manifest_schema(reader.fieldnames)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Audio router manifest is empty: {manifest_path}")
    return [_normalize_manifest_row(row, schema) for row in rows], schema


def _resolve_input_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(manifest_path).resolve().parent / path).resolve()


def _extract_feature_rows(manifest_path: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    manifest_rows, schema = _load_manifest(manifest_path)
    feature_rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="audio-router-features-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for manifest_row in manifest_rows:
            input_path = _resolve_input_path(manifest_row["input_path"], manifest_path)
            row = {
                "sample_id": manifest_row["sample_id"],
                "input_path": str(input_path),
                "router_label": manifest_row["router_label"],
                "source": manifest_row["source"],
                "split": manifest_row["split"] or "train",
                "status": "failed",
                "notes": manifest_row["notes"],
                "error": "",
            }
            row.update({column: "" for column in FEATURE_COLUMNS})
            try:
                if not input_path.exists():
                    raise FileNotFoundError(f"Input file not found: {input_path}")
                audio, sample_rate = _audio_for_features(input_path, temp_dir)
                features = _features(audio, sample_rate)
                row["status"] = "success"
                row["duration_sec"] = f"{audio.size / sample_rate:.6f}"
                for column in FEATURE_COLUMNS:
                    if column != "duration_sec":
                        row[column] = f"{features[column]:.10f}"
            except Exception as exc:
                row["error"] = str(exc)
            feature_rows.append(row)
    return feature_rows, schema


def _write_feature_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FEATURE_ROW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _successful_training_rows(feature_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = [row for row in feature_rows if row["status"] == "success"]
    if not rows:
        raise ValueError("No valid audio feature rows were extracted.")
    if not any(row["split"] == "train" for row in rows):
        raise ValueError("Audio router training requires at least one train row.")
    if not any(row["split"] == "test" for row in rows):
        raise ValueError("Audio router training requires at least one test row.")
    return rows


def _feature_matrix(rows: list[dict[str, str]], feature_columns: list[str]) -> np.ndarray:
    return np.asarray([[float(row[column]) for column in feature_columns] for row in rows], dtype=np.float32)


def _normalize(
    train_features: np.ndarray,
    all_features: np.ndarray,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (all_features - mean) / std
    stats = {
        column: {"mean": float(mean[index]), "std": float(std[index])}
        for index, column in enumerate(feature_columns)
    }
    return normalized.astype(np.float32), train_features, stats


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _accuracy_from_indices(truth: list[int], predictions: list[int]) -> float:
    if not truth:
        return 0.0
    correct = sum(1 for true_index, predicted_index in zip(truth, predictions) if true_index == predicted_index)
    return _safe_divide(correct, len(truth))


def _confusion_matrix(
    truth: list[int],
    predictions: list[int],
    index_to_label: dict[int, str],
) -> dict[str, dict[str, int]]:
    matrix = {
        true_label: {predicted_label: 0 for predicted_label in index_to_label.values()}
        for true_label in index_to_label.values()
    }
    for true_index, predicted_index in zip(truth, predictions):
        matrix[index_to_label[true_index]][index_to_label[predicted_index]] += 1
    return matrix


def _classification_report(
    *,
    truth: list[int],
    predictions: list[int],
    index_to_label: dict[int, str],
    macro_labels: list[str] | None = None,
) -> dict[str, object]:
    matrix = _confusion_matrix(truth, predictions, index_to_label)
    labels = list(index_to_label.values())
    macro_label_set = set(macro_labels or labels)
    per_class: dict[str, dict[str, float | int]] = {}
    total_support = 0
    weighted_f1_sum = 0.0
    macro_precisions: list[float] = []
    macro_recalls: list[float] = []
    macro_f1_scores: list[float] = []

    for label in labels:
        true_positive = matrix[label][label]
        false_negative = sum(matrix[label].values()) - true_positive
        false_positive = sum(matrix[other_label][label] for other_label in labels if other_label != label)
        support = true_positive + false_negative
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, support)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        total_support += support
        weighted_f1_sum += f1 * support
        if label in macro_label_set:
            macro_precisions.append(precision)
            macro_recalls.append(recall)
            macro_f1_scores.append(f1)

    return {
        "per_class": per_class,
        "macro_precision": _safe_divide(sum(macro_precisions), len(macro_precisions)),
        "macro_recall": _safe_divide(sum(macro_recalls), len(macro_recalls)),
        "macro_f1": _safe_divide(sum(macro_f1_scores), len(macro_f1_scores)),
        "weighted_f1": _safe_divide(weighted_f1_sum, total_support),
        "accuracy": _accuracy_from_indices(truth, predictions),
        "confusion_matrix": matrix,
    }


def _tensor_indices(tensor: torch.Tensor) -> list[int]:
    return tensor.cpu().numpy().astype(int).tolist()


def _logit_predictions(logits: torch.Tensor) -> list[int]:
    return torch.argmax(logits, dim=1).cpu().numpy().astype(int).tolist()


def _count_by_fields(rows: list[dict[str, str]], fields: tuple[str, ...]) -> dict[str, int]:
    counts: Counter[str] = Counter("|".join(row[field] for field in fields) for row in rows)
    return dict(sorted(counts.items()))


def _test_metrics_by_source(
    *,
    rows: list[dict[str, str]],
    truth: list[int],
    predictions: list[int],
    index_to_label: dict[int, str],
) -> dict[str, dict[str, object]]:
    metrics: dict[str, dict[str, object]] = {}
    labels = list(index_to_label.values())
    sources = sorted({row["source"] for row in rows})
    for source in sources:
        source_indices = [index for index, row in enumerate(rows) if row["source"] == source]
        source_truth = [truth[index] for index in source_indices]
        source_predictions = [predictions[index] for index in source_indices]
        support_by_label = Counter(index_to_label[index] for index in source_truth)
        predicted_by_label = Counter(index_to_label[index] for index in source_predictions)
        source_report = _classification_report(
            truth=source_truth,
            predictions=source_predictions,
            index_to_label=index_to_label,
            macro_labels=sorted(support_by_label),
        )
        metrics[source] = {
            "row_count": len(source_indices),
            "accuracy": source_report["accuracy"],
            "support_by_label": {label: support_by_label.get(label, 0) for label in labels},
            "predicted_by_label": {label: predicted_by_label.get(label, 0) for label in labels},
            "confusion_matrix": source_report["confusion_matrix"],
        }
    return metrics


def train_audio_router(
    *,
    manifest_path: Path,
    output_dir: Path,
    epochs: int = 50,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> Path:
    """Train and evaluate a small audio router baseline."""
    if epochs <= 0:
        raise ValueError("--epochs must be greater than zero.")
    if learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than zero.")

    _set_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    feature_rows, manifest_schema = _extract_feature_rows(manifest_path)
    _write_feature_rows(output / "feature_rows.csv", feature_rows)
    failed_feature_rows = sum(1 for row in feature_rows if row["status"] != "success")
    rows = _successful_training_rows(feature_rows)

    labels = sorted({row["router_label"] for row in rows})
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}

    training_feature_columns = list(TRAINING_FEATURE_COLUMNS)
    all_features = _feature_matrix(rows, training_feature_columns)
    train_indices = [index for index, row in enumerate(rows) if row["split"] == "train"]
    test_indices = [index for index, row in enumerate(rows) if row["split"] == "test"]
    normalized_features, _raw_train_features, feature_stats = _normalize(
        all_features[train_indices],
        all_features,
        training_feature_columns,
    )

    x = torch.tensor(normalized_features, dtype=torch.float32)
    y = torch.tensor([label_to_index[row["router_label"]] for row in rows], dtype=torch.long)
    train_index_tensor = torch.tensor(train_indices, dtype=torch.long)
    test_index_tensor = torch.tensor(test_indices, dtype=torch.long)

    model = AudioRouterMLP(input_dim=len(training_feature_columns), num_classes=len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_curve: list[dict[str, str]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_logits = model(x[train_index_tensor])
        train_loss = F.cross_entropy(train_logits, y[train_index_tensor])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(x[test_index_tensor])
            test_loss = F.cross_entropy(test_logits, y[test_index_tensor])
        loss_curve.append(
            {
                "epoch": str(epoch),
                "train_loss": f"{float(train_loss.item()):.8f}",
                "test_loss": f"{float(test_loss.item()):.8f}",
            }
        )

    model.eval()
    with torch.no_grad():
        logits = model(x)
        train_logits = logits[train_index_tensor]
        test_logits = logits[test_index_tensor]
        train_truth = _tensor_indices(y[train_index_tensor])
        test_truth = _tensor_indices(y[test_index_tensor])
        train_predictions = _logit_predictions(train_logits)
        test_predictions = _logit_predictions(test_logits)
        train_accuracy = _accuracy_from_indices(train_truth, train_predictions)
        test_accuracy = _accuracy_from_indices(test_truth, test_predictions)

    test_rows = [rows[index] for index in test_indices]
    train_classification_report = _classification_report(
        truth=train_truth,
        predictions=train_predictions,
        index_to_label=index_to_label,
    )
    test_classification_report = _classification_report(
        truth=test_truth,
        predictions=test_predictions,
        index_to_label=index_to_label,
    )

    metrics = {
        "status": "success",
        "manifest_schema": manifest_schema["manifest_schema"],
        "label_column": manifest_schema["label_column"],
        "source_column": manifest_schema["source_column"],
        "label_values": labels,
        "feature_columns": training_feature_columns,
        "excluded_feature_columns": EXCLUDED_FEATURE_COLUMNS,
        "feature_policy_notes": FEATURE_POLICY_NOTES,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_classification_report": train_classification_report,
        "test_classification_report": test_classification_report,
        "manifest_rows": len(feature_rows),
        "successful_feature_rows": len(rows),
        "failed_feature_rows": failed_feature_rows,
        "total_rows": len(rows),
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "counts_by_label": dict(sorted(Counter(row["router_label"] for row in rows).items())),
        "counts_by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "counts_by_source": dict(sorted(Counter(row["source"] for row in rows).items())),
        "counts_by_split_label": _count_by_fields(rows, ("split", "router_label")),
        "counts_by_source_label": _count_by_fields(rows, ("source", "router_label")),
        "test_metrics_by_source": _test_metrics_by_source(
            rows=test_rows,
            truth=test_truth,
            predictions=test_predictions,
            index_to_label=index_to_label,
        ),
        "confusion_matrix": test_classification_report["confusion_matrix"],
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "feature_columns": training_feature_columns,
                "label_to_index": label_to_index,
                "feature_stats": feature_stats,
            },
        },
        output / "checkpoint.pt",
    )
    with (output / "label_mapping.json").open("w", encoding="utf-8") as json_file:
        json.dump({"label_to_index": label_to_index, "index_to_label": {str(key): value for key, value in index_to_label.items()}}, json_file, indent=2)
        json_file.write("\n")
    with (output / "feature_stats.json").open("w", encoding="utf-8") as json_file:
        json.dump(feature_stats, json_file, indent=2)
        json_file.write("\n")
    with (output / "metrics.json").open("w", encoding="utf-8") as json_file:
        json.dump(metrics, json_file, indent=2)
        json_file.write("\n")
    with (output / "loss_curve.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "test_loss"])
        writer.writeheader()
        writer.writerows(loss_curve)

    return output.resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the audio router trainer parser."""
    parser = argparse.ArgumentParser(description="Train a lightweight audio router classifier.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        output_dir = train_audio_router(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        print(f"Wrote audio router training artifacts: {output_dir}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
