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


REQUIRED_COLUMNS = {"sample_id", "input_path", "router_label", "source", "split", "notes"}
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


def _load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with Path(manifest_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Audio router manifest is missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Audio router manifest is empty: {manifest_path}")
    return rows


def _resolve_input_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(manifest_path).resolve().parent / path).resolve()


def _extract_feature_rows(manifest_path: Path) -> list[dict[str, str]]:
    manifest_rows = _load_manifest(manifest_path)
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
    return feature_rows


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


def _feature_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(row[column]) for column in FEATURE_COLUMNS] for row in rows], dtype=np.float32)


def _normalize(train_features: np.ndarray, all_features: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (all_features - mean) / std
    stats = {
        column: {"mean": float(mean[index]), "std": float(std[index])}
        for index, column in enumerate(FEATURE_COLUMNS)
    }
    return normalized.astype(np.float32), train_features, stats


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return float((predictions == labels).float().mean().item())


def _confusion_matrix(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    index_to_label: dict[int, str],
) -> dict[str, dict[str, int]]:
    predictions = torch.argmax(logits, dim=1).cpu().numpy().tolist()
    truth = labels.cpu().numpy().tolist()
    matrix = {
        true_label: {predicted_label: 0 for predicted_label in index_to_label.values()}
        for true_label in index_to_label.values()
    }
    for true_index, predicted_index in zip(truth, predictions):
        matrix[index_to_label[true_index]][index_to_label[predicted_index]] += 1
    return matrix


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

    feature_rows = _extract_feature_rows(manifest_path)
    _write_feature_rows(output / "feature_rows.csv", feature_rows)
    failed_feature_rows = sum(1 for row in feature_rows if row["status"] != "success")
    rows = _successful_training_rows(feature_rows)

    labels = sorted({row["router_label"] for row in rows})
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}

    all_features = _feature_matrix(rows)
    train_indices = [index for index, row in enumerate(rows) if row["split"] == "train"]
    test_indices = [index for index, row in enumerate(rows) if row["split"] == "test"]
    normalized_features, _raw_train_features, feature_stats = _normalize(all_features[train_indices], all_features)

    x = torch.tensor(normalized_features, dtype=torch.float32)
    y = torch.tensor([label_to_index[row["router_label"]] for row in rows], dtype=torch.long)
    train_index_tensor = torch.tensor(train_indices, dtype=torch.long)
    test_index_tensor = torch.tensor(test_indices, dtype=torch.long)

    model = AudioRouterMLP(input_dim=len(FEATURE_COLUMNS), num_classes=len(labels))
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
        train_accuracy = _accuracy(train_logits, y[train_index_tensor])
        test_accuracy = _accuracy(test_logits, y[test_index_tensor])

    metrics = {
        "status": "success",
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "manifest_rows": len(feature_rows),
        "successful_feature_rows": len(rows),
        "failed_feature_rows": failed_feature_rows,
        "total_rows": len(rows),
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "counts_by_label": dict(sorted(Counter(row["router_label"] for row in rows).items())),
        "confusion_matrix": _confusion_matrix(logits=logits[test_index_tensor], labels=y[test_index_tensor], index_to_label=index_to_label),
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "feature_columns": FEATURE_COLUMNS,
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
