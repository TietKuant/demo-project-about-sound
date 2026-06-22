from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import train_human_labeled_router as trainer


def _write_manifest(
    path: Path,
    *,
    blank_label: bool = False,
    duplicate_path: bool = False,
    duplicate_sample_id: bool = False,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    labels = [*trainer.DEFAULT_LABELS, trainer.UNKNOWN_LABEL]
    for label in labels:
        for index in range(2):
            media_path = path.parent / f"{label}-{index}.wav"
            media_path.write_bytes(b"audio")
            rows.append(
                {
                    "sample_id": f"{label}-{index}",
                    "path": str(media_path),
                    "filename": media_path.name,
                    "human_label": label,
                    "workflow_label": "safe_abstain",
                }
            )
    if blank_label:
        rows[0]["human_label"] = ""
    if duplicate_path:
        rows[1]["path"] = rows[0]["path"]
    if duplicate_sample_id:
        rows[1]["sample_id"] = rows[0]["sample_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _fake_feature_rows(rows):
    output = []
    for row_index, row in enumerate(rows):
        feature_row = {
            **row,
            "status": "success",
            "error": "",
        }
        for feature_index, column in enumerate(trainer.FEATURE_COLUMNS):
            feature_row[column] = str(row_index + feature_index / 100)
        output.append(feature_row)
    return output


def _fake_fit_model(
    *,
    train_rows,
    test_rows,
    hard_unknown_rows,
    labels,
    **_kwargs,
):
    probabilities = []
    for row in test_rows:
        true_index = labels.index(row["human_label"])
        values = [0.01] * len(labels)
        values[true_index] = 0.96
        probabilities.append(values)
    hard_probabilities = []
    for _ in hard_unknown_rows:
        values = [0.01] * len(labels)
        values[0] = 0.96
        hard_probabilities.append(values)
    return {
        "model_state_dict": {},
        "feature_stats": {
            column: {"mean": 0.0, "std": 1.0}
            for column in trainer.TRAINING_FEATURE_COLUMNS
        },
        "test_probabilities": probabilities,
        "hard_unknown_probabilities": hard_probabilities,
    }


def test_training_writes_expected_artifacts_and_excludes_unknown_by_default(
    tmp_path,
    monkeypatch,
):
    manifest = tmp_path / "labeled.csv"
    _write_manifest(manifest)
    monkeypatch.setattr(trainer, "_extract_feature_rows", _fake_feature_rows)
    monkeypatch.setattr(trainer, "_fit_model", _fake_fit_model)
    output_dir = tmp_path / "output"

    result = trainer.train_human_labeled_router(
        manifest_path=manifest,
        output_dir=output_dir,
        max_epochs=1,
        test_ratio=0.5,
    )

    assert result == output_dir.resolve()
    expected_files = {
        "checkpoint.pt",
        "label_mapping.json",
        "feature_stats.json",
        "feature_rows.csv",
        "split_manifest.csv",
        "metrics.json",
        "test_predictions.csv",
        "confusion_matrix.csv",
        "hard_unknown_predictions.csv",
        "threshold_metrics.csv",
        "all_error_rows.csv",
        "accepted_error_rows.csv",
        "hard_unknown_accepted.csv",
    }
    assert expected_files <= {path.name for path in output_dir.iterdir()}
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["excluded_unknown_rows"] == 2
    assert metrics["labels"] == trainer.DEFAULT_LABELS
    assert metrics["accuracy"] == 1.0
    assert "per_class" in metrics
    assert "confusion_matrix" in metrics
    assert "accepted_accuracy_at_threshold" in metrics
    assert "accepted_errors_at_threshold" in metrics
    assert metrics["hard_unknown_accepted_at_threshold"] == 2
    assert set(metrics["per_class"]) == set(trainer.DEFAULT_LABELS)


def test_include_unknown_class_makes_unknown_trainable(tmp_path):
    manifest = tmp_path / "labeled.csv"
    _write_manifest(manifest)
    rows = trainer._load_and_validate_manifest(manifest)

    split_rows, hard_rows, labels = trainer._prepare_split_rows(
        rows,
        include_unknown_class=True,
        test_ratio=0.5,
        seed=42,
        allow_small_classes=False,
    )

    assert trainer.UNKNOWN_LABEL in labels
    assert hard_rows == []
    assert any(row["human_label"] == trainer.UNKNOWN_LABEL for row in split_rows)
    assert {
        row["split"]
        for row in split_rows
        if row["human_label"] == trainer.UNKNOWN_LABEL
    } == {"train", "test"}


def test_validation_rejects_blank_labels(tmp_path):
    manifest = tmp_path / "blank.csv"
    _write_manifest(manifest, blank_label=True)

    with pytest.raises(ValueError, match="blank human_label"):
        trainer._load_and_validate_manifest(manifest)


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("duplicate_path", "Duplicate path"),
        ("duplicate_sample_id", "Duplicate sample_id"),
    ],
)
def test_validation_rejects_duplicates(tmp_path, option, message):
    manifest = tmp_path / f"{option}.csv"
    _write_manifest(manifest, **{option: True})

    with pytest.raises(ValueError, match=message):
        trainer._load_and_validate_manifest(manifest)
