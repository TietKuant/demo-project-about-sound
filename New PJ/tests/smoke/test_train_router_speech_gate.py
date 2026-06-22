from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import train_router_speech_gate as gate
from scripts.train_human_labeled_router import WORKFLOW_BY_LABEL


def _write_manifest(
    path: Path,
    *,
    duplicate_path: bool = False,
    duplicate_sample_id: bool = False,
    bad_workflow: bool = False,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    labels = [
        "speech_clean",
        "speech_noisy_general",
        "speech_target_noise",
        "music_with_vocals",
        "environment_only",
        "unknown_mixed",
    ]
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
                    "workflow_label": WORKFLOW_BY_LABEL[label],
                }
            )
    if duplicate_path:
        rows[1]["path"] = rows[0]["path"]
    if duplicate_sample_id:
        rows[1]["sample_id"] = rows[0]["sample_id"]
    if bad_workflow:
        rows[0]["workflow_label"] = "safe_abstain"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_baseline_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.mkdir(parents=True)
    fieldnames = [
        "sample_id",
        "path",
        "filename",
        "true_workflow",
        "predicted_workflow",
        "confidence",
        "accepted",
    ]
    with (path / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            predicted_workflow = row["workflow_label"]
            if row["human_label"] in gate.NON_SPEECH_LABELS:
                predicted_workflow = "speech_cleanup"
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "path": row["path"],
                    "filename": row["filename"],
                    "true_workflow": row["workflow_label"],
                    "predicted_workflow": predicted_workflow,
                    "confidence": "0.95",
                    "accepted": "true",
                }
            )


def _fake_feature_rows(rows):
    output = []
    for row_index, row in enumerate(rows):
        feature_row = {**row, "status": "success", "error": ""}
        for feature_index, column in enumerate(gate.TRAINING_FEATURE_COLUMNS):
            feature_row[column] = str(row_index + feature_index / 100)
        output.append(feature_row)
    return output


def _fake_fit_gate_model(
    *,
    test_rows,
    hard_unknown_rows,
    extra_rows,
    **_kwargs,
):
    test_probabilities = []
    for row in test_rows:
        if row["gate_label"] == "speech_present":
            test_probabilities.append([0.95, 0.05])
        else:
            test_probabilities.append([0.05, 0.95])
    return {
        "model_state_dict": {},
        "feature_stats": {
            column: {"mean": 0.0, "std": 1.0}
            for column in gate.TRAINING_FEATURE_COLUMNS
        },
        "test_probabilities": test_probabilities,
        "hard_unknown_probabilities": [
            [0.55, 0.45] for _ in hard_unknown_rows
        ],
        "extra_probabilities": [
            (
                [0.95, 0.05]
                if row["gate_label"] == "speech_present"
                else [0.05, 0.95]
            )
            for row in extra_rows
        ],
    }


def test_gate_writes_artifacts_and_blocks_dangerous_speech_invocation(
    tmp_path,
    monkeypatch,
):
    manifest = tmp_path / "labeled.csv"
    rows = _write_manifest(manifest)
    baseline_rows = [row for index, row in enumerate(rows) if index % 2 == 0]
    baseline_dir = tmp_path / "baseline"
    _write_baseline_predictions(baseline_dir, baseline_rows)
    monkeypatch.setattr(gate, "_extract_feature_rows", _fake_feature_rows)
    monkeypatch.setattr(gate, "_fit_gate_model", _fake_fit_gate_model)
    output_dir = tmp_path / "gate-output"

    gate.train_router_speech_gate(
        manifest_path=manifest,
        router_output_dir=baseline_dir,
        output_dir=output_dir,
        test_ratio=0.5,
        max_epochs=1,
        threshold=0.70,
    )

    expected = {
        "checkpoint.pt",
        "feature_stats.json",
        "metrics.json",
        "split_manifest.csv",
        "test_predictions.csv",
        "threshold_metrics.csv",
        "hard_unknown_predictions.csv",
        "combined_guarded_predictions.csv",
        "combined_guarded_metrics.json",
        "combined_accepted_dangerous_errors.csv",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    metrics = json.loads(
        (output_dir / "combined_guarded_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert metrics["baseline_dangerous_speech_invocations"] > 0
    assert metrics["guarded_dangerous_speech_invocations"] == 0
    assert metrics["baseline_rows"] == len(baseline_rows)
    assert metrics["combined_rows"] == len(baseline_rows)
    assert metrics["missing_gate_rows"] == 0
    assert metrics["excluded_from_gate_training_for_combined_eval"] == len(
        {row["sample_id"] for row in baseline_rows}
    )
    with (output_dir / "combined_guarded_predictions.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        combined_rows = list(csv.DictReader(handle))
    guarded_non_speech = [
        row
        for row in combined_rows
        if row["true_workflow"] in gate.DANGEROUS_TRUE_WORKFLOWS
        and row["baseline_workflow"] == "speech_cleanup"
    ]
    assert guarded_non_speech
    assert all(row["guarded_workflow"] == "safe_abstain" for row in guarded_non_speech)
    with (output_dir / "split_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        gate_split_rows = list(csv.DictReader(handle))
    gate_train_ids = {
        row["sample_id"]
        for row in gate_split_rows
        if row["split"] == "train"
    }
    baseline_ids = {row["sample_id"] for row in baseline_rows}
    assert gate_train_ids.isdisjoint(baseline_ids)
    assert any(
        row["sample_id"] in baseline_ids
        and row["guarded_workflow"] == "safe_abstain"
        for row in guarded_non_speech
    )

    gate_metrics = json.loads(
        (output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert gate_metrics["excluded_unknown_rows"] == 1


def test_gate_rejects_baseline_sample_missing_from_manifest(tmp_path):
    manifest = tmp_path / "labeled.csv"
    rows = _write_manifest(manifest)
    missing_media = tmp_path / "missing-from-manifest.wav"
    missing_media.write_bytes(b"audio")
    rows.append(
        {
            "sample_id": "missing-from-manifest",
            "path": str(missing_media),
            "filename": missing_media.name,
            "human_label": "environment_only",
            "workflow_label": "no_process",
        }
    )
    baseline_dir = tmp_path / "baseline"
    _write_baseline_predictions(baseline_dir, rows)

    with pytest.raises(ValueError, match="missing from the validated manifest"):
        gate.train_router_speech_gate(
            manifest_path=manifest,
            router_output_dir=baseline_dir,
            output_dir=tmp_path / "output",
            max_epochs=1,
        )


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("duplicate_path", "Duplicate path"),
        ("duplicate_sample_id", "Duplicate sample_id"),
        ("bad_workflow", "workflow_label"),
    ],
)
def test_gate_validation_reuses_human_manifest_safety_checks(
    tmp_path,
    option,
    message,
):
    manifest = tmp_path / f"{option}.csv"
    _write_manifest(manifest, **{option: True})

    with pytest.raises(ValueError, match=message):
        gate._load_and_validate_manifest(manifest)
