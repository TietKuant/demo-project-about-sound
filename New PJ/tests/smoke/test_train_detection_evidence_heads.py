from __future__ import annotations

import csv
import json

import pytest

from scripts.train_detection_evidence_heads import (
    FEATURE_COLUMNS,
    HEADS,
    train_detection_evidence_heads,
)


def _evidence_row(
    sample_id: str,
    split: str,
    *,
    speech: bool,
    music: bool,
    target: bool,
) -> dict[str, str]:
    row = {
        "sample_id": sample_id,
        "input_path": f"/tmp/{sample_id}.wav",
        "split": split,
        "status": "success",
        "content_label": (
            "music_with_vocals"
            if music
            else "speech_target_noise"
            if speech and target
            else "speech_clean"
            if speech
            else "environment_only"
        ),
        "contains_speech": str(speech).lower(),
        "contains_music": str(music).lower(),
        "contains_target_noise": str(target).lower(),
        "target_noise_label": "siren" if target else "",
    }
    base = 1.0 if speech else 0.1
    for index, column in enumerate(FEATURE_COLUMNS):
        row[column] = str(base + index / 100)
    return row


def _write_evidence(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _balanced_rows():
    combinations = [
        (True, False, False),
        (False, True, False),
        (True, False, True),
        (False, False, True),
        (False, False, False),
    ]
    rows = []
    for split in ("train", "test"):
        for index, (speech, music, target) in enumerate(combinations):
            rows.append(
                _evidence_row(
                    f"{split}-{index}",
                    split,
                    speech=speech,
                    music=music,
                    target=target,
                )
            )
    return rows


def test_trainer_rejects_missing_class_coverage(tmp_path):
    evidence_path = tmp_path / "evidence.csv"
    rows = [
        _evidence_row(
            f"{split}-{index}",
            split,
            speech=True,
            music=False,
            target=False,
        )
        for split in ("train", "test")
        for index in range(2)
    ]
    _write_evidence(evidence_path, rows)

    with pytest.raises(ValueError, match="lacks positive or negative"):
        train_detection_evidence_heads(
            evidence_csv=evidence_path,
            output_dir=tmp_path / "output",
            epochs=1,
        )


def test_trainer_writes_checkpoint_metrics_and_predictions(tmp_path):
    evidence_path = tmp_path / "evidence.csv"
    _write_evidence(evidence_path, _balanced_rows())
    output_dir = tmp_path / "output"

    result = train_detection_evidence_heads(
        evidence_csv=evidence_path,
        output_dir=output_dir,
        epochs=2,
        seed=7,
    )

    assert result == output_dir.resolve()
    assert {
        "checkpoint.pt",
        "metrics.json",
        "test_predictions.csv",
        "feature_columns.json",
    } <= {path.name for path in output_dir.iterdir()}
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["heads"]) == set(HEADS)
    for head in HEADS:
        assert {"accuracy", "precision", "recall", "f1", "confusion"} <= set(
            metrics["heads"][head]
        )
        assert metrics["counts_by_head_and_split"][head]["train"]["positive"] > 0
        assert metrics["counts_by_head_and_split"][head]["train"]["negative"] > 0
    feature_payload = json.loads(
        (output_dir / "feature_columns.json").read_text(encoding="utf-8")
    )
    assert feature_payload["feature_columns"] == FEATURE_COLUMNS
    with (output_dir / "test_predictions.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert len(prediction_rows) == 5
    assert "speech_present_probability" in prediction_rows[0]
