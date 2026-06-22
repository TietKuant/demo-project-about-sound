from __future__ import annotations

import csv
import json

from scripts.evaluate_detection_fusion import evaluate_detection_fusion


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evidence_rows():
    return [
        {
            "input_path": "/audio/ignored-train.wav",
            "content_label": "environment_only",
            "split": "train",
        },
        {
            "input_path": "/audio/environment.wav",
            "content_label": "environment_only",
            "split": "test",
        },
        {
            "input_path": "/audio/music.wav",
            "content_label": "music_with_vocals",
            "split": "test",
        },
        {
            "input_path": "/audio/clean.wav",
            "content_label": "speech_clean",
            "split": "test",
        },
        {
            "input_path": "/audio/noisy.wav",
            "content_label": "speech_noisy_general",
            "split": "test",
        },
        {
            "input_path": "/audio/target.wav",
            "content_label": "speech_target_noise",
            "split": "test",
        },
    ]


def _prediction_rows(*, target_probability: float = 0.9):
    return [
        {
            "input_path": "/audio/environment.wav",
            "speech_present_probability": "0.1",
            "music_present_probability": "0.1",
            "target_event_present_probability": "0.1",
        },
        {
            "input_path": "/audio/music.wav",
            "speech_present_probability": "0.2",
            "music_present_probability": "0.9",
            "target_event_present_probability": "0.1",
        },
        {
            "input_path": "/audio/clean.wav",
            "speech_present_probability": "0.9",
            "music_present_probability": "0.1",
            "target_event_present_probability": "0.1",
        },
        {
            "input_path": "/audio/noisy.wav",
            "speech_present_probability": "0.9",
            "music_present_probability": "0.1",
            "target_event_present_probability": "0.2",
        },
        {
            "input_path": "/audio/target.wav",
            "speech_present_probability": "0.9",
            "music_present_probability": "0.1",
            "target_event_present_probability": str(target_probability),
        },
    ]


def test_evaluator_joins_by_path_computes_metrics_and_selects_candidate(tmp_path):
    evidence_csv = tmp_path / "evidence.csv"
    predictions_csv = tmp_path / "predictions.csv"
    output_json = tmp_path / "fusion.json"
    output_csv = tmp_path / "fusion.csv"
    _write_csv(evidence_csv, _evidence_rows())
    _write_csv(predictions_csv, list(reversed(_prediction_rows())))

    report = evaluate_detection_fusion(
        evidence_csv=evidence_csv,
        predictions_csv=predictions_csv,
        output_json=output_json,
        output_csv=output_csv,
        speech_thresholds="0.7",
        music_thresholds="0.5",
        target_thresholds="0.7",
    )

    assert report["row_count"] == 5
    result = report["threshold_results"][0]
    assert result["macro_recall"] == 1.0
    assert result["recall_by_label"]["speech_target_noise"] == 1.0
    assert result["confusion"]["speech_clean"]["speech_clean"] == 1
    recommendation = report["recommended_research_thresholds"]
    assert recommendation == {
        "speech_threshold": 0.7,
        "music_threshold": 0.5,
        "target_threshold": 0.7,
    }
    assert json.loads(output_json.read_text(encoding="utf-8"))["row_count"] == 5
    with output_csv.open(newline="", encoding="utf-8") as handle:
        output_rows = list(csv.DictReader(handle))
    assert len(output_rows) == 5
    clean_row = next(
        row for row in output_rows if row["input_path"] == "/audio/clean.wav"
    )
    assert clean_row["raw_fusion_label"] == "speech_present_general"
    assert clean_row["normalized_fusion_label"] == "speech_clean"


def test_evaluator_returns_null_recommendation_when_constraints_fail(tmp_path):
    evidence_csv = tmp_path / "evidence.csv"
    predictions_csv = tmp_path / "predictions.csv"
    output_json = tmp_path / "fusion.json"
    output_csv = tmp_path / "fusion.csv"
    _write_csv(evidence_csv, _evidence_rows())
    _write_csv(
        predictions_csv,
        _prediction_rows(target_probability=0.1),
    )

    report = evaluate_detection_fusion(
        evidence_csv=evidence_csv,
        predictions_csv=predictions_csv,
        output_json=output_json,
        output_csv=output_csv,
        speech_thresholds="0.7",
        music_thresholds="0.5",
        target_thresholds="0.7",
    )

    recommendation = report["recommended_research_thresholds"]
    assert recommendation is None
    assert "No threshold combination" in report["recommended_research_reason"]
    with output_csv.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []
