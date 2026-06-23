from __future__ import annotations

import csv
import json

from scripts.evaluate_detection_evidence import (
    REVIEW_COLUMNS,
    evaluate_detection_evidence,
)


def test_evaluator_writes_json_and_review_csv(tmp_path):
    evidence_path = tmp_path / "evidence.csv"
    rows = [
        {
            "input_path": "/tmp/a.wav",
            "content_label": "speech_target_noise",
            "split": "test",
            "status": "success",
            "router_label": "speech_noisy_general",
            "router_confidence": "0.90",
            "router_accepted": "true",
            "speech_gate_label": "speech_present",
            "speech_gate_confidence": "0.80",
            "top_label": "speech_noisy_general",
            "top_score": "0.90",
            "second_label": "speech_target_noise",
            "second_score": "0.08",
            "score_margin": "0.82",
            "target_noise_label": "siren",
            "snr_db": "5",
            "needs_review": "true",
        },
        {
            "input_path": "/tmp/b.wav",
            "content_label": "environment_only",
            "split": "train",
            "status": "success",
            "router_label": "environment_only",
            "router_confidence": "0.85",
            "router_accepted": "true",
            "speech_gate_label": "speech_present",
            "speech_gate_confidence": "0.75",
            "top_label": "environment_only",
            "top_score": "0.85",
            "second_label": "speech_clean",
            "second_score": "0.10",
            "score_margin": "0.75",
            "target_noise_label": "",
            "snr_db": "",
            "needs_review": "false",
        },
    ]
    with evidence_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output_json = tmp_path / "report.json"
    review_csv = tmp_path / "review.csv"

    report = evaluate_detection_evidence(
        evidence_csv=evidence_path,
        output_json=output_json,
        review_csv=review_csv,
    )

    assert report["row_count"] == 2
    assert report["hard_error_count"] == 1
    assert report["router_recall_by_label"]["speech_target_noise"] == 0.0
    assert report["speech_gate_by_content_label"]["environment_only"] == {
        "speech_present": 1
    }
    assert report["speech_target_noise_by_snr_router_label"]["5"] == {
        "speech_noisy_general": 1
    }
    assert json.loads(output_json.read_text(encoding="utf-8"))["row_count"] == 2
    with review_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        review_rows = list(reader)
    assert reader.fieldnames == REVIEW_COLUMNS
    assert len(review_rows) == 1
    assert review_rows[0]["target_noise_label"] == "siren"
