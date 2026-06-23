"""Smoke tests for Audio Router real-audio validation report export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.export_audio_router_real_validation_report import (
    OUTPUT_COLUMNS,
    export_audio_router_real_validation_report,
)


def _write_batch_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "input_path",
        "status",
        "predicted_label",
        "confidence",
        "confidence_threshold",
        "accepted",
        "route_target",
        "engine_target",
        "recommended_task",
        "decision_reason",
        "warnings",
        "prob_environment_only",
        "prob_music_with_vocals",
        "prob_speech_clean",
        "prob_speech_target_noise",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, *, input_path: Path, expected_label: str, accepted: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "success",
        "input_path": str(input_path),
        "predicted_label": expected_label,
        "confidence": 0.91,
        "confidence_threshold": 0.90,
        "accepted": accepted,
        "route_target": "manual_required" if expected_label == "music_with_vocals" else "no_process",
        "engine_target": "demucs" if expected_label == "music_with_vocals" else "none",
        "recommended_task": "extract_vocals" if expected_label == "music_with_vocals" else None,
        "decision_reason": "accepted_router_prediction" if accepted else "low_confidence",
        "warnings": ["manual_music_task_selection_required"] if expected_label == "music_with_vocals" else [],
        "probabilities": {
            "environment_only": 0.01,
            "music_with_vocals": 0.91 if expected_label == "music_with_vocals" else 0.02,
            "speech_clean": 0.94 if expected_label == "speech_clean" else 0.03,
            "speech_target_noise": 0.04,
        },
        "features": {
            "rms_energy": 0.1,
            "zero_crossing_rate": 0.02,
            "spectral_centroid_hz": 1234.5,
            "spectral_bandwidth_hz": 2345.6,
            "spectral_rolloff_hz": 3456.7,
            "spectral_flatness": 0.001,
            "low_band_energy_ratio": 0.2,
            "mid_band_energy_ratio": 0.5,
            "high_band_energy_ratio": 0.3,
            "rms_std": 0.01,
            "zcr_std": 0.002,
            "silence_ratio": 0.0,
        },
        "error": "",
    }
    path.write_text(json.dumps(summary), encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def test_export_audio_router_real_validation_report_merges_summaries(tmp_path: Path) -> None:
    input_root = tmp_path / "real_test_audio"
    speech_clean = input_root / "speech_clean" / "clean.wav"
    music = input_root / "music_with_vocals" / "song.wav"
    missing_summary = input_root / "environment_only" / "noise.wav"
    batch_csv = tmp_path / "batch.csv"
    summaries_dir = tmp_path / "router_summaries"
    output_csv = tmp_path / "merged.csv"
    output_md = tmp_path / "report.md"

    _write_batch_csv(
        batch_csv,
        [
            {
                "input_path": str(speech_clean),
                "status": "success",
                "predicted_label": "speech_clean",
                "confidence": "0.9000000000",
                "confidence_threshold": "0.9000000000",
                "accepted": "true",
                "route_target": "no_process",
                "engine_target": "none",
                "recommended_task": "",
                "decision_reason": "accepted_router_prediction",
                "warnings": "",
                "prob_environment_only": "0.0100000000",
                "prob_music_with_vocals": "0.0200000000",
                "prob_speech_clean": "0.9400000000",
                "prob_speech_target_noise": "0.0300000000",
                "error": "",
            },
            {
                "input_path": str(music),
                "status": "success",
                "predicted_label": "music_with_vocals",
                "confidence": "0.9100000000",
                "confidence_threshold": "0.9000000000",
                "accepted": "true",
                "route_target": "manual_required",
                "engine_target": "demucs",
                "recommended_task": "extract_vocals",
                "decision_reason": "accepted_router_prediction",
                "warnings": "manual_music_task_selection_required",
                "prob_environment_only": "0.0100000000",
                "prob_music_with_vocals": "0.9100000000",
                "prob_speech_clean": "0.0300000000",
                "prob_speech_target_noise": "0.0400000000",
                "error": "",
            },
            {
                "input_path": str(missing_summary),
                "status": "success",
                "predicted_label": "environment_only",
                "confidence": "0.5000000000",
                "confidence_threshold": "0.9000000000",
                "accepted": "false",
                "route_target": "manual_required",
                "engine_target": "none",
                "recommended_task": "",
                "decision_reason": "low_confidence",
                "warnings": "low_confidence_router_prediction",
                "prob_environment_only": "0.5000000000",
                "prob_music_with_vocals": "0.3000000000",
                "prob_speech_clean": "0.1000000000",
                "prob_speech_target_noise": "0.1000000000",
                "error": "",
            },
        ],
    )
    _write_summary(summaries_dir / "speech_clean_clean.router_summary.json", input_path=speech_clean, expected_label="speech_clean", accepted=True)
    _write_summary(summaries_dir / "music_with_vocals_song.router_summary.json", input_path=music, expected_label="music_with_vocals", accepted=True)

    export_audio_router_real_validation_report(
        batch_csv=batch_csv,
        summaries_dir=summaries_dir,
        output_csv=output_csv,
        output_md=output_md,
    )

    rows = _read_rows(output_csv)
    assert output_csv.exists()
    assert output_md.exists()
    assert len(rows) == 3
    assert list(rows[0].keys()) == OUTPUT_COLUMNS
    assert rows[0]["expected_group"] == "speech_clean"
    assert rows[0]["rms_energy"] == "0.1000000000"
    assert rows[1]["warnings"] == "manual_music_task_selection_required"
    assert rows[2]["expected_group"] == "environment_only"
    assert rows[2]["rms_energy"] == ""
    assert rows[2]["prob_environment_only"] == "0.5000000000"

    markdown = output_md.read_text(encoding="utf-8")
    assert "experimental baseline" in markdown
    assert "domain shift" in markdown
    assert "dangerous" in markdown.lower()
    assert "expected_group vs predicted_label" in markdown
