from __future__ import annotations

import csv
import math
import wave
from pathlib import Path

from scripts import build_detection_evidence_dataset as evidence


def _write_wav(path: Path, frequency_hz: float = 440.0) -> None:
    sample_rate = 8000
    samples = 800
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(samples):
            sample = int(
                12000
                * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate)
            )
            frames.extend(sample.to_bytes(2, "little", signed=True))
        handle.writeframes(bytes(frames))


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "sample_id",
        "input_path",
        "source_dataset",
        "split",
        "group_id",
        "content_label",
        "workflow_label",
        "contains_speech",
        "contains_music",
        "contains_target_noise",
        "target_noise_label",
        "snr_db",
        "is_synthetic",
        "clean_source_id",
        "noise_source_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(sample_id: str, input_path: str, label: str = "speech_clean") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "input_path": input_path,
        "source_dataset": "fixture",
        "split": "test",
        "group_id": "group-1",
        "content_label": label,
        "workflow_label": "no_process",
        "contains_speech": "true",
        "contains_music": "false",
        "contains_target_noise": "false",
        "target_noise_label": "",
        "snr_db": "",
        "is_synthetic": "false",
        "clean_source_id": "",
        "noise_source_id": "",
    }


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return list(reader.fieldnames or []), rows


def test_signal_only_mode_writes_schema_and_features(tmp_path):
    audio_path = tmp_path / "audio" / "sample.wav"
    _write_wav(audio_path)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [_row("sample-1", "audio/sample.wav")])
    output = tmp_path / "evidence.csv"

    evidence.build_detection_evidence_dataset(
        manifest_path=manifest,
        output_csv=output,
    )

    fieldnames, rows = _read(output)
    assert fieldnames == evidence.OUTPUT_COLUMNS
    assert len(rows) == 1
    assert rows[0]["status"] == "success"
    assert rows[0]["duration_sec"]
    assert rows[0]["spectral_rolloff_hz"]
    assert rows[0]["router_confidence"] == ""
    assert rows[0]["speech_gate_confidence"] == ""
    assert rows[0]["needs_review"] == "false"


def test_skip_missing_marks_row_skipped_and_reviewable(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [_row("missing", "audio/missing.wav")])
    output = tmp_path / "evidence.csv"

    evidence.build_detection_evidence_dataset(
        manifest_path=manifest,
        output_csv=output,
        skip_missing=True,
    )

    _, rows = _read(output)
    assert rows[0]["status"] == "skipped"
    assert rows[0]["needs_review"] == "true"
    assert "Input file not found" in rows[0]["error"]


def test_guarded_probabilities_compute_ranking_margin_and_review(
    tmp_path,
    monkeypatch,
):
    audio_path = tmp_path / "sample.wav"
    _write_wav(audio_path)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [_row("sample-1", str(audio_path), label="speech_clean")],
    )
    router_checkpoint = tmp_path / "router.pt"
    gate_checkpoint = tmp_path / "gate.pt"
    router_checkpoint.touch()
    gate_checkpoint.touch()
    guarded_result = {
        "router_label": "speech_noisy_general",
        "router_confidence": 0.55,
        "router_accepted": True,
        "reason": "accepted",
        "router_probabilities": {
            "speech_noisy_general": 0.55,
            "speech_clean": 0.45,
        },
        "speech_gate_label": "speech_present",
        "speech_gate_confidence": 0.9,
        "speech_gate_probabilities": {
            "non_speech": 0.1,
            "speech_present": 0.9,
        },
    }
    monkeypatch.setattr(
        evidence,
        "run_guarded_router",
        lambda **_kwargs: guarded_result,
    )
    output = tmp_path / "evidence.csv"

    evidence.build_detection_evidence_dataset(
        manifest_path=manifest,
        output_csv=output,
        router_checkpoint=router_checkpoint,
        speech_gate_checkpoint=gate_checkpoint,
        router_threshold=0.70,
    )

    _, rows = _read(output)
    row = rows[0]
    assert row["top_label"] == "speech_noisy_general"
    assert float(row["top_score"]) == 0.55
    assert row["second_label"] == "speech_clean"
    assert float(row["score_margin"]) == 0.10
    assert row["router_prob_speech_noisy_general"] == "0.5500000000"
    assert row["speech_gate_prob_speech_present"] == "0.9000000000"
    assert row["needs_review"] == "true"
