"""Smoke tests for lightweight audio feature extraction."""

from __future__ import annotations

import csv
import math
import subprocess
import wave
from pathlib import Path
from unittest.mock import patch

from scripts.extract_audio_features import extract_audio_features


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "sample_id",
        "category",
        "input_path",
        "input_type",
        "language",
        "expected_task",
        "has_clean_reference",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _manifest_row(sample_id: str, input_path: Path) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "category": "vietnamese_speech_quiet",
        "input_path": str(input_path),
        "input_type": "audio",
        "language": "vi",
        "expected_task": "clean_voice",
        "has_clean_reference": "false",
        "notes": f"notes for {sample_id}",
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_sine_wav(path: Path, frequency_hz: float = 440.0, duration_sec: float = 0.1) -> None:
    sample_rate = 16000
    sample_count = int(sample_rate * duration_sec)
    amplitude = 12000
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for index in range(sample_count):
            sample = int(amplitude * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate))
            frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(frames))


def test_extract_audio_features_writes_features_for_wav(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.wav"
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "features.csv"
    _write_sine_wav(input_path)
    _write_manifest(manifest_path, [_manifest_row("vi_sine_001", input_path)])

    with patch(
        "scripts.extract_audio_features.subprocess.run",
        return_value=subprocess.CompletedProcess(["ffprobe"], 0, "0.100000\n", ""),
    ) as subprocess_mock:
        result_path = extract_audio_features(manifest_path=manifest_path, output_path=output_path)

    rows = _read_csv_rows(result_path)
    assert result_path == output_path.resolve()
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "vi_sine_001"
    assert rows[0]["status"] == "success"
    assert rows[0]["duration_sec"] == "0.100000"
    assert rows[0]["rms_energy"]
    assert rows[0]["zero_crossing_rate"]
    assert rows[0]["spectral_centroid_hz"]
    assert rows[0]["spectral_bandwidth_hz"]
    assert float(rows[0]["rms_energy"]) > 0
    assert float(rows[0]["zero_crossing_rate"]) > 0
    subprocess_mock.assert_called_once()
    command = subprocess_mock.call_args.args[0]
    assert command[:6] == ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of"]


def test_extract_audio_features_skips_missing_when_requested(tmp_path: Path) -> None:
    missing_input = tmp_path / "missing.wav"
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "features.csv"
    _write_manifest(manifest_path, [_manifest_row("vi_missing", missing_input)])

    with patch("scripts.extract_audio_features.subprocess.run") as subprocess_mock:
        result_path = extract_audio_features(
            manifest_path=manifest_path,
            output_path=output_path,
            skip_missing=True,
        )

    rows = _read_csv_rows(result_path)
    subprocess_mock.assert_not_called()
    assert rows[0]["status"] == "skipped"
    assert rows[0]["rms_energy"] == ""
    assert rows[0]["zero_crossing_rate"] == ""
    assert "Input file not found" in rows[0]["error"]


def test_extract_audio_features_marks_missing_failed_by_default(tmp_path: Path) -> None:
    missing_input = tmp_path / "missing.wav"
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "features.csv"
    _write_manifest(manifest_path, [_manifest_row("vi_missing", missing_input)])

    result_path = extract_audio_features(manifest_path=manifest_path, output_path=output_path)

    rows = _read_csv_rows(result_path)
    assert rows[0]["status"] == "failed"
    assert rows[0]["duration_sec"] == ""
    assert rows[0]["rms_energy"] == ""
    assert rows[0]["zero_crossing_rate"] == ""
    assert "Input file not found" in rows[0]["error"]
