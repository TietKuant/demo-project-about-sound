"""Smoke tests for the audio router baseline trainer."""

from __future__ import annotations

import csv
import json
import math
import wave
from pathlib import Path

import numpy as np

from scripts.train_audio_router import FEATURE_COLUMNS, train_audio_router


def _write_sine_wav(path: Path, frequency: float, duration_sec: float = 0.12, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.arange(int(duration_sec * sample_rate), dtype=np.float64)
    audio = 0.4 * np.sin(2.0 * math.pi * frequency * samples / sample_rate)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _write_manifest(manifest: Path) -> None:
    audio_dir = manifest.parent / "audio"
    rows = [
        ("speech_train", "speech_noise", "train", 220.0),
        ("speech_test", "speech_noise", "test", 260.0),
        ("music_train", "music", "train", 880.0),
        ("music_test", "music", "test", 930.0),
        ("env_train", "environment_noise", "train", 1800.0),
        ("env_test", "environment_noise", "test", 1900.0),
    ]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["sample_id", "input_path", "router_label", "source", "split", "notes"])
        writer.writeheader()
        for sample_id, label, split, frequency in rows:
            audio_path = audio_dir / f"{sample_id}.wav"
            _write_sine_wav(audio_path, frequency)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "input_path": f"audio/{audio_path.name}",
                    "router_label": label,
                    "source": "synthetic",
                    "split": split,
                    "notes": f"frequency={frequency}",
                }
            )


def test_train_audio_router_writes_artifacts(tmp_path: Path) -> None:
    manifest = tmp_path / "data" / "manifests" / "audio_router.local.csv"
    output_dir = tmp_path / "outputs" / "audio-router"
    _write_manifest(manifest)

    result = train_audio_router(
        manifest_path=manifest,
        output_dir=output_dir,
        epochs=3,
        learning_rate=0.01,
        seed=123,
    )

    assert result == output_dir.resolve()
    for filename in (
        "checkpoint.pt",
        "label_mapping.json",
        "metrics.json",
        "feature_stats.json",
        "feature_rows.csv",
        "loss_curve.csv",
    ):
        assert (output_dir / filename).exists()

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "success"
    assert "train_accuracy" in metrics
    assert "test_accuracy" in metrics
    assert metrics["manifest_rows"] == 6
    assert metrics["successful_feature_rows"] == 6
    assert metrics["failed_feature_rows"] == 0
    assert metrics["total_rows"] == 6
    assert metrics["train_rows"] == 3
    assert metrics["test_rows"] == 3
    assert set(metrics["counts_by_label"]) == {"speech_noise", "music", "environment_noise"}
    assert "confusion_matrix" in metrics

    label_mapping = json.loads((output_dir / "label_mapping.json").read_text(encoding="utf-8"))
    assert set(label_mapping["label_to_index"]) == {"speech_noise", "music", "environment_noise"}

    with (output_dir / "feature_rows.csv").open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames is not None
        for column in FEATURE_COLUMNS:
            assert column in reader.fieldnames
        feature_rows = list(reader)
    assert feature_rows[0]["spectral_rolloff_hz"]
    assert feature_rows[0]["spectral_flatness"]
    assert feature_rows[0]["low_band_energy_ratio"]
    assert feature_rows[0]["mid_band_energy_ratio"]
    assert feature_rows[0]["high_band_energy_ratio"]
    assert feature_rows[0]["rms_std"]
    assert feature_rows[0]["zcr_std"]
    assert feature_rows[0]["silence_ratio"]


def test_train_audio_router_requires_test_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "audio_router.local.csv"
    audio_path = tmp_path / "audio" / "speech.wav"
    _write_sine_wav(audio_path, 220.0)
    with manifest.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["sample_id", "input_path", "router_label", "source", "split", "notes"])
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "speech",
                "input_path": "audio/speech.wav",
                "router_label": "speech_noise",
                "source": "synthetic",
                "split": "train",
                "notes": "",
            }
        )

    try:
        train_audio_router(manifest_path=manifest, output_dir=tmp_path / "out", epochs=1)
    except ValueError as exc:
        assert "test row" in str(exc)
    else:
        raise AssertionError("Expected missing test split to raise ValueError.")
