"""Smoke tests for the audio router baseline trainer."""

from __future__ import annotations

import csv
import json
import math
import wave
from pathlib import Path

import numpy as np
import torch

from scripts.train_audio_router import FEATURE_COLUMNS, _classification_report, train_audio_router


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
        ("speech_train", "speech_noise", "synthetic_speech", "train", 220.0),
        ("speech_test", "speech_noise", "synthetic_speech", "test", 260.0),
        ("music_train", "music", "synthetic_music", "train", 880.0),
        ("music_test", "music", "synthetic_music", "test", 930.0),
        ("env_train", "environment_noise", "synthetic_env", "train", 1800.0),
        ("env_test", "environment_noise", "synthetic_env", "test", 1900.0),
    ]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["sample_id", "input_path", "router_label", "source", "split", "notes"])
        writer.writeheader()
        for sample_id, label, source, split, frequency in rows:
            audio_path = audio_dir / f"{sample_id}.wav"
            _write_sine_wav(audio_path, frequency)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "input_path": f"audio/{audio_path.name}",
                    "router_label": label,
                    "source": source,
                    "split": split,
                    "notes": f"frequency={frequency}",
                }
            )


def _write_v4_manifest(manifest: Path) -> None:
    audio_dir = manifest.parent / "audio"
    rows = [
        ("speech_clean_train", "speech_clean", "voicebank_clean", "train", 240.0),
        ("speech_clean_test", "speech_clean", "voicebank_clean", "test", 280.0),
        ("music_vocals_train", "music_with_vocals", "musdb18_preview", "train", 720.0),
        ("music_vocals_test", "music_with_vocals", "musdb18_preview", "test", 760.0),
        ("env_only_train", "environment_only", "esc50", "train", 1600.0),
        ("env_only_test", "environment_only", "esc50", "test", 1700.0),
    ]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["sample_id", "input_path", "source_dataset", "split", "content_label", "notes"],
        )
        writer.writeheader()
        for sample_id, label, source_dataset, split, frequency in rows:
            audio_path = audio_dir / f"{sample_id}.wav"
            _write_sine_wav(audio_path, frequency)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "input_path": f"audio/{audio_path.name}",
                    "source_dataset": source_dataset,
                    "split": split,
                    "content_label": label,
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
    assert metrics["manifest_schema"] == "audio_router_v3"
    assert metrics["label_column"] == "router_label"
    assert metrics["source_column"] == "source"
    assert set(metrics["label_values"]) == {"speech_noise", "music", "environment_noise"}
    assert "train_accuracy" in metrics
    assert "test_accuracy" in metrics
    assert "duration_sec" not in metrics["feature_columns"]
    assert "duration_sec" in metrics["excluded_feature_columns"]
    assert metrics["feature_policy_notes"]
    assert metrics["manifest_rows"] == 6
    assert metrics["successful_feature_rows"] == 6
    assert metrics["failed_feature_rows"] == 0
    assert metrics["total_rows"] == 6
    assert metrics["train_rows"] == 3
    assert metrics["test_rows"] == 3
    assert set(metrics["counts_by_label"]) == {"speech_noise", "music", "environment_noise"}
    assert metrics["counts_by_split"] == {"test": 3, "train": 3}
    assert metrics["counts_by_source"] == {
        "synthetic_env": 2,
        "synthetic_music": 2,
        "synthetic_speech": 2,
    }
    assert metrics["counts_by_split_label"]["train|speech_noise"] == 1
    assert metrics["counts_by_split_label"]["test|music"] == 1
    assert metrics["counts_by_source_label"]["synthetic_speech|speech_noise"] == 2
    assert metrics["counts_by_source_label"]["synthetic_music|music"] == 2
    assert metrics["counts_by_source_label"]["synthetic_env|environment_noise"] == 2
    assert "train_classification_report" in metrics
    assert "test_classification_report" in metrics
    test_report = metrics["test_classification_report"]
    assert "per_class" in test_report
    assert "macro_f1" in test_report
    assert "weighted_f1" in test_report
    assert "accuracy" in test_report
    assert set(test_report["per_class"]) == {"speech_noise", "music", "environment_noise"}
    assert "test_metrics_by_source" in metrics
    assert set(metrics["test_metrics_by_source"]) == {"synthetic_speech", "synthetic_music", "synthetic_env"}
    for source in ("synthetic_speech", "synthetic_music", "synthetic_env"):
        source_metrics = metrics["test_metrics_by_source"][source]
        assert source_metrics["row_count"] == 1
        assert "accuracy" in source_metrics
        assert "support_by_label" in source_metrics
        assert "predicted_by_label" in source_metrics
        assert "confusion_matrix" in source_metrics
    assert "confusion_matrix" in metrics

    label_mapping = json.loads((output_dir / "label_mapping.json").read_text(encoding="utf-8"))
    assert set(label_mapping["label_to_index"]) == {"speech_noise", "music", "environment_noise"}

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu")
    assert checkpoint["config"]["feature_columns"] == metrics["feature_columns"]
    assert "duration_sec" not in checkpoint["config"]["feature_columns"]
    assert set(checkpoint["config"]["feature_stats"]) == set(metrics["feature_columns"])

    with (output_dir / "feature_rows.csv").open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames is not None
        for column in FEATURE_COLUMNS:
            assert column in reader.fieldnames
        feature_rows = list(reader)
    assert "duration_sec" in feature_rows[0]
    assert feature_rows[0]["duration_sec"]
    assert feature_rows[0]["spectral_rolloff_hz"]
    assert feature_rows[0]["spectral_flatness"]
    assert feature_rows[0]["low_band_energy_ratio"]
    assert feature_rows[0]["mid_band_energy_ratio"]
    assert feature_rows[0]["high_band_energy_ratio"]
    assert feature_rows[0]["rms_std"]
    assert feature_rows[0]["zcr_std"]
    assert feature_rows[0]["silence_ratio"]


def test_train_audio_router_supports_v4_content_label_schema(tmp_path: Path) -> None:
    manifest = tmp_path / "data" / "manifests" / "audio_router_v4.local.csv"
    output_dir = tmp_path / "outputs" / "audio-router-v4"
    expected_labels = {"speech_clean", "music_with_vocals", "environment_only"}
    _write_v4_manifest(manifest)

    train_audio_router(
        manifest_path=manifest,
        output_dir=output_dir,
        epochs=2,
        learning_rate=0.01,
        seed=321,
    )

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["manifest_schema"] == "audio_router_v4"
    assert metrics["label_column"] == "content_label"
    assert metrics["source_column"] == "source_dataset"
    assert set(metrics["label_values"]) == expected_labels
    assert set(metrics["counts_by_label"]) == expected_labels
    assert "duration_sec" not in metrics["feature_columns"]
    assert "duration_sec" in metrics["excluded_feature_columns"]

    label_mapping = json.loads((output_dir / "label_mapping.json").read_text(encoding="utf-8"))
    assert set(label_mapping["label_to_index"]) == expected_labels

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu")
    assert set(checkpoint["config"]["label_to_index"]) == expected_labels
    assert checkpoint["config"]["feature_columns"] == metrics["feature_columns"]
    assert "duration_sec" not in checkpoint["config"]["feature_columns"]

    with (output_dir / "feature_rows.csv").open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames is not None
        assert "router_label" in reader.fieldnames
        assert "duration_sec" in reader.fieldnames
        feature_rows = list(reader)
    assert {row["router_label"] for row in feature_rows} == expected_labels
    assert feature_rows[0]["duration_sec"]


def test_classification_report_handles_zero_division() -> None:
    report = _classification_report(
        truth=[0, 0, 1],
        predictions=[1, 1, 1],
        index_to_label={0: "speech_noise", 1: "music"},
    )

    assert report["accuracy"] == 1 / 3
    assert "macro_f1" in report
    assert "weighted_f1" in report
    assert report["per_class"]["speech_noise"]["support"] == 2
    assert report["per_class"]["speech_noise"]["precision"] == 0.0
    assert report["per_class"]["speech_noise"]["recall"] == 0.0
    assert report["per_class"]["speech_noise"]["f1"] == 0.0
    assert report["per_class"]["music"]["support"] == 1
    for metric in ("precision", "recall", "f1"):
        assert isinstance(report["per_class"]["speech_noise"][metric], float)
        assert isinstance(report["per_class"]["music"][metric], float)
    for metric in ("macro_precision", "macro_recall", "macro_f1", "weighted_f1", "accuracy"):
        assert isinstance(report[metric], float)
        assert not math.isnan(report[metric])


def test_train_audio_router_rejects_unknown_manifest_schema(tmp_path: Path) -> None:
    manifest = tmp_path / "bad_router_manifest.csv"
    output_dir = tmp_path / "out"
    with manifest.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["sample_id", "input_path", "split", "notes"])
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "bad",
                "input_path": "missing.wav",
                "split": "train",
                "notes": "",
            }
        )

    try:
        train_audio_router(manifest_path=manifest, output_dir=output_dir, epochs=1)
    except ValueError as exc:
        message = str(exc)
        assert "must match either the V3 schema" in message
        assert "audio_router_v3 missing" in message
        assert "audio_router_v4 missing" in message
    else:
        raise AssertionError("Expected unknown manifest schema to raise ValueError.")


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
