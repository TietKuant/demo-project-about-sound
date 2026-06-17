"""Smoke tests for Audio Router batch inference."""

from __future__ import annotations

import csv
import math
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_audio_router_batch import BATCH_COLUMNS, run_audio_router_batch
from scripts.train_audio_router import FEATURE_COLUMNS
from src.router.audio_router_model import AudioRouterMLP


def _write_sine_wav(path: Path, frequency: float = 440.0, duration_sec: float = 0.1, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.arange(int(duration_sec * sample_rate), dtype=np.float64)
    audio = 0.3 * np.sin(2.0 * math.pi * frequency * samples / sample_rate)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _write_checkpoint(path: Path) -> None:
    labels = {
        "environment_only": 0,
        "music_with_vocals": 1,
        "speech_clean": 2,
        "speech_target_noise": 3,
    }
    model = AudioRouterMLP(input_dim=len(FEATURE_COLUMNS), num_classes=len(labels))
    feature_stats = {column: {"mean": 0.0, "std": 1.0} for column in FEATURE_COLUMNS}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "feature_columns": FEATURE_COLUMNS,
                "label_to_index": labels,
                "feature_stats": feature_stats,
            },
        },
        path,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def test_run_audio_router_batch_writes_csv_and_summaries(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    input_dir = tmp_path / "inputs"
    output_csv = tmp_path / "reports" / "router_batch.csv"
    _write_checkpoint(checkpoint)
    _write_sine_wav(input_dir / "first.wav", frequency=220.0)
    _write_sine_wav(input_dir / "nested" / "second.wav", frequency=880.0)

    result = run_audio_router_batch(
        checkpoint_path=checkpoint,
        input_dir=input_dir,
        output_csv=output_csv,
        confidence_threshold=0.4,
    )

    assert result == output_csv.resolve()
    rows = _read_rows(output_csv)
    assert len(rows) == 2
    assert list(rows[0].keys()) == BATCH_COLUMNS
    for row in rows:
        assert row["status"] == "success"
        assert row["confidence_threshold"] == "0.4000000000"
        assert row["predicted_label"]
        assert row["prob_environment_only"]
        assert row["prob_music_with_vocals"]
        assert row["prob_speech_clean"]
        assert row["prob_speech_target_noise"]

    summary_dir = output_csv.parent / "router_summaries"
    assert (summary_dir / "first.router_summary.json").exists()
    assert (summary_dir / "nested_second.router_summary.json").exists()


def test_run_audio_router_batch_avoids_summary_name_collisions(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    input_dir = tmp_path / "inputs"
    output_csv = tmp_path / "reports" / "router_batch.csv"
    _write_checkpoint(checkpoint)
    _write_sine_wav(input_dir / "a" / "test.wav", frequency=220.0)
    _write_sine_wav(input_dir / "b" / "test.wav", frequency=880.0)

    run_audio_router_batch(
        checkpoint_path=checkpoint,
        input_dir=input_dir,
        output_csv=output_csv,
    )

    summary_dir = output_csv.parent / "router_summaries"
    assert (summary_dir / "a_test.router_summary.json").exists()
    assert (summary_dir / "b_test.router_summary.json").exists()
    assert not (summary_dir / "test.router_summary.json").exists()


def test_run_audio_router_batch_raises_for_missing_input_dir(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)

    with pytest.raises(FileNotFoundError, match="input_dir not found"):
        run_audio_router_batch(
            checkpoint_path=checkpoint,
            input_dir=tmp_path / "missing",
            output_csv=tmp_path / "reports" / "router_batch.csv",
        )


def test_run_audio_router_batch_raises_for_non_directory_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    input_file = tmp_path / "input.wav"
    _write_checkpoint(checkpoint)
    _write_sine_wav(input_file)

    with pytest.raises(NotADirectoryError, match="input_dir is not a directory"):
        run_audio_router_batch(
            checkpoint_path=checkpoint,
            input_dir=input_file,
            output_csv=tmp_path / "reports" / "router_batch.csv",
        )


def test_run_audio_router_batch_raises_for_empty_input_dir(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    input_dir = tmp_path / "empty_inputs"
    input_dir.mkdir()
    _write_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="no supported media files") as exc_info:
        run_audio_router_batch(
            checkpoint_path=checkpoint,
            input_dir=input_dir,
            output_csv=tmp_path / "reports" / "router_batch.csv",
        )

    assert str(input_dir) in str(exc_info.value)
    assert ".wav" in str(exc_info.value)
