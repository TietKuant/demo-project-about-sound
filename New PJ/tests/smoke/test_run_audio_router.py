"""Smoke tests for standalone audio router inference."""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_audio_router import run_audio_router
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
    labels = {"speech_noise": 0, "music": 1, "environment_noise": 2}
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


def test_run_audio_router_writes_summary(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    input_path = tmp_path / "input.wav"
    summary_path = tmp_path / "router_summary.json"
    _write_checkpoint(checkpoint)
    _write_sine_wav(input_path)

    summary = run_audio_router(
        checkpoint_path=checkpoint,
        input_path=input_path,
        output_summary=summary_path,
    )

    assert summary_path.exists()
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "success"
    assert written["status"] == "success"
    assert written["predicted_label"] in {"speech_noise", "music", "environment_noise"}
    assert set(written["probabilities"]) == {"speech_noise", "music", "environment_noise"}
    assert isinstance(written["confidence"], float)
    assert "rms_energy" in written["features"]
    assert "duration_sec" in written["features"]


def test_run_audio_router_raises_for_missing_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)

    with pytest.raises(FileNotFoundError, match="input file not found"):
        run_audio_router(
            checkpoint_path=checkpoint,
            input_path=tmp_path / "missing.wav",
            output_summary=tmp_path / "summary.json",
        )

def test_run_audio_router_raises_for_missing_checkpoint(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    _write_sine_wav(input_path)

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        run_audio_router(
            checkpoint_path=tmp_path / "missing_checkpoint.pt",
            input_path=input_path,
            output_summary=tmp_path / "summary.json",
        )
\n