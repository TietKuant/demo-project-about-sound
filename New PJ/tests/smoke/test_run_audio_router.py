"""Smoke tests for standalone audio router inference."""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_audio_router import build_arg_parser, build_router_decision, run_audio_router
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


def _write_v4_checkpoint(path: Path) -> None:
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
                "class_weighting": "balanced",
                "class_weights": {
                    "environment_only": 1.0,
                    "music_with_vocals": 1.0,
                    "speech_clean": 1.0,
                    "speech_target_noise": 1.0,
                },
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
    assert written["confidence_threshold"] == 0.90
    assert "accepted" in written
    assert "route_target" in written
    assert "engine_target" in written
    assert "recommended_task" in written
    assert "decision_reason" in written
    assert "warnings" in written
    assert written["decision_reason"] in {"unknown_router_label", "low_confidence", "accepted_router_prediction"}
    assert written["model_metadata"]["feature_columns"] == FEATURE_COLUMNS
    assert written["model_metadata"]["label_to_index"] == {"speech_noise": 0, "music": 1, "environment_noise": 2}
    assert written["model_metadata"]["class_weights"] == {}
    assert "rms_energy" in written["features"]
    assert "duration_sec" in written["features"]
    assert "spectral_rolloff_hz" in written["features"]
    assert "spectral_flatness" in written["features"]
    assert "low_band_energy_ratio" in written["features"]
    assert "mid_band_energy_ratio" in written["features"]
    assert "high_band_energy_ratio" in written["features"]
    assert "rms_std" in written["features"]
    assert "zcr_std" in written["features"]
    assert "silence_ratio" in written["features"]


def test_build_router_decision_low_confidence_abstains() -> None:
    decision = build_router_decision("music_with_vocals", 0.75)

    assert decision["accepted"] is False
    assert decision["route_target"] == "manual_required"
    assert decision["engine_target"] == "none"
    assert decision["recommended_task"] is None
    assert decision["decision_reason"] == "low_confidence"
    assert decision["warnings"] == ["low_confidence_router_prediction"]


def test_build_router_decision_target_noise_maps_to_experimental_task() -> None:
    decision = build_router_decision("speech_target_noise", 0.95)

    assert decision["accepted"] is True
    assert decision["route_target"] == "target_noise_suppression"
    assert decision["engine_target"] == "target_noise_suppressor"
    assert decision["recommended_task"] == "target_noise_suppression"
    assert decision["decision_reason"] == "accepted_router_prediction"
    assert decision["warnings"] == ["target_noise_suppression_is_experimental"]


def test_run_audio_router_summary_includes_v4_model_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    input_path = tmp_path / "input.wav"
    summary_path = tmp_path / "router_summary.json"
    _write_v4_checkpoint(checkpoint)
    _write_sine_wav(input_path)

    summary = run_audio_router(
        checkpoint_path=checkpoint,
        input_path=input_path,
        output_summary=summary_path,
        confidence_threshold=0.1,
    )

    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["confidence_threshold"] == 0.1
    assert written["model_metadata"]["class_weighting"] == "balanced"
    assert set(written["model_metadata"]["class_weights"]) == {
        "environment_only",
        "music_with_vocals",
        "speech_clean",
        "speech_target_noise",
    }
    assert set(written["model_metadata"]["label_to_index"]) == {
        "environment_only",
        "music_with_vocals",
        "speech_clean",
        "speech_target_noise",
    }


def test_parser_accepts_confidence_threshold() -> None:
    args = build_arg_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--input",
            "input.wav",
            "--confidence-threshold",
            "0.72",
        ]
    )

    assert args.confidence_threshold == 0.72


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
