"""Smoke tests for target-noise suppressor inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from scripts.run_target_noise_suppressor import run_target_noise_suppressor
from src.models.target_noise_suppressor import TinyWaveformDenoiser


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate)


def _write_checkpoint(path: Path, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = TinyWaveformDenoiser()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"sample_rate": sample_rate},
            "metrics": {},
        },
        path,
    )


def test_run_target_noise_suppressor_writes_output_and_summary_with_target_metrics(tmp_path: Path) -> None:
    sample_rate = 8000
    t = np.arange(800, dtype=np.float32) / sample_rate
    target = 0.2 * np.sin(2 * np.pi * 220 * t)
    mixed = target + 0.05
    checkpoint_path = tmp_path / "checkpoint.pt"
    input_path = tmp_path / "input.wav"
    target_path = tmp_path / "target.wav"
    output_path = tmp_path / "enhanced.wav"
    summary_path = tmp_path / "summary.json"
    _write_checkpoint(checkpoint_path, sample_rate=sample_rate)
    _write_wav(input_path, mixed, sample_rate)
    _write_wav(target_path, target, sample_rate)

    paths = run_target_noise_suppressor(
        checkpoint_path=checkpoint_path,
        input_path=input_path,
        output_path=output_path,
        target_path=target_path,
        summary_path=summary_path,
        device="cpu",
    )

    assert paths["output"] == output_path.resolve()
    assert paths["summary"] == summary_path.resolve()
    assert output_path.exists()
    assert summary_path.exists()
    enhanced, output_rate = sf.read(output_path)
    assert output_rate == sample_rate
    assert enhanced.ndim == 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "success"
    assert summary["model_sample_rate"] == sample_rate
    assert summary["input_sample_rate"] == sample_rate
    assert summary["output_sample_rate"] == sample_rate
    assert summary["target_path"] == str(target_path.resolve())
    assert "baseline_mixed_l1" in summary
    assert "model_output_l1" in summary
    assert "output_mse" in summary


def test_run_target_noise_suppressor_uses_default_summary_next_to_output(tmp_path: Path) -> None:
    sample_rate = 8000
    checkpoint_path = tmp_path / "checkpoint.pt"
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "nested" / "enhanced.wav"
    _write_checkpoint(checkpoint_path, sample_rate=sample_rate)
    _write_wav(input_path, np.zeros(400, dtype=np.float32), sample_rate)

    paths = run_target_noise_suppressor(
        checkpoint_path=checkpoint_path,
        input_path=input_path,
        output_path=output_path,
        device="cpu",
    )

    assert paths["summary"] == (output_path.parent / "summary.json").resolve()
    assert (output_path.parent / "summary.json").exists()
    summary = json.loads((output_path.parent / "summary.json").read_text(encoding="utf-8"))
    assert summary["target_path"] == ""
    assert "baseline_mixed_l1" not in summary


def test_run_target_noise_suppressor_missing_input_raises_clear_error(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint_path, sample_rate=8000)

    try:
        run_target_noise_suppressor(
            checkpoint_path=checkpoint_path,
            input_path=tmp_path / "missing.wav",
            output_path=tmp_path / "enhanced.wav",
            device="cpu",
        )
    except FileNotFoundError as exc:
        assert "Input audio file not found" in str(exc)
    else:
        raise AssertionError("Expected missing input to raise FileNotFoundError.")
