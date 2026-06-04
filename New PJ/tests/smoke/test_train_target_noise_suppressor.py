"""Smoke tests for the target-noise suppressor training baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from scripts.train_target_noise_suppressor import TargetNoiseDataset, train_target_noise_suppressor
from src.models.target_noise_suppressor import TinyWaveformDenoiser


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["sample_id", "mixed_path", "target_path", "split", "noise_label", "snr_db"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_synthetic_dataset(root: Path, sample_rate: int = 8000) -> Path:
    rng = np.random.default_rng(123)
    manifest_path = root / "target_noise_suppression.csv"
    rows: list[dict[str, str]] = []
    samples = int(sample_rate * 0.12)
    t = np.arange(samples, dtype=np.float32) / sample_rate
    for index, split in enumerate(["train", "train", "test", "test"]):
        clean = 0.2 * np.sin(2 * np.pi * (220 + index * 20) * t)
        noise = 0.03 * rng.standard_normal(samples).astype(np.float32)
        mixed = clean + noise
        mixed_path = root / f"mixed_{index}.wav"
        target_path = root / f"target_{index}.wav"
        _write_wav(mixed_path, mixed, sample_rate)
        _write_wav(target_path, clean, sample_rate)
        rows.append(
            {
                "sample_id": f"sample_{index}",
                "mixed_path": mixed_path.name,
                "target_path": target_path.name,
                "split": split,
                "noise_label": "dog_bark",
                "snr_db": "5",
            }
        )
    _write_manifest(manifest_path, rows)
    return manifest_path


def test_tiny_waveform_denoiser_preserves_input_shape() -> None:
    model = TinyWaveformDenoiser(channels=4)
    waveform = torch.zeros(2, 1, 160)

    output = model(waveform)

    assert output.shape == waveform.shape


def test_target_noise_dataset_random_crop_keeps_mixed_and_target_aligned(tmp_path: Path) -> None:
    sample_rate = 1000
    clean = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    mixed = clean + 0.125
    mixed_path = tmp_path / "mixed.wav"
    target_path = tmp_path / "target.wav"
    manifest_path = tmp_path / "target_noise_suppression.csv"
    _write_wav(mixed_path, mixed, sample_rate)
    _write_wav(target_path, clean, sample_rate)
    _write_manifest(
        manifest_path,
        [
            {
                "sample_id": "aligned",
                "mixed_path": mixed_path.name,
                "target_path": target_path.name,
                "split": "train",
                "noise_label": "dog_bark",
                "snr_db": "5",
            }
        ],
    )
    dataset = TargetNoiseDataset(
        rows=[
            {
                "mixed_path": mixed_path.name,
                "target_path": target_path.name,
                "split": "train",
                "noise_label": "dog_bark",
                "snr_db": "5",
            }
        ],
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        segment_samples=100,
        random_crop=True,
    )

    mixed_segment, target_segment = dataset[0]

    offset = mixed_segment.squeeze(0) - target_segment.squeeze(0)
    assert torch.allclose(offset, torch.full_like(offset, 0.125), atol=1e-4)


def test_train_target_noise_suppressor_writes_expected_artifacts(tmp_path: Path) -> None:
    manifest_path = _make_synthetic_dataset(tmp_path / "dataset")
    output_dir = tmp_path / "training"

    paths = train_target_noise_suppressor(
        manifest=manifest_path,
        output_dir=output_dir,
        sample_rate=8000,
        segment_seconds=0.05,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        max_train_samples=2,
        max_test_samples=2,
        seed=7,
        device="cpu",
    )

    checkpoint_path = output_dir / "checkpoint.pt"
    config_path = output_dir / "config.json"
    metrics_path = output_dir / "metrics.json"
    loss_curve_path = output_dir / "loss_curve.csv"
    assert paths["checkpoint"] == checkpoint_path.resolve()
    assert paths["config"] == config_path.resolve()
    assert paths["metrics"] == metrics_path.resolve()
    assert paths["loss_curve"] == loss_curve_path.resolve()
    assert checkpoint_path.exists()
    assert config_path.exists()
    assert metrics_path.exists()
    assert loss_curve_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "baseline_mixed_l1" in metrics
    assert "model_output_l1" in metrics
    assert "test_mse" in metrics
    assert metrics["train_rows"] == 2
    assert metrics["test_rows"] == 2

    with loss_curve_path.open(newline="", encoding="utf-8") as csv_file:
        loss_rows = list(csv.DictReader(csv_file))
    assert loss_rows[0]["epoch"] == "1"
    assert loss_rows[0]["train_loss"]
