"""Smoke tests for aggregate target-noise suppressor evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from scripts.evaluate_target_noise_suppressor import evaluate_target_noise_suppressor
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


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["sample_id", "mixed_path", "target_path", "split", "noise_label", "snr_db"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_dataset(root: Path, sample_rate: int = 8000) -> Path:
    rows: list[dict[str, str]] = []
    t = np.arange(400, dtype=np.float32) / sample_rate
    for index, split in enumerate(["train", "test", "test"]):
        target = 0.2 * np.sin(2 * np.pi * (220 + index * 20) * t)
        mixed = target + 0.05
        mixed_path = root / f"mixed_{index}.wav"
        target_path = root / f"target_{index}.wav"
        _write_wav(mixed_path, mixed, sample_rate)
        _write_wav(target_path, target, sample_rate)
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
    manifest_path = root / "target_noise_suppression.csv"
    _write_manifest(manifest_path, rows)
    return manifest_path


def test_evaluate_target_noise_suppressor_writes_summary_and_per_sample_metrics(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    manifest_path = _make_dataset(tmp_path / "dataset")
    output_dir = tmp_path / "eval"
    _write_checkpoint(checkpoint_path)

    paths = evaluate_target_noise_suppressor(
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        split="test",
        max_samples=2,
        device="cpu",
    )

    summary_path = output_dir / "evaluation_summary.json"
    per_sample_path = output_dir / "per_sample_metrics.csv"
    assert paths["summary"] == summary_path.resolve()
    assert paths["per_sample_metrics"] == per_sample_path.resolve()
    assert summary_path.exists()
    assert per_sample_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_samples"] == 2
    assert "improvement_rate" in summary
    assert "mean_baseline_mixed_l1" in summary
    assert "mean_model_output_l1" in summary
    assert summary["split"] == "test"
    assert summary["checkpoint_path"] == str(checkpoint_path.resolve())
    assert summary["manifest_path"] == str(manifest_path.resolve())

    with per_sample_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2
    assert rows[0]["split"] == "test"
    assert rows[0]["baseline_mixed_l1"]
    assert rows[0]["model_output_l1"]
    assert rows[0]["output_mse"]
    assert rows[0]["improved"] in {"true", "false"}
