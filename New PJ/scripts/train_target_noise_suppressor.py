"""Train a tiny PyTorch baseline for target-noise suppression."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover - exercised only when runtime deps are missing
    raise RuntimeError(
        "Training requires numpy, soundfile, torch, and torchaudio to be installed in the active environment."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.target_noise_suppressor import TinyWaveformDenoiser


REQUIRED_COLUMNS = {"mixed_path", "target_path", "split", "noise_label", "snr_db"}


@dataclass(frozen=True)
class TrainConfig:
    manifest: str
    output_dir: str
    sample_rate: int = 16000
    segment_seconds: float = 1.0
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-3
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    seed: int = 42
    device: str = "cpu"


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with Path(manifest_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Target-noise manifest is missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Target-noise manifest is empty: {manifest_path}")
    return rows


def _resolve_audio_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (Path(manifest_path).resolve().parent / path).resolve()


def _read_mono_audio(path: Path, sample_rate: int) -> torch.Tensor:
    data, source_rate = sf.read(path, always_2d=False)
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    waveform = torch.from_numpy(audio).float()
    if int(source_rate) != sample_rate:
        waveform = torchaudio.functional.resample(waveform.unsqueeze(0), int(source_rate), sample_rate).squeeze(0)
    return waveform


def _pad_to_min_length(waveform: torch.Tensor, min_samples: int) -> torch.Tensor:
    if waveform.numel() >= min_samples:
        return waveform
    return torch.nn.functional.pad(waveform, (0, min_samples - waveform.numel()))


def _crop_or_pad_pair(
    mixed: torch.Tensor,
    target: torch.Tensor,
    segment_samples: int,
    *,
    random_crop: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad and crop paired waveforms with one shared crop start."""
    mixed = _pad_to_min_length(mixed, segment_samples)
    target = _pad_to_min_length(target, segment_samples)
    shared_length = min(mixed.numel(), target.numel())
    max_start = shared_length - segment_samples
    start = random.randint(0, max_start) if random_crop and max_start > 0 else max_start // 2
    end = start + segment_samples
    return mixed[start:end], target[start:end]


class TargetNoiseDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Manifest-backed waveform pair dataset."""

    def __init__(
        self,
        *,
        rows: list[dict[str, str]],
        manifest_path: Path,
        sample_rate: int,
        segment_samples: int,
        random_crop: bool,
    ) -> None:
        self.rows = rows
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        mixed = _read_mono_audio(_resolve_audio_path(row["mixed_path"], self.manifest_path), self.sample_rate)
        target = _read_mono_audio(_resolve_audio_path(row["target_path"], self.manifest_path), self.sample_rate)
        mixed, target = _crop_or_pad_pair(mixed, target, self.segment_samples, random_crop=self.random_crop)
        return mixed.unsqueeze(0), target.unsqueeze(0)


def _split_rows(
    rows: list[dict[str, str]],
    *,
    split: str,
    max_samples: int | None,
) -> list[dict[str, str]]:
    selected = [row for row in rows if row["split"].strip().lower() == split]
    if max_samples is not None:
        selected = selected[:max_samples]
    if not selected:
        raise ValueError(f"No {split} rows available after filtering.")
    return selected


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for mixed, target in loader:
        mixed = mixed.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        predicted = model(mixed)
        loss = loss_fn(predicted, target)
        loss.backward()
        optimizer.step()
        batch_size = mixed.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def _evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_output_l1 = 0.0
    total_baseline_l1 = 0.0
    total_mse = 0.0
    total_items = 0
    with torch.no_grad():
        for mixed, target in loader:
            mixed = mixed.to(device)
            target = target.to(device)
            predicted = model(mixed)
            batch_size = mixed.shape[0]
            total_output_l1 += float(torch.nn.functional.l1_loss(predicted, target).cpu()) * batch_size
            total_baseline_l1 += float(torch.nn.functional.l1_loss(mixed, target).cpu()) * batch_size
            total_mse += float(torch.nn.functional.mse_loss(predicted, target).cpu()) * batch_size
            total_items += batch_size
    denominator = max(total_items, 1)
    return {
        "model_output_l1": total_output_l1 / denominator,
        "baseline_mixed_l1": total_baseline_l1 / denominator,
        "test_mse": total_mse / denominator,
    }


def train_target_noise_suppressor(
    *,
    manifest: Path,
    output_dir: Path,
    sample_rate: int = 16000,
    segment_seconds: float = 1.0,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Path]:
    """Train the tiny baseline and write checkpoint/config/metrics artifacts."""
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    _set_deterministic_seed(seed)
    manifest_path = Path(manifest).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        manifest=str(manifest_path),
        output_dir=str(output.resolve()),
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        seed=seed,
        device=device,
    )
    segment_samples = int(round(segment_seconds * sample_rate))
    if segment_samples <= 0:
        raise ValueError("segment_seconds and sample_rate produce an empty segment.")

    rows = _load_manifest_rows(manifest_path)
    train_rows = _split_rows(rows, split="train", max_samples=max_train_samples)
    test_rows = _split_rows(rows, split="test", max_samples=max_test_samples)
    train_dataset = TargetNoiseDataset(
        rows=train_rows,
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        segment_samples=segment_samples,
        random_crop=True,
    )
    test_dataset = TargetNoiseDataset(
        rows=test_rows,
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        segment_samples=segment_samples,
        random_crop=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    torch_device = torch.device(device)
    model = TinyWaveformDenoiser().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()
    loss_rows: list[dict[str, str]] = []

    last_train_loss = 0.0
    for epoch in range(1, epochs + 1):
        last_train_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn, torch_device)
        loss_rows.append({"epoch": str(epoch), "train_loss": f"{last_train_loss:.8f}"})

    train_eval = _evaluate(model, train_loader, torch_device)
    test_eval = _evaluate(model, test_loader, torch_device)
    metrics = {
        "train_loss": float(train_eval["model_output_l1"]),
        "last_epoch_train_loss": float(last_train_loss),
        "test_loss": float(test_eval["model_output_l1"]),
        "baseline_mixed_l1": float(test_eval["baseline_mixed_l1"]),
        "model_output_l1": float(test_eval["model_output_l1"]),
        "test_mse": float(test_eval["test_mse"]),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
    }

    checkpoint_path = output / "checkpoint.pt"
    config_path = output / "config.json"
    metrics_path = output / "metrics.json"
    loss_curve_path = output / "loss_curve.csv"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    with config_path.open("w", encoding="utf-8") as json_file:
        json.dump(asdict(config), json_file, indent=2)
        json_file.write("\n")
    with metrics_path.open("w", encoding="utf-8") as json_file:
        json.dump(metrics, json_file, indent=2)
        json_file.write("\n")
    with loss_curve_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(loss_rows)

    return {
        "checkpoint": checkpoint_path.resolve(),
        "config": config_path.resolve(),
        "metrics": metrics_path.resolve(),
        "loss_curve": loss_curve_path.resolve(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the training CLI parser."""
    parser = argparse.ArgumentParser(description="Train a tiny target-noise suppression baseline.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        paths = train_target_noise_suppressor(
            manifest=args.manifest,
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
            segment_seconds=args.segment_seconds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            seed=args.seed,
            device=args.device,
        )
        for label, path in paths.items():
            print(f"{label}: {path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
