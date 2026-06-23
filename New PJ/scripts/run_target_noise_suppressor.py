"""Run inference with a trained target-noise suppressor checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
except Exception as exc:  # pragma: no cover - exercised only when runtime deps are missing
    raise RuntimeError(
        "Inference requires numpy, soundfile, torch, and torchaudio to be installed in the active environment."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.target_noise_suppressor import TinyWaveformDenoiser


def _read_mono_audio(path: Path, model_sample_rate: int) -> tuple[torch.Tensor, int]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Input audio file not found: {path}")
    data, source_rate = sf.read(path, always_2d=False)
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    waveform = torch.from_numpy(audio).float()
    if int(source_rate) != model_sample_rate:
        waveform = torchaudio.functional.resample(waveform.unsqueeze(0), int(source_rate), model_sample_rate).squeeze(0)
    return waveform, int(source_rate)


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[TinyWaveformDenoiser, dict[str, object]]:
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint is missing model_state_dict: {checkpoint_path}")
    config = checkpoint.get("config") or {}
    if "sample_rate" not in config:
        raise ValueError(f"Checkpoint config is missing sample_rate: {checkpoint_path}")
    model = TinyWaveformDenoiser().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def _duration_sec(waveform: torch.Tensor, sample_rate: int) -> float:
    return float(waveform.numel() / sample_rate)


def _paired_metrics(input_waveform: torch.Tensor, output_waveform: torch.Tensor, target_waveform: torch.Tensor) -> dict[str, float]:
    min_length = min(input_waveform.numel(), output_waveform.numel(), target_waveform.numel())
    if min_length <= 0:
        raise ValueError("Cannot compute target metrics for empty aligned audio.")
    aligned_input = input_waveform[:min_length]
    aligned_output = output_waveform[:min_length]
    aligned_target = target_waveform[:min_length]
    return {
        "baseline_mixed_l1": float(torch.nn.functional.l1_loss(aligned_input, aligned_target).cpu()),
        "model_output_l1": float(torch.nn.functional.l1_loss(aligned_output, aligned_target).cpu()),
        "output_mse": float(torch.nn.functional.mse_loss(aligned_output, aligned_target).cpu()),
    }


def run_target_noise_suppressor(
    checkpoint_path: Path,
    input_path: Path,
    output_path: Path,
    target_path: Path | None = None,
    summary_path: Path | None = None,
    device: str = "cpu",
) -> dict[str, Path]:
    """Run checkpoint inference and write enhanced WAV plus summary JSON."""
    torch_device = torch.device(device)
    checkpoint = Path(checkpoint_path).expanduser()
    source = Path(input_path).expanduser()
    output = Path(output_path).expanduser()
    summary = Path(summary_path).expanduser() if summary_path is not None else output.parent / "summary.json"

    model, config = _load_checkpoint(checkpoint, torch_device)
    model_sample_rate = int(config["sample_rate"])
    input_waveform, input_sample_rate = _read_mono_audio(source, model_sample_rate)
    output.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model_input = input_waveform.to(torch_device).view(1, 1, -1)
        enhanced = model(model_input).squeeze(0).squeeze(0).detach().cpu().clamp(-1.0, 1.0)

    sf.write(output, enhanced.numpy(), model_sample_rate)

    summary_data: dict[str, object] = {
        "checkpoint_path": str(checkpoint.resolve()),
        "input_path": str(source.resolve()),
        "output_path": str(output.resolve()),
        "target_path": "" if target_path is None else str(Path(target_path).expanduser().resolve()),
        "model_sample_rate": model_sample_rate,
        "input_sample_rate": input_sample_rate,
        "output_sample_rate": model_sample_rate,
        "input_duration_sec": _duration_sec(input_waveform, model_sample_rate),
        "output_duration_sec": _duration_sec(enhanced, model_sample_rate),
        "status": "success",
        "error": "",
    }

    if target_path is not None:
        target = Path(target_path).expanduser()
        target_waveform, _ = _read_mono_audio(target, model_sample_rate)
        summary_data.update(_paired_metrics(input_waveform, enhanced, target_waveform))

    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", encoding="utf-8") as json_file:
        json.dump(summary_data, json_file, indent=2)
        json_file.write("\n")

    return {"output": output.resolve(), "summary": summary.resolve()}


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the target-noise suppressor inference parser."""
    parser = argparse.ArgumentParser(description="Run target-noise suppressor inference.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", default=None, type=Path)
    parser.add_argument("--summary", default=None, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        paths = run_target_noise_suppressor(
            checkpoint_path=args.checkpoint,
            input_path=args.input,
            output_path=args.output,
            target_path=args.target,
            summary_path=args.summary,
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
