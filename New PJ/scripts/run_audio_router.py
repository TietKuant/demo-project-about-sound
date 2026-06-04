"""Run standalone inference for the baseline audio router."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except Exception as exc:  # pragma: no cover - exercised only when runtime lacks torch.
    raise RuntimeError("Audio router inference requires torch to be installed.") from exc

from scripts.extract_audio_features import _audio_for_features, _features
from src.router.audio_router_model import AudioRouterMLP


def _load_checkpoint(checkpoint_path: Path, device: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    required_keys = {"model_state_dict", "config"}
    missing_keys = required_keys - set(checkpoint)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Audio router checkpoint is missing required keys: {missing}")
    return checkpoint


def _extract_input_features(input_path: Path, feature_columns: list[str]) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="audio-router-inference-") as temp_dir_name:
        audio, sample_rate = _audio_for_features(input_path, Path(temp_dir_name))
    values = _features(audio, sample_rate)
    values["duration_sec"] = float(audio.size / sample_rate)
    return {column: float(values[column]) for column in feature_columns}


def _normalized_feature_vector(features: dict[str, float], feature_columns: list[str], feature_stats: dict) -> np.ndarray:
    values = []
    for column in feature_columns:
        if column not in feature_stats:
            raise ValueError(f"Feature stats are missing column: {column}")
        mean = float(feature_stats[column]["mean"])
        std = float(feature_stats[column]["std"])
        if abs(std) < 1e-8:
            std = 1.0
        values.append((features[column] - mean) / std)
    return np.asarray([values], dtype=np.float32)


def run_audio_router(
    checkpoint_path: Path,
    input_path: Path,
    output_summary: Path | None = None,
    device: str = "cpu",
) -> dict:
    """Predict broad content type for one audio/video input."""
    checkpoint = Path(checkpoint_path)
    input_file = Path(input_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Audio router checkpoint not found: {checkpoint}")
    if not input_file.exists():
        raise FileNotFoundError(f"Audio router input file not found: {input_file}")

    loaded = _load_checkpoint(checkpoint, device)
    config = loaded["config"]
    feature_columns = list(config["feature_columns"])
    label_to_index = dict(config["label_to_index"])
    feature_stats = dict(config["feature_stats"])
    index_to_label = {int(index): label for label, index in label_to_index.items()}

    features = _extract_input_features(input_file, feature_columns)
    normalized = _normalized_feature_vector(features, feature_columns, feature_stats)

    model = AudioRouterMLP(input_dim=len(feature_columns), num_classes=len(label_to_index)).to(device)
    model.load_state_dict(loaded["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(normalized, dtype=torch.float32, device=device))
        probabilities_tensor = torch.softmax(logits, dim=1)[0].cpu()

    probabilities = {
        index_to_label[index]: float(probabilities_tensor[index].item())
        for index in range(len(index_to_label))
    }
    predicted_label = max(probabilities, key=probabilities.get)
    summary = {
        "status": "success",
        "input_path": str(input_file),
        "checkpoint_path": str(checkpoint),
        "predicted_label": predicted_label,
        "confidence": probabilities[predicted_label],
        "probabilities": probabilities,
        "features": features,
        "error": "",
    }

    summary_path = Path(output_summary) if output_summary is not None else input_file.parent / "router_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
        json_file.write("\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the audio router inference parser."""
    parser = argparse.ArgumentParser(description="Run baseline audio router inference.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        summary = run_audio_router(
            checkpoint_path=args.checkpoint,
            input_path=args.input,
            output_summary=args.output_summary,
            device=args.device,
        )
        print(f"Predicted router label: {summary['predicted_label']} ({summary['confidence']:.4f})")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
