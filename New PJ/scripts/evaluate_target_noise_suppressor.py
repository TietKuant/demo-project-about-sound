"""Evaluate a target-noise suppressor checkpoint across held-out manifest rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import torch
except Exception as exc:  # pragma: no cover - exercised only when torch is missing
    raise RuntimeError("Evaluation requires torch to be installed in the active environment.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_target_noise_suppressor import _load_checkpoint, _paired_metrics, _read_mono_audio


REQUIRED_COLUMNS = {"sample_id", "mixed_path", "target_path", "split", "noise_label", "snr_db"}
PER_SAMPLE_COLUMNS = [
    "sample_id",
    "split",
    "noise_label",
    "snr_db",
    "mixed_path",
    "target_path",
    "baseline_mixed_l1",
    "model_output_l1",
    "output_mse",
    "improved",
]


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


def _resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (Path(manifest_path).resolve().parent / path).resolve()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def evaluate_target_noise_suppressor(
    *,
    manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    split: str = "test",
    max_samples: int | None = 64,
    device: str = "cpu",
) -> dict[str, Path]:
    """Evaluate a checkpoint on selected manifest rows and write aggregate artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    torch_device = torch.device(device)
    model, config = _load_checkpoint(checkpoint, torch_device)
    model_sample_rate = int(config["sample_rate"])

    rows = [row for row in _load_manifest_rows(manifest) if row["split"].strip().lower() == split]
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"No rows found for split: {split}")

    per_sample_rows: list[dict[str, str]] = []
    baseline_values: list[float] = []
    output_values: list[float] = []
    mse_values: list[float] = []
    improved_count = 0

    with torch.no_grad():
        for row in rows:
            mixed_path = _resolve_manifest_path(row["mixed_path"], manifest)
            target_path = _resolve_manifest_path(row["target_path"], manifest)
            mixed_waveform, _ = _read_mono_audio(mixed_path, model_sample_rate)
            target_waveform, _ = _read_mono_audio(target_path, model_sample_rate)
            model_input = mixed_waveform.to(torch_device).view(1, 1, -1)
            enhanced = model(model_input).squeeze(0).squeeze(0).detach().cpu().clamp(-1.0, 1.0)
            metrics = _paired_metrics(mixed_waveform, enhanced, target_waveform)
            improved = metrics["model_output_l1"] < metrics["baseline_mixed_l1"]
            improved_count += int(improved)
            baseline_values.append(metrics["baseline_mixed_l1"])
            output_values.append(metrics["model_output_l1"])
            mse_values.append(metrics["output_mse"])
            per_sample_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "noise_label": row["noise_label"],
                    "snr_db": row["snr_db"],
                    "mixed_path": row["mixed_path"],
                    "target_path": row["target_path"],
                    "baseline_mixed_l1": f"{metrics['baseline_mixed_l1']:.8f}",
                    "model_output_l1": f"{metrics['model_output_l1']:.8f}",
                    "output_mse": f"{metrics['output_mse']:.8f}",
                    "improved": str(improved).lower(),
                }
            )

    total_samples = len(per_sample_rows)
    mean_baseline = _mean(baseline_values)
    mean_model = _mean(output_values)
    relative_improvement = (mean_baseline - mean_model) / mean_baseline if mean_baseline > 0 else 0.0
    summary = {
        "total_samples": total_samples,
        "improved_samples": improved_count,
        "worsened_samples": total_samples - improved_count,
        "improvement_rate": float(improved_count / total_samples),
        "mean_baseline_mixed_l1": mean_baseline,
        "mean_model_output_l1": mean_model,
        "mean_output_mse": _mean(mse_values),
        "relative_l1_improvement": float(relative_improvement),
        "split": split,
        "checkpoint_path": str(checkpoint),
        "manifest_path": str(manifest),
    }

    per_sample_path = output / "per_sample_metrics.csv"
    with per_sample_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PER_SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(per_sample_rows)

    summary_path = output / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
        json_file.write("\n")

    return {"summary": summary_path.resolve(), "per_sample_metrics": per_sample_path.resolve()}


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the aggregate evaluator CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate a target-noise suppressor checkpoint.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        paths = evaluate_target_noise_suppressor(
            manifest_path=args.manifest,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            split=args.split,
            max_samples=args.max_samples,
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
