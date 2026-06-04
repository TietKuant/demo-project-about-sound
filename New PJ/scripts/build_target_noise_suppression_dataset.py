"""Build a target-noise suppression dataset manifest from clean speech and noise manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CLEAN_COLUMNS = {"sample_id", "path", "split", "notes"}
NOISE_COLUMNS = {"noise_id", "path", "noise_label", "source", "notes"}
OUTPUT_COLUMNS = [
    "sample_id",
    "clean_path",
    "noise_path",
    "mixed_path",
    "target_path",
    "noise_label",
    "snr_db",
    "split",
]
DEFAULT_SPLIT_RANGES = (0.8, 0.9)


def _readable_path(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), Path.cwd().resolve())).as_posix()


def _load_manifest(path: Path, required_columns: set[str], manifest_name: str) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{manifest_name} manifest is missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{manifest_name} manifest is empty: {path}")
    return rows


def _resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(manifest_path).resolve().parent / path).resolve()


def _assign_split(sample_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    if score < DEFAULT_SPLIT_RANGES[0]:
        return "train"
    if score < DEFAULT_SPLIT_RANGES[1]:
        return "val"
    return "test"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def mix_clean_with_noise(clean_path: Path, noise_path: Path, mixed_path: Path, snr_db: float) -> None:
    """Create a mixed audio file with ffmpeg using approximate noise amplitude scaling.

    This is not measured RMS normalization. It treats SNR as an amplitude ratio:
    noise_volume = 10 ** (-snr_db / 20). The helper remains mockable for tests
    and avoids adding audio dependencies in this sprint.
    """
    mixed_path.parent.mkdir(parents=True, exist_ok=True)
    noise_volume = 10 ** (-float(snr_db) / 20.0)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(clean_path),
            "-i",
            str(noise_path),
            "-filter_complex",
            f"[1:a]volume={noise_volume:.6f}[noise];[0:a][noise]amix=inputs=2:duration=first:dropout_transition=0",
            "-acodec",
            "pcm_s16le",
            str(mixed_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip() or "unknown ffmpeg mixing error"
        raise RuntimeError(f"ffmpeg failed while mixing at SNR {snr_db}: {details}")


def _build_rows(
    *,
    clean_rows: list[dict[str, str]],
    noise_rows: list[dict[str, str]],
    clean_manifest: Path,
    noise_manifest: Path,
    output_root: Path,
    snr_db_values: list[float],
    seed: int,
    max_samples: int | None,
) -> list[dict[str, str]]:
    candidates: list[tuple[dict[str, str], dict[str, str], float]] = []
    for clean_row in sorted(clean_rows, key=lambda row: row["sample_id"]):
        for noise_row in sorted(noise_rows, key=lambda row: (row["noise_label"], row["noise_id"])):
            for snr_db in snr_db_values:
                candidates.append((clean_row, noise_row, snr_db))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    if max_samples is not None:
        candidates = candidates[:max_samples]

    rows: list[dict[str, str]] = []
    for index, (clean_row, noise_row, snr_db) in enumerate(candidates, start=1):
        split = clean_row.get("split", "").strip() or _assign_split(clean_row["sample_id"], seed)
        clean_path = _resolve_manifest_path(clean_row["path"], clean_manifest)
        noise_path = _resolve_manifest_path(noise_row["path"], noise_manifest)
        mixed_name = (
            f"{index:05d}_{_slug(clean_row['sample_id'])}_{_slug(noise_row['noise_label'])}_"
            f"{_slug(noise_row['noise_id'])}_snr{snr_db:g}.wav"
        )
        mixed_path = output_root / "mixed" / split / mixed_name
        mix_clean_with_noise(clean_path, noise_path, mixed_path, snr_db)
        rows.append(
            {
                "sample_id": clean_row["sample_id"],
                "clean_path": _readable_path(clean_path),
                "noise_path": _readable_path(noise_path),
                "mixed_path": _readable_path(mixed_path),
                "target_path": _readable_path(clean_path),
                "noise_label": noise_row["noise_label"],
                "snr_db": f"{snr_db:g}",
                "split": split,
            }
        )
    return rows


def _write_summary(
    *,
    output_root: Path,
    rows: list[dict[str, str]],
    snr_db_values: list[float],
    selected_classes: list[str],
) -> Path:
    counts_by_split = Counter(row["split"] for row in rows)
    counts_by_noise_label = Counter(row["noise_label"] for row in rows)
    summary = {
        "total_rows": len(rows),
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "counts_by_noise_label": dict(sorted(counts_by_noise_label.items())),
        "snr_db_values": [float(value) for value in snr_db_values],
        "selected_classes": selected_classes,
    }
    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
        json_file.write("\n")
    return summary_path


def build_target_noise_suppression_dataset(
    *,
    clean_manifest: Path,
    noise_manifest: Path,
    output_root: Path,
    classes: list[str],
    snr_db_values: list[float],
    max_samples: int | None = None,
    seed: int = 13,
) -> Path:
    """Create mixed audio rows and a structured manifest for future target-noise suppression."""
    if not classes:
        raise ValueError("At least one noise class must be selected.")
    if not snr_db_values:
        raise ValueError("At least one SNR value must be provided.")

    output = Path(output_root)
    (output / "mixed").mkdir(parents=True, exist_ok=True)
    manifests_dir = output / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    clean_rows = _load_manifest(clean_manifest, CLEAN_COLUMNS, "Clean")
    all_noise_rows = _load_manifest(noise_manifest, NOISE_COLUMNS, "Noise")
    selected_classes = sorted(set(classes))
    noise_rows = [row for row in all_noise_rows if row["noise_label"] in selected_classes]
    if not noise_rows:
        raise ValueError(f"No noise rows found for selected classes: {', '.join(selected_classes)}")

    rows = _build_rows(
        clean_rows=clean_rows,
        noise_rows=noise_rows,
        clean_manifest=clean_manifest,
        noise_manifest=noise_manifest,
        output_root=output,
        snr_db_values=snr_db_values,
        seed=seed,
        max_samples=max_samples,
    )
    if not rows:
        raise ValueError("No dataset rows were generated.")

    manifest_path = manifests_dir / "target_noise_suppression.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    _write_summary(
        output_root=output,
        rows=rows,
        snr_db_values=snr_db_values,
        selected_classes=selected_classes,
    )
    return manifest_path.resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the target-noise suppression dataset builder parser."""
    parser = argparse.ArgumentParser(description="Build a target-noise suppression dataset manifest.")
    parser.add_argument("--clean-manifest", required=True, type=Path)
    parser.add_argument("--noise-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--classes", required=True, nargs="+", help="Noise labels to include.")
    parser.add_argument("--snr-db-values", required=True, nargs="+", type=float)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        manifest_path = build_target_noise_suppression_dataset(
            clean_manifest=args.clean_manifest,
            noise_manifest=args.noise_manifest,
            output_root=args.output_root,
            classes=args.classes,
            snr_db_values=args.snr_db_values,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        print(f"Wrote target-noise suppression manifest: {manifest_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
