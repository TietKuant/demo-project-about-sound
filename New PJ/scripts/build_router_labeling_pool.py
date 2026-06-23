#!/usr/bin/env python3
"""Build a balanced local pool for the router labeling UI."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.create_router_labeling_manifest import (
    MEDIA_SUFFIXES,
    OUTPUT_COLUMNS,
    stable_sample_id,
)


DEFAULT_SOURCES: dict[str, tuple[Path, ...]] = {
    "speech_clean": (
        PROJECT_ROOT / "data/external/voicebank/clean_testset_wav",
        PROJECT_ROOT / "data/external/voicebank/clean_trainset_28spk_wav",
    ),
    "speech_noisy_general": (
        PROJECT_ROOT / "data/external/voicebank/noisy_testset_wav",
        PROJECT_ROOT / "data/generated/audio_router_v5_speech_noisy_general/train",
        PROJECT_ROOT / "data/generated/audio_router_v5_speech_noisy_general/val",
        PROJECT_ROOT / "data/generated/audio_router_v5_speech_noisy_general/test",
    ),
    "speech_target_noise": (
        PROJECT_ROOT / "outputs/target-noise-v2-source-disjoint/mixed/train",
        PROJECT_ROOT / "outputs/target-noise-v2-source-disjoint/mixed/test",
        PROJECT_ROOT / "outputs/target-noise-v1/mixed/train",
        PROJECT_ROOT / "outputs/target-noise-v1/mixed/test",
    ),
    "music_with_vocals": (
        PROJECT_ROOT / "data/external/musdb18-preview/train",
        PROJECT_ROOT / "data/external/musdb18-preview/test",
    ),
    "environment_only": (
        PROJECT_ROOT / "data/external/ESC-50-master/audio",
        PROJECT_ROOT / "data/external/UrbanSound8K/audio",
    ),
    "unknown_mixed": (
        PROJECT_ROOT / "data/real_test_audio/hard_cases",
    ),
}


def _source_name(source_dir: Path) -> str:
    resolved = source_dir.expanduser().resolve(strict=False)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _collect_candidates(
    sources: Mapping[str, Sequence[str | Path]],
) -> dict[str, list[tuple[Path, str]]]:
    collected: dict[str, list[tuple[Path, str]]] = {
        label: [] for label in sources
    }
    seen_paths: set[Path] = set()
    for label, source_dirs in sources.items():
        for source_dir_value in source_dirs:
            source_dir = Path(source_dir_value).expanduser()
            if not source_dir.is_dir():
                continue
            source_name = _source_name(source_dir)
            for path in sorted(source_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
                    continue
                resolved_path = path.resolve()
                if resolved_path in seen_paths:
                    continue
                seen_paths.add(resolved_path)
                collected[label].append((resolved_path, source_name))
    return collected


def build_router_labeling_pool(
    output_path: str | Path,
    *,
    max_per_label: int = 30,
    seed: int = 42,
    sources: Mapping[str, Sequence[str | Path]] | None = None,
) -> Path:
    """Sample a balanced, unlabeled pool from media already on disk."""
    if max_per_label <= 0:
        raise ValueError("max_per_label must be greater than zero.")

    source_mapping = sources if sources is not None else DEFAULT_SOURCES
    candidates = _collect_candidates(source_mapping)
    rng = random.Random(seed)
    label_order = {label: index for index, label in enumerate(source_mapping)}
    rows: list[dict[str, str]] = []

    for label, label_candidates in candidates.items():
        sample_size = min(max_per_label, len(label_candidates))
        sampled = rng.sample(label_candidates, sample_size)
        for path, source_name in sampled:
            rows.append(
                {
                    "sample_id": stable_sample_id(path),
                    "path": str(path),
                    "filename": path.name,
                    "file_id": "",
                    "router_label": "",
                    "router_confidence": "",
                    "router_accepted": "",
                    "router_reason": "",
                    "source": source_name,
                    "duration_sec": "",
                    "rms_energy": "",
                    "zero_crossing_rate": "",
                    "spectral_centroid_hz": "",
                    "spectral_bandwidth_hz": "",
                    "suggested_label": label,
                    "human_label": "",
                    "workflow_label": "",
                    "split": "",
                    "notes": f"source_suggested={label}; verify_by_listening",
                }
            )

    if not rows:
        raise ValueError("No media files were found in the configured local sources.")

    rows.sort(
        key=lambda row: (
            label_order[row["suggested_label"]],
            row["filename"].lower(),
            row["path"],
        )
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _count_rows(path: Path) -> Counter[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return Counter(
            row["suggested_label"]
            for row in csv.DictReader(handle)
            if row.get("suggested_label")
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a balanced local router labeling pool."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-per-label", default=30, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    output_path = build_router_labeling_pool(
        args.output,
        max_per_label=args.max_per_label,
        seed=args.seed,
    )
    counts = _count_rows(output_path)
    print(f"Wrote router labeling pool: {output_path}")
    for label in DEFAULT_SOURCES:
        print(f"{label}: {counts.get(label, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
