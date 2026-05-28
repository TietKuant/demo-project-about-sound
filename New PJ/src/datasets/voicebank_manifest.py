"""Manifest helpers for VoiceBank-DEMAND paired speech data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = {"sample_id", "noisy_path", "clean_path", "split"}


@dataclass(slots=True)
class VoiceBankPair:
    """One clean/noisy speech pair from a VoiceBank-style manifest."""

    sample_id: str
    noisy_path: Path
    clean_path: Path
    split: str


def _resolve_manifest_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_voicebank_manifest(manifest_path: Path, base_dir: Path | None = None) -> list[VoiceBankPair]:
    """Load VoiceBank clean/noisy pairs from a CSV manifest."""
    manifest = Path(manifest_path).resolve()
    path_base = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()

    with manifest.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - fieldnames
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"VoiceBank manifest is missing required columns: {missing}")

        pairs = [
            VoiceBankPair(
                sample_id=row["sample_id"],
                noisy_path=_resolve_manifest_path(row["noisy_path"], path_base),
                clean_path=_resolve_manifest_path(row["clean_path"], path_base),
                split=row["split"],
            )
            for row in reader
        ]

    if not pairs:
        raise ValueError("VoiceBank manifest must contain at least one row.")
    return pairs


def validate_voicebank_pairs(pairs: list[VoiceBankPair]) -> list[str]:
    """Return human-readable missing-file errors without raising."""
    errors: list[str] = []
    for pair in pairs:
        if not pair.noisy_path.exists():
            errors.append(f"{pair.sample_id}: missing noisy file: {pair.noisy_path}")
        if not pair.clean_path.exists():
            errors.append(f"{pair.sample_id}: missing clean file: {pair.clean_path}")
    return errors
