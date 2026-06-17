"""Build Audio Router V5 general/background noise source manifests."""

from __future__ import annotations

import argparse
import csv
import os
import re
import wave
from collections import Counter
from pathlib import Path


OUTPUT_COLUMNS = [
    "path",
    "source_corpus",
    "source_id",
    "recording_id",
    "split",
    "noise_label",
    "duration_sec",
    "notes",
]
URBANSOUND8K_REQUIRED_COLUMNS = {"slice_file_name", "fold", "class"}
ESC50_REQUIRED_COLUMNS = {"filename", "fold", "category"}
TARGET_NOISE_LABELS = {"dog_bark", "car_horn", "siren"}
EXCLUDED_LABELS = TARGET_NOISE_LABELS | {
    "children_playing",
    "street_music",
    "human_voice",
    "laughter",
    "crying_baby",
    "sneezing",
    "breathing",
    "clapping",
    "coughing",
    "laughing",
}
URBANSOUND8K_ALLOWED_LABELS = {
    "air_conditioner",
    "drilling",
    "engine_idling",
    "jackhammer",
}
ESC50_ALLOWED_LABELS = {
    "rain",
    "sea_waves",
    "crackling_fire",
    "crickets",
    "chirping_birds",
    "water_drops",
    "wind",
    "pouring_water",
    "toilet_flush",
    "thunderstorm",
    "airplane",
    "train",
    "clock_tick",
    "chainsaw",
    "helicopter",
    "vacuum_cleaner",
    "washing_machine",
    "keyboard_typing",
    "door_wood_knock",
    "mouse_click",
    "can_opening",
}
NOTE_VALUE_LIMIT = 240


def _label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _readable_path(path: Path, output_csv: Path) -> str:
    resolved = Path(path).resolve()
    cwd = Path.cwd().resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        base_dir = Path(output_csv).resolve().parent
        return os.path.relpath(resolved, base_dir).replace(os.sep, "/")


def _normalize_note_value(value: str, limit: int = NOTE_VALUE_LIMIT) -> str:
    normalized = re.sub(r"\s+", " ", str(value)).strip().replace(";", ",")
    if len(normalized) <= limit:
        return normalized
    suffix = "...[truncated]"
    return normalized[: max(0, limit - len(suffix))].rstrip() + suffix


def _note(parts: list[tuple[str, str]]) -> str:
    return "; ".join(f"{key}={_normalize_note_value(value)}" for key, value in parts if value)


def _duration_from_wav(path: Path) -> str:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
    except (OSError, wave.Error):
        return ""
    if sample_rate <= 0:
        return ""
    return f"{frames / sample_rate:.6f}"


def _read_csv(path: Path, required_columns: set[str], source_name: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source_name} metadata is missing required columns: {', '.join(sorted(missing))}")
        return list(reader)


def _first_match(root: Path, name: str) -> Path | None:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    return matches[0] if matches else None


def _esc50_metadata_and_audio(root: Path) -> tuple[Path, Path] | None:
    metadata = _first_match(root, "esc50.csv")
    if metadata is None:
        return None
    candidate_audio = metadata.parent.parent / "audio"
    if candidate_audio.exists():
        return metadata, candidate_audio
    audio_dirs = sorted(path for path in root.rglob("audio") if path.is_dir())
    return (metadata, audio_dirs[0]) if audio_dirs else None


def _urbansound8k_metadata_and_audio(root: Path) -> tuple[Path, Path] | None:
    metadata = _first_match(root, "UrbanSound8K.csv")
    if metadata is None:
        return None
    candidate_audio = metadata.parent.parent / "audio"
    if candidate_audio.exists():
        return metadata, candidate_audio
    audio_dirs = sorted(path for path in root.rglob("audio") if path.is_dir())
    return (metadata, audio_dirs[0]) if audio_dirs else None


def _esc50_split(fold: str) -> str:
    if fold == "4":
        return "val"
    if fold == "5":
        return "test"
    return "train"


def _urbansound8k_split(fold: str) -> str:
    if fold == "8":
        return "val"
    if fold in {"9", "10"}:
        return "test"
    return "train"


def _esc50_rows(root: Path, output_csv: Path) -> list[dict[str, str]]:
    discovered = _esc50_metadata_and_audio(root)
    if discovered is None:
        return []
    metadata_path, audio_dir = discovered
    rows: list[dict[str, str]] = []
    for row in _read_csv(metadata_path, ESC50_REQUIRED_COLUMNS, "ESC-50"):
        noise_label = _label(row["category"])
        if noise_label not in ESC50_ALLOWED_LABELS or noise_label in EXCLUDED_LABELS:
            continue
        filename = row["filename"].strip()
        audio_path = audio_dir / filename
        if not audio_path.exists():
            continue
        fold = row["fold"].strip()
        rows.append(
            {
                "path": _readable_path(audio_path, output_csv),
                "source_corpus": "esc50",
                "source_id": Path(filename).stem,
                "recording_id": Path(filename).stem,
                "split": _esc50_split(fold),
                "noise_label": noise_label,
                "duration_sec": _duration_from_wav(audio_path),
                "notes": _note([("source_split", fold), ("fold", fold), ("category", row["category"].strip())]),
            }
        )
    return rows


def _urbansound8k_rows(root: Path, output_csv: Path) -> list[dict[str, str]]:
    discovered = _urbansound8k_metadata_and_audio(root)
    if discovered is None:
        return []
    metadata_path, audio_dir = discovered
    rows: list[dict[str, str]] = []
    for row in _read_csv(metadata_path, URBANSOUND8K_REQUIRED_COLUMNS, "UrbanSound8K"):
        noise_label = _label(row["class"])
        if noise_label not in URBANSOUND8K_ALLOWED_LABELS or noise_label in EXCLUDED_LABELS:
            continue
        fold = row["fold"].strip()
        filename = row["slice_file_name"].strip()
        audio_path = audio_dir / f"fold{fold}" / filename
        if not audio_path.exists():
            continue
        rows.append(
            {
                "path": _readable_path(audio_path, output_csv),
                "source_corpus": "urbansound8k",
                "source_id": Path(filename).stem,
                "recording_id": f"fold{fold}:{Path(filename).stem}",
                "split": _urbansound8k_split(fold),
                "noise_label": noise_label,
                "duration_sec": _duration_from_wav(audio_path),
                "notes": _note([("source_split", fold), ("fold", fold), ("class", row["class"].strip())]),
            }
        )
    return rows


def build_audio_router_v5_general_noise_manifest(
    *,
    output_csv: Path,
    esc50_root: Path | None = None,
    urbansound8k_root: Path | None = None,
) -> dict[str, object]:
    """Build a conservative general/background noise source manifest."""
    if esc50_root is None and urbansound8k_root is None:
        raise ValueError("At least one of --esc50-root or --urbansound8k-root must be provided.")

    output_path = Path(output_csv)
    rows: list[dict[str, str]] = []
    if esc50_root is not None:
        root = Path(esc50_root)
        if not root.exists():
            raise FileNotFoundError(f"ESC-50 root not found: {root}")
        rows.extend(_esc50_rows(root, output_path))
    if urbansound8k_root is not None:
        root = Path(urbansound8k_root)
        if not root.exists():
            raise FileNotFoundError(f"UrbanSound8K root not found: {root}")
        rows.extend(_urbansound8k_rows(root, output_path))

    if not rows:
        raise ValueError(
            "No general noise rows were built. Check source roots, metadata files, and allowed label filters."
        )

    rows.sort(key=lambda row: (row["source_corpus"], row["split"], row["noise_label"], row["source_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    counts_by_source = Counter(row["source_corpus"] for row in rows)
    counts_by_split = Counter(row["split"] for row in rows)
    counts_by_noise_label = Counter(row["noise_label"] for row in rows)
    summary = {
        "rows": len(rows),
        "counts_by_source_corpus": dict(sorted(counts_by_source.items())),
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "counts_by_noise_label": dict(sorted(counts_by_noise_label.items())),
    }
    print(f"rows={summary['rows']}")
    print(f"counts_by_source_corpus={summary['counts_by_source_corpus']}")
    print(f"counts_by_split={summary['counts_by_split']}")
    print(f"counts_by_noise_label={summary['counts_by_noise_label']}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Audio Router V5 general noise source manifest.")
    parser.add_argument("--esc50-root", type=Path, default=None)
    parser.add_argument("--urbansound8k-root", type=Path, default=None)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        build_audio_router_v5_general_noise_manifest(
            esc50_root=args.esc50_root,
            urbansound8k_root=args.urbansound8k_root,
            output_csv=args.output_csv,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
