"""Create a project noise manifest from UrbanSound8K metadata."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_COLUMNS = {"slice_file_name", "fold", "class"}
OUTPUT_COLUMNS = ["noise_id", "path", "noise_label", "source", "notes"]
DEFAULT_CLASSES = ["dog_bark", "car_horn", "siren"]


def _readable_path(path: Path, output_manifest: Path) -> str:
    base_dir = Path(output_manifest).resolve().parent
    return Path(os.path.relpath(path.resolve(), base_dir)).as_posix()


def _load_metadata(metadata_csv: Path) -> list[dict[str, str]]:
    with Path(metadata_csv).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"UrbanSound8K metadata is missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"UrbanSound8K metadata is empty: {metadata_csv}")
    return rows


def create_urbansound8k_noise_manifest(
    *,
    metadata_csv: Path,
    audio_root: Path,
    output_manifest: Path,
    classes: list[str],
) -> Path:
    """Convert UrbanSound8K metadata rows into the project noise manifest format."""
    selected_classes = {value.strip() for value in classes if value.strip()}
    if not selected_classes:
        raise ValueError("At least one UrbanSound8K class must be selected.")
    rows = []
    for row in _load_metadata(metadata_csv):
        noise_label = row["class"].strip()
        if noise_label not in selected_classes:
            continue
        slice_file_name = row["slice_file_name"].strip()
        fold = row["fold"].strip()
        audio_path = Path(audio_root).resolve() / f"fold{fold}" / slice_file_name
        rows.append(
            {
                "noise_id": Path(slice_file_name).stem,
                "path": _readable_path(audio_path, output_manifest),
                "noise_label": noise_label,
                "source": "UrbanSound8K",
                "notes": f"fold={fold}",
                "_sort_label": noise_label,
                "_sort_fold": int(fold) if fold.isdigit() else fold,
                "_sort_name": slice_file_name,
            }
        )
    if not rows:
        selected = ", ".join(sorted(selected_classes))
        raise ValueError(f"No UrbanSound8K metadata rows found for selected classes: {selected}")

    rows.sort(key=lambda item: (item["_sort_label"], item["_sort_fold"], item["_sort_name"]))
    output = Path(output_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in OUTPUT_COLUMNS})
    return output.resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the UrbanSound8K noise manifest parser."""
    parser = argparse.ArgumentParser(description="Create a noise manifest from UrbanSound8K metadata.")
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        output_path = create_urbansound8k_noise_manifest(
            metadata_csv=args.metadata_csv,
            audio_root=args.audio_root,
            output_manifest=args.output_manifest,
            classes=args.classes,
        )
        print(f"Wrote UrbanSound8K noise manifest: {output_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
