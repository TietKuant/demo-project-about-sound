"""Build a manifest for local MUSDB18 preview stem containers."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


FIELDNAMES = ["track_id", "split", "input_path", "dataset", "source"]


def _infer_split(path: Path) -> str:
    if "train" in path.parts:
        return "train"
    if "test" in path.parts:
        return "test"
    return "unknown"


def _readable_path(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), Path.cwd().resolve())).as_posix()


def build_musdb_preview_manifest(*, dataset_root: Path, output_path: Path, limit: int | None = None) -> Path:
    """Scan MUSDB preview stem containers and write a deterministic CSV manifest."""
    root = Path(dataset_root).resolve()
    rows = [
        {
            "track_id": path.name.removesuffix(".stem.mp4"),
            "split": _infer_split(path.relative_to(root)),
            "input_path": _readable_path(path),
            "dataset": "MUSDB18-7-STEMS",
            "source": "sigsep-mus-db release preview",
        }
        for path in root.rglob("*.stem.mp4")
    ]
    rows.sort(key=lambda row: (row["split"], row["track_id"]))
    if limit is not None:
        rows = rows[:limit]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return output.resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the MUSDB preview manifest CLI parser."""
    parser = argparse.ArgumentParser(description="Build a MUSDB18 preview manifest.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/manifests/musdb_preview.csv"), type=Path)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    output_path = build_musdb_preview_manifest(
        dataset_root=args.dataset_root,
        output_path=args.output,
        limit=args.limit,
    )
    print(f"Wrote MUSDB preview manifest: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
