"""Smoke tests for the MUSDB preview manifest builder."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_musdb_preview_manifest import FIELDNAMES, build_musdb_preview_manifest


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def test_musdb_preview_manifest_header_split_order_and_limit(tmp_path: Path) -> None:
    dataset_root = tmp_path / "musdb18-preview"
    output_path = tmp_path / "reports" / "manifest.csv"
    for relative_path in (
        "train/Zebra.stem.mp4",
        "test/Beta.stem.mp4",
        "test/Alpha.stem.mp4",
        "other/Gamma.stem.mp4",
    ):
        path = dataset_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    build_musdb_preview_manifest(dataset_root=dataset_root, output_path=output_path)

    with output_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames == FIELDNAMES
        rows = list(reader)
    assert [(row["split"], row["track_id"]) for row in rows] == [
        ("test", "Alpha"),
        ("test", "Beta"),
        ("train", "Zebra"),
        ("unknown", "Gamma"),
    ]
    assert all(row["dataset"] == "MUSDB18-7-STEMS" for row in rows)
    assert all(row["source"] == "sigsep-mus-db release preview" for row in rows)
    assert all(not Path(row["input_path"]).is_absolute() for row in rows)

    limited_output = tmp_path / "limited.csv"
    build_musdb_preview_manifest(dataset_root=dataset_root, output_path=limited_output, limit=2)
    assert [row["track_id"] for row in _read_rows(limited_output)] == ["Alpha", "Beta"]
