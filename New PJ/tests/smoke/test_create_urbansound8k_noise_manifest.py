"""Smoke tests for UrbanSound8K noise manifest creation."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.create_urbansound8k_noise_manifest import (
    OUTPUT_COLUMNS,
    create_urbansound8k_noise_manifest,
)


def _write_metadata(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["slice_file_name", "fold", "class"])
        writer.writeheader()
        writer.writerow({"slice_file_name": "siren_b.wav", "fold": "2", "class": "siren"})
        writer.writerow({"slice_file_name": "dog_b.wav", "fold": "1", "class": "dog_bark"})
        writer.writerow({"slice_file_name": "air.wav", "fold": "1", "class": "air_conditioner"})
        writer.writerow({"slice_file_name": "horn_a.wav", "fold": "3", "class": "car_horn"})
        writer.writerow({"slice_file_name": "dog_a.wav", "fold": "1", "class": "dog_bark"})


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def test_create_manifest_filters_classes_and_writes_expected_columns(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    audio_root = tmp_path / "data" / "external" / "UrbanSound8K" / "audio"
    output_manifest = tmp_path / "data" / "manifests" / "noise_sources.local.csv"
    _write_metadata(metadata)

    result = create_urbansound8k_noise_manifest(
        metadata_csv=metadata,
        audio_root=audio_root,
        output_manifest=output_manifest,
        classes=["dog_bark", "car_horn", "siren"],
    )

    rows = _read_rows(result)
    assert result == output_manifest.resolve()
    assert list(rows[0].keys()) == OUTPUT_COLUMNS
    assert [row["noise_label"] for row in rows] == ["car_horn", "dog_bark", "dog_bark", "siren"]
    assert {row["noise_label"] for row in rows} == {"dog_bark", "car_horn", "siren"}
    assert all(row["source"] == "UrbanSound8K" for row in rows)
    assert rows[0]["noise_id"] == "horn_a"
    assert rows[0]["notes"] == "fold=3"


def test_create_manifest_uses_paths_relative_to_output_manifest(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    audio_root = tmp_path / "data" / "external" / "UrbanSound8K" / "audio"
    output_manifest = tmp_path / "data" / "manifests" / "noise_sources.local.csv"
    _write_metadata(metadata)

    result = create_urbansound8k_noise_manifest(
        metadata_csv=metadata,
        audio_root=audio_root,
        output_manifest=output_manifest,
        classes=["siren"],
    )

    rows = _read_rows(result)
    assert rows == [
        {
            "noise_id": "siren_b",
            "path": "../external/UrbanSound8K/audio/fold2/siren_b.wav",
            "noise_label": "siren",
            "source": "UrbanSound8K",
            "notes": "fold=2",
        }
    ]


def test_create_manifest_order_is_deterministic(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    audio_root = tmp_path / "UrbanSound8K" / "audio"
    first_output = tmp_path / "first" / "noise.csv"
    second_output = tmp_path / "second" / "noise.csv"
    _write_metadata(metadata)

    create_urbansound8k_noise_manifest(
        metadata_csv=metadata,
        audio_root=audio_root,
        output_manifest=first_output,
        classes=["dog_bark", "car_horn", "siren"],
    )
    create_urbansound8k_noise_manifest(
        metadata_csv=metadata,
        audio_root=audio_root,
        output_manifest=second_output,
        classes=["dog_bark", "car_horn", "siren"],
    )

    first_rows = _read_rows(first_output)
    second_rows = _read_rows(second_output)
    first_projection = [(row["noise_label"], row["notes"], row["noise_id"]) for row in first_rows]
    second_projection = [(row["noise_label"], row["notes"], row["noise_id"]) for row in second_rows]
    assert first_projection == second_projection


def test_create_manifest_raises_for_missing_required_metadata_columns(tmp_path: Path) -> None:
    metadata = tmp_path / "bad_metadata.csv"
    metadata.write_text("slice_file_name,fold\nclip.wav,1\n", encoding="utf-8")

    try:
        create_urbansound8k_noise_manifest(
            metadata_csv=metadata,
            audio_root=tmp_path / "audio",
            output_manifest=tmp_path / "noise.csv",
            classes=["dog_bark"],
        )
    except ValueError as exc:
        assert "missing required columns" in str(exc)
        assert "class" in str(exc)
    else:
        raise AssertionError("Expected missing metadata column to raise ValueError.")


def test_create_manifest_raises_for_empty_selected_classes(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    _write_metadata(metadata)

    try:
        create_urbansound8k_noise_manifest(
            metadata_csv=metadata,
            audio_root=tmp_path / "audio",
            output_manifest=tmp_path / "noise.csv",
            classes=[" ", ""],
        )
    except ValueError as exc:
        assert "At least one UrbanSound8K class" in str(exc)
    else:
        raise AssertionError("Expected empty selected classes to raise ValueError.")


def test_create_manifest_raises_when_selected_classes_match_no_rows(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    _write_metadata(metadata)

    try:
        create_urbansound8k_noise_manifest(
            metadata_csv=metadata,
            audio_root=tmp_path / "audio",
            output_manifest=tmp_path / "noise.csv",
            classes=["children_playing"],
        )
    except ValueError as exc:
        assert "No UrbanSound8K metadata rows found" in str(exc)
        assert "children_playing" in str(exc)
    else:
        raise AssertionError("Expected no matching selected classes to raise ValueError.")
