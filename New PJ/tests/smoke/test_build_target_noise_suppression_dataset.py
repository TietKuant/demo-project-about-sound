"""Smoke tests for target-noise suppression dataset manifest building."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

from scripts.build_target_noise_suppression_dataset import (
    OUTPUT_COLUMNS,
    build_target_noise_suppression_dataset,
    mix_clean_with_noise,
    validate_no_source_split_leakage,
)


def _write_clean_manifest(path: Path, *, include_empty_split: bool = True) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["sample_id", "path", "split", "notes"])
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "clean_a",
                "path": "audio/clean_a.wav",
                "split": "train",
                "notes": "fixed train split",
            }
        )
        writer.writerow(
            {
                "sample_id": "clean_b",
                "path": "audio/clean_b.wav",
                "split": "" if include_empty_split else "val",
                "notes": "missing split is assigned deterministically",
            }
        )


def _write_noise_manifest(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["noise_id", "path", "noise_label", "source", "notes", "split"])
        writer.writeheader()
        writer.writerow(
            {
                "noise_id": "dog_001",
                "path": "noise/dog_001.wav",
                "noise_label": "dog_bark",
                "source": "example",
                "notes": "selected",
                "split": "train",
            }
        )
        writer.writerow(
            {
                "noise_id": "horn_001",
                "path": "noise/horn_001.wav",
                "noise_label": "car_horn",
                "source": "example",
                "notes": "selected",
                "split": "val",
            }
        )
        writer.writerow(
            {
                "noise_id": "rain_001",
                "path": "noise/rain_001.wav",
                "noise_label": "rain",
                "source": "example",
                "notes": "not selected",
                "split": "train",
            }
        )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _mock_mix(clean_path: Path, noise_path: Path, mixed_path: Path, snr_db: float) -> None:
    mixed_path.parent.mkdir(parents=True, exist_ok=True)
    mixed_path.write_text(f"{clean_path.name}+{noise_path.name}@{snr_db}", encoding="utf-8")


def _source_splits(rows: list[dict[str, str]], field: str) -> dict[str, set[str]]:
    splits: dict[str, set[str]] = {}
    for row in rows:
        splits.setdefault(row[field], set()).add(row["split"])
    return splits


def test_output_columns_append_source_disjoint_metadata() -> None:
    assert OUTPUT_COLUMNS[:8] == [
        "sample_id",
        "clean_path",
        "noise_path",
        "mixed_path",
        "target_path",
        "noise_label",
        "snr_db",
        "split",
    ]
    assert OUTPUT_COLUMNS[-4:] == ["clean_source_id", "noise_source_id", "mix_seed", "split_policy"]


def test_build_dataset_filters_noise_classes_and_writes_manifest_and_summary(tmp_path: Path) -> None:
    clean_manifest = tmp_path / "clean.csv"
    noise_manifest = tmp_path / "noise.csv"
    output_root = tmp_path / "dataset"
    _write_clean_manifest(clean_manifest)
    _write_noise_manifest(noise_manifest)

    with patch("scripts.build_target_noise_suppression_dataset.mix_clean_with_noise", side_effect=_mock_mix):
        manifest_path = build_target_noise_suppression_dataset(
            clean_manifest=clean_manifest,
            noise_manifest=noise_manifest,
            output_root=output_root,
            classes=["dog_bark"],
            snr_db_values=[0.0, 5.0],
            max_samples=2,
            seed=123,
        )

    rows = _read_rows(manifest_path)
    assert manifest_path == (output_root / "manifests" / "target_noise_suppression.csv").resolve()
    assert list(rows[0].keys()) == OUTPUT_COLUMNS
    assert len(rows) == 2
    assert {row["noise_label"] for row in rows} == {"dog_bark"}
    assert {row["snr_db"] for row in rows} <= {"0", "5"}
    assert all(row["target_path"] == row["clean_path"] for row in rows)
    assert all((Path.cwd() / row["mixed_path"]).exists() for row in rows)
    assert {row["split_policy"] for row in rows} == {"source_disjoint"}
    assert {row["clean_source_id"] for row in rows} <= {"clean_a", "clean_b"}
    assert {row["noise_source_id"] for row in rows} == {"dog_001"}
    assert {row["mix_seed"] for row in rows} == {"123"}

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["split_policy"] == "source_disjoint"
    assert summary["total_rows"] == 2
    assert summary["counts_by_noise_label"] == {"dog_bark": 2}
    assert summary["snr_db_values"] == [0.0, 5.0]
    assert summary["selected_classes"] == ["dog_bark"]
    assert summary["strict_leakage_validated"] is True


def test_build_dataset_is_deterministic_with_seed(tmp_path: Path) -> None:
    clean_manifest = tmp_path / "clean.csv"
    noise_manifest = tmp_path / "noise.csv"
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_clean_manifest(clean_manifest)
    _write_noise_manifest(noise_manifest)

    with patch("scripts.build_target_noise_suppression_dataset.mix_clean_with_noise", side_effect=_mock_mix):
        first_manifest = build_target_noise_suppression_dataset(
            clean_manifest=clean_manifest,
            noise_manifest=noise_manifest,
            output_root=first_root,
            classes=["dog_bark", "car_horn"],
            snr_db_values=[-5.0, 0.0],
            max_samples=4,
            seed=77,
            split_policy="legacy_clean_split",
        )
        second_manifest = build_target_noise_suppression_dataset(
            clean_manifest=clean_manifest,
            noise_manifest=noise_manifest,
            output_root=second_root,
            classes=["dog_bark", "car_horn"],
            snr_db_values=[-5.0, 0.0],
            max_samples=4,
            seed=77,
            split_policy="legacy_clean_split",
        )

    first_rows = _read_rows(first_manifest)
    second_rows = _read_rows(second_manifest)
    first_projection = [(row["sample_id"], row["noise_label"], row["snr_db"], row["split"]) for row in first_rows]
    second_projection = [(row["sample_id"], row["noise_label"], row["snr_db"], row["split"]) for row in second_rows]
    assert first_projection == second_projection


def test_build_dataset_assigns_missing_split_with_seed(tmp_path: Path) -> None:
    clean_manifest = tmp_path / "clean.csv"
    noise_manifest = tmp_path / "noise.csv"
    output_root = tmp_path / "dataset"
    _write_clean_manifest(clean_manifest)
    _write_noise_manifest(noise_manifest)

    with patch("scripts.build_target_noise_suppression_dataset.mix_clean_with_noise", side_effect=_mock_mix):
        manifest_path = build_target_noise_suppression_dataset(
            clean_manifest=clean_manifest,
            noise_manifest=noise_manifest,
            output_root=output_root,
            classes=["car_horn"],
            snr_db_values=[0.0],
            seed=5,
            split_policy="legacy_clean_split",
        )

    rows = _read_rows(manifest_path)
    clean_b_splits = {row["split"] for row in rows if row["sample_id"] == "clean_b"}
    assert len(clean_b_splits) == 1
    assert clean_b_splits <= {"train", "val", "test"}


def test_default_split_policy_is_source_disjoint(tmp_path: Path) -> None:
    clean_manifest = tmp_path / "clean.csv"
    noise_manifest = tmp_path / "noise.csv"
    output_root = tmp_path / "dataset"
    _write_clean_manifest(clean_manifest)
    _write_noise_manifest(noise_manifest)

    with patch("scripts.build_target_noise_suppression_dataset.mix_clean_with_noise", side_effect=_mock_mix):
        manifest_path = build_target_noise_suppression_dataset(
            clean_manifest=clean_manifest,
            noise_manifest=noise_manifest,
            output_root=output_root,
            classes=["dog_bark"],
            snr_db_values=[0.0],
            seed=13,
        )

    rows = _read_rows(manifest_path)
    assert rows
    assert {row["split_policy"] for row in rows} == {"source_disjoint"}


def test_source_disjoint_has_no_source_id_crossing_splits(tmp_path: Path) -> None:
    clean_manifest = tmp_path / "clean.csv"
    noise_manifest = tmp_path / "noise.csv"
    output_root = tmp_path / "dataset"
    _write_clean_manifest(clean_manifest)
    _write_noise_manifest(noise_manifest)

    with patch("scripts.build_target_noise_suppression_dataset.mix_clean_with_noise", side_effect=_mock_mix):
        manifest_path = build_target_noise_suppression_dataset(
            clean_manifest=clean_manifest,
            noise_manifest=noise_manifest,
            output_root=output_root,
            classes=["dog_bark", "car_horn"],
            snr_db_values=[0.0],
            seed=13,
        )

    rows = _read_rows(manifest_path)
    assert validate_no_source_split_leakage(rows) == []
    assert all(len(splits) == 1 for splits in _source_splits(rows, "clean_source_id").values())
    assert all(len(splits) == 1 for splits in _source_splits(rows, "noise_source_id").values())


def test_source_disjoint_only_mixes_sources_from_same_split(tmp_path: Path) -> None:
    clean_manifest = tmp_path / "clean.csv"
    noise_manifest = tmp_path / "noise.csv"
    output_root = tmp_path / "dataset"
    _write_clean_manifest(clean_manifest, include_empty_split=False)
    _write_noise_manifest(noise_manifest)

    with patch("scripts.build_target_noise_suppression_dataset.mix_clean_with_noise", side_effect=_mock_mix):
        manifest_path = build_target_noise_suppression_dataset(
            clean_manifest=clean_manifest,
            noise_manifest=noise_manifest,
            output_root=output_root,
            classes=["dog_bark", "car_horn"],
            snr_db_values=[0.0],
            seed=13,
        )

    rows = _read_rows(manifest_path)
    assert {row["clean_source_id"] for row in rows if row["noise_source_id"] == "dog_001"} == {"clean_a"}
    assert {row["clean_source_id"] for row in rows if row["noise_source_id"] == "horn_001"} == {"clean_b"}


def test_legacy_clean_split_still_works_and_marks_policy(tmp_path: Path) -> None:
    clean_manifest = tmp_path / "clean.csv"
    noise_manifest = tmp_path / "noise.csv"
    output_root = tmp_path / "dataset"
    _write_clean_manifest(clean_manifest)
    _write_noise_manifest(noise_manifest)

    with patch("scripts.build_target_noise_suppression_dataset.mix_clean_with_noise", side_effect=_mock_mix):
        manifest_path = build_target_noise_suppression_dataset(
            clean_manifest=clean_manifest,
            noise_manifest=noise_manifest,
            output_root=output_root,
            classes=["dog_bark", "car_horn"],
            snr_db_values=[0.0],
            max_samples=3,
            seed=17,
            split_policy="legacy_clean_split",
        )

    rows = _read_rows(manifest_path)
    assert rows
    assert {row["split_policy"] for row in rows} == {"legacy_clean_split"}
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["split_policy"] == "legacy_clean_split"
    assert summary["strict_leakage_validated"] is False


def test_validate_no_source_split_leakage_catches_repeated_noise_source() -> None:
    errors = validate_no_source_split_leakage(
        [
            {"split": "train", "clean_source_id": "clean_a", "noise_source_id": "noise_a"},
            {"split": "test", "clean_source_id": "clean_b", "noise_source_id": "noise_a"},
        ]
    )
    assert any("noise_source_id noise_a" in error for error in errors)


def test_validate_no_source_split_leakage_catches_repeated_clean_source() -> None:
    errors = validate_no_source_split_leakage(
        [
            {"split": "train", "clean_source_id": "clean_a", "noise_source_id": "noise_a"},
            {"split": "test", "clean_source_id": "clean_a", "noise_source_id": "noise_b"},
        ]
    )
    assert any("clean_source_id clean_a" in error for error in errors)


def test_build_dataset_raises_for_missing_selected_classes(tmp_path: Path) -> None:
    clean_manifest = tmp_path / "clean.csv"
    noise_manifest = tmp_path / "noise.csv"
    _write_clean_manifest(clean_manifest)
    _write_noise_manifest(noise_manifest)

    with patch("scripts.build_target_noise_suppression_dataset.mix_clean_with_noise") as mix_mock:
        try:
            build_target_noise_suppression_dataset(
                clean_manifest=clean_manifest,
                noise_manifest=noise_manifest,
                output_root=tmp_path / "dataset",
                classes=["siren"],
                snr_db_values=[0.0],
            )
        except ValueError as exc:
            assert "No noise rows found" in str(exc)
        else:
            raise AssertionError("Expected ValueError for missing selected noise classes.")

    mix_mock.assert_not_called()


def test_mix_clean_with_noise_applies_expected_ffmpeg_volume_scale(tmp_path: Path) -> None:
    clean_path = tmp_path / "clean.wav"
    noise_path = tmp_path / "noise.wav"
    mixed_path = tmp_path / "mixed.wav"
    clean_path.write_bytes(b"clean")
    noise_path.write_bytes(b"noise")

    class Result:
        returncode = 0
        stderr = ""
        stdout = ""

    with patch("scripts.build_target_noise_suppression_dataset.subprocess.run", return_value=Result()) as run_mock:
        mix_clean_with_noise(clean_path, noise_path, mixed_path, 5.0)

    command = run_mock.call_args.args[0]
    filter_index = command.index("-filter_complex") + 1
    filter_graph = command[filter_index]
    assert "volume=0.562341" in filter_graph
    assert "[0:a][noise]amix=inputs=2:duration=first:dropout_transition=0" in filter_graph
    assert command[-1] == str(mixed_path)
