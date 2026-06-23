"""Smoke tests for clean-speech Router V4 split manifest building."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_clean_speech_router_v4_manifest import (
    OUTPUT_COLUMNS,
    build_clean_speech_router_v4_manifest,
    main,
    validate_clean_router_v4_rows,
)


def _write_clean_manifest(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for speaker in ("p226", "p227"):
            for index in range(3):
                writer.writerow(
                    {
                        "sample_id": f"{speaker}_{index:03d}",
                        "path": f"clean/{speaker}_{index:03d}.wav",
                        "split": "train",
                        "notes": f"speaker={speaker}",
                    }
                )
        for index in range(2):
            writer.writerow(
                {
                    "sample_id": f"p232_{index:03d}",
                    "path": f"clean/p232_{index:03d}.wav",
                    "split": "test",
                    "notes": "official test",
                }
            )
        writer.writerow(
            {
                "sample_id": "custom_001",
                "path": "clean/custom_001.wav",
                "split": "",
                "notes": "missing split",
            }
        )


def _write_local_shaped_manifest(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for index in range(353):
            writer.writerow(
                {
                    "sample_id": f"p226_{index:03d}",
                    "path": f"clean/p226_{index:03d}.wav",
                    "split": "train",
                    "notes": "speaker=p226",
                }
            )
        for index in range(147):
            writer.writerow(
                {
                    "sample_id": f"p227_{index:03d}",
                    "path": f"clean/p227_{index:03d}.wav",
                    "split": "train",
                    "notes": "speaker=p227",
                }
            )
        for index in range(100):
            writer.writerow(
                {
                    "sample_id": f"p232_{index:03d}",
                    "path": f"clean/p232_{index:03d}.wav",
                    "split": "test",
                    "notes": "speaker=p232",
                }
            )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def test_writes_exact_columns_and_preserves_test_rows(tmp_path: Path) -> None:
    input_manifest = tmp_path / "clean.csv"
    output_manifest = tmp_path / "router_v4_clean.csv"
    _write_clean_manifest(input_manifest)

    build_clean_speech_router_v4_manifest(
        input_manifest=input_manifest,
        output_manifest=output_manifest,
        seed=13,
        val_ratio=0.5,
    )

    rows = _read_rows(output_manifest)
    assert list(rows[0].keys()) == OUTPUT_COLUMNS
    assert {row["split"] for row in rows if row["sample_id"].startswith("p232_")} == {"test"}


def test_creates_val_rows_from_train_rows(tmp_path: Path) -> None:
    input_manifest = tmp_path / "clean.csv"
    output_manifest = tmp_path / "router_v4_clean.csv"
    _write_clean_manifest(input_manifest)

    build_clean_speech_router_v4_manifest(
        input_manifest=input_manifest,
        output_manifest=output_manifest,
        seed=13,
        val_ratio=0.5,
    )

    rows = _read_rows(output_manifest)
    assert any(row["split"] == "val" for row in rows if not row["sample_id"].startswith("p232_"))


def test_speaker_policy_does_not_split_same_speaker_between_train_and_val(tmp_path: Path) -> None:
    input_manifest = tmp_path / "clean.csv"
    output_manifest = tmp_path / "router_v4_clean.csv"
    _write_clean_manifest(input_manifest)

    build_clean_speech_router_v4_manifest(
        input_manifest=input_manifest,
        output_manifest=output_manifest,
        seed=99,
        val_ratio=0.5,
        group_policy="speaker",
    )

    rows = _read_rows(output_manifest)
    speaker_splits: dict[str, set[str]] = {}
    for row in rows:
        if row["split"] in {"train", "val"} and row["sample_id"].startswith("p"):
            speaker = row["sample_id"].split("_", 1)[0]
            speaker_splits.setdefault(speaker, set()).add(row["split"])
    assert all(not {"train", "val"} <= splits for splits in speaker_splits.values())


def test_local_shaped_manifest_selects_smaller_train_speaker_for_validation(tmp_path: Path) -> None:
    input_manifest = tmp_path / "clean.csv"
    output_manifest = tmp_path / "router_v4_clean.csv"
    _write_local_shaped_manifest(input_manifest)

    build_clean_speech_router_v4_manifest(
        input_manifest=input_manifest,
        output_manifest=output_manifest,
        seed=13,
        val_ratio=0.15,
        group_policy="speaker",
    )

    rows = _read_rows(output_manifest)
    assert {row["split"] for row in rows if row["sample_id"].startswith("p227_")} == {"val"}
    assert {row["split"] for row in rows if row["sample_id"].startswith("p226_")} == {"train"}
    assert {row["split"] for row in rows if row["sample_id"].startswith("p232_")} == {"test"}
    speaker_train_val_splits: dict[str, set[str]] = {}
    for row in rows:
        if row["split"] in {"train", "val"}:
            speaker = row["sample_id"].split("_", 1)[0]
            speaker_train_val_splits.setdefault(speaker, set()).add(row["split"])
    assert all(not {"train", "val"} <= splits for splits in speaker_train_val_splits.values())


def test_validation_selection_leaves_train_group_when_multiple_groups_exist(tmp_path: Path) -> None:
    input_manifest = tmp_path / "clean.csv"
    output_manifest = tmp_path / "router_v4_clean.csv"
    _write_clean_manifest(input_manifest)

    build_clean_speech_router_v4_manifest(
        input_manifest=input_manifest,
        output_manifest=output_manifest,
        seed=13,
        val_ratio=1.0,
        group_policy="speaker",
    )

    rows = _read_rows(output_manifest)
    train_val_splits = {row["split"] for row in rows if not row["sample_id"].startswith("p232_")}
    assert "train" in train_val_splits
    assert "val" in train_val_splits


def test_sample_group_policy_works(tmp_path: Path) -> None:
    input_manifest = tmp_path / "clean.csv"
    output_manifest = tmp_path / "router_v4_clean.csv"
    _write_clean_manifest(input_manifest)

    build_clean_speech_router_v4_manifest(
        input_manifest=input_manifest,
        output_manifest=output_manifest,
        seed=7,
        val_ratio=0.3,
        group_policy="sample",
    )

    rows = _read_rows(output_manifest)
    assert any(row["split"] == "val" for row in rows)
    assert validate_clean_router_v4_rows(rows, group_policy="sample") == []


def test_output_is_deterministic_with_same_seed(tmp_path: Path) -> None:
    input_manifest = tmp_path / "clean.csv"
    first_output = tmp_path / "first.csv"
    second_output = tmp_path / "second.csv"
    _write_clean_manifest(input_manifest)

    build_clean_speech_router_v4_manifest(input_manifest=input_manifest, output_manifest=first_output, seed=21)
    build_clean_speech_router_v4_manifest(input_manifest=input_manifest, output_manifest=second_output, seed=21)

    assert first_output.read_text(encoding="utf-8") == second_output.read_text(encoding="utf-8")


def test_notes_preserve_original_and_append_metadata(tmp_path: Path) -> None:
    input_manifest = tmp_path / "clean.csv"
    output_manifest = tmp_path / "router_v4_clean.csv"
    _write_clean_manifest(input_manifest)

    build_clean_speech_router_v4_manifest(input_manifest=input_manifest, output_manifest=output_manifest, seed=31)

    row = _read_rows(output_manifest)[0]
    assert "speaker=p226" in row["notes"]
    assert "router_v4_split_policy=train_val_from_train_keep_test" in row["notes"]
    assert "router_v4_group_policy=speaker" in row["notes"]
    assert "router_v4_seed=31" in row["notes"]
    assert "original_split=train" in row["notes"]
    assert ";;" not in row["notes"]


def test_validation_catches_duplicate_sample_id_across_splits() -> None:
    rows = [
        {"sample_id": "p226_001", "path": "a.wav", "split": "train", "notes": ""},
        {"sample_id": "p226_001", "path": "a.wav", "split": "test", "notes": ""},
    ]

    errors = validate_clean_router_v4_rows(rows)

    assert any("sample_id p226_001 appears in multiple splits" in error for error in errors)


def test_validation_catches_invalid_split() -> None:
    rows = [{"sample_id": "p226_001", "path": "a.wav", "split": "dev", "notes": ""}]

    errors = validate_clean_router_v4_rows(rows)

    assert any("invalid split" in error for error in errors)


def test_validation_catches_same_speaker_in_train_and_val() -> None:
    rows = [
        {"sample_id": "p226_001", "path": "a.wav", "split": "train", "notes": ""},
        {"sample_id": "p226_002", "path": "b.wav", "split": "val", "notes": ""},
        {"sample_id": "p226_003", "path": "c.wav", "split": "test", "notes": ""},
    ]

    errors = validate_clean_router_v4_rows(rows, group_policy="speaker")

    assert any("speaker p226 appears in both train and val" in error for error in errors)


def test_cli_writes_output_manifest(tmp_path: Path) -> None:
    input_manifest = tmp_path / "clean.csv"
    output_manifest = tmp_path / "router_v4_clean.csv"
    _write_clean_manifest(input_manifest)

    exit_code = main(
        [
            "--input-manifest",
            str(input_manifest),
            "--output-manifest",
            str(output_manifest),
            "--seed",
            "5",
            "--val-ratio",
            "0.5",
        ]
    )

    assert exit_code == 0
    rows = _read_rows(output_manifest)
    assert rows
    assert list(rows[0].keys()) == OUTPUT_COLUMNS
