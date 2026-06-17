"""Smoke tests for Audio Router V5 speech_noisy_general dataset builder."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_audio_router_v5_speech_noisy_general_dataset import (
    V5_COLUMNS,
    build_audio_router_v5_speech_noisy_general_dataset,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _clean_row(sample_id: str, split: str) -> dict[str, str]:
    return {
        "path": f"clean/{split}/{sample_id}.wav",
        "label": "speech_clean",
        "source_corpus": "vivos",
        "source_id": sample_id,
        "recording_id": sample_id.split("_", 1)[0],
        "split": split,
        "is_synthetic": "false",
        "mix_clean_source": "",
        "mix_noise_source": "",
        "snr_db": "",
        "route_target": "no_process",
        "engine_target": "none",
        "duration_sec": "1.000000",
        "notes": "language=vi",
    }


def _non_clean_row(sample_id: str, split: str) -> dict[str, str]:
    row = _clean_row(sample_id, split)
    row["label"] = "music_with_vocals"
    row["source_corpus"] = "musdb18_preview"
    row["route_target"] = "manual_required"
    row["engine_target"] = "demucs"
    return row


def _noise_row(source_id: str, split: str, noise_label: str = "rain") -> dict[str, str]:
    return {
        "path": f"noise/{split}/{source_id}.wav",
        "source_corpus": "esc50",
        "source_id": source_id,
        "recording_id": source_id,
        "split": split,
        "noise_label": noise_label,
        "duration_sec": "1.000000",
        "notes": "fold=1",
    }


def test_build_audio_router_v5_speech_noisy_general_dataset_dry_run(tmp_path: Path) -> None:
    clean_manifest = tmp_path / "manifests" / "clean.csv"
    noise_manifest = tmp_path / "manifests" / "noise.csv"
    output_dir = tmp_path / "generated" / "speech_noisy_general"
    output_csv = tmp_path / "manifests" / "speech_noisy_general.csv"
    clean_rows = [
        _clean_row("clean_train_001", "train"),
        _clean_row("clean_train_002", "train"),
        _clean_row("clean_train_003", "train"),
        _clean_row("clean_val_001", "val"),
        _clean_row("clean_test_001", "test"),
    ]
    noise_rows = [
        _noise_row("noise_train_001", "train", "rain"),
        _noise_row("noise_train_002", "train", "wind"),
        _noise_row("noise_val_001", "val", "air_conditioner"),
        _noise_row("noise_test_001", "test", "engine_idling"),
    ]
    _write_csv(clean_manifest, V5_COLUMNS, clean_rows)
    _write_csv(
        noise_manifest,
        ["path", "source_corpus", "source_id", "recording_id", "split", "noise_label", "duration_sec", "notes"],
        noise_rows,
    )

    summary = build_audio_router_v5_speech_noisy_general_dataset(
        clean_speech_manifest=clean_manifest,
        general_noise_manifest=noise_manifest,
        output_dir=output_dir,
        output_csv=output_csv,
        max_per_split=2,
        snr_db_values=[0.0, 5.0, 10.0],
        dry_run=True,
    )

    rows = _read_rows(output_csv)
    assert rows
    assert list(rows[0].keys()) == V5_COLUMNS
    assert summary["counts_by_split"] == {"test": 1, "train": 2, "val": 1}
    assert all(row["label"] == "speech_noisy_general" for row in rows)
    assert all(row["source_corpus"] == "synthetic_speech_noisy_general" for row in rows)
    assert all(row["is_synthetic"] == "true" for row in rows)
    assert all(row["route_target"] == "clean_voice" for row in rows)
    assert all(row["engine_target"] == "deepfilternet" for row in rows)
    assert all(row["mix_clean_source"].startswith("vivos:") for row in rows)
    assert all(row["mix_noise_source"].startswith("esc50:") for row in rows)

    by_split = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append(row)
        assert row["path"].endswith(".wav")
        assert f"/{row['split']}/" in row["path"]
        assert f"clean_{row['split']}" in row["mix_clean_source"]
        assert f"noise_{row['split']}" in row["mix_noise_source"]

    assert [row["snr_db"] for row in by_split["train"]] == ["0", "5"]
    assert [row["snr_db"] for row in by_split["val"]] == ["0"]
    assert [row["snr_db"] for row in by_split["test"]] == ["0"]
    assert len(by_split["train"]) == 2


def test_build_audio_router_v5_speech_noisy_general_validates_max_per_split(tmp_path: Path) -> None:
    clean_manifest = tmp_path / "clean.csv"
    noise_manifest = tmp_path / "noise.csv"
    _write_csv(clean_manifest, V5_COLUMNS, [_clean_row("clean_train_001", "train")])
    _write_csv(
        noise_manifest,
        ["path", "source_corpus", "source_id", "recording_id", "split", "noise_label", "duration_sec", "notes"],
        [_noise_row("noise_train_001", "train")],
    )

    try:
        build_audio_router_v5_speech_noisy_general_dataset(
            clean_speech_manifest=clean_manifest,
            general_noise_manifest=noise_manifest,
            output_dir=tmp_path / "generated",
            output_csv=tmp_path / "out.csv",
            max_per_split=0,
            snr_db_values=[0.0],
            dry_run=True,
        )
    except ValueError as exc:
        assert "max-per-split" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError")


def test_build_audio_router_v5_speech_noisy_general_uses_only_speech_clean_rows(tmp_path: Path) -> None:
    clean_manifest = tmp_path / "clean.csv"
    noise_manifest = tmp_path / "noise.csv"
    output_csv = tmp_path / "out.csv"
    _write_csv(
        clean_manifest,
        V5_COLUMNS,
        [
            _clean_row("clean_train_001", "train"),
            _non_clean_row("music_train_001", "train"),
        ],
    )
    _write_csv(
        noise_manifest,
        ["path", "source_corpus", "source_id", "recording_id", "split", "noise_label", "duration_sec", "notes"],
        [_noise_row("noise_train_001", "train")],
    )

    build_audio_router_v5_speech_noisy_general_dataset(
        clean_speech_manifest=clean_manifest,
        general_noise_manifest=noise_manifest,
        output_dir=tmp_path / "generated",
        output_csv=output_csv,
        max_per_split=5,
        snr_db_values=[0.0],
        dry_run=True,
    )

    rows = _read_rows(output_csv)
    assert len(rows) == 1
    assert rows[0]["mix_clean_source"] == "vivos:clean_train_001"
    assert "music_train_001" not in rows[0]["mix_clean_source"]


def test_build_audio_router_v5_speech_noisy_general_fails_without_speech_clean_rows(tmp_path: Path) -> None:
    clean_manifest = tmp_path / "clean.csv"
    noise_manifest = tmp_path / "noise.csv"
    _write_csv(clean_manifest, V5_COLUMNS, [_non_clean_row("music_train_001", "train")])
    _write_csv(
        noise_manifest,
        ["path", "source_corpus", "source_id", "recording_id", "split", "noise_label", "duration_sec", "notes"],
        [_noise_row("noise_train_001", "train")],
    )

    try:
        build_audio_router_v5_speech_noisy_general_dataset(
            clean_speech_manifest=clean_manifest,
            general_noise_manifest=noise_manifest,
            output_dir=tmp_path / "generated",
            output_csv=tmp_path / "out.csv",
            max_per_split=1,
            snr_db_values=[0.0],
            dry_run=True,
        )
    except ValueError as exc:
        assert "No speech_clean rows" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError")


def test_build_audio_router_v5_speech_noisy_general_fails_without_noise_rows(tmp_path: Path) -> None:
    clean_manifest = tmp_path / "clean.csv"
    noise_manifest = tmp_path / "noise.csv"
    _write_csv(clean_manifest, V5_COLUMNS, [_clean_row("clean_train_001", "train")])
    _write_csv(
        noise_manifest,
        ["path", "source_corpus", "source_id", "recording_id", "split", "noise_label", "duration_sec", "notes"],
        [],
    )

    try:
        build_audio_router_v5_speech_noisy_general_dataset(
            clean_speech_manifest=clean_manifest,
            general_noise_manifest=noise_manifest,
            output_dir=tmp_path / "generated",
            output_csv=tmp_path / "out.csv",
            max_per_split=1,
            snr_db_values=[0.0],
            dry_run=True,
        )
    except ValueError as exc:
        assert "No general noise rows" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError")
