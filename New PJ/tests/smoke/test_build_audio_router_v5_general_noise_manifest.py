"""Smoke tests for Audio Router V5 general noise manifest builder."""

from __future__ import annotations

import csv
import wave
from pathlib import Path

from scripts.build_audio_router_v5_general_noise_manifest import (
    OUTPUT_COLUMNS,
    build_audio_router_v5_general_noise_manifest,
)


def _write_wav(path: Path, *, sample_rate: int = 8000, frames: int = 800) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fake_esc50(root: Path) -> None:
    metadata = root / "nested" / "ESC-50-master" / "meta" / "esc50.csv"
    audio = metadata.parent.parent / "audio"
    _write_csv(
        metadata,
        ["filename", "fold", "category"],
        [
            {"filename": "rain-1.wav", "fold": "1", "category": "rain"},
            {"filename": "wind-4.wav", "fold": "4", "category": "wind"},
            {"filename": "train-5.wav", "fold": "5", "category": "train"},
            {"filename": "dog-5.wav", "fold": "5", "category": "dog"},
            {"filename": "siren-1.wav", "fold": "1", "category": "siren"},
            {"filename": "laugh-1.wav", "fold": "1", "category": "laughing"},
        ],
    )
    for filename in ("rain-1.wav", "wind-4.wav", "train-5.wav", "dog-5.wav", "siren-1.wav", "laugh-1.wav"):
        _write_wav(audio / filename)


def _fake_urbansound8k(root: Path) -> None:
    metadata = root / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
    audio = metadata.parent.parent / "audio"
    _write_csv(
        metadata,
        ["slice_file_name", "fold", "class"],
        [
            {"slice_file_name": "air.wav", "fold": "1", "class": "air_conditioner"},
            {"slice_file_name": "drill.wav", "fold": "8", "class": "drilling"},
            {"slice_file_name": "engine.wav", "fold": "9", "class": "engine_idling"},
            {"slice_file_name": "dog.wav", "fold": "10", "class": "dog_bark"},
            {"slice_file_name": "music.wav", "fold": "2", "class": "street_music"},
        ],
    )
    for fold, filename in (("1", "air.wav"), ("8", "drill.wav"), ("9", "engine.wav"), ("10", "dog.wav"), ("2", "music.wav")):
        _write_wav(audio / f"fold{fold}" / filename)


def test_build_audio_router_v5_general_noise_manifest_filters_and_maps_splits(tmp_path: Path) -> None:
    esc50_root = tmp_path / "ESC-50-root"
    urban_root = tmp_path / "Urban-root"
    output_csv = tmp_path / "data" / "manifests" / "general_noise.csv"
    _fake_esc50(esc50_root)
    _fake_urbansound8k(urban_root)

    summary = build_audio_router_v5_general_noise_manifest(
        esc50_root=esc50_root,
        urbansound8k_root=urban_root,
        output_csv=output_csv,
    )

    rows = _read_rows(output_csv)
    assert rows
    assert list(rows[0].keys()) == OUTPUT_COLUMNS
    labels = {row["noise_label"] for row in rows}
    assert {"rain", "wind", "train", "air_conditioner", "drilling", "engine_idling"} <= labels
    assert "dog_bark" not in labels
    assert "siren" not in labels
    assert "street_music" not in labels
    assert "laughing" not in labels

    by_source_id = {row["source_id"]: row for row in rows}
    assert by_source_id["rain-1"]["split"] == "train"
    assert by_source_id["wind-4"]["split"] == "val"
    assert by_source_id["train-5"]["split"] == "test"
    assert by_source_id["air"]["split"] == "train"
    assert by_source_id["drill"]["split"] == "val"
    assert by_source_id["engine"]["split"] == "test"
    assert by_source_id["rain-1"]["duration_sec"] == "0.100000"
    assert summary["rows"] == len(rows)


def test_build_audio_router_v5_general_noise_requires_one_source(tmp_path: Path) -> None:
    try:
        build_audio_router_v5_general_noise_manifest(output_csv=tmp_path / "noise.csv")
    except ValueError as exc:
        assert "At least one" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError")


def test_build_audio_router_v5_general_noise_fails_when_filters_remove_all_rows(tmp_path: Path) -> None:
    esc50_root = tmp_path / "ESC-50-root"
    metadata = esc50_root / "ESC-50-master" / "meta" / "esc50.csv"
    audio = metadata.parent.parent / "audio"
    _write_csv(
        metadata,
        ["filename", "fold", "category"],
        [
            {"filename": "siren.wav", "fold": "1", "category": "siren"},
            {"filename": "laugh.wav", "fold": "2", "category": "laughing"},
        ],
    )
    _write_wav(audio / "siren.wav")
    _write_wav(audio / "laugh.wav")

    try:
        build_audio_router_v5_general_noise_manifest(
            esc50_root=esc50_root,
            output_csv=tmp_path / "noise.csv",
        )
    except ValueError as exc:
        assert "No general noise rows were built" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError")
