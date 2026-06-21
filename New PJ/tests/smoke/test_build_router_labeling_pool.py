from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.build_router_labeling_pool import build_router_labeling_pool
from scripts.create_router_labeling_manifest import OUTPUT_COLUMNS


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return list(reader.fieldnames or []), rows


def _make_media(directory: Path, names: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(name.encode("utf-8"))


def test_build_pool_is_balanced_deterministic_and_ui_compatible(tmp_path):
    clean_dir = tmp_path / "clean"
    music_dir = tmp_path / "music"
    environment_dir = tmp_path / "environment"
    missing_dir = tmp_path / "missing"
    _make_media(clean_dir, ["clean-c.wav", "clean-a.wav", "clean-b.wav"])
    _make_media(music_dir, ["song-c.mp3", "song-a.mp3", "song-b.mp3"])
    _make_media(environment_dir, ["env-c.wav", "env-a.wav", "env-b.wav"])
    sources = {
        "speech_clean": [clean_dir],
        "music_with_vocals": [music_dir],
        "environment_only": [environment_dir, missing_dir],
    }

    first_output = tmp_path / "first.csv"
    second_output = tmp_path / "second.csv"
    build_router_labeling_pool(
        first_output,
        max_per_label=2,
        seed=17,
        sources=sources,
    )
    build_router_labeling_pool(
        second_output,
        max_per_label=2,
        seed=17,
        sources=sources,
    )

    fieldnames, rows = _read_rows(first_output)
    _, repeated_rows = _read_rows(second_output)
    assert fieldnames == OUTPUT_COLUMNS
    assert rows == repeated_rows
    assert len(rows) == 6
    for label in sources:
        assert sum(row["suggested_label"] == label for row in rows) == 2
    assert [row["suggested_label"] for row in rows] == [
        "speech_clean",
        "speech_clean",
        "music_with_vocals",
        "music_with_vocals",
        "environment_only",
        "environment_only",
    ]
    assert all(row["human_label"] == "" for row in rows)
    assert all(row["workflow_label"] == "" for row in rows)
    assert all(row["suggested_label"] for row in rows)
    assert all(Path(row["path"]).is_absolute() for row in rows)
    assert all(
        row["notes"]
        == f"source_suggested={row['suggested_label']}; verify_by_listening"
        for row in rows
    )


def test_pool_deduplicates_resolved_paths(tmp_path):
    media_dir = tmp_path / "media"
    _make_media(media_dir, ["shared.wav"])
    output_path = tmp_path / "pool.csv"

    build_router_labeling_pool(
        output_path,
        sources={
            "speech_clean": [media_dir],
            "environment_only": [media_dir],
        },
    )

    _, rows = _read_rows(output_path)
    assert len(rows) == 1
    assert rows[0]["suggested_label"] == "speech_clean"


def test_missing_sources_are_skipped_but_empty_pool_fails(tmp_path):
    with pytest.raises(ValueError, match="No media files"):
        build_router_labeling_pool(
            tmp_path / "pool.csv",
            sources={
                "speech_clean": [tmp_path / "missing-clean"],
                "music_with_vocals": [tmp_path / "missing-music"],
            },
        )


def test_max_per_label_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="greater than zero"):
        build_router_labeling_pool(
            tmp_path / "pool.csv",
            max_per_label=0,
            sources={},
        )
