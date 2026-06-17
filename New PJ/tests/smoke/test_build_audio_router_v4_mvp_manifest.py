"""Smoke tests for the Audio Router V4 MVP manifest filter."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_audio_router_v4_mvp_manifest import (
    DEFAULT_NOTES_SUFFIX,
    build_audio_router_v4_mvp_manifest,
    main,
    validate_train_test_coverage,
)


FIELDNAMES = [
    "sample_id",
    "input_path",
    "source_dataset",
    "split",
    "content_label",
    "route_target",
    "notes",
]


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _row(sample_id: str, label: str, split: str, notes: str = "source=test") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "input_path": f"audio/{sample_id}.wav",
        "source_dataset": "fixture",
        "split": split,
        "content_label": label,
        "route_target": "test_route",
        "notes": notes,
    }


def _valid_rows() -> list[dict[str, str]]:
    return [
        _row("speech_clean_train", "speech_clean", "train"),
        _row("speech_clean_test", "speech_clean", "test"),
        _row("target_train", "speech_target_noise", "train"),
        _row("target_test", "speech_target_noise", "test"),
        _row("music_train", "music_with_vocals", "train"),
        _row("music_test", "music_with_vocals", "test"),
        _row("env_train", "environment_only", "train"),
        _row("env_test", "environment_only", "test"),
        _row("noisy_test", "speech_noisy_general", "test"),
    ]


def test_filters_speech_noisy_general_and_preserves_columns(tmp_path: Path) -> None:
    input_manifest = tmp_path / "audio_router_v4.csv"
    output_manifest = tmp_path / "audio_router_v4_mvp.csv"
    _write_manifest(input_manifest, _valid_rows())

    build_audio_router_v4_mvp_manifest(input_manifest=input_manifest, output_manifest=output_manifest)

    rows = _read_rows(output_manifest)
    assert rows
    assert list(rows[0].keys()) == FIELDNAMES
    assert "speech_noisy_general" not in {row["content_label"] for row in rows}
    assert [row["sample_id"] for row in rows] == [
        "speech_clean_train",
        "speech_clean_test",
        "target_train",
        "target_test",
        "music_train",
        "music_test",
        "env_train",
        "env_test",
    ]


def test_appends_notes_suffix_without_duplicate_separators(tmp_path: Path) -> None:
    input_manifest = tmp_path / "audio_router_v4.csv"
    output_manifest = tmp_path / "audio_router_v4_mvp.csv"
    rows = _valid_rows()
    rows[0]["notes"] = "source=test; " + DEFAULT_NOTES_SUFFIX
    rows[1]["notes"] = ""
    _write_manifest(input_manifest, rows)

    build_audio_router_v4_mvp_manifest(input_manifest=input_manifest, output_manifest=output_manifest)

    output_rows = _read_rows(output_manifest)
    assert output_rows[0]["notes"] == f"source=test; {DEFAULT_NOTES_SUFFIX}"
    assert output_rows[1]["notes"] == DEFAULT_NOTES_SUFFIX
    assert ";;" not in output_rows[0]["notes"]


def test_validates_remaining_labels_have_train_and_test() -> None:
    errors = validate_train_test_coverage(
        [
            _row("music_train", "music_with_vocals", "train"),
            _row("music_test", "music_with_vocals", "test"),
            _row("env_test", "environment_only", "test"),
        ]
    )

    assert errors == ["content_label environment_only lacks train/test coverage; observed splits: test"]


def test_raises_when_remaining_label_has_only_test(tmp_path: Path) -> None:
    input_manifest = tmp_path / "audio_router_v4.csv"
    output_manifest = tmp_path / "audio_router_v4_mvp.csv"
    _write_manifest(
        input_manifest,
        [
            _row("music_train", "music_with_vocals", "train"),
            _row("music_test", "music_with_vocals", "test"),
            _row("env_test", "environment_only", "test"),
            _row("noisy_test", "speech_noisy_general", "test"),
        ],
    )

    try:
        build_audio_router_v4_mvp_manifest(input_manifest=input_manifest, output_manifest=output_manifest)
    except ValueError as exc:
        assert "environment_only" in str(exc)
        assert "observed splits: test" in str(exc)
    else:
        raise AssertionError("Expected missing train split to fail.")


def test_raises_when_remaining_label_has_only_train(tmp_path: Path) -> None:
    input_manifest = tmp_path / "audio_router_v4.csv"
    output_manifest = tmp_path / "audio_router_v4_mvp.csv"
    _write_manifest(
        input_manifest,
        [
            _row("music_train", "music_with_vocals", "train"),
            _row("music_test", "music_with_vocals", "test"),
            _row("env_train", "environment_only", "train"),
            _row("noisy_test", "speech_noisy_general", "test"),
        ],
    )

    try:
        build_audio_router_v4_mvp_manifest(input_manifest=input_manifest, output_manifest=output_manifest)
    except ValueError as exc:
        assert "environment_only" in str(exc)
        assert "observed splits: train" in str(exc)
    else:
        raise AssertionError("Expected missing test split to fail.")


def test_cli_writes_output_manifest(tmp_path: Path) -> None:
    input_manifest = tmp_path / "audio_router_v4.csv"
    output_manifest = tmp_path / "nested" / "audio_router_v4_mvp.csv"
    _write_manifest(input_manifest, _valid_rows())

    exit_code = main(
        [
            "--input-manifest",
            str(input_manifest),
            "--output-manifest",
            str(output_manifest),
            "--exclude-label",
            "speech_noisy_general",
        ]
    )

    assert exit_code == 0
    assert output_manifest.exists()
    assert "speech_noisy_general" not in {row["content_label"] for row in _read_rows(output_manifest)}
