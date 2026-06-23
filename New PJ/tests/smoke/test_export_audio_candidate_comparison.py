"""Smoke tests for the human-labeled audio candidate comparison exporter."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.export_audio_candidate_comparison import (
    MANIFEST_COLUMNS,
    export_audio_candidate_comparison,
)


def _write_manifest(path: Path, rows: list[dict[str, str]], columns: list[str] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns or MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_candidate_comparison_validates_missing_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [], columns=["case_id", "candidate_path"])

    with pytest.raises(ValueError, match="missing columns"):
        export_audio_candidate_comparison(manifest_path=manifest, output_dir=tmp_path / "output")


def test_candidate_comparison_writes_csv_and_grouped_markdown(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    deepfilter_path = tmp_path / "deepfilter.wav"
    target_path = tmp_path / "target.wav"
    for path in (input_path, deepfilter_path, target_path):
        path.write_bytes(b"audio")

    manifest = tmp_path / "manifest.csv"
    rows = [
        {
            "case_id": "siren-case",
            "input_path": input_path.name,
            "candidate_name": "DeepFilterNet",
            "candidate_path": deepfilter_path.name,
            "candidate_role": "deepfilternet",
            "human_rating": "4",
            "human_notes": "Speech remains understandable.",
            "is_selected": "yes",
            "reject_reason": "",
        },
        {
            "case_id": "siren-case",
            "input_path": input_path.name,
            "candidate_name": "Target baseline",
            "candidate_path": target_path.name,
            "candidate_role": "target_suppressor_baseline",
            "human_rating": "1",
            "human_notes": "Siren remains and speech is degraded.",
            "is_selected": "no",
            "reject_reason": "speech damage",
        },
    ]
    _write_manifest(manifest, rows)

    output_dir = export_audio_candidate_comparison(
        manifest_path=manifest,
        output_dir=tmp_path / "comparison",
    )

    with (output_dir / "comparison.csv").open(newline="", encoding="utf-8") as csv_file:
        written_rows = list(csv.DictReader(csv_file))
    assert written_rows == rows

    markdown = (output_dir / "comparison.md").read_text(encoding="utf-8")
    assert "## siren-case" in markdown
    assert "| DeepFilterNet | deepfilternet | 4 | yes | - | Speech remains understandable. |" in markdown
    assert "| Target baseline | target_suppressor_baseline | 1 | no | speech damage |" in markdown
    assert "no automated quality metric is computed" in markdown


def test_candidate_comparison_rejects_missing_candidate_file(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "case_id": "missing-output",
                "input_path": str(input_path),
                "candidate_name": "Missing",
                "candidate_path": "missing.wav",
                "candidate_role": "target_suppressor_v2",
                "human_rating": "",
                "human_notes": "",
                "is_selected": "",
                "reject_reason": "",
            }
        ],
    )

    with pytest.raises(FileNotFoundError, match="missing candidate_path"):
        export_audio_candidate_comparison(manifest_path=manifest, output_dir=tmp_path / "output")
