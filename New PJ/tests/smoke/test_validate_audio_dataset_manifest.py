"""Smoke tests for audio dataset manifest validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

from scripts.validate_audio_dataset_manifest import AUDIT_COLUMNS, validate_audio_dataset_manifest


def _write_noise_manifest(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["noise_id", "path", "noise_label", "source", "notes"])
        writer.writeheader()
        writer.writerow(
            {
                "noise_id": "dog_001",
                "path": "audio/dog.wav",
                "noise_label": "dog_bark",
                "source": "tmp",
                "notes": "valid",
            }
        )
        writer.writerow(
            {
                "noise_id": "dog_dup",
                "path": "audio/dog.wav",
                "noise_label": "dog_bark",
                "source": "tmp",
                "notes": "duplicate path",
            }
        )
        writer.writerow(
            {
                "noise_id": "rain_001",
                "path": "audio/rain.wav",
                "noise_label": "rain",
                "source": "tmp",
                "notes": "bad label",
            }
        )
        writer.writerow(
            {
                "noise_id": "missing_001",
                "path": "audio/missing.wav",
                "noise_label": "car_horn",
                "source": "tmp",
                "notes": "missing file",
            }
        )


def _read_audit(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _mock_metadata(path: Path) -> dict[str, str]:
    if path.name == "rain.wav":
        return {"duration_sec": "0.100000", "sample_rate": "44100", "channels": "1"}
    return {"duration_sec": "1.500000", "sample_rate": "44100", "channels": "1"}


def test_validate_manifest_writes_audit_and_summary(tmp_path: Path) -> None:
    manifest = tmp_path / "noise.csv"
    audio_dir = tmp_path / "audio"
    output_root = tmp_path / "audit"
    audio_dir.mkdir()
    (audio_dir / "dog.wav").write_bytes(b"dog")
    (audio_dir / "rain.wav").write_bytes(b"rain")
    _write_noise_manifest(manifest)

    with patch("scripts.validate_audio_dataset_manifest._probe_audio_metadata", side_effect=_mock_metadata):
        audit_path, summary_path = validate_audio_dataset_manifest(
            manifest_path=manifest,
            path_column="path",
            label_column="noise_label",
            allowed_labels=["dog_bark", "car_horn"],
            output_root=output_root,
            min_duration_sec=0.25,
            max_duration_sec=60.0,
        )

    rows = _read_audit(audit_path)
    assert audit_path == (output_root / "manifest_audit.csv").resolve()
    assert summary_path == (output_root / "manifest_audit_summary.json").resolve()
    assert list(rows[0].keys()) == AUDIT_COLUMNS
    assert rows[0]["status"] == "valid"
    assert rows[0]["duration_sec"] == "1.500000"
    assert rows[0]["sample_rate"] == "44100"
    assert rows[0]["channels"] == "1"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_rows"] == 4
    assert summary["valid_rows"] == 1
    assert summary["invalid_rows"] == 3
    assert summary["duplicate_path_count"] == 1
    assert summary["counts_by_label"] == {"car_horn": 1, "dog_bark": 2, "rain": 1}
    assert summary["counts_by_status"] == {"invalid": 3, "valid": 1}


def test_validate_manifest_flags_duplicates_bad_labels_and_missing_files(tmp_path: Path) -> None:
    manifest = tmp_path / "noise.csv"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "dog.wav").write_bytes(b"dog")
    (audio_dir / "rain.wav").write_bytes(b"rain")
    _write_noise_manifest(manifest)

    with patch("scripts.validate_audio_dataset_manifest._probe_audio_metadata", side_effect=_mock_metadata):
        audit_path, _ = validate_audio_dataset_manifest(
            manifest_path=manifest,
            path_column="path",
            label_column="noise_label",
            allowed_labels=["dog_bark", "car_horn"],
            output_root=tmp_path / "audit",
        )

    rows = _read_audit(audit_path)
    duplicate_row = rows[1]
    bad_label_row = rows[2]
    missing_row = rows[3]
    assert duplicate_row["status"] == "invalid"
    assert "duplicate path" in duplicate_row["error"]
    assert bad_label_row["status"] == "invalid"
    assert "label not allowed: rain" in bad_label_row["error"]
    assert "duration below minimum" in bad_label_row["error"]
    assert missing_row["status"] == "invalid"
    assert missing_row["exists"] == "false"
    assert "file does not exist" in missing_row["error"]


def test_validate_manifest_raises_for_missing_path_column(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.csv"
    manifest.write_text("sample_id,file\none,a.wav\n", encoding="utf-8")

    try:
        validate_audio_dataset_manifest(
            manifest_path=manifest,
            path_column="path",
            output_root=tmp_path / "audit",
        )
    except ValueError as exc:
        assert "missing required path column" in str(exc)
    else:
        raise AssertionError("Expected missing path column to raise ValueError.")


def test_validate_manifest_raises_for_empty_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "empty.csv"
    manifest.write_text("noise_id,path,noise_label,source,notes\n", encoding="utf-8")

    try:
        validate_audio_dataset_manifest(
            manifest_path=manifest,
            path_column="path",
            output_root=tmp_path / "audit",
        )
    except ValueError as exc:
        assert "Manifest is empty" in str(exc)
    else:
        raise AssertionError("Expected empty manifest to raise ValueError.")


def test_validate_manifest_raises_for_missing_label_column(tmp_path: Path) -> None:
    manifest = tmp_path / "missing_label.csv"
    manifest.write_text("noise_id,path,source,notes\none,a.wav,tmp,no label\n", encoding="utf-8")

    try:
        validate_audio_dataset_manifest(
            manifest_path=manifest,
            path_column="path",
            label_column="noise_label",
            output_root=tmp_path / "audit",
        )
    except ValueError as exc:
        assert "missing label column" in str(exc)
    else:
        raise AssertionError("Expected missing label column to raise ValueError.")


def test_validate_manifest_empty_path_is_invalid_without_metadata_probe(tmp_path: Path) -> None:
    manifest = tmp_path / "empty_path.csv"
    manifest.write_text(
        "noise_id,path,noise_label,source,notes\nempty,,dog_bark,tmp,empty path\n",
        encoding="utf-8",
    )

    with patch("scripts.validate_audio_dataset_manifest._probe_audio_metadata") as probe_mock:
        audit_path, summary_path = validate_audio_dataset_manifest(
            manifest_path=manifest,
            path_column="path",
            label_column="noise_label",
            allowed_labels=["dog_bark"],
            output_root=tmp_path / "audit",
        )

    probe_mock.assert_not_called()
    rows = _read_audit(audit_path)
    assert rows[0]["status"] == "invalid"
    assert rows[0]["exists"] == "false"
    assert "empty path" in rows[0]["error"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["invalid_rows"] == 1
