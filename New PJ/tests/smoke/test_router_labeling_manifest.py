from __future__ import annotations

import csv

import pytest
from scripts import create_router_labeling_manifest as labeling


def _fake_feature_rows(candidates):
    return {
        labeling.stable_sample_id(path): {
            "sample_id": labeling.stable_sample_id(path),
            "status": "success",
            "duration_sec": "1.25",
            "rms_energy": "0.10",
            "zero_crossing_rate": "0.20",
            "spectral_centroid_hz": "1200.0",
            "spectral_bandwidth_hz": "800.0",
        }
        for path in candidates
    }


def test_label_constants_are_exposed():
    assert labeling.ALLOWED_HUMAN_LABELS == {
        "speech_clean",
        "speech_noisy_general",
        "speech_target_noise",
        "music_with_vocals",
        "environment_only",
        "unknown_mixed",
    }
    assert labeling.ALLOWED_WORKFLOW_LABELS == {
        "speech_cleanup",
        "music_separation_package",
        "no_process",
        "target_noise_guard",
        "safe_abstain",
    }


def test_stable_sample_id_uses_normalized_path(tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.touch()

    assert labeling.stable_sample_id(audio_path) == labeling.stable_sample_id(
        tmp_path / "." / "clip.wav"
    )
    assert labeling.stable_sample_id(audio_path).startswith("router_")


def test_create_manifest_writes_columns_and_deduplicates_existing_paths(
    tmp_path,
    monkeypatch,
):
    media_dir = tmp_path / "media"
    nested_dir = media_dir / "nested"
    nested_dir.mkdir(parents=True)
    first_audio = media_dir / "first.wav"
    second_audio = nested_dir / "second.mp3"
    ignored_file = media_dir / "readme.txt"
    first_audio.touch()
    second_audio.touch()
    ignored_file.touch()

    existing_manifest = tmp_path / "existing.csv"
    with existing_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "source",
                "suggested_label",
                "human_label",
                "workflow_label",
                "split",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "path": str(first_audio),
                "source": "existing_dataset",
                "suggested_label": "speech_noisy_general",
                "human_label": "speech_noisy_general",
                "workflow_label": "speech_cleanup",
                "split": "train",
                "notes": "keep this label",
            }
        )

    monkeypatch.setattr(labeling, "_extract_feature_rows", _fake_feature_rows)
    output_path = tmp_path / "router_labels.csv"
    labeling.create_router_labeling_manifest(
        [media_dir],
        output_path,
        existing_manifests=[existing_manifest],
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == labeling.OUTPUT_COLUMNS
    assert len(rows) == 2
    assert len({row["sample_id"] for row in rows}) == 2
    assert len({row["path"] for row in rows}) == 2
    first_row = next(row for row in rows if row["path"] == str(first_audio.resolve()))
    assert first_row["sample_id"] == labeling.stable_sample_id(first_audio)
    assert first_row["source"] == "existing_dataset"
    assert first_row["suggested_label"] == "speech_noisy_general"
    assert first_row["human_label"] == "speech_noisy_general"
    assert first_row["workflow_label"] == "speech_cleanup"
    assert first_row["split"] == "train"
    assert first_row["notes"] == "keep this label"
    second_row = next(row for row in rows if row["path"] == str(second_audio.resolve()))
    assert second_row["human_label"] == ""
    assert second_row["workflow_label"] == ""


def test_unanalyzable_files_are_skipped(tmp_path, monkeypatch):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    good_path = media_dir / "good.wav"
    bad_path = media_dir / "bad.wav"
    good_path.touch()
    bad_path.touch()

    def feature_rows(candidates):
        return {
            labeling.stable_sample_id(good_path): {
                "sample_id": labeling.stable_sample_id(good_path),
                "status": "success",
                "duration_sec": "0.5",
                "rms_energy": "0.1",
                "zero_crossing_rate": "0.2",
                "spectral_centroid_hz": "1000",
                "spectral_bandwidth_hz": "500",
            },
            labeling.stable_sample_id(bad_path): {
                "sample_id": labeling.stable_sample_id(bad_path),
                "status": "failed",
                "error": "unsupported media",
            },
        }

    monkeypatch.setattr(labeling, "_extract_feature_rows", feature_rows)
    output_path = tmp_path / "labels.csv"
    labeling.create_router_labeling_manifest([media_dir], output_path)

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["path"] for row in rows] == [str(good_path.resolve())]


def test_feature_manifest_uses_audio_and_video_input_types(tmp_path, monkeypatch):
    audio_path = tmp_path / "clip.wav"
    video_path = tmp_path / "clip.mp4"
    audio_path.touch()
    video_path.touch()
    captured_rows = []

    def fake_extract_audio_features(manifest_path, output_path, **_kwargs):
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            captured_rows.extend(csv.DictReader(handle))
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "status"])
            writer.writeheader()
            for row in captured_rows:
                writer.writerow(
                    {"sample_id": row["sample_id"], "status": "success"}
                )

    monkeypatch.setattr(
        labeling,
        "extract_audio_features",
        fake_extract_audio_features,
    )
    candidates = {audio_path: {}, video_path: {}}

    labeling._extract_feature_rows(candidates)

    input_types = {row["input_path"]: row["input_type"] for row in captured_rows}
    assert input_types[str(audio_path)] == "audio"
    assert input_types[str(video_path)] == "video"


def test_existing_manifest_can_be_used_without_input_directory(
    tmp_path,
    monkeypatch,
):
    audio_path = tmp_path / "existing.wav"
    audio_path.touch()
    existing_manifest = tmp_path / "existing.csv"
    with existing_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path"])
        writer.writeheader()
        writer.writerow({"path": str(audio_path)})

    monkeypatch.setattr(labeling, "_extract_feature_rows", _fake_feature_rows)
    output_path = tmp_path / "labels.csv"
    labeling.create_router_labeling_manifest(
        [],
        output_path,
        existing_manifests=[existing_manifest],
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["path"] for row in rows] == [str(audio_path.resolve())]


def test_manifest_requires_at_least_one_input_source(tmp_path):
    with pytest.raises(ValueError, match="input directory or existing manifest"):
        labeling.create_router_labeling_manifest([], tmp_path / "labels.csv")
