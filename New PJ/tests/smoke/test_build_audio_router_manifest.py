"""Smoke tests for the audio router manifest builder."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.build_audio_router_manifest import (
    ENVIRONMENT_NOISE,
    MUSIC,
    OUTPUT_COLUMNS,
    SPEECH_NOISE,
    build_audio_router_manifest,
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_target_noise_manifest(path: Path, count: int = 2) -> None:
    mixed_dir = path.parent / "mixed"
    mixed_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["sample_id", "mixed_path", "target_path", "split", "noise_label", "snr_db"])
        writer.writeheader()
        for index in range(count):
            mixed_path = mixed_dir / f"speech_{index}.wav"
            mixed_path.write_bytes(b"placeholder")
            writer.writerow(
                {
                    "sample_id": f"speech_{index}",
                    "mixed_path": f"mixed/{mixed_path.name}",
                    "target_path": f"clean_{index}.wav",
                    "split": "test" if index == 0 else "train",
                    "noise_label": "dog_bark",
                    "snr_db": "5",
                }
            )


def _create_musdb(root: Path, count: int = 2) -> None:
    for index in range(count):
        split = "test" if index == 0 else "train"
        path = root / split / f"Track {index}.stem.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")


def _create_voicebank(root: Path) -> None:
    for folder_name, filenames in {
        "clean_trainset_28spk_wav": ["clean_train_0.wav", "clean_train_1.wav"],
        "clean_testset_wav": ["clean_test_0.wav"],
        "noisy_trainset_28spk_wav": ["noisy_train_0.wav", "noisy_train_1.wav"],
        "noisy_testset_wav": ["noisy_test_0.wav"],
    }.items():
        folder = root / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            (folder / filename).write_bytes(b"placeholder")


def _create_esc50(root: Path, count: int = 2) -> None:
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    with (root / "meta" / "esc50.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["filename", "fold", "category"])
        writer.writeheader()
        for index in range(count):
            filename = f"esc_{index}.wav"
            (root / "audio" / filename).write_bytes(b"placeholder")
            writer.writerow({"filename": filename, "fold": "5" if index == 0 else "1", "category": "dog"})


def _create_urbansound(root: Path, count: int = 2) -> tuple[Path, Path]:
    metadata = root / "metadata" / "UrbanSound8K.csv"
    audio_root = root / "audio"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    with metadata.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["slice_file_name", "fold", "class"])
        writer.writeheader()
        for index in range(count):
            fold = "10" if index == 0 else "1"
            filename = f"urban_{index}.wav"
            path = audio_root / f"fold{fold}" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"placeholder")
            writer.writerow({"slice_file_name": filename, "fold": fold, "class": "siren"})
    return metadata, audio_root


def test_build_audio_router_manifest_from_all_sources(tmp_path: Path) -> None:
    output = tmp_path / "data" / "manifests" / "audio_router.local.csv"
    target_manifest = tmp_path / "outputs" / "target-noise-v1" / "manifests" / "target_noise_suppression.csv"
    musdb_root = tmp_path / "data" / "external" / "musdb18-preview"
    voicebank_root = tmp_path / "data" / "external" / "voicebank"
    esc50_root = tmp_path / "data" / "external" / "ESC-50-master"
    urbansound_metadata, urbansound_audio = _create_urbansound(tmp_path / "data" / "external" / "UrbanSound8K")
    _write_target_noise_manifest(target_manifest)
    _create_musdb(musdb_root)
    _create_voicebank(voicebank_root)
    _create_esc50(esc50_root)

    result = build_audio_router_manifest(
        target_noise_manifest=target_manifest,
        voicebank_root=voicebank_root,
        musdb_root=musdb_root,
        esc50_root=esc50_root,
        urbansound8k_metadata=urbansound_metadata,
        urbansound8k_audio_root=urbansound_audio,
        output=output,
        max_per_label=10,
        seed=7,
    )

    rows = _read_rows(result)
    assert result == output.resolve()
    assert list(rows[0].keys()) == OUTPUT_COLUMNS
    assert {row["router_label"] for row in rows} == {SPEECH_NOISE, MUSIC, ENVIRONMENT_NOISE}
    assert {row["source"] for row in rows} == {
        "target_noise_v1",
        "voicebank_clean",
        "voicebank_noisy",
        "musdb18_preview",
        "esc50",
        "urbansound8k",
    }
    assert all(not Path(row["input_path"]).is_absolute() for row in rows)


def test_max_per_label_is_respected(tmp_path: Path) -> None:
    output = tmp_path / "data" / "manifests" / "audio_router.local.csv"
    target_manifest = tmp_path / "target" / "target_noise_suppression.csv"
    musdb_root = tmp_path / "musdb"
    esc50_root = tmp_path / "esc50"
    _write_target_noise_manifest(target_manifest, count=4)
    _create_musdb(musdb_root, count=4)
    _create_esc50(esc50_root, count=4)

    build_audio_router_manifest(
        target_noise_manifest=target_manifest,
        musdb_root=musdb_root,
        esc50_root=esc50_root,
        output=output,
        max_per_label=1,
        seed=99,
    )

    counts = {label: 0 for label in (SPEECH_NOISE, MUSIC, ENVIRONMENT_NOISE)}
    for row in _read_rows(output):
        counts[row["router_label"]] += 1
    assert counts == {SPEECH_NOISE: 1, MUSIC: 1, ENVIRONMENT_NOISE: 1}


def test_build_audio_router_manifest_works_with_one_source(tmp_path: Path) -> None:
    output = tmp_path / "data" / "manifests" / "audio_router.local.csv"
    musdb_root = tmp_path / "musdb"
    _create_musdb(musdb_root, count=1)

    build_audio_router_manifest(musdb_root=musdb_root, output=output)

    rows = _read_rows(output)
    assert len(rows) == 1
    assert rows[0]["router_label"] == MUSIC
    assert rows[0]["source"] == "musdb18_preview"


def test_voicebank_root_adds_speech_noise_rows(tmp_path: Path) -> None:
    output = tmp_path / "data" / "manifests" / "audio_router.local.csv"
    voicebank_root = tmp_path / "data" / "external" / "voicebank"
    _create_voicebank(voicebank_root)

    build_audio_router_manifest(voicebank_root=voicebank_root, output=output, max_per_label=20)

    rows = _read_rows(output)
    assert {row["router_label"] for row in rows} == {SPEECH_NOISE}
    assert {row["source"] for row in rows} == {"voicebank_clean", "voicebank_noisy"}
    assert {row["split"] for row in rows} == {"train", "test"}
    assert all(row["notes"].startswith("folder=") for row in rows)


def test_speech_noise_sampling_is_balanced_by_source(tmp_path: Path) -> None:
    output = tmp_path / "data" / "manifests" / "audio_router.local.csv"
    target_manifest = tmp_path / "target" / "target_noise_suppression.csv"
    voicebank_root = tmp_path / "voicebank"
    _write_target_noise_manifest(target_manifest, count=10)
    _create_voicebank(voicebank_root)

    build_audio_router_manifest(
        target_noise_manifest=target_manifest,
        voicebank_root=voicebank_root,
        output=output,
        max_per_label=3,
        seed=5,
    )

    speech_sources = [row["source"] for row in _read_rows(output) if row["router_label"] == SPEECH_NOISE]
    assert len(speech_sources) == 3
    assert set(speech_sources) == {"target_noise_v1", "voicebank_clean", "voicebank_noisy"}


def test_target_noise_project_relative_path_resolves_from_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    manifest = repo_root / "outputs" / "target-noise-v1" / "manifests" / "target_noise_suppression.csv"
    mixed = repo_root / "outputs" / "target-noise-v1" / "mixed" / "test" / "sample.wav"
    output = repo_root / "data" / "manifests" / "audio_router.local.csv"
    mixed.parent.mkdir(parents=True, exist_ok=True)
    mixed.write_bytes(b"placeholder")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["sample_id", "mixed_path", "split"])
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample",
                "mixed_path": "outputs/target-noise-v1/mixed/test/sample.wav",
                "split": "test",
            }
        )

    monkeypatch.chdir(repo_root)
    build_audio_router_manifest(target_noise_manifest=manifest, output=output)

    rows = _read_rows(output)
    assert rows[0]["input_path"] == "../../outputs/target-noise-v1/mixed/test/sample.wav"
    assert "manifests/outputs" not in rows[0]["input_path"]
    assert (output.parent / rows[0]["input_path"]).resolve() == mixed.resolve()


def test_target_noise_source_name_can_be_configured(tmp_path: Path) -> None:
    output = tmp_path / "data" / "manifests" / "audio_router.local.csv"
    target_manifest = tmp_path / "outputs" / "target-noise-v2" / "manifests" / "target_noise_suppression.csv"
    _write_target_noise_manifest(target_manifest, count=3)

    build_audio_router_manifest(
        target_noise_manifest=target_manifest,
        target_noise_source="target_noise_v2_source_disjoint",
        output=output,
        max_per_label=10,
        seed=3,
    )

    rows = _read_rows(output)
    assert rows
    assert {row["router_label"] for row in rows} == {SPEECH_NOISE}
    assert {row["source"] for row in rows} == {"target_noise_v2_source_disjoint"}


def test_build_audio_router_manifest_raises_when_no_rows_are_produced(tmp_path: Path) -> None:
    try:
        build_audio_router_manifest(output=tmp_path / "audio_router.csv")
    except ValueError as exc:
        assert "No audio router rows" in str(exc)
    else:
        raise AssertionError("Expected empty source set to raise ValueError.")


def test_urbansound8k_metadata_requires_audio_root(tmp_path: Path) -> None:
    metadata, _audio_root = _create_urbansound(tmp_path / "UrbanSound8K")

    try:
        build_audio_router_manifest(
            urbansound8k_metadata=metadata,
            output=tmp_path / "audio_router.csv",
        )
    except ValueError as exc:
        assert "--urbansound8k-audio-root" in str(exc)
    else:
        raise AssertionError("Expected missing UrbanSound8K audio root to raise ValueError.")
