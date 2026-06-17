"""Smoke tests for the Audio Router V4 manifest builder."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_audio_router_v4_manifest import (
    CONTENT_LABELS,
    ENGINE_TARGETS,
    MANIFEST_V4_COLUMNS,
    ROUTE_TARGETS,
    SUPPORTED_TARGET_NOISE_LABELS,
    build_audio_router_v4_manifest,
    collapse_content_label,
    convert_v3_row_to_v4,
    encode_candidate_tasks,
    main,
    parse_target_noise_filename,
    route_mapping_for_content_label,
    validate_manifest_rows,
    validate_no_split_leakage,
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _v4_row(**overrides: str) -> dict[str, str]:
    row = {column: "" for column in MANIFEST_V4_COLUMNS}
    row.update(
        {
            "sample_id": "sample",
            "input_path": "sample.wav",
            "source_dataset": "test",
            "split": "train",
            "split_key": "sample",
            "group_id": "group",
            "taxonomy_version": "audio-router-v4",
            "normalization_version": "pending",
            "content_label": "speech_noisy_general",
            "route_target": "clean_voice",
            "engine_target": "deepfilternet",
            "candidate_tasks": "clean_voice",
            "label_method": "test",
            "label_confidence": "1.0",
            "contains_speech": "true",
            "contains_music": "false",
            "contains_vocals": "false",
            "contains_environment_noise": "true",
            "contains_target_noise": "false",
            "is_synthetic": "false",
            "features_version": "pending",
            "feature_status": "pending",
        }
    )
    row.update(overrides)
    return row


def test_constants_and_enums_are_defined() -> None:
    assert CONTENT_LABELS == [
        "speech_clean",
        "speech_noisy_general",
        "speech_target_noise",
        "music_with_vocals",
        "music_instrumental",
        "environment_only",
        "unknown_mixed",
    ]
    assert "out_of_scope" in ROUTE_TARGETS
    assert ENGINE_TARGETS == ["deepfilternet", "demucs", "target_noise_suppressor", "none"]
    assert SUPPORTED_TARGET_NOISE_LABELS == ["dog_bark", "car_horn", "siren"]


def test_candidate_tasks_are_semicolon_encoded() -> None:
    assert encode_candidate_tasks(["extract_vocals", "remove_vocals"]) == "extract_vocals;remove_vocals"
    assert encode_candidate_tasks([]) == ""


def test_route_mapping_for_all_content_labels() -> None:
    expected = {
        "speech_clean": ("no_process", "none", "clean_voice", "", ""),
        "speech_noisy_general": ("clean_voice", "deepfilternet", "clean_voice", "", ""),
        "speech_target_noise": (
            "target_noise_suppression",
            "target_noise_suppressor",
            "target_noise_suppression",
            "clean_voice",
            "deepfilternet",
        ),
        "music_with_vocals": ("manual_required", "demucs", "extract_vocals;remove_vocals", "", ""),
        "music_instrumental": ("no_process", "none", "extract_vocals;remove_vocals", "", ""),
        "environment_only": ("out_of_scope", "none", "", "", ""),
        "unknown_mixed": ("abstain", "none", "", "manual_required", ""),
    }
    for content_label, values in expected.items():
        mapping = route_mapping_for_content_label(content_label)
        assert (
            mapping["route_target"],
            mapping["engine_target"],
            mapping["candidate_tasks"],
            mapping["fallback_route_target"],
            mapping["fallback_engine_target"],
        ) == values
        assert mapping["fallback_route_target"] != "none"


def test_collapse_content_label_representative_cases() -> None:
    assert collapse_content_label({"confidence": "0.1", "contains_speech": "true"}) == "unknown_mixed"
    assert (
        collapse_content_label(
            {
                "contains_speech": "true",
                "contains_target_noise": "true",
                "target_noise_label": "dog_bark",
            }
        )
        == "speech_target_noise"
    )
    assert collapse_content_label({"contains_speech": "true", "is_noisy": "true"}) == "speech_noisy_general"
    assert collapse_content_label({"contains_speech": "true"}) == "speech_clean"
    assert collapse_content_label({"contains_music": "true", "contains_vocals": "true"}) == "music_with_vocals"
    assert collapse_content_label({"contains_music": "true", "contains_vocals": "false"}) == "music_instrumental"
    assert collapse_content_label({"contains_environment_noise": "true"}) == "environment_only"
    assert collapse_content_label({"contains_speech": "true", "contains_music": "true"}) == "unknown_mixed"


def test_v3_style_row_conversion_for_target_noise() -> None:
    row = convert_v3_row_to_v4(
        {
            "sample_id": "tn_001",
            "input_path": "mixed/tn_001.wav",
            "router_label": "speech_noise",
            "source": "target_noise_v1",
            "split": "test",
            "notes": "noise_label=car_horn; snr_db=5",
        },
        source_dataset="fallback",
        taxonomy_version="audio-router-v4",
        features_version="pending",
        normalization_version="pending",
    )

    assert list(row.keys()) == MANIFEST_V4_COLUMNS
    assert row["content_label"] == "speech_target_noise"
    assert row["route_target"] == "target_noise_suppression"
    assert row["engine_target"] == "target_noise_suppressor"
    assert row["candidate_tasks"] == "target_noise_suppression"
    assert row["fallback_route_target"] == "clean_voice"
    assert row["fallback_engine_target"] == "deepfilternet"
    assert row["target_noise_label"] == "car_horn"
    assert row["snr_db"] == "5"
    assert row["clean_source_id"] == "tn_001"
    assert row["noise_source_id"] == ""
    assert row["source_dataset"] == "target_noise_v1"
    assert row["is_synthetic"] == "true"


def test_v3_conversion_for_renamed_target_noise_source() -> None:
    row = convert_v3_row_to_v4(
        {
            "sample_id": "00544_p232_012_car_horn_71309-1-0-1_snr0",
            "input_path": "mixed/test/00544_p232_012_car_horn_71309-1-0-1_snr0.wav",
            "router_label": "speech_noise",
            "source": "target_noise_v2_source_disjoint",
            "split": "test",
            "notes": "snr_db=0",
        },
        source_dataset="fallback",
        taxonomy_version="audio-router-v4",
        features_version="pending",
        normalization_version="pending",
    )

    assert row["source_dataset"] == "target_noise_v2_source_disjoint"
    assert row["content_label"] == "speech_target_noise"
    assert row["route_target"] == "target_noise_suppression"
    assert row["engine_target"] == "target_noise_suppressor"
    assert row["candidate_tasks"] == "target_noise_suppression"
    assert row["target_noise_label"] == "car_horn"
    assert row["noise_source_id"] == "71309-1-0-1"
    assert row["clean_source_id"] == "00544_p232_012_car_horn_71309-1-0-1_snr0"
    assert row["is_synthetic"] == "true"


def test_v3_conversion_unknown_non_target_source_stays_unknown() -> None:
    row = convert_v3_row_to_v4(
        {
            "sample_id": "unknown_001",
            "input_path": "unknown/file.wav",
            "router_label": "speech_noise",
            "source": "new_dataset",
            "split": "test",
            "notes": "",
        },
        source_dataset="fallback",
        taxonomy_version="audio-router-v4",
        features_version="pending",
        normalization_version="pending",
    )

    assert row["source_dataset"] == "new_dataset"
    assert row["content_label"] == "unknown_mixed"
    assert row["route_target"] == "abstain"
    assert row["engine_target"] == "none"
    assert row["is_synthetic"] == "false"


def test_target_noise_filename_parser_extracts_label_and_noise_source_id() -> None:
    assert parse_target_noise_filename("00304_p232_009_siren_22601-8-0-37_snr5.wav") == (
        "siren",
        "22601-8-0-37",
    )
    assert parse_target_noise_filename("00193_p232_009_dog_bark_125791-3-0-9_snr-5.wav") == (
        "dog_bark",
        "125791-3-0-9",
    )
    assert parse_target_noise_filename("00544_p232_012_car_horn_71309-1-0-1_snr0.wav") == (
        "car_horn",
        "71309-1-0-1",
    )


def test_target_noise_conversion_parses_noise_source_id_from_filename() -> None:
    row = convert_v3_row_to_v4(
        {
            "sample_id": "00193_p232_009_dog_bark_125791-3-0-9_snr-5",
            "input_path": "mixed/test/00193_p232_009_dog_bark_125791-3-0-9_snr-5.wav",
            "router_label": "speech_noise",
            "source": "target_noise_v1",
            "split": "test",
            "notes": "snr_db=-5",
        },
        source_dataset="fallback",
        taxonomy_version="audio-router-v4",
        features_version="pending",
        normalization_version="pending",
    )

    assert row["target_noise_label"] == "dog_bark"
    assert row["noise_source_id"] == "125791-3-0-9"
    assert row["noise_source_id"] not in {"dog_bark", "car_horn", "siren"}
    assert row["clean_source_id"] == "00193_p232_009_dog_bark_125791-3-0-9_snr-5"


def test_v3_conversion_prefixes_split_key_and_group_id() -> None:
    row = convert_v3_row_to_v4(
        {
            "sample_id": "shared_id",
            "input_path": "voicebank/clean.wav",
            "router_label": "speech_noise",
            "source": "voicebank_clean",
            "split": "train",
            "notes": "",
        },
        source_dataset="fallback",
        taxonomy_version="audio-router-v4",
        features_version="pending",
        normalization_version="pending",
    )

    assert row["split_key"] == "voicebank_clean:shared_id"
    assert row["group_id"] == "voicebank_clean:shared_id"


def test_v3_conversion_preserves_source_fold_and_original_split_from_notes() -> None:
    row = convert_v3_row_to_v4(
        {
            "sample_id": "env_001",
            "input_path": "esc50/dog.wav",
            "router_label": "environment_noise",
            "source": "esc50",
            "split": "holdout",
            "notes": "category=dog; fold=1; original_split=official_train",
        },
        source_dataset="fallback",
        taxonomy_version="audio-router-v4",
        features_version="pending",
        normalization_version="pending",
    )

    assert row["split"] == "holdout"
    assert row["source_fold"] == "1"
    assert row["original_split"] == "official_train"


def test_validation_rejects_unsupported_target_noise_label() -> None:
    row = _v4_row(
        content_label="speech_target_noise",
        route_target="target_noise_suppression",
        engine_target="target_noise_suppressor",
        candidate_tasks="target_noise_suppression",
        contains_target_noise="true",
        target_noise_label="rain",
    )

    errors = validate_manifest_rows([row])

    assert any("unsupported target_noise_label" in error for error in errors)


def test_validation_requires_environment_only_out_of_scope() -> None:
    row = _v4_row(
        content_label="environment_only",
        route_target="no_process",
        engine_target="none",
        candidate_tasks="",
        contains_speech="false",
        contains_environment_noise="true",
    )

    errors = validate_manifest_rows([row])

    assert any("environment_only must map to out_of_scope" in error for error in errors)


def test_validation_rejects_none_fallback_route_target() -> None:
    row = _v4_row(fallback_route_target="none")

    errors = validate_manifest_rows([row])

    assert any("fallback_route_target must be empty" in error for error in errors)


def test_validation_catches_route_mapping_mismatch() -> None:
    row = _v4_row(
        content_label="speech_clean",
        route_target="clean_voice",
        engine_target="deepfilternet",
        candidate_tasks="clean_voice",
    )

    errors = validate_manifest_rows([row])

    assert any("route_target mismatch for speech_clean" in error for error in errors)
    assert any("engine_target mismatch for speech_clean" in error for error in errors)


def test_validation_passes_correctly_mapped_row() -> None:
    row = _v4_row(
        content_label="speech_clean",
        route_target="no_process",
        engine_target="none",
        candidate_tasks="clean_voice",
        contains_speech="true",
        contains_environment_noise="false",
    )

    assert validate_manifest_rows([row]) == []


def test_leakage_validation_catches_clean_source_across_splits() -> None:
    rows = [
        _v4_row(sample_id="a", split="train", clean_source_id="clean_1", group_id=""),
        _v4_row(sample_id="b", split="test", clean_source_id="clean_1", group_id=""),
    ]

    errors = validate_no_split_leakage(rows)

    assert any("clean_source_id clean_1" in error for error in errors)


def test_leakage_validation_catches_noise_source_across_splits() -> None:
    rows = [
        _v4_row(sample_id="a", split="train", noise_source_id="noise_1", group_id=""),
        _v4_row(sample_id="b", split="test", noise_source_id="noise_1", group_id=""),
    ]

    errors = validate_no_split_leakage(rows)

    assert any("noise_source_id noise_1" in error for error in errors)


def test_leakage_validation_passes_disjoint_examples() -> None:
    rows = [
        _v4_row(sample_id="a", split="train", clean_source_id="clean_1", noise_source_id="noise_1", group_id="g1"),
        _v4_row(sample_id="b", split="test", clean_source_id="clean_2", noise_source_id="noise_2", group_id="g2"),
        _v4_row(sample_id="c", split="val", clean_source_id="clean_3", noise_source_id="noise_3", group_id="g3"),
    ]

    assert validate_no_split_leakage(rows) == []


def test_cli_writes_exact_manifest_columns(tmp_path: Path) -> None:
    input_manifest = tmp_path / "audio_router_v3.csv"
    output_manifest = tmp_path / "audio_router_v4.csv"
    with input_manifest.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["sample_id", "input_path", "router_label", "source", "split", "notes"])
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "speech_clean_001",
                "input_path": "voicebank/clean.wav",
                "router_label": "speech_noise",
                "source": "voicebank_clean",
                "split": "train",
                "notes": "folder=clean_trainset_28spk_wav",
            }
        )
        writer.writerow(
            {
                "sample_id": "song_001",
                "input_path": "musdb/song.stem.mp4",
                "router_label": "music",
                "source": "musdb18_preview",
                "split": "test",
                "notes": "MUSDB18 preview stem container",
            }
        )
        writer.writerow(
            {
                "sample_id": "env_001",
                "input_path": "esc50/dog.wav",
                "router_label": "environment_noise",
                "source": "esc50",
                "split": "holdout",
                "notes": "category=dog; fold=1",
            }
        )

    exit_code = main(
        [
            "--input-manifest",
            str(input_manifest),
            "--output-manifest",
            str(output_manifest),
            "--source-dataset",
            "fallback",
        ]
    )

    assert exit_code == 0
    rows = _read_rows(output_manifest)
    assert rows
    assert list(rows[0].keys()) == MANIFEST_V4_COLUMNS
    assert [row["content_label"] for row in rows] == ["speech_clean", "music_with_vocals", "environment_only"]
    assert rows[2]["route_target"] == "out_of_scope"
