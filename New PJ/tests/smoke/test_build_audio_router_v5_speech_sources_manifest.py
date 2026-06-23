"""Smoke tests for Audio Router V5 Vietnamese clean-speech manifest builder."""

from __future__ import annotations

import csv
import wave
from pathlib import Path

from scripts.build_audio_router_v5_speech_sources_manifest import (
    V5_COLUMNS,
    _readable_path,
    build_arg_parser,
    build_audio_router_v5_speech_sources_manifest,
)


def _write_wav(path: Path, *, sample_rate: int = 8000, frames: int = 800) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as tsv_file:
        writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _fake_vivos(root: Path) -> None:
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir(parents=True)
    (root / "train" / "prompts.txt").write_text(
        "VIVOSSPK01_R001 xin chao viet nam\n"
        "VIVOSSPK02_R999 missing audio should be skipped\n",
        encoding="utf-8",
    )
    (root / "train" / "genders.txt").write_text("VIVOSSPK01 m\nVIVOSSPK02 f\n", encoding="utf-8")
    _write_wav(root / "train" / "waves" / "VIVOSSPK01" / "VIVOSSPK01_R001.wav")

    (root / "test" / "prompts.txt").write_text("VIVOSDEV01_R001 cau thu nghiem\n", encoding="utf-8")
    (root / "test" / "genders.txt").write_text("VIVOSDEV01 f\n", encoding="utf-8")
    _write_wav(root / "test" / "waves" / "VIVOSDEV01" / "VIVOSDEV01_R001.wav")


def _fake_common_voice(root: Path) -> None:
    clips = root / "clips"
    clips.mkdir(parents=True)
    for name in ("train_clip.mp3", "dev_clip.mp3", "test_clip.mp3"):
        (clips / name).write_bytes(b"placeholder")

    fieldnames = ["client_id", "path", "sentence", "age", "gender", "accent"]
    _write_tsv(
        root / "train.tsv",
        fieldnames,
        [
            {
                "client_id": "cv-client-train",
                "path": "train_clip.mp3",
                "sentence": "xin chao",
                "age": "twenties",
                "gender": "male",
                "accent": "north",
            },
            {
                "client_id": "cv-client-missing",
                "path": "missing_clip.mp3",
                "sentence": "missing audio",
                "age": "",
                "gender": "",
                "accent": "",
            },
        ],
    )
    _write_tsv(
        root / "dev.tsv",
        fieldnames,
        [
            {
                "client_id": "cv-client-dev",
                "path": "dev_clip.mp3",
                "sentence": " ".join(["phat trien"] * 80) + ";\nnewline\tand tab",
                "age": "thirties",
                "gender": "female",
                "accent": "south",
            }
        ],
    )
    _write_tsv(
        root / "test.tsv",
        fieldnames,
        [
            {
                "client_id": "cv-client-test",
                "path": "test_clip.mp3",
                "sentence": "kiem thu",
                "age": "",
                "gender": "other",
                "accent": "",
            }
        ],
    )
    _write_tsv(
        root / "validated.tsv",
        fieldnames,
        [
            {
                "client_id": "cv-client-validated",
                "path": "validated_clip.mp3",
                "sentence": "should not be used",
                "age": "",
                "gender": "",
                "accent": "",
            }
        ],
    )
    _write_tsv(
        root / "clip_durations.tsv",
        ["clip", "duration[ms]"],
        [
            {"clip": "train_clip.mp3", "duration[ms]": "1234"},
            {"clip": "dev_clip.mp3", "duration[ms]": "2.5"},
            {"clip": "test_clip.mp3", "duration[ms]": "3000"},
        ],
    )


def test_build_audio_router_v5_speech_sources_manifest(tmp_path: Path) -> None:
    vivos_root = tmp_path / "external" / "vivos" / "vivos"
    common_voice_root = tmp_path / "external" / "common_voice_vi" / "vi"
    output_csv = tmp_path / "data" / "manifests" / "audio_router_v5_speech_sources.local.csv"
    _fake_vivos(vivos_root)
    _fake_common_voice(common_voice_root)

    summary = build_audio_router_v5_speech_sources_manifest(
        vivos_root=vivos_root,
        common_voice_root=common_voice_root,
        output_csv=output_csv,
    )

    rows = _read_rows(output_csv)
    assert output_csv.exists()
    assert rows
    assert list(rows[0].keys()) == V5_COLUMNS
    assert summary["skipped_missing_audio"] == 2

    by_source_id = {row["source_id"]: row for row in rows}
    vivos_train = by_source_id["VIVOSSPK01_R001"]
    assert vivos_train["split"] == "train"
    assert vivos_train["source_corpus"] == "vivos"
    assert vivos_train["recording_id"] == "VIVOSSPK01"
    assert vivos_train["label"] == "speech_clean"
    assert vivos_train["route_target"] == "no_process"
    assert vivos_train["engine_target"] == "none"
    assert vivos_train["is_synthetic"] == "false"
    assert vivos_train["duration_sec"] == "0.100000"
    assert "gender=m" in vivos_train["notes"]
    assert "transcript=xin chao viet nam" in vivos_train["notes"]

    assert by_source_id["VIVOSDEV01_R001"]["split"] == "test"
    assert by_source_id["dev_clip"]["split"] == "val"
    assert by_source_id["train_clip"]["split"] == "train"
    assert by_source_id["test_clip"]["split"] == "test"
    assert by_source_id["dev_clip"]["source_corpus"] == "common_voice_vi"
    assert by_source_id["dev_clip"]["recording_id"] == "cv-client-dev"
    assert by_source_id["dev_clip"]["duration_sec"] == "2.500000"
    assert by_source_id["train_clip"]["duration_sec"] == "1.234000"
    assert by_source_id["test_clip"]["duration_sec"] == "3.000000"
    assert "sentence=phat trien" in by_source_id["dev_clip"]["notes"]
    assert len(by_source_id["dev_clip"]["notes"]) < 1000
    assert "[truncated]" in by_source_id["dev_clip"]["notes"]
    assert "\n" not in by_source_id["dev_clip"]["notes"]
    assert "\t" not in by_source_id["dev_clip"]["notes"]
    assert "validated_clip" not in by_source_id


def test_readable_path_prefers_repo_relative_when_under_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    audio_path = tmp_path / "data" / "external" / "vivos" / "vivos" / "train" / "waves" / "speaker" / "utt.wav"
    output_csv = tmp_path / "data" / "manifests" / "manifest.local.csv"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"audio")

    assert _readable_path(audio_path, output_csv) == "data/external/vivos/vivos/train/waves/speaker/utt.wav"


def test_build_audio_router_v5_requires_at_least_one_source(tmp_path: Path) -> None:
    output_csv = tmp_path / "manifest.csv"

    try:
        build_audio_router_v5_speech_sources_manifest(output_csv=output_csv)
    except ValueError as exc:
        assert "At least one" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError")


def test_build_audio_router_v5_fails_for_missing_root(tmp_path: Path) -> None:
    output_csv = tmp_path / "manifest.csv"

    try:
        build_audio_router_v5_speech_sources_manifest(vivos_root=tmp_path / "missing", output_csv=output_csv)
    except FileNotFoundError as exc:
        assert "VIVOS root not found" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected FileNotFoundError")


def test_build_audio_router_v5_cli_args() -> None:
    args = build_arg_parser().parse_args(
        [
            "--vivos-root",
            "data/external/vivos/vivos",
            "--common-voice-root",
            "data/external/common_voice_vi/cv-corpus-25.0-2026-03-09/vi",
            "--output-csv",
            "data/manifests/audio_router_v5_speech_sources.local.csv",
        ]
    )

    assert args.vivos_root == Path("data/external/vivos/vivos")
    assert args.common_voice_root == Path("data/external/common_voice_vi/cv-corpus-25.0-2026-03-09/vi")
    assert args.output_csv == Path("data/manifests/audio_router_v5_speech_sources.local.csv")
