"""Build Audio Router V5 clean-speech manifests from VIVOS and Common Voice VI."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import wave
from collections import Counter
from pathlib import Path


csv.field_size_limit(sys.maxsize)

V5_COLUMNS = [
    "path",
    "label",
    "source_corpus",
    "source_id",
    "recording_id",
    "split",
    "is_synthetic",
    "mix_clean_source",
    "mix_noise_source",
    "snr_db",
    "route_target",
    "engine_target",
    "duration_sec",
    "notes",
]
VIVOS_SPLITS = {"train": "train", "test": "test"}
COMMON_VOICE_SPLITS = {"train": "train", "dev": "val", "test": "test"}
NOTE_VALUE_LIMIT = 240


def _readable_path(path: Path, output_csv: Path) -> str:
    resolved = Path(path).resolve()
    cwd = Path.cwd().resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        base_dir = Path(output_csv).resolve().parent
        return os.path.relpath(resolved, base_dir).replace(os.sep, "/")


def _normalize_note_value(value: str, limit: int = NOTE_VALUE_LIMIT) -> str:
    normalized = re.sub(r"\s+", " ", str(value)).strip().replace(";", ",")
    if len(normalized) <= limit:
        return normalized
    suffix = "...[truncated]"
    return normalized[: max(0, limit - len(suffix))].rstrip() + suffix


def _note(parts: list[tuple[str, str]]) -> str:
    return "; ".join(f"{key}={_normalize_note_value(value)}" for key, value in parts if value)


def _duration_from_wav(path: Path) -> str:
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
    if sample_rate <= 0:
        return ""
    return f"{frames / sample_rate:.6f}"


def _read_vivos_prompts(path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    if not path.exists():
        return prompts
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        utterance_id = parts[0]
        transcript = parts[1] if len(parts) > 1 else ""
        prompts[utterance_id] = transcript
    return prompts


def _read_vivos_genders(path: Path) -> dict[str, str]:
    genders: dict[str, str] = {}
    if not path.exists():
        return genders
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            genders[parts[0]] = parts[1]
    return genders


def _base_clean_speech_row(*, path: Path, output_csv: Path, source_corpus: str, source_id: str, recording_id: str, split: str, duration_sec: str, notes: str) -> dict[str, str]:
    return {
        "path": _readable_path(path, output_csv),
        "label": "speech_clean",
        "source_corpus": source_corpus,
        "source_id": source_id,
        "recording_id": recording_id,
        "split": split,
        "is_synthetic": "false",
        "mix_clean_source": "",
        "mix_noise_source": "",
        "snr_db": "",
        "route_target": "no_process",
        "engine_target": "none",
        "duration_sec": duration_sec,
        "notes": notes,
    }


def _speaker_id_from_utterance(utterance_id: str) -> str:
    return utterance_id.split("_", 1)[0]


def _vivos_rows(vivos_root: Path, output_csv: Path) -> tuple[list[dict[str, str]], int]:
    root = Path(vivos_root)
    if not root.exists():
        raise FileNotFoundError(f"VIVOS root not found: {root}")

    rows: list[dict[str, str]] = []
    skipped_missing_audio = 0
    for source_split, manifest_split in VIVOS_SPLITS.items():
        split_dir = root / source_split
        prompts = _read_vivos_prompts(split_dir / "prompts.txt")
        genders = _read_vivos_genders(split_dir / "genders.txt")
        for utterance_id, transcript in sorted(prompts.items()):
            speaker_id = _speaker_id_from_utterance(utterance_id)
            audio_path = split_dir / "waves" / speaker_id / f"{utterance_id}.wav"
            if not audio_path.exists():
                skipped_missing_audio += 1
                continue
            notes = _note(
                [
                    ("language", "vi"),
                    ("source_split", source_split),
                    ("gender", genders.get(speaker_id, "")),
                    ("transcript", transcript),
                ]
            )
            rows.append(
                _base_clean_speech_row(
                    path=audio_path,
                    output_csv=output_csv,
                    source_corpus="vivos",
                    source_id=utterance_id,
                    recording_id=speaker_id,
                    split=manifest_split,
                    duration_sec=_duration_from_wav(audio_path),
                    notes=notes,
                )
            )
    return rows, skipped_missing_audio


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as tsv_file:
        return list(csv.DictReader(tsv_file, delimiter="\t"))


def _duration_keys(clip_path: str) -> set[str]:
    path = Path(clip_path)
    return {clip_path, path.as_posix(), path.name, path.stem}


def _duration_value(row: dict[str, str]) -> str:
    for key in ("duration", "duration_sec", "duration_seconds", "seconds", "duration[ms]", "duration_ms", "clip_duration"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    for key, value in row.items():
        if "duration" in key.lower() and value:
            return str(value)
    return ""


def _duration_path_value(row: dict[str, str]) -> str:
    for key in ("path", "clip", "filename", "file", "clip_path"):
        value = row.get(key)
        if value:
            return str(value)
    for key, value in row.items():
        if key.lower() not in {"duration", "duration_sec", "duration_seconds", "seconds", "duration[ms]", "duration_ms"} and value:
            return str(value)
    return ""


def _normalize_duration(raw_value: str) -> str:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return ""
    if value > 1000.0:
        value = value / 1000.0
    return f"{value:.6f}" if value > 0 else ""


def _read_common_voice_durations(path: Path) -> dict[str, str]:
    durations: dict[str, str] = {}
    if not path.exists():
        return durations
    with path.open(newline="", encoding="utf-8") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        for row in reader:
            clip_value = _duration_path_value(row).strip()
            duration = _normalize_duration(_duration_value(row).strip())
            if not clip_value or not duration:
                continue
            for key in _duration_keys(clip_value):
                durations[key] = duration
    return durations


def _common_voice_duration(durations: dict[str, str], clip_path: str) -> str:
    for key in _duration_keys(clip_path):
        if key in durations:
            return durations[key]
    return ""


def _common_voice_source_id(clip_path: str) -> str:
    return Path(clip_path).stem or clip_path


def _common_voice_rows(common_voice_root: Path, output_csv: Path) -> tuple[list[dict[str, str]], int]:
    root = Path(common_voice_root)
    if not root.exists():
        raise FileNotFoundError(f"Common Voice VI root not found: {root}")

    durations = _read_common_voice_durations(root / "clip_durations.tsv")
    rows: list[dict[str, str]] = []
    skipped_missing_audio = 0
    for source_split, manifest_split in COMMON_VOICE_SPLITS.items():
        for row in _read_tsv(root / f"{source_split}.tsv"):
            clip_name = row.get("path", "").strip()
            if not clip_name:
                continue
            audio_path = root / "clips" / clip_name
            if not audio_path.exists():
                skipped_missing_audio += 1
                continue
            accent = row.get("accent", "").strip() or row.get("accents", "").strip()
            notes = _note(
                [
                    ("language", "vi"),
                    ("source_split", source_split),
                    ("age", row.get("age", "").strip()),
                    ("gender", row.get("gender", "").strip()),
                    ("accent", accent),
                    ("sentence", row.get("sentence", "").strip()),
                ]
            )
            rows.append(
                _base_clean_speech_row(
                    path=audio_path,
                    output_csv=output_csv,
                    source_corpus="common_voice_vi",
                    source_id=_common_voice_source_id(clip_name),
                    recording_id=row.get("client_id", "").strip(),
                    split=manifest_split,
                    duration_sec=_common_voice_duration(durations, clip_name),
                    notes=notes,
                )
            )
    return rows, skipped_missing_audio


def _dedupe_rows_by_path(rows: list[dict[str, str]], output_csv: Path) -> list[dict[str, str]]:
    seen: set[Path] = set()
    deduped: list[dict[str, str]] = []
    base_dir = Path(output_csv).resolve().parent
    for row in rows:
        manifest_path = Path(row["path"])
        cwd_path = manifest_path.resolve()
        resolved = cwd_path if cwd_path.exists() else (base_dir / manifest_path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(row)
    return deduped


def build_audio_router_v5_speech_sources_manifest(
    *,
    output_csv: Path,
    vivos_root: Path | None = None,
    common_voice_root: Path | None = None,
) -> dict[str, object]:
    """Build a V5 clean-speech manifest from provided source roots."""
    if vivos_root is None and common_voice_root is None:
        raise ValueError("At least one of --vivos-root or --common-voice-root must be provided.")

    output_path = Path(output_csv)
    rows: list[dict[str, str]] = []
    skipped_missing_audio = 0
    if vivos_root is not None:
        source_rows, skipped = _vivos_rows(Path(vivos_root), output_path)
        rows.extend(source_rows)
        skipped_missing_audio += skipped
    if common_voice_root is not None:
        source_rows, skipped = _common_voice_rows(Path(common_voice_root), output_path)
        rows.extend(source_rows)
        skipped_missing_audio += skipped

    rows = _dedupe_rows_by_path(rows, output_path)
    rows.sort(key=lambda row: (row["source_corpus"], row["split"], row["recording_id"], row["source_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=V5_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    counts_by_source = Counter(row["source_corpus"] for row in rows)
    counts_by_split = Counter((row["source_corpus"], row["split"]) for row in rows)
    summary = {
        "rows": len(rows),
        "counts_by_source_corpus": dict(sorted(counts_by_source.items())),
        "counts_by_source_split": {f"{source}:{split}": count for (source, split), count in sorted(counts_by_split.items())},
        "skipped_missing_audio": skipped_missing_audio,
    }
    print(f"rows={summary['rows']}")
    print(f"counts_by_source_corpus={summary['counts_by_source_corpus']}")
    print(f"counts_by_source_split={summary['counts_by_source_split']}")
    print(f"skipped_missing_audio={summary['skipped_missing_audio']}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Audio Router V5 clean-speech manifest from VIVOS and Common Voice VI.")
    parser.add_argument("--vivos-root", type=Path, default=None)
    parser.add_argument("--common-voice-root", type=Path, default=None)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        build_audio_router_v5_speech_sources_manifest(
            vivos_root=args.vivos_root,
            common_voice_root=args.common_voice_root,
            output_csv=args.output_csv,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
