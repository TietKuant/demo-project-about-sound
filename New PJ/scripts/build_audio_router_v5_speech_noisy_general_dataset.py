"""Build synthetic Audio Router V5 speech_noisy_general examples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path


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
GENERAL_NOISE_COLUMNS = [
    "path",
    "source_corpus",
    "source_id",
    "recording_id",
    "split",
    "noise_label",
    "duration_sec",
    "notes",
]
VALID_SPLITS = ("train", "val", "test")
NOTE_VALUE_LIMIT = 240


def _readable_path(path: Path, output_csv: Path) -> str:
    resolved = Path(path).resolve()
    cwd = Path.cwd().resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        base_dir = Path(output_csv).resolve().parent
        return os.path.relpath(resolved, base_dir).replace(os.sep, "/")


def _resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (Path(manifest_path).resolve().parent / path).resolve()


def _normalize_note_value(value: str, limit: int = NOTE_VALUE_LIMIT) -> str:
    normalized = re.sub(r"\s+", " ", str(value)).strip().replace(";", ",")
    if len(normalized) <= limit:
        return normalized
    suffix = "...[truncated]"
    return normalized[: max(0, limit - len(suffix))].rstrip() + suffix


def _note(parts: list[tuple[str, str]]) -> str:
    return "; ".join(f"{key}={_normalize_note_value(value)}" for key, value in parts if value)


def _read_csv(path: Path, required_columns: list[str], source_name: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(required_columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source_name} manifest is missing required columns: {', '.join(sorted(missing))}")
        return list(reader)


def _parse_snr_values(value: str) -> list[float]:
    snrs = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not snrs:
        raise ValueError("--snr-db must contain at least one value.")
    return snrs


def _snr_text(snr_db: float) -> str:
    return f"{snr_db:g}"


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def _source_ref(row: dict[str, str]) -> str:
    return f"{row['source_corpus']}:{row['source_id']}"


def _split_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped = {split: [] for split in VALID_SPLITS}
    for row in rows:
        split = row.get("split", "").strip()
        if split in grouped:
            grouped[split].append(row)
    for split in grouped:
        grouped[split].sort(key=lambda row: (row.get("source_corpus", ""), row.get("recording_id", ""), row.get("source_id", ""), row.get("path", "")))
    return grouped


def mix_clean_with_general_noise(clean_path: Path, noise_path: Path, output_path: Path, snr_db: float) -> None:
    """Mix clean speech with background noise using ffmpeg and approximate SNR scaling."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to generate speech_noisy_general WAV files.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    noise_volume = 10 ** (-float(snr_db) / 20.0)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(clean_path),
            "-stream_loop",
            "-1",
            "-i",
            str(noise_path),
            "-filter_complex",
            (
                "[0:a]aformat=sample_fmts=s16:channel_layouts=mono,aresample=16000[clean];"
                f"[1:a]aformat=sample_fmts=s16:channel_layouts=mono,aresample=16000,volume={noise_volume:.6f}[noise];"
                "[clean][noise]amix=inputs=2:duration=first:dropout_transition=0,alimiter=limit=0.99[out]"
            ),
            "-map",
            "[out]",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip() or "unknown ffmpeg mixing error"
        raise RuntimeError(f"ffmpeg failed while mixing speech_noisy_general at SNR {snr_db}: {details}")


def _build_rows(
    *,
    clean_rows: list[dict[str, str]],
    noise_rows: list[dict[str, str]],
    clean_manifest: Path,
    noise_manifest: Path,
    output_dir: Path,
    output_csv: Path,
    max_per_split: int,
    snr_db_values: list[float],
    dry_run: bool,
) -> list[dict[str, str]]:
    if max_per_split <= 0:
        raise ValueError("--max-per-split must be greater than zero.")

    clean_by_split = _split_rows(clean_rows)
    noise_by_split = _split_rows(noise_rows)
    output_rows: list[dict[str, str]] = []
    for split in VALID_SPLITS:
        clean_split_rows = clean_by_split[split]
        noise_split_rows = noise_by_split[split]
        if not clean_split_rows or not noise_split_rows:
            continue
        count = min(max_per_split, len(clean_split_rows))
        for index in range(count):
            clean_row = clean_split_rows[index % len(clean_split_rows)]
            noise_row = noise_split_rows[index % len(noise_split_rows)]
            snr_db = snr_db_values[index % len(snr_db_values)]
            clean_ref = _source_ref(clean_row)
            noise_ref = _source_ref(noise_row)
            source_id = f"{split}_{index:05d}_{_short_hash(clean_ref)}_{_short_hash(noise_ref)}_snr{_snr_text(snr_db)}"
            output_path = output_dir / split / f"{_safe_slug(source_id)}.wav"
            clean_path = _resolve_manifest_path(clean_row["path"], clean_manifest)
            noise_path = _resolve_manifest_path(noise_row["path"], noise_manifest)
            if not dry_run:
                mix_clean_with_general_noise(clean_path, noise_path, output_path, snr_db)
            output_rows.append(
                {
                    "path": _readable_path(output_path, output_csv),
                    "label": "speech_noisy_general",
                    "source_corpus": "synthetic_speech_noisy_general",
                    "source_id": source_id,
                    "recording_id": clean_row.get("recording_id", ""),
                    "split": split,
                    "is_synthetic": "true",
                    "mix_clean_source": clean_ref,
                    "mix_noise_source": noise_ref,
                    "snr_db": _snr_text(snr_db),
                    "route_target": "clean_voice",
                    "engine_target": "deepfilternet",
                    "duration_sec": clean_row.get("duration_sec", ""),
                    "notes": _note(
                        [
                            ("clean_split", clean_row.get("split", "")),
                            ("noise_split", noise_row.get("split", "")),
                            ("noise_label", noise_row.get("noise_label", "")),
                            ("dry_run", str(dry_run).lower()),
                        ]
                    ),
                }
            )
    output_rows.sort(key=lambda row: (row["split"], row["source_id"]))
    return output_rows


def build_audio_router_v5_speech_noisy_general_dataset(
    *,
    clean_speech_manifest: Path,
    general_noise_manifest: Path,
    output_dir: Path,
    output_csv: Path,
    max_per_split: int,
    snr_db_values: list[float],
    dry_run: bool = False,
) -> dict[str, object]:
    """Build synthetic speech_noisy_general examples from clean speech and general noise manifests."""
    clean_manifest = Path(clean_speech_manifest)
    noise_manifest = Path(general_noise_manifest)
    output_path = Path(output_csv)
    clean_rows = _read_csv(clean_manifest, V5_COLUMNS, "Clean speech")
    noise_rows = _read_csv(noise_manifest, GENERAL_NOISE_COLUMNS, "General noise")
    clean_rows = [row for row in clean_rows if row.get("label", "").strip() == "speech_clean"]
    if not clean_rows:
        raise ValueError("No speech_clean rows are available in the clean speech manifest.")
    if not noise_rows:
        raise ValueError("No general noise rows are available in the general noise manifest.")
    rows = _build_rows(
        clean_rows=clean_rows,
        noise_rows=noise_rows,
        clean_manifest=clean_manifest,
        noise_manifest=noise_manifest,
        output_dir=Path(output_dir),
        output_csv=output_path,
        max_per_split=max_per_split,
        snr_db_values=snr_db_values,
        dry_run=dry_run,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=V5_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    counts_by_split = Counter(row["split"] for row in rows)
    counts_by_snr = Counter(row["snr_db"] for row in rows)
    summary = {
        "rows": len(rows),
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "counts_by_snr_db": dict(sorted(counts_by_snr.items())),
        "dry_run": dry_run,
    }
    print(f"rows={summary['rows']}")
    print(f"counts_by_split={summary['counts_by_split']}")
    print(f"counts_by_snr_db={summary['counts_by_snr_db']}")
    print(f"dry_run={str(dry_run).lower()}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Audio Router V5 speech_noisy_general dataset.")
    parser.add_argument("--clean-speech-manifest", required=True, type=Path)
    parser.add_argument("--general-noise-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--max-per-split", required=True, type=int)
    parser.add_argument("--snr-db", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        build_audio_router_v5_speech_noisy_general_dataset(
            clean_speech_manifest=args.clean_speech_manifest,
            general_noise_manifest=args.general_noise_manifest,
            output_dir=args.output_dir,
            output_csv=args.output_csv,
            max_per_split=args.max_per_split,
            snr_db_values=_parse_snr_values(args.snr_db),
            dry_run=args.dry_run,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
