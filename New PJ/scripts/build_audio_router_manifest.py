"""Build a training manifest for the future audio task router."""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path


SPEECH_NOISE = "speech_noise"
MUSIC = "music"
ENVIRONMENT_NOISE = "environment_noise"

LABEL_ORDER = [SPEECH_NOISE, MUSIC, ENVIRONMENT_NOISE]
OUTPUT_COLUMNS = ["sample_id", "input_path", "router_label", "source", "split", "notes"]

TARGET_NOISE_COLUMNS = {"mixed_path"}
URBANSOUND8K_COLUMNS = {"slice_file_name", "fold", "class"}
ESC50_COLUMNS = {"filename", "fold", "category"}
VOICEBANK_FOLDERS = {
    "clean_trainset_28spk_wav": ("voicebank_clean", "train"),
    "clean_testset_wav": ("voicebank_clean", "test"),
    "noisy_trainset_28spk_wav": ("voicebank_noisy", "train"),
    "noisy_testset_wav": ("voicebank_noisy", "test"),
}


def _readable_path(path: Path, output_manifest: Path) -> str:
    base_dir = Path(output_manifest).resolve().parent
    return Path(os.path.relpath(path.resolve(), base_dir)).as_posix()


def _resolve_audio_input_path(value: str, manifest_path: Path) -> Path:
    """Resolve dataset audio paths from either project root or manifest location.

    Some generated manifests use project-root relative paths such as
    outputs/target-noise-v1/mixed/test/file.wav. Others use paths relative to
    the manifest CSV. Prefer the current working directory when that path
    exists, then fall back to manifest-relative resolution.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (Path(manifest_path).resolve().parent / path).resolve()


def _read_csv(path: Path, required_columns: set[str], source_name: str) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{source_name} manifest is missing required columns: {missing}")
        return list(reader)


def _target_noise_rows(manifest_path: Path, output_manifest: Path, target_noise_source: str) -> list[dict[str, str]]:
    rows = []
    for index, row in enumerate(_read_csv(manifest_path, TARGET_NOISE_COLUMNS, "Target-noise"), start=1):
        mixed_path = row.get("mixed_path", "").strip()
        if not mixed_path:
            continue
        sample_id = row.get("sample_id", "").strip() or Path(mixed_path).stem or f"target_noise_{index:05d}"
        noise_label = row.get("noise_label", "").strip()
        snr_db = row.get("snr_db", "").strip()
        notes = "; ".join(part for part in (f"noise_label={noise_label}" if noise_label else "", f"snr_db={snr_db}" if snr_db else "") if part)
        rows.append(
            {
                "sample_id": sample_id,
                "input_path": _readable_path(_resolve_audio_input_path(mixed_path, manifest_path), output_manifest),
                "router_label": SPEECH_NOISE,
                "source": target_noise_source,
                "split": row.get("split", "").strip() or "train",
                "notes": notes,
            }
        )
    return rows


def _infer_split_from_parts(path: Path, default: str = "train") -> str:
    parts = {part.lower() for part in path.parts}
    if "test" in parts:
        return "test"
    if "val" in parts or "validation" in parts:
        return "val"
    if "train" in parts:
        return "train"
    return default


def _musdb_rows(musdb_root: Path, output_manifest: Path) -> list[dict[str, str]]:
    root = Path(musdb_root).resolve()
    rows = []
    for path in sorted(root.rglob("*.stem.mp4")):
        relative = path.relative_to(root)
        rows.append(
            {
                "sample_id": path.name.removesuffix(".stem.mp4"),
                "input_path": _readable_path(path, output_manifest),
                "router_label": MUSIC,
                "source": "musdb18_preview",
                "split": _infer_split_from_parts(relative),
                "notes": "MUSDB18 preview stem container",
            }
        )
    return rows


def _voicebank_rows(voicebank_root: Path, output_manifest: Path) -> list[dict[str, str]]:
    root = Path(voicebank_root).resolve()
    rows = []
    for folder_name, (source, split) in VOICEBANK_FOLDERS.items():
        folder = root / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.wav")):
            rows.append(
                {
                    "sample_id": path.stem,
                    "input_path": _readable_path(path, output_manifest),
                    "router_label": SPEECH_NOISE,
                    "source": source,
                    "split": split,
                    "notes": f"folder={folder_name}",
                }
            )
    return rows


def _esc50_rows(esc50_root: Path, output_manifest: Path) -> list[dict[str, str]]:
    root = Path(esc50_root).resolve()
    metadata_path = root / "meta" / "esc50.csv"
    if not metadata_path.exists():
        return []

    rows = []
    for row in _read_csv(metadata_path, ESC50_COLUMNS, "ESC-50"):
        filename = row["filename"].strip()
        fold = row["fold"].strip()
        category = row["category"].strip()
        input_path = root / "audio" / filename
        rows.append(
            {
                "sample_id": Path(filename).stem,
                "input_path": _readable_path(input_path, output_manifest),
                "router_label": ENVIRONMENT_NOISE,
                "source": "esc50",
                "split": "test" if fold == "5" else "train",
                "notes": f"category={category}; fold={fold}",
            }
        )
    return rows


def _urbansound8k_rows(metadata_path: Path, audio_root: Path, output_manifest: Path) -> list[dict[str, str]]:
    rows = []
    for row in _read_csv(metadata_path, URBANSOUND8K_COLUMNS, "UrbanSound8K"):
        slice_file_name = row["slice_file_name"].strip()
        fold = row["fold"].strip()
        class_name = row["class"].strip()
        input_path = Path(audio_root).resolve() / f"fold{fold}" / slice_file_name
        rows.append(
            {
                "sample_id": Path(slice_file_name).stem,
                "input_path": _readable_path(input_path, output_manifest),
                "router_label": ENVIRONMENT_NOISE,
                "source": "urbansound8k",
                "split": "test" if fold == "10" else "train",
                "notes": f"class={class_name}; fold={fold}",
            }
        )
    return rows


def _sample_rows(rows: list[dict[str, str]], max_per_label: int, seed: int) -> list[dict[str, str]]:
    if max_per_label <= 0:
        raise ValueError("--max-per-label must be greater than zero.")

    rows_by_label_and_source: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        rows_by_label_and_source[row["router_label"]][row["source"]].append(row)

    sampled_rows: list[dict[str, str]] = []
    for label in LABEL_ORDER:
        source_groups = rows_by_label_and_source.get(label, {})
        if not source_groups:
            continue

        shuffled_by_source: dict[str, list[dict[str, str]]] = {}
        for source, source_rows in sorted(source_groups.items()):
            shuffled_rows = list(source_rows)
            rng = random.Random(f"{seed}:{label}:{source}")
            rng.shuffle(shuffled_rows)
            shuffled_by_source[source] = shuffled_rows

        sources = sorted(shuffled_by_source)
        base_quota = max_per_label // len(sources)
        remainder = max_per_label % len(sources)
        selected: list[dict[str, str]] = []
        leftovers: list[dict[str, str]] = []
        for index, source in enumerate(sources):
            source_rows = shuffled_by_source[source]
            quota = base_quota + (1 if index < remainder else 0)
            selected.extend(source_rows[:quota])
            leftovers.extend(source_rows[quota:])

        rng = random.Random(f"{seed}:{label}:fill")
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max(0, max_per_label - len(selected))])
        sampled_rows.extend(selected[:max_per_label])

    return sorted(sampled_rows, key=lambda row: (LABEL_ORDER.index(row["router_label"]), row["source"], row["split"], row["sample_id"]))


def build_audio_router_manifest(
    *,
    output: Path,
    target_noise_manifest: Path | None = None,
    voicebank_root: Path | None = None,
    musdb_root: Path | None = None,
    esc50_root: Path | None = None,
    urbansound8k_metadata: Path | None = None,
    urbansound8k_audio_root: Path | None = None,
    target_noise_source: str = "target_noise_v1",
    max_per_label: int = 300,
    seed: int = 42,
) -> Path:
    """Build a router training manifest from any available subset of sources."""
    output_path = Path(output)
    rows: list[dict[str, str]] = []

    if target_noise_manifest is not None:
        rows.extend(_target_noise_rows(target_noise_manifest, output_path, target_noise_source))
    if voicebank_root is not None:
        rows.extend(_voicebank_rows(voicebank_root, output_path))
    if musdb_root is not None:
        rows.extend(_musdb_rows(musdb_root, output_path))
    if esc50_root is not None:
        rows.extend(_esc50_rows(esc50_root, output_path))
    if urbansound8k_metadata is not None:
        if urbansound8k_audio_root is None:
            raise ValueError("--urbansound8k-audio-root is required when --urbansound8k-metadata is provided.")
        rows.extend(_urbansound8k_rows(urbansound8k_metadata, urbansound8k_audio_root, output_path))

    sampled_rows = _sample_rows(rows, max_per_label=max_per_label, seed=seed)
    if not sampled_rows:
        raise ValueError("No audio router rows were produced from the provided sources.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(sampled_rows)

    return output_path.resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the audio router manifest CLI parser."""
    parser = argparse.ArgumentParser(description="Build an audio router training manifest.")
    parser.add_argument("--target-noise-manifest", type=Path, default=None)
    parser.add_argument("--voicebank-root", type=Path, default=None)
    parser.add_argument("--musdb-root", type=Path, default=None)
    parser.add_argument("--esc50-root", type=Path, default=None)
    parser.add_argument("--urbansound8k-metadata", type=Path, default=None)
    parser.add_argument("--urbansound8k-audio-root", type=Path, default=None)
    parser.add_argument("--target-noise-source", default="target_noise_v1")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-per-label", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        output_path = build_audio_router_manifest(
            target_noise_manifest=args.target_noise_manifest,
            voicebank_root=args.voicebank_root,
            musdb_root=args.musdb_root,
            esc50_root=args.esc50_root,
            urbansound8k_metadata=args.urbansound8k_metadata,
            urbansound8k_audio_root=args.urbansound8k_audio_root,
            target_noise_source=args.target_noise_source,
            output=args.output,
            max_per_label=args.max_per_label,
            seed=args.seed,
        )
        print(f"Wrote audio router manifest: {output_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
