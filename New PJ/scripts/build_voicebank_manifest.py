"""Build a small VoiceBank-DEMAND clean/noisy pair manifest."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


SPLIT_FOLDERS = {
    "test": ("noisy_testset_wav", "clean_testset_wav"),
    "train": ("noisy_trainset_28spk_wav", "clean_trainset_28spk_wav"),
}


def _readable_path(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), Path.cwd().resolve())).as_posix()


def build_voicebank_manifest(
    *,
    voicebank_root: Path,
    output_path: Path,
    split: str = "test",
    limit: int | None = 20,
) -> Path:
    """Build a CSV manifest from a local VoiceBank-DEMAND folder."""
    root = Path(voicebank_root).resolve()
    noisy_folder_name, clean_folder_name = SPLIT_FOLDERS[split]
    noisy_dir = root / noisy_folder_name
    clean_dir = root / clean_folder_name

    if not noisy_dir.is_dir():
        raise ValueError(f"Expected noisy folder is missing: {noisy_dir}")
    if not clean_dir.is_dir():
        raise ValueError(f"Expected clean folder is missing: {clean_dir}")

    rows: list[dict[str, str]] = []
    for noisy_path in sorted(noisy_dir.glob("*.wav"), key=lambda path: path.name):
        clean_path = clean_dir / noisy_path.name
        if not clean_path.exists():
            continue
        rows.append(
            {
                "sample_id": noisy_path.stem,
                "noisy_path": _readable_path(noisy_path),
                "clean_path": _readable_path(clean_path),
                "split": split,
            }
        )

    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No clean/noisy VoiceBank pairs found for split: {split}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["sample_id", "noisy_path", "clean_path", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return output.resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the manifest builder CLI parser."""
    parser = argparse.ArgumentParser(description="Build a VoiceBank-DEMAND subset manifest.")
    parser.add_argument("--voicebank-root", required=True, type=Path, help="Path to local VoiceBank-DEMAND folder.")
    parser.add_argument("--output", default=Path("data/manifests/voicebank_subset.csv"), type=Path)
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--limit", type=int, default=20)
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        output_path = build_voicebank_manifest(
            voicebank_root=args.voicebank_root,
            output_path=args.output,
            split=args.split,
            limit=args.limit,
        )
        print(f"Wrote VoiceBank manifest: {output_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
