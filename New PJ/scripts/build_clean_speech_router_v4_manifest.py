"""Build a clean-speech Router V4 manifest with train/val/test splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
import re


OUTPUT_COLUMNS = ["sample_id", "path", "split", "notes"]
VALID_SPLITS = {"train", "val", "test"}
SPLIT_POLICY = "train_val_from_train_keep_test"
GROUP_POLICIES = {"speaker", "sample"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(OUTPUT_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input manifest is missing required columns: {', '.join(sorted(missing))}")
        return list(reader)


def _speaker_id(sample_id: str) -> str:
    match = re.match(r"^(p\d+)_", sample_id)
    return match.group(1) if match else sample_id


def _group_id(row: dict[str, str], group_policy: str) -> str:
    sample_id = row.get("sample_id", "").strip()
    if group_policy == "speaker":
        return _speaker_id(sample_id)
    if group_policy == "sample":
        return sample_id
    raise ValueError(f"Unsupported group policy: {group_policy}")


def _stable_score(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}:clean-router-v4:{group_id}".encode("utf-8")).hexdigest()


def _normalized_split(value: str) -> str:
    split = value.strip().lower()
    return split if split in VALID_SPLITS else "train"


def _append_notes(notes: str, metadata: list[str]) -> str:
    parts = [part.strip() for part in notes.split(";") if part.strip()]
    parts.extend(metadata)
    return "; ".join(parts)


def _select_validation_groups(
    train_rows: list[dict[str, str]],
    *,
    seed: int,
    val_ratio: float,
    group_policy: str,
) -> set[str]:
    groups: dict[str, int] = Counter(_group_id(row, group_policy) for row in train_rows)
    if not groups or val_ratio <= 0:
        return set()

    total = sum(groups.values())
    target = total * val_ratio
    ordered = sorted(groups.items(), key=lambda item: (_stable_score(seed, item[0]), item[0]))
    dp: dict[int, tuple[str, ...]] = {0: ()}
    for group_id, count in ordered:
        updates: dict[int, tuple[str, ...]] = {}
        for current_count, selected_groups in dp.items():
            new_count = current_count + count
            new_groups = tuple(sorted((*selected_groups, group_id)))
            existing = dp.get(new_count) or updates.get(new_count)
            if existing is None or _stable_score(seed, "|".join(new_groups)) < _stable_score(seed, "|".join(existing)):
                updates[new_count] = new_groups
        dp.update(updates)

    candidates: list[tuple[float, str, int, tuple[str, ...]]] = []
    for count, selected_groups in dp.items():
        if count == 0:
            continue
        if len(groups) > 1 and count == total:
            continue
        tie_key = _stable_score(seed, "|".join(selected_groups))
        candidates.append((abs(count - target), tie_key, count, selected_groups))

    if not candidates:
        return set()

    _distance, _tie_key, _count, selected_groups = min(candidates)
    return set(selected_groups)


def validate_clean_router_v4_rows(rows: list[dict[str, str]], group_policy: str = "speaker") -> list[str]:
    """Validate clean-speech Router V4 rows."""
    errors: list[str] = []
    sample_splits: dict[str, set[str]] = defaultdict(set)
    speaker_train_val_splits: dict[str, set[str]] = defaultdict(set)

    for index, row in enumerate(rows, start=1):
        missing = [column for column in OUTPUT_COLUMNS if column not in row]
        if missing:
            errors.append(f"row {index}: missing columns: {', '.join(missing)}")
            continue

        sample_id = row["sample_id"].strip()
        path = row["path"].strip()
        split = row["split"].strip()
        if not sample_id:
            errors.append(f"row {index}: sample_id is required")
        if not path:
            errors.append(f"row {index}: path is required")
        if split not in VALID_SPLITS:
            errors.append(f"row {index}: invalid split: {split}")
        if sample_id:
            sample_splits[sample_id].add(split)
        if group_policy == "speaker" and sample_id and split in {"train", "val"}:
            speaker_train_val_splits[_speaker_id(sample_id)].add(split)

    for sample_id, splits in sorted(sample_splits.items()):
        valid_splits = splits & VALID_SPLITS
        if len(valid_splits) > 1:
            errors.append(f"sample_id {sample_id} appears in multiple splits: {', '.join(sorted(valid_splits))}")
    for speaker_id, splits in sorted(speaker_train_val_splits.items()):
        if {"train", "val"} <= splits:
            errors.append(f"speaker {speaker_id} appears in both train and val")

    return errors


def build_clean_speech_router_v4_manifest(
    *,
    input_manifest: Path,
    output_manifest: Path,
    seed: int = 13,
    val_ratio: float = 0.15,
    group_policy: str = "speaker",
    split_policy: str = SPLIT_POLICY,
) -> Path:
    """Write a clean speech manifest with deterministic Router V4 train/val/test splits."""
    if group_policy not in GROUP_POLICIES:
        raise ValueError(f"Unsupported group policy: {group_policy}")
    if split_policy != SPLIT_POLICY:
        raise ValueError(f"Unsupported split policy: {split_policy}")

    input_rows = _read_rows(input_manifest)
    train_candidates = [row for row in input_rows if _normalized_split(row.get("split", "")) == "train"]
    validation_groups = _select_validation_groups(
        train_candidates,
        seed=seed,
        val_ratio=val_ratio,
        group_policy=group_policy,
    )

    output_rows: list[dict[str, str]] = []
    for row in input_rows:
        sample_id = row.get("sample_id", "").strip()
        original_split = row.get("split", "").strip()
        normalized_split = _normalized_split(original_split)
        if normalized_split == "test":
            split = "test"
        elif normalized_split == "val":
            split = "val"
        elif _group_id(row, group_policy) in validation_groups:
            split = "val"
        else:
            split = "train"

        notes = _append_notes(
            row.get("notes", ""),
            [
                f"router_v4_split_policy={split_policy}",
                f"router_v4_group_policy={group_policy}",
                f"router_v4_seed={seed}",
                f"original_split={original_split or 'train'}",
            ],
        )
        output_rows.append(
            {
                "sample_id": sample_id,
                "path": row.get("path", "").strip(),
                "split": split,
                "notes": notes,
            }
        )

    errors = validate_clean_router_v4_rows(output_rows, group_policy=group_policy)
    if errors:
        raise ValueError("; ".join(errors))

    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    counts = Counter(row["split"] for row in output_rows)
    print(f"total rows: {len(output_rows)}")
    print(f"counts by split: {dict(sorted(counts.items()))}")
    print(f"number of train-candidate groups: {len({_group_id(row, group_policy) for row in train_candidates})}")
    print(f"validation groups selected: {', '.join(sorted(validation_groups))}")
    print(f"group_policy: {group_policy}")
    print(f"split_policy: {split_policy}")
    return output_path.resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a clean-speech Router V4 split manifest.")
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--group-policy", choices=sorted(GROUP_POLICIES), default="speaker")
    parser.add_argument("--split-policy", choices=[SPLIT_POLICY], default=SPLIT_POLICY)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        build_clean_speech_router_v4_manifest(
            input_manifest=args.input_manifest,
            output_manifest=args.output_manifest,
            seed=args.seed,
            val_ratio=args.val_ratio,
            group_policy=args.group_policy,
            split_policy=args.split_policy,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
