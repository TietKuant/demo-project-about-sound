"""Build a reproducible MVP subset from an Audio Router V4 manifest."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_EXCLUDED_LABELS = ["speech_noisy_general"]
DEFAULT_NOTES_SUFFIX = "mvp_exclusion=speech_noisy_general_missing_train_split"
REQUIRED_COLUMNS = {"sample_id", "input_path", "content_label", "split", "notes"}


def _read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ValueError(f"Audio Router V4 manifest is missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Audio Router V4 manifest is empty: {path}")
    return fieldnames, rows


def _append_notes_suffix(notes: str, suffix: str) -> str:
    suffix = suffix.strip()
    if not suffix:
        return notes.strip()
    parts = [part.strip() for part in notes.split(";") if part.strip()]
    if suffix not in parts:
        parts.append(suffix)
    return "; ".join(parts)


def _flatten_excluded_labels(values: list[list[str]] | None) -> list[str]:
    if not values:
        return list(DEFAULT_EXCLUDED_LABELS)
    labels = [label.strip() for group in values for label in group if label.strip()]
    if not labels:
        raise ValueError("At least one non-empty excluded label is required.")
    return sorted(set(labels))


def validate_train_test_coverage(rows: list[dict[str, str]]) -> list[str]:
    """Return errors for labels that lack both train and test rows."""
    splits_by_label: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label = row.get("content_label", "").strip()
        split = row.get("split", "").strip()
        if label:
            splits_by_label[label].add(split)

    errors: list[str] = []
    for label, splits in sorted(splits_by_label.items()):
        if not {"train", "test"} <= splits:
            observed = ", ".join(sorted(split for split in splits if split)) or "none"
            errors.append(f"content_label {label} lacks train/test coverage; observed splits: {observed}")
    return errors


def build_audio_router_v4_mvp_manifest(
    *,
    input_manifest: Path,
    output_manifest: Path,
    exclude_labels: list[str] | None = None,
    require_train_test_coverage: bool = True,
    notes_suffix: str = DEFAULT_NOTES_SUFFIX,
) -> Path:
    """Filter a V4 manifest into a reproducible MVP subset."""
    fieldnames, rows = _read_manifest(input_manifest)
    excluded = set(exclude_labels if exclude_labels is not None else DEFAULT_EXCLUDED_LABELS)
    filtered_rows: list[dict[str, str]] = []
    for row in rows:
        if row.get("content_label", "").strip() in excluded:
            continue
        filtered_row = dict(row)
        filtered_row["notes"] = _append_notes_suffix(filtered_row.get("notes", ""), notes_suffix)
        filtered_rows.append(filtered_row)

    if not filtered_rows:
        raise ValueError("No rows remain after applying MVP label exclusions.")

    if require_train_test_coverage:
        coverage_errors = validate_train_test_coverage(filtered_rows)
        if coverage_errors:
            raise ValueError("; ".join(coverage_errors))

    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    return output_path.resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an Audio Router V4 MVP manifest.")
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--exclude-label", action="append", nargs="+", default=None)
    parser.add_argument("--require-train-test-coverage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--notes-suffix", default=DEFAULT_NOTES_SUFFIX)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        output_path = build_audio_router_v4_mvp_manifest(
            input_manifest=args.input_manifest,
            output_manifest=args.output_manifest,
            exclude_labels=_flatten_excluded_labels(args.exclude_label),
            require_train_test_coverage=args.require_train_test_coverage,
            notes_suffix=args.notes_suffix,
        )
        print(f"Wrote Audio Router V4 MVP manifest: {output_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
