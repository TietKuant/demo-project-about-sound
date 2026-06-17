"""Build and validate Audio Router V4 manifests."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


CONTENT_LABELS = [
    "speech_clean",
    "speech_noisy_general",
    "speech_target_noise",
    "music_with_vocals",
    "music_instrumental",
    "environment_only",
    "unknown_mixed",
]

ROUTE_TARGETS = [
    "clean_voice",
    "target_noise_suppression",
    "extract_vocals",
    "remove_vocals",
    "no_process",
    "manual_required",
    "abstain",
    "out_of_scope",
]

ENGINE_TARGETS = [
    "deepfilternet",
    "demucs",
    "target_noise_suppressor",
    "none",
]

SUPPORTED_TARGET_NOISE_LABELS = [
    "dog_bark",
    "car_horn",
    "siren",
]

VALID_TASK_IDS = {
    "clean_voice",
    "target_noise_suppression",
    "extract_vocals",
    "remove_vocals",
}

VALID_SPLITS = {"train", "val", "test", "holdout"}

MANIFEST_V4_COLUMNS = [
    "sample_id",
    "input_path",
    "source_dataset",
    "split",
    "split_key",
    "group_id",
    "duration_sec",
    "sample_rate",
    "channels",
    "taxonomy_version",
    "original_sample_rate",
    "normalized_sample_rate",
    "normalization_version",
    "analysis_window_sec",
    "source_fold",
    "original_split",
    "content_label",
    "route_target",
    "engine_target",
    "candidate_tasks",
    "fallback_route_target",
    "fallback_engine_target",
    "dominant_source_policy",
    "label_method",
    "label_confidence",
    "contains_speech",
    "contains_music",
    "contains_vocals",
    "contains_environment_noise",
    "contains_target_noise",
    "target_noise_label",
    "snr_db",
    "is_synthetic",
    "clean_source_id",
    "noise_source_id",
    "mix_seed",
    "language",
    "domain",
    "checksum",
    "features_version",
    "event_count",
    "event_time_spans",
    "feature_status",
    "notes",
]

SOURCE_TO_CONTENT_LABEL = {
    "voicebank_clean": "speech_clean",
    "voicebank_noisy": "speech_noisy_general",
    "target_noise_v1": "speech_target_noise",
    "musdb18_preview": "music_with_vocals",
    "esc50": "environment_only",
    "urbansound8k": "environment_only",
}


def _is_target_noise_source(source: str) -> bool:
    return source.startswith("target_noise")


def _content_label_for_source(source: str) -> str:
    if _is_target_noise_source(source):
        return "speech_target_noise"
    return SOURCE_TO_CONTENT_LABEL.get(source, "unknown_mixed")


def encode_candidate_tasks(tasks: list[str]) -> str:
    """Encode task ids as a semicolon-separated CSV value."""
    invalid = [task for task in tasks if task not in VALID_TASK_IDS]
    if invalid:
        raise ValueError(f"Invalid candidate task ids: {', '.join(invalid)}")
    return ";".join(tasks)


def route_mapping_for_content_label(content_label: str) -> dict[str, str]:
    """Return controlled route fields for one V4 content label."""
    if content_label not in CONTENT_LABELS:
        raise ValueError(f"Unsupported content_label: {content_label}")

    mappings = {
        "speech_clean": {
            "route_target": "no_process",
            "engine_target": "none",
            "candidate_tasks": encode_candidate_tasks(["clean_voice"]),
            "fallback_route_target": "",
            "fallback_engine_target": "",
            "notes": "clean_voice is optional; clean speech should not be over-processed.",
        },
        "speech_noisy_general": {
            "route_target": "clean_voice",
            "engine_target": "deepfilternet",
            "candidate_tasks": encode_candidate_tasks(["clean_voice"]),
            "fallback_route_target": "",
            "fallback_engine_target": "",
            "notes": "main speech enhancement route.",
        },
        "speech_target_noise": {
            "route_target": "target_noise_suppression",
            "engine_target": "target_noise_suppressor",
            "candidate_tasks": encode_candidate_tasks(["target_noise_suppression"]),
            "fallback_route_target": "clean_voice",
            "fallback_engine_target": "deepfilternet",
            "notes": "project-supported target-noise only; not universal noise removal.",
        },
        "music_with_vocals": {
            "route_target": "manual_required",
            "engine_target": "demucs",
            "candidate_tasks": encode_candidate_tasks(["extract_vocals", "remove_vocals"]),
            "fallback_route_target": "",
            "fallback_engine_target": "",
            "notes": "user goal decides whether to keep vocals or accompaniment.",
        },
        "music_instrumental": {
            "route_target": "no_process",
            "engine_target": "none",
            "candidate_tasks": encode_candidate_tasks(["extract_vocals", "remove_vocals"]),
            "fallback_route_target": "",
            "fallback_engine_target": "",
            "notes": "Demucs can remain available as manual optional processing, but no automatic processing is needed.",
        },
        "environment_only": {
            "route_target": "out_of_scope",
            "engine_target": "none",
            "candidate_tasks": "",
            "fallback_route_target": "",
            "fallback_engine_target": "",
            "notes": "MVP does not claim universal environmental sound removal.",
        },
        "unknown_mixed": {
            "route_target": "abstain",
            "engine_target": "none",
            "candidate_tasks": "",
            "fallback_route_target": "manual_required",
            "fallback_engine_target": "",
            "notes": "do not auto-route.",
        },
    }
    return dict(mappings[content_label])


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _is_low_confidence(evidence: dict[str, object]) -> bool:
    for key in ("confidence", "label_confidence"):
        confidence = _float_or_none(evidence.get(key))
        if confidence is not None and confidence < 0.5:
            return True
    return False


def _is_noisy(evidence: dict[str, object]) -> bool:
    if _truthy(evidence.get("is_noisy")):
        return True
    return _float_or_none(evidence.get("snr_db")) is not None


def collapse_content_label(evidence: dict[str, object]) -> str:
    """Collapse multi-label evidence into the single V4 baseline content label."""
    contains_speech = _truthy(evidence.get("contains_speech"))
    contains_music = _truthy(evidence.get("contains_music"))
    contains_vocals = _truthy(evidence.get("contains_vocals"))
    contains_environment_noise = _truthy(evidence.get("contains_environment_noise"))
    contains_target_noise = _truthy(evidence.get("contains_target_noise"))
    target_noise_label = str(evidence.get("target_noise_label") or "").strip()

    if _is_low_confidence(evidence):
        return "unknown_mixed"
    if contains_speech and contains_music:
        return "unknown_mixed"
    if contains_speech and contains_target_noise and target_noise_label in SUPPORTED_TARGET_NOISE_LABELS:
        return "speech_target_noise"
    if contains_speech and _is_noisy(evidence):
        return "speech_noisy_general"
    if contains_speech and not _is_noisy(evidence) and not contains_music:
        return "speech_clean"
    if contains_music and contains_vocals:
        return "music_with_vocals"
    if contains_music and not contains_vocals:
        return "music_instrumental"
    if contains_environment_noise and not contains_speech and not contains_music:
        return "environment_only"
    return "unknown_mixed"


def validate_manifest_rows(rows: list[dict[str, str]]) -> list[str]:
    """Validate Audio Router V4 manifest rows and return human-readable errors."""
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        missing_columns = [column for column in MANIFEST_V4_COLUMNS if column not in row]
        if missing_columns:
            errors.append(f"row {index}: missing columns: {', '.join(missing_columns)}")
            continue

        content_label = row["content_label"].strip()
        route_target = row["route_target"].strip()
        engine_target = row["engine_target"].strip()
        fallback_route_target = row["fallback_route_target"].strip()
        fallback_engine_target = row["fallback_engine_target"].strip()
        candidate_tasks = [task for task in row["candidate_tasks"].split(";") if task]

        if content_label not in CONTENT_LABELS:
            errors.append(f"row {index}: invalid content_label: {content_label}")
        if route_target not in ROUTE_TARGETS:
            errors.append(f"row {index}: invalid route_target: {route_target}")
        if engine_target not in ENGINE_TARGETS:
            errors.append(f"row {index}: invalid engine_target: {engine_target}")
        if fallback_route_target == "none":
            errors.append(f"row {index}: fallback_route_target must be empty, not none")
        if fallback_route_target and fallback_route_target not in ROUTE_TARGETS:
            errors.append(f"row {index}: invalid fallback_route_target: {fallback_route_target}")
        if fallback_engine_target and fallback_engine_target not in ENGINE_TARGETS:
            errors.append(f"row {index}: invalid fallback_engine_target: {fallback_engine_target}")
        for task in candidate_tasks:
            if task not in VALID_TASK_IDS:
                errors.append(f"row {index}: invalid candidate task: {task}")
        if content_label in CONTENT_LABELS:
            expected_mapping = route_mapping_for_content_label(content_label)
            for field in (
                "route_target",
                "engine_target",
                "candidate_tasks",
                "fallback_route_target",
                "fallback_engine_target",
            ):
                actual = row[field].strip()
                expected = expected_mapping[field]
                if actual != expected:
                    errors.append(
                        f"row {index}: {field} mismatch for {content_label}: expected {expected!r}, got {actual!r}"
                    )
        if content_label == "speech_target_noise":
            target_noise_label = row["target_noise_label"].strip()
            if target_noise_label not in SUPPORTED_TARGET_NOISE_LABELS:
                errors.append(f"row {index}: speech_target_noise has unsupported target_noise_label: {target_noise_label}")
            if not _truthy(row["contains_target_noise"]):
                errors.append(f"row {index}: speech_target_noise must have contains_target_noise true")
        if content_label == "environment_only" and route_target != "out_of_scope":
            errors.append(f"row {index}: environment_only must map to out_of_scope")
    return errors


def validate_no_split_leakage(rows: list[dict[str, str]]) -> list[str]:
    """Validate simple split leakage constraints for source ids and groups."""
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        split = row.get("split", "").strip()
        if split not in VALID_SPLITS:
            errors.append(f"row {index}: invalid split: {split}")

    for field in ("clean_source_id", "noise_source_id", "group_id"):
        splits_by_value: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = row.get(field, "").strip()
            split = row.get("split", "").strip()
            if value:
                splits_by_value[value].add(split)
        for value, splits in sorted(splits_by_value.items()):
            valid_splits = splits & VALID_SPLITS
            if len(valid_splits) > 1:
                errors.append(f"{field} {value} appears in multiple splits: {', '.join(sorted(valid_splits))}")
    return errors


def _parse_notes(notes: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+)\s*=\s*([^;]+)", notes):
        parsed[key.strip()] = value.strip()
    return parsed


def parse_target_noise_filename(input_path: str, target_noise_label: str = "") -> tuple[str, str]:
    """Parse target-noise class and source clip id from a generated mixed filename."""
    stem = Path(input_path).stem
    labels = [target_noise_label] if target_noise_label else SUPPORTED_TARGET_NOISE_LABELS
    for label in labels:
        if label not in SUPPORTED_TARGET_NOISE_LABELS:
            continue
        marker = f"_{label}_"
        start = stem.find(marker)
        if start < 0:
            continue
        noise_start = start + len(marker)
        snr_marker = stem.find("_snr", noise_start)
        if snr_marker < 0:
            continue
        noise_source_id = stem[noise_start:snr_marker].strip("_")
        if noise_source_id and noise_source_id not in SUPPORTED_TARGET_NOISE_LABELS:
            return label, noise_source_id
    return target_noise_label, ""


def _contains_flags_for_label(content_label: str, target_noise_label: str) -> dict[str, str]:
    return {
        "contains_speech": "true" if content_label.startswith("speech_") else "false",
        "contains_music": "true" if content_label.startswith("music_") else "false",
        "contains_vocals": "true" if content_label == "music_with_vocals" else "false",
        "contains_environment_noise": "true" if content_label in {"environment_only", "speech_noisy_general", "speech_target_noise"} else "false",
        "contains_target_noise": "true" if content_label == "speech_target_noise" and target_noise_label in SUPPORTED_TARGET_NOISE_LABELS else "false",
    }


def _empty_v4_row() -> dict[str, str]:
    return {column: "" for column in MANIFEST_V4_COLUMNS}


def convert_v3_row_to_v4(
    row: dict[str, str],
    *,
    source_dataset: str,
    taxonomy_version: str,
    features_version: str,
    normalization_version: str,
) -> dict[str, str]:
    """Convert one current router manifest row to the V4 schema."""
    source = (row.get("source") or source_dataset).strip()
    notes = row.get("notes", "").strip()
    parsed_notes = _parse_notes(notes)
    content_label = _content_label_for_source(source)
    sample_id = row.get("sample_id", "").strip()
    input_path = row.get("input_path", "").strip()
    target_noise_label = parsed_notes.get("noise_label", "")
    parsed_label, parsed_noise_source_id = parse_target_noise_filename(input_path, target_noise_label)
    if not target_noise_label:
        target_noise_label = parsed_label
    if content_label == "speech_target_noise" and target_noise_label not in SUPPORTED_TARGET_NOISE_LABELS:
        content_label = "unknown_mixed"

    route_fields = route_mapping_for_content_label(content_label)
    v4_row = _empty_v4_row()
    v4_row.update(route_fields)
    v4_row.update(_contains_flags_for_label(content_label, target_noise_label))
    v4_row.update(
        {
            "sample_id": sample_id,
            "input_path": input_path,
            "source_dataset": source,
            "split": row.get("split", "").strip() or "train",
            "split_key": f"{source}:{sample_id}",
            "group_id": f"{source}:{sample_id}",
            "taxonomy_version": taxonomy_version,
            "normalization_version": normalization_version,
            "source_fold": parsed_notes.get("fold", ""),
            "original_split": parsed_notes.get("original_split", "")
            or parsed_notes.get("dataset_split", "")
            or parsed_notes.get("subset", ""),
            "content_label": content_label,
            "label_method": "v3_source_mapping",
            "label_confidence": "1.0" if content_label != "unknown_mixed" else "0.5",
            "target_noise_label": target_noise_label,
            "snr_db": parsed_notes.get("snr_db", ""),
            "is_synthetic": "true" if _is_target_noise_source(source) else "false",
            "features_version": features_version,
            "feature_status": "pending",
            "notes": notes,
        }
    )
    if _is_target_noise_source(source):
        v4_row["clean_source_id"] = sample_id
        v4_row["noise_source_id"] = parsed_noise_source_id
    return {column: v4_row.get(column, "") for column in MANIFEST_V4_COLUMNS}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def write_v4_manifest(rows: list[dict[str, str]], output_manifest: Path) -> Path:
    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_V4_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path.resolve()


def build_audio_router_v4_manifest(
    *,
    input_manifest: Path,
    output_manifest: Path,
    source_dataset: str,
    taxonomy_version: str = "audio-router-v4",
    features_version: str = "pending",
    normalization_version: str = "pending",
) -> Path:
    """Convert a transitional v3-style router manifest to V4."""
    rows = [
        convert_v3_row_to_v4(
            row,
            source_dataset=source_dataset,
            taxonomy_version=taxonomy_version,
            features_version=features_version,
            normalization_version=normalization_version,
        )
        for row in read_csv_rows(input_manifest)
    ]
    validation_errors = validate_manifest_rows(rows) + validate_no_split_leakage(rows)
    if validation_errors:
        raise ValueError("; ".join(validation_errors))
    return write_v4_manifest(rows, output_manifest)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an Audio Router V4 manifest.")
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--taxonomy-version", default="audio-router-v4")
    parser.add_argument("--features-version", default="pending")
    parser.add_argument("--normalization-version", default="pending")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        output_path = build_audio_router_v4_manifest(
            input_manifest=args.input_manifest,
            output_manifest=args.output_manifest,
            source_dataset=args.source_dataset,
            taxonomy_version=args.taxonomy_version,
            features_version=args.features_version,
            normalization_version=args.normalization_version,
        )
        print(f"Wrote Audio Router V4 manifest: {output_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
