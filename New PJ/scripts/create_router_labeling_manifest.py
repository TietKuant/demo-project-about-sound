#!/usr/bin/env python3
"""Create a human-labeling manifest for future audio router training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.extract_audio_features import extract_audio_features
from src.io.paths import infer_input_type


OUTPUT_COLUMNS = [
    "sample_id",
    "path",
    "filename",
    "file_id",
    "router_label",
    "router_confidence",
    "router_accepted",
    "router_reason",
    "source",
    "duration_sec",
    "rms_energy",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "suggested_label",
    "human_label",
    "workflow_label",
    "split",
    "notes",
]

ALLOWED_HUMAN_LABELS = frozenset(
    {
        "speech_clean",
        "speech_noisy_general",
        "speech_target_noise",
        "music_with_vocals",
        "environment_only",
        "unknown_mixed",
    }
)

ALLOWED_WORKFLOW_LABELS = frozenset(
    {
        "speech_cleanup",
        "music_separation_package",
        "no_process",
        "target_noise_guard",
        "safe_abstain",
    }
)

MEDIA_SUFFIXES = frozenset(
    {
        ".aac",
        ".aif",
        ".aiff",
        ".avi",
        ".flac",
        ".m4a",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".ogg",
        ".opus",
        ".wav",
        ".webm",
        ".wma",
    }
)

PATH_COLUMNS = ("path", "input_path", "mixed_path", "noisy_path")
SUGGESTED_LABEL_COLUMNS = (
    "suggested_label",
    "predicted_label",
    "router_label",
    "content_label",
)


def stable_sample_id(path: str | Path) -> str:
    """Return a deterministic identifier derived from a normalized path."""
    normalized = str(Path(path).expanduser().resolve(strict=False))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    return f"router_{digest}"


def _normalized_path(path: str | Path, *, base_dir: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    return candidate.resolve(strict=False)


def _is_media_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES


def _first_value(row: Mapping[str, str], columns: Iterable[str]) -> str:
    for column in columns:
        value = (row.get(column) or "").strip()
        if value:
            return value
    return ""


def _valid_or_blank(value: str, allowed: frozenset[str]) -> str:
    normalized = value.strip()
    return normalized if normalized in allowed else ""


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _api_upload_metadata(path: Path) -> dict[str, str]:
    fields = {
        "filename": path.name,
        "file_id": "",
        "router_label": "",
        "router_confidence": "",
        "router_accepted": "",
        "router_reason": "",
        "suggested_label": "",
    }
    metadata_path = path.parent / "metadata.json"
    if not metadata_path.is_file():
        return fields
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return fields
    if not isinstance(metadata, dict):
        return fields

    router = metadata.get("router")
    if not isinstance(router, dict):
        router = {}
    router_label = _csv_value(router.get("predicted_label")).strip()
    fields.update(
        {
            "filename": _csv_value(metadata.get("filename")).strip() or path.name,
            "file_id": _csv_value(metadata.get("file_id")).strip(),
            "router_label": router_label,
            "router_confidence": _csv_value(router.get("confidence")).strip(),
            "router_accepted": _csv_value(router.get("accepted")).strip(),
            "router_reason": _csv_value(router.get("decision_reason")).strip(),
            "suggested_label": _valid_or_blank(
                router_label,
                ALLOWED_HUMAN_LABELS,
            ),
        }
    )
    return fields


def _load_existing_candidates(manifest_paths: Sequence[Path]) -> dict[Path, dict[str, str]]:
    candidates: dict[Path, dict[str, str]] = {}
    for manifest_path in manifest_paths:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            for row in reader:
                raw_path = _first_value(row, PATH_COLUMNS)
                if not raw_path:
                    continue
                path = _normalized_path(raw_path, base_dir=manifest_path.parent)
                if not _is_media_file(path):
                    continue
                suggested_label = _valid_or_blank(
                    _first_value(row, SUGGESTED_LABEL_COLUMNS),
                    ALLOWED_HUMAN_LABELS,
                )
                candidates.setdefault(
                    path,
                    {
                        "filename": (row.get("filename") or path.name).strip(),
                        "file_id": (row.get("file_id") or "").strip(),
                        "router_label": (row.get("router_label") or "").strip(),
                        "router_confidence": (
                            row.get("router_confidence") or ""
                        ).strip(),
                        "router_accepted": (
                            row.get("router_accepted") or ""
                        ).strip(),
                        "router_reason": (row.get("router_reason") or "").strip(),
                        "source": (row.get("source") or manifest_path.stem).strip(),
                        "suggested_label": suggested_label,
                        "human_label": _valid_or_blank(
                            row.get("human_label") or "",
                            ALLOWED_HUMAN_LABELS,
                        ),
                        "workflow_label": _valid_or_blank(
                            row.get("workflow_label") or "",
                            ALLOWED_WORKFLOW_LABELS,
                        ),
                        "split": (row.get("split") or "").strip(),
                        "notes": (row.get("notes") or "").strip(),
                    },
                )
    return candidates


def _discover_directory_candidates(
    input_dirs: Sequence[Path],
    candidates: dict[Path, dict[str, str]],
) -> None:
    for input_dir in input_dirs:
        resolved_dir = input_dir.expanduser().resolve(strict=True)
        if not resolved_dir.is_dir():
            raise ValueError(f"Input directory is not a directory: {input_dir}")
        for path in sorted(resolved_dir.rglob("*")):
            if not _is_media_file(path):
                continue
            resolved_path = path.resolve()
            metadata = _api_upload_metadata(resolved_path)
            existing = candidates.get(resolved_path)
            if existing is not None:
                for field in (
                    "filename",
                    "file_id",
                    "router_label",
                    "router_confidence",
                    "router_accepted",
                    "router_reason",
                ):
                    if metadata[field] and not existing.get(field):
                        existing[field] = metadata[field]
                if metadata["suggested_label"]:
                    existing["suggested_label"] = metadata["suggested_label"]
                continue
            candidates[resolved_path] = {
                **metadata,
                "source": resolved_dir.name,
                "human_label": "",
                "workflow_label": "",
                "split": "",
                "notes": "",
            }


def _extract_feature_rows(
    candidates: Mapping[Path, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    if not candidates:
        return {}

    with tempfile.TemporaryDirectory(prefix="router-labeling-") as temp_dir:
        temp_path = Path(temp_dir)
        extraction_manifest = temp_path / "inputs.csv"
        feature_output = temp_path / "features.csv"
        fieldnames = [
            "sample_id",
            "category",
            "input_path",
            "input_type",
            "language",
            "expected_task",
            "has_clean_reference",
            "notes",
        ]
        with extraction_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for path in sorted(candidates, key=str):
                writer.writerow(
                    {
                        "sample_id": stable_sample_id(path),
                        "category": "router_labeling",
                        "input_path": str(path),
                        "input_type": infer_input_type(path),
                        "language": "",
                        "expected_task": "auto",
                        "has_clean_reference": "false",
                        "notes": candidates[path].get("notes", ""),
                    }
                )

        extract_audio_features(
            manifest_path=extraction_manifest,
            output_path=feature_output,
            skip_missing=True,
        )
        with feature_output.open(newline="", encoding="utf-8") as handle:
            return {
                row["sample_id"]: row
                for row in csv.DictReader(handle)
                if row.get("sample_id")
            }


def _router_suggestion(path: Path, checkpoint_path: Path | None) -> str:
    if checkpoint_path is None or not checkpoint_path.is_file():
        return ""
    try:
        from scripts.run_audio_router import run_audio_router

        with tempfile.TemporaryDirectory(prefix="router-suggestion-") as temp_dir:
            summary_path = Path(temp_dir) / "summary.csv"
            result = run_audio_router(
                checkpoint_path=checkpoint_path,
                input_path=path,
                output_summary=summary_path,
            )
        label = str(result.get("predicted_label") or "").strip()
        return label if label in ALLOWED_HUMAN_LABELS else ""
    except Exception:
        return ""


def create_router_labeling_manifest(
    input_dirs: Sequence[str | Path],
    output_path: str | Path,
    *,
    existing_manifests: Sequence[str | Path] | None = None,
    router_checkpoint: str | Path | None = None,
) -> Path:
    """Create a deduplicated feature manifest without copying source media."""
    directory_paths = [Path(path) for path in input_dirs]
    manifest_paths = [Path(path) for path in (existing_manifests or [])]
    if not directory_paths and not manifest_paths:
        raise ValueError(
            "At least one input directory or existing manifest is required."
        )
    candidates = _load_existing_candidates(manifest_paths)
    _discover_directory_candidates(directory_paths, candidates)
    feature_rows = _extract_feature_rows(candidates)

    checkpoint = (
        Path(router_checkpoint).expanduser()
        if router_checkpoint
        else (
            Path(os.environ["AUDIO_ROUTER_CHECKPOINT"]).expanduser()
            if os.environ.get("AUDIO_ROUTER_CHECKPOINT")
            else None
        )
    )
    rows: list[dict[str, str]] = []
    for path in sorted(candidates, key=str):
        sample_id = stable_sample_id(path)
        features = feature_rows.get(sample_id)
        if not features or features.get("status") != "success":
            continue
        existing = candidates[path]
        suggested_label = existing.get("suggested_label", "") or _router_suggestion(
            path,
            checkpoint,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "path": str(path),
                "filename": existing.get("filename", path.name),
                "file_id": existing.get("file_id", ""),
                "router_label": existing.get("router_label", ""),
                "router_confidence": existing.get("router_confidence", ""),
                "router_accepted": existing.get("router_accepted", ""),
                "router_reason": existing.get("router_reason", ""),
                "source": existing.get("source", ""),
                "duration_sec": features.get("duration_sec", ""),
                "rms_energy": features.get("rms_energy", ""),
                "zero_crossing_rate": features.get("zero_crossing_rate", ""),
                "spectral_centroid_hz": features.get("spectral_centroid_hz", ""),
                "spectral_bandwidth_hz": features.get("spectral_bandwidth_hz", ""),
                "suggested_label": suggested_label,
                "human_label": existing.get("human_label", ""),
                "workflow_label": existing.get("workflow_label", ""),
                "split": existing.get("split", ""),
                "notes": existing.get("notes", ""),
            }
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a human-labeling CSV for future router training."
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        dest="input_dirs",
        help="Directory to scan recursively; may be supplied more than once.",
    )
    parser.add_argument(
        "--existing-manifest",
        action="append",
        default=[],
        dest="existing_manifests",
        help="Existing CSV whose media paths and labels should be retained.",
    )
    parser.add_argument("--output", required=True, help="Output labeling CSV path.")
    parser.add_argument(
        "--router-checkpoint",
        help="Optional router checkpoint used to populate suggested_label.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.input_dirs and not args.existing_manifests:
        parser.error(
            "at least one --input-dir or --existing-manifest must be supplied"
        )
    output_path = create_router_labeling_manifest(
        args.input_dirs,
        args.output,
        existing_manifests=args.existing_manifests,
        router_checkpoint=args.router_checkpoint,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
