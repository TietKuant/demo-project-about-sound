#!/usr/bin/env python3
"""Build Detection Evidence Dataset v1 from an audio-router manifest."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.extract_audio_features import _audio_for_features, _features
from scripts.run_guarded_router import run_guarded_router


MANIFEST_REQUIRED_COLUMNS = {"sample_id", "input_path", "content_label", "split"}
METADATA_COLUMNS = [
    "sample_id",
    "input_path",
    "source_dataset",
    "split",
    "group_id",
    "content_label",
    "workflow_label",
    "contains_speech",
    "contains_music",
    "contains_target_noise",
    "target_noise_label",
    "snr_db",
    "is_synthetic",
    "clean_source_id",
    "noise_source_id",
]
SIGNAL_FEATURE_COLUMNS = [
    "duration_sec",
    "rms_energy",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff_hz",
    "spectral_flatness",
    "low_band_energy_ratio",
    "mid_band_energy_ratio",
    "high_band_energy_ratio",
    "rms_std",
    "zcr_std",
    "silence_ratio",
]
ROUTER_LABELS = [
    "environment_only",
    "music_with_vocals",
    "speech_clean",
    "speech_noisy_general",
    "speech_target_noise",
]
ROUTER_COLUMNS = [
    "router_label",
    "router_confidence",
    "router_accepted",
    "router_decision_reason",
    *[f"router_prob_{label}" for label in ROUTER_LABELS],
]
SPEECH_GATE_COLUMNS = [
    "speech_gate_label",
    "speech_gate_confidence",
    "speech_gate_prob_non_speech",
    "speech_gate_prob_speech_present",
]
RANKING_COLUMNS = [
    "top_label",
    "top_score",
    "second_label",
    "second_score",
    "score_margin",
    "needs_review",
]
OUTPUT_COLUMNS = [
    *METADATA_COLUMNS,
    "status",
    "error",
    *SIGNAL_FEATURE_COLUMNS,
    *ROUTER_COLUMNS,
    *SPEECH_GATE_COLUMNS,
    *RANKING_COLUMNS,
]


def _load_manifest(manifest_path: Path, limit: int | None) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = MANIFEST_REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Detection evidence manifest is missing columns: "
                + ", ".join(sorted(missing))
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Detection evidence manifest is empty: {manifest_path}")
    return rows if limit is None else rows[:limit]


def _resolve_input_path(value: str, manifest_path: Path) -> Path:
    expanded = Path(value).expanduser()
    candidates = (
        expanded,
        manifest_path.parent / expanded,
        Path.cwd() / expanded,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (manifest_path.parent / expanded).resolve(strict=False)


def _blank_output_row(manifest_row: Mapping[str, str], input_path: Path) -> dict[str, str]:
    row = {column: "" for column in OUTPUT_COLUMNS}
    for column in METADATA_COLUMNS:
        row[column] = str(manifest_row.get(column, "") or "")
    row["input_path"] = str(input_path)
    row["status"] = "failed"
    row["needs_review"] = "true"
    return row


def _format_float(value: object) -> str:
    return f"{float(value):.10f}"


def _rank_probabilities(
    probabilities: Mapping[str, object],
) -> tuple[str, str, str, str, str]:
    scored = sorted(
        (
            (str(label), float(score))
            for label, score in probabilities.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not scored:
        return "", "", "", "", ""
    top_label, top_score = scored[0]
    if len(scored) == 1:
        return top_label, _format_float(top_score), "", "", ""
    second_label, second_score = scored[1]
    return (
        top_label,
        _format_float(top_score),
        second_label,
        _format_float(second_score),
        _format_float(top_score - second_score),
    )


def _needs_review(
    row: Mapping[str, str],
    *,
    router_threshold: float,
) -> bool:
    if row.get("status") != "success":
        return True
    router_label = str(row.get("router_label") or "")
    router_accepted = str(row.get("router_accepted") or "").lower() == "true"
    content_label = str(row.get("content_label") or "")
    if router_accepted and router_label and router_label != content_label:
        return True
    margin = str(row.get("score_margin") or "")
    if margin and float(margin) < 0.15:
        return True
    confidence = str(row.get("router_confidence") or "")
    if confidence and float(confidence) < router_threshold:
        return True
    return False


def build_detection_evidence_dataset(
    *,
    manifest_path: str | Path,
    output_csv: str | Path,
    router_checkpoint: str | Path | None = None,
    speech_gate_checkpoint: str | Path | None = None,
    router_threshold: float = 0.70,
    speech_threshold: float = 0.70,
    limit: int | None = None,
    skip_missing: bool = False,
) -> Path:
    """Build signal and optional detector evidence without training."""
    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    if (router_checkpoint is None) != (speech_gate_checkpoint is None):
        raise ValueError(
            "--router-checkpoint and --speech-gate-checkpoint must be provided together."
        )
    rows = _load_manifest(manifest, limit)
    output = Path(output_csv).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_rows: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="detection-evidence-") as temp_name:
        temp_dir = Path(temp_name)
        for manifest_row in rows:
            input_path = _resolve_input_path(manifest_row["input_path"], manifest)
            row = _blank_output_row(manifest_row, input_path)
            if not input_path.is_file():
                row["status"] = "skipped" if skip_missing else "failed"
                row["error"] = f"Input file not found: {input_path}"
                evidence_rows.append(row)
                continue
            try:
                audio, sample_rate = _audio_for_features(input_path, temp_dir)
                features = _features(audio, sample_rate)
                row["status"] = "success"
                row["duration_sec"] = _format_float(audio.size / sample_rate)
                for column in SIGNAL_FEATURE_COLUMNS:
                    if column != "duration_sec":
                        row[column] = _format_float(features[column])

                if router_checkpoint is not None and speech_gate_checkpoint is not None:
                    guarded = run_guarded_router(
                        input_path=input_path,
                        router_checkpoint=Path(router_checkpoint),
                        speech_gate_checkpoint=Path(speech_gate_checkpoint),
                        router_threshold=router_threshold,
                        speech_threshold=speech_threshold,
                    )
                    row["router_label"] = str(guarded.get("router_label") or "")
                    confidence = guarded.get("router_confidence")
                    row["router_confidence"] = (
                        _format_float(confidence) if confidence is not None else ""
                    )
                    row["router_accepted"] = str(
                        guarded.get("router_accepted") is True
                    ).lower()
                    row["router_decision_reason"] = str(
                        guarded.get("reason") or ""
                    )
                    router_probabilities = dict(
                        guarded.get("router_probabilities") or {}
                    )
                    for label in ROUTER_LABELS:
                        value = router_probabilities.get(label)
                        row[f"router_prob_{label}"] = (
                            _format_float(value) if value is not None else ""
                        )
                    row["speech_gate_label"] = str(
                        guarded.get("speech_gate_label") or ""
                    )
                    gate_confidence = guarded.get("speech_gate_confidence")
                    row["speech_gate_confidence"] = (
                        _format_float(gate_confidence)
                        if gate_confidence is not None
                        else ""
                    )
                    gate_probabilities = dict(
                        guarded.get("speech_gate_probabilities") or {}
                    )
                    for label in ("non_speech", "speech_present"):
                        value = gate_probabilities.get(label)
                        row[f"speech_gate_prob_{label}"] = (
                            _format_float(value) if value is not None else ""
                        )
                    (
                        row["top_label"],
                        row["top_score"],
                        row["second_label"],
                        row["second_score"],
                        row["score_margin"],
                    ) = _rank_probabilities(router_probabilities)
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
            row["needs_review"] = str(
                _needs_review(row, router_threshold=router_threshold)
            ).lower()
            evidence_rows.append(row)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(evidence_rows)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Detection Evidence Dataset v1."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--router-checkpoint", type=Path)
    parser.add_argument("--speech-gate-checkpoint", type=Path)
    parser.add_argument("--router-threshold", default=0.70, type=float)
    parser.add_argument("--speech-threshold", default=0.70, type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-missing", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        output = build_detection_evidence_dataset(
            manifest_path=args.manifest,
            output_csv=args.output_csv,
            router_checkpoint=args.router_checkpoint,
            speech_gate_checkpoint=args.speech_gate_checkpoint,
            router_threshold=args.router_threshold,
            speech_threshold=args.speech_threshold,
            limit=args.limit,
            skip_missing=args.skip_missing,
        )
        print(f"Wrote detection evidence CSV: {output}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
