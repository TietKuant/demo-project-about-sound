#!/usr/bin/env python3
"""Evaluate Detection Evidence Dataset v1 with dependency-light summaries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REVIEW_COLUMNS = [
    "input_path",
    "content_label",
    "split",
    "router_label",
    "router_confidence",
    "speech_gate_label",
    "speech_gate_confidence",
    "top_label",
    "top_score",
    "second_label",
    "second_score",
    "score_margin",
    "target_noise_label",
    "snr_db",
]


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _optional_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "content_label",
            "split",
            "router_label",
            "router_confidence",
            "router_accepted",
            "speech_gate_label",
            "needs_review",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Detection evidence CSV is missing columns: "
                + ", ".join(sorted(missing))
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Detection evidence CSV is empty: {path}")
    return rows


def _hard_error(row: Mapping[str, str], confidence_threshold: float) -> bool:
    confidence = _optional_float(row.get("router_confidence"))
    return (
        row.get("status") == "success"
        and _truthy(row.get("router_accepted"))
        and confidence is not None
        and confidence >= confidence_threshold
        and bool(row.get("router_label"))
        and row.get("router_label") != row.get("content_label")
    )


def _review_reasons(
    row: Mapping[str, str],
    *,
    confidence_threshold: float,
    margin_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    if row.get("status") != "success":
        reasons.append("status_not_success")
    if (
        _truthy(row.get("router_accepted"))
        and row.get("router_label")
        and row.get("router_label") != row.get("content_label")
    ):
        reasons.append("accepted_label_conflict")
    margin = _optional_float(row.get("score_margin"))
    if margin is not None and margin < margin_threshold:
        reasons.append("low_score_margin")
    confidence = _optional_float(row.get("router_confidence"))
    if confidence is not None and confidence < confidence_threshold:
        reasons.append("low_router_confidence")
    if _hard_error(row, confidence_threshold):
        reasons.append("hard_error")
    if _truthy(row.get("needs_review")) and not reasons:
        reasons.append("source_marked_review")
    return reasons


def evaluate_detection_evidence(
    *,
    evidence_csv: str | Path,
    output_json: str | Path | None = None,
    review_csv: str | Path | None = None,
    confidence_threshold: float = 0.70,
    margin_threshold: float = 0.15,
) -> dict[str, object]:
    """Summarize router and speech-gate evidence without model dependencies."""
    source = Path(evidence_csv).expanduser().resolve(strict=True)
    rows = _load_rows(source)
    content_labels = sorted({row["content_label"] for row in rows})
    router_labels = sorted(
        {row["router_label"] for row in rows if row.get("router_label")}
    )
    confusion = {
        label: {predicted: 0 for predicted in router_labels}
        for label in content_labels
    }
    speech_gate_counts: dict[str, Counter[str]] = defaultdict(Counter)
    snr_router_counts: dict[str, Counter[str]] = defaultdict(Counter)
    review_counts: Counter[str] = Counter()
    review_rows: list[dict[str, str]] = []
    hard_error_count = 0

    for row in rows:
        true_label = row["content_label"]
        predicted_label = row.get("router_label") or ""
        if predicted_label:
            confusion[true_label].setdefault(predicted_label, 0)
            confusion[true_label][predicted_label] += 1
        gate_label = row.get("speech_gate_label") or "missing"
        speech_gate_counts[true_label][gate_label] += 1
        if true_label == "speech_target_noise":
            snr_key = row.get("snr_db") or "unknown"
            snr_router_counts[snr_key][predicted_label or "missing"] += 1
        is_hard_error = _hard_error(row, confidence_threshold)
        hard_error_count += int(is_hard_error)
        reasons = _review_reasons(
            row,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
        )
        review_counts.update(reasons)
        if _truthy(row.get("needs_review")) or is_hard_error:
            review_rows.append(
                {column: row.get(column, "") for column in REVIEW_COLUMNS}
            )

    recall_by_label: dict[str, float] = {}
    for label in content_labels:
        total = sum(confusion[label].values())
        recall_by_label[label] = (
            float(confusion[label].get(label, 0) / total) if total else 0.0
        )

    report: dict[str, object] = {
        "row_count": len(rows),
        "counts_by_content_label": dict(
            sorted(Counter(row["content_label"] for row in rows).items())
        ),
        "counts_by_split": dict(
            sorted(Counter(row["split"] for row in rows).items())
        ),
        "router_confusion": confusion,
        "router_recall_by_label": recall_by_label,
        "speech_gate_by_content_label": {
            label: dict(sorted(counts.items()))
            for label, counts in sorted(speech_gate_counts.items())
        },
        "review_reason_counts": dict(sorted(review_counts.items())),
        "hard_error_count": hard_error_count,
        "speech_target_noise_by_snr_router_label": {
            snr: dict(sorted(counts.items()))
            for snr, counts in sorted(snr_router_counts.items())
        },
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
    }
    if output_json is not None:
        destination = Path(output_json).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
    if review_csv is not None:
        destination = Path(review_csv).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerows(review_rows)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Detection Evidence Dataset v1."
    )
    parser.add_argument("--evidence-csv", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--confidence-threshold", default=0.70, type=float)
    parser.add_argument("--margin-threshold", default=0.15, type=float)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        report = evaluate_detection_evidence(
            evidence_csv=args.evidence_csv,
            output_json=args.output_json,
            review_csv=args.review_csv,
            confidence_threshold=args.confidence_threshold,
            margin_threshold=args.margin_threshold,
        )
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
