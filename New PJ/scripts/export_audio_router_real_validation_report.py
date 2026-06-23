"""Export a merged Audio Router real-audio validation report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROBABILITY_FIELDS = [
    "prob_environment_only",
    "prob_music_with_vocals",
    "prob_speech_clean",
    "prob_speech_target_noise",
]
FEATURE_FIELDS = [
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
OUTPUT_COLUMNS = [
    "input_path",
    "expected_group",
    "status",
    "predicted_label",
    "confidence",
    "confidence_threshold",
    "accepted",
    "route_target",
    "engine_target",
    "recommended_task",
    "decision_reason",
    "warnings",
    *PROBABILITY_FIELDS,
    *FEATURE_FIELDS,
    "error",
]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data if isinstance(data, dict) else {}


def _path_keys(path_value: str) -> set[str]:
    if not path_value:
        return set()
    path = Path(path_value)
    keys = {path_value, path.as_posix(), str(path)}
    try:
        keys.add(str(path.expanduser().resolve()))
    except OSError:
        pass
    return keys


def _load_summaries_by_input(summaries_dir: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    if not summaries_dir.exists():
        return summaries
    for summary_path in sorted(summaries_dir.rglob("*.json")):
        try:
            summary = _read_json(summary_path)
        except (OSError, json.JSONDecodeError):
            continue
        for key in _path_keys(str(summary.get("input_path", ""))):
            summaries[key] = summary
    return summaries


def _summary_for_row(row: dict[str, str], summaries_by_input: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for key in _path_keys(row.get("input_path", "")):
        if key in summaries_by_input:
            return summaries_by_input[key]
    return None


def _format_bool(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value).lower() if value not in (None, "") else ""


def _format_float(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.10f}"
    except (TypeError, ValueError):
        return str(value)


def _format_warnings(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return "" if value is None else str(value)


def _expected_group(input_path: str) -> str:
    return Path(input_path).parent.name if input_path else ""


def _merged_row(row: dict[str, str], summary: dict[str, Any] | None) -> dict[str, str]:
    probabilities = summary.get("probabilities", {}) if summary else {}
    features = summary.get("features", {}) if summary else {}
    merged = {
        "input_path": row.get("input_path", ""),
        "expected_group": _expected_group(row.get("input_path", "")),
        "status": str(summary.get("status", row.get("status", ""))) if summary else row.get("status", ""),
        "predicted_label": str(summary.get("predicted_label", row.get("predicted_label", ""))) if summary else row.get("predicted_label", ""),
        "confidence": _format_float(summary.get("confidence", row.get("confidence", ""))) if summary else row.get("confidence", ""),
        "confidence_threshold": _format_float(summary.get("confidence_threshold", row.get("confidence_threshold", "")))
        if summary
        else row.get("confidence_threshold", ""),
        "accepted": _format_bool(summary.get("accepted", row.get("accepted", ""))) if summary else row.get("accepted", ""),
        "route_target": str(summary.get("route_target", row.get("route_target", ""))) if summary else row.get("route_target", ""),
        "engine_target": str(summary.get("engine_target", row.get("engine_target", ""))) if summary else row.get("engine_target", ""),
        "recommended_task": str(summary.get("recommended_task", row.get("recommended_task", "")) or "") if summary else row.get("recommended_task", ""),
        "decision_reason": str(summary.get("decision_reason", row.get("decision_reason", ""))) if summary else row.get("decision_reason", ""),
        "warnings": _format_warnings(summary.get("warnings", row.get("warnings", ""))) if summary else row.get("warnings", ""),
        "error": str(summary.get("error", row.get("error", ""))) if summary else row.get("error", ""),
    }
    for field in PROBABILITY_FIELDS:
        label = field.removeprefix("prob_")
        merged[field] = _format_float(probabilities.get(label, row.get(field, ""))) if summary else row.get(field, "")
    for field in FEATURE_FIELDS:
        merged[field] = _format_float(features.get(field, "")) if summary else ""
    return merged


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _counter_table(counter: Counter[str]) -> str:
    if not counter:
        return "- none"
    return "\n".join(f"- {key or '(blank)'}: {count}" for key, count in sorted(counter.items()))


def _matrix_lines(rows: list[dict[str, str]]) -> list[str]:
    labels = sorted({row["predicted_label"] for row in rows if row["predicted_label"]})
    groups = sorted({row["expected_group"] for row in rows if row["expected_group"]})
    if not labels or not groups:
        return ["No expected_group vs predicted_label matrix available."]
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        counts[(row["expected_group"], row["predicted_label"])] += 1
    lines = ["| expected_group | " + " | ".join(labels) + " |", "| --- | " + " | ".join("---" for _ in labels) + " |"]
    for group in groups:
        values = [str(counts[(group, label)]) for label in labels]
        lines.append("| " + group + " | " + " | ".join(values) + " |")
    return lines


def _is_accepted(row: dict[str, str]) -> bool:
    return row.get("accepted", "").strip().lower() == "true"


def _dangerous_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    dangerous = []
    for row in rows:
        if not _is_accepted(row):
            continue
        expected_group = row.get("expected_group", "")
        route_target = row.get("route_target", "")
        recommended_task = row.get("recommended_task", "")
        if expected_group in {"environment_only", "hard_cases"} and route_target == "target_noise_suppression":
            dangerous.append(row)
        elif expected_group == "speech_clean" and (
            route_target == "target_noise_suppression"
            or (route_target == "manual_required" and recommended_task == "extract_vocals")
        ):
            dangerous.append(row)
    return dangerous


def _write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(row["status"] for row in rows)
    accepted_counts = Counter("accepted" if _is_accepted(row) else "rejected" for row in rows)
    predicted_counts = Counter(row["predicted_label"] for row in rows)
    accepted_rows = [row for row in rows if _is_accepted(row)]
    dangerous = _dangerous_rows(rows)

    lines = [
        "# Audio Router Real-Audio Validation",
        "",
        "This report summarizes local real-audio validation without exposing private raw audio.",
        "",
        "## Summary",
        f"- Total files: {len(rows)}",
        "",
        "## Status Counts",
        _counter_table(status_counts),
        "",
        "## Accepted / Rejected Counts",
        _counter_table(accepted_counts),
        "",
        "## Predicted-Label Counts",
        _counter_table(predicted_counts),
        "",
        "## expected_group vs predicted_label",
        *_matrix_lines(rows),
        "",
        "## Accepted Rows",
    ]
    if accepted_rows:
        lines.extend(
            f"- {Path(row['input_path']).name}: expected={row['expected_group']}, predicted={row['predicted_label']}, "
            f"route={row['route_target']}, confidence={row['confidence']}"
            for row in accepted_rows
        )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Dangerous Route Analysis",
            f"- Dangerous accepted routes: {len(dangerous)}",
        ]
    )
    if dangerous:
        lines.extend(
            f"- {Path(row['input_path']).name}: expected={row['expected_group']}, route={row['route_target']}, "
            f"recommended={row['recommended_task']}, confidence={row['confidence']}"
            for row in dangerous
        )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Conclusion",
            "V4 is an experimental baseline. Real-audio validation exposed domain shift and "
            "overconfident misclassification. The 0.90 threshold makes runtime conservative but "
            "does not fix model generalization. V5 should use better taxonomy and pretrained embeddings.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def export_audio_router_real_validation_report(
    *,
    batch_csv: Path,
    summaries_dir: Path,
    output_csv: Path,
    output_md: Path,
) -> tuple[Path, Path]:
    """Merge batch CSV rows with per-file router JSON summaries and write reports."""
    batch_rows = _read_csv_rows(Path(batch_csv))
    summaries_by_input = _load_summaries_by_input(Path(summaries_dir))
    merged_rows = [_merged_row(row, _summary_for_row(row, summaries_by_input)) for row in batch_rows]
    _write_csv(Path(output_csv), merged_rows)
    _write_markdown(Path(output_md), merged_rows)
    return Path(output_csv), Path(output_md)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Audio Router real-audio validation report.")
    parser.add_argument("--batch-csv", required=True, type=Path)
    parser.add_argument("--summaries-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    export_audio_router_real_validation_report(
        batch_csv=args.batch_csv,
        summaries_dir=args.summaries_dir,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )
    print(f"Wrote merged CSV: {args.output_csv}")
    print(f"Wrote markdown report: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
