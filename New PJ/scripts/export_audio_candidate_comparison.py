"""Export a human-labeled comparison of audio processing candidates."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


MANIFEST_COLUMNS = [
    "case_id",
    "input_path",
    "candidate_name",
    "candidate_path",
    "candidate_role",
    "human_rating",
    "human_notes",
    "is_selected",
    "reject_reason",
]
TRUE_VALUES = {"1", "true", "yes", "y"}


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate comparison manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Candidate comparison manifest is missing columns: {', '.join(sorted(missing))}")
        rows = [{column: row.get(column, "") for column in MANIFEST_COLUMNS} for row in reader]
    if not rows:
        raise ValueError("Candidate comparison manifest is empty.")
    return rows


def _resolve_reference(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(manifest_path).resolve().parent / path
    return path.resolve()


def _validate_rows(rows: list[dict[str, str]], manifest_path: Path) -> None:
    for index, row in enumerate(rows, start=1):
        if not row["case_id"].strip():
            raise ValueError(f"Candidate comparison manifest row {index} has an empty case_id.")
        for column in ("input_path", "candidate_path"):
            value = row[column].strip()
            if not value:
                raise ValueError(f"Candidate comparison manifest row {index} has an empty {column}.")
            resolved = _resolve_reference(value, manifest_path)
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"Candidate comparison manifest row {index} references missing {column}: {resolved}"
                )


def _is_selected(value: str) -> bool:
    return value.strip().casefold() in TRUE_VALUES


def _markdown_cell(value: str) -> str:
    return str(value or "").replace("|", r"\|").replace("\n", " ").strip() or "-"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    cases: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cases[row["case_id"]].append(row)

    lines = [
        "# Audio Candidate Comparison",
        "",
        "Human-labeled candidate comparison; no automated quality metric is computed.",
        "",
    ]
    for case_id, candidates in cases.items():
        lines.extend(
            [
                f"## {_markdown_cell(case_id)}",
                "",
                f"- Input: `{_markdown_cell(candidates[0]['input_path'])}`",
                "",
                "| Candidate | Role | Rating | Selected | Reject reason | Notes |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in candidates:
            selected = "yes" if _is_selected(row["is_selected"]) else "no"
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(row["candidate_name"]),
                        _markdown_cell(row["candidate_role"]),
                        _markdown_cell(row["human_rating"]),
                        selected,
                        _markdown_cell(row["reject_reason"]),
                        _markdown_cell(row["human_notes"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_audio_candidate_comparison(
    *,
    manifest_path: Path,
    output_dir: Path = Path("outputs/audio-candidate-comparison"),
) -> Path:
    """Validate candidate references and write CSV/Markdown comparison artifacts."""
    manifest = Path(manifest_path)
    rows = _read_manifest(manifest)
    _validate_rows(rows, manifest)

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "comparison.csv", rows)
    _write_markdown(output / "comparison.md", rows)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a human-labeled audio candidate comparison.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/audio-candidate-comparison"), type=Path)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        output_dir = export_audio_candidate_comparison(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
        )
        print(f"Wrote audio candidate comparison: {output_dir}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
