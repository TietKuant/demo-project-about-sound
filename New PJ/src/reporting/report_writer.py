"""Write small JSON and Markdown artifacts for a restore run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_restore_report(
    *,
    report_dir: Path,
    input_path: Path,
    output_path: Path,
    engine_name: str,
    runtime_sec: float,
    audio_duration_sec: float | None,
    rtf: float | None,
    plot_paths: list[str],
    warning: str = "",
    error: str = "",
) -> tuple[Path, Path]:
    """Write restore run summary JSON and Markdown files."""
    output_dir = Path(report_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "engine_name": engine_name,
        "runtime_sec": runtime_sec,
        "audio_duration_sec": audio_duration_sec,
        "rtf": rtf,
        "plot_paths": plot_paths,
        "warning": warning,
        "error": error,
    }

    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    plot_lines = "\n".join(f"- `{path}`" for path in plot_paths) or "- None"
    markdown = (
        "# Restore Run Summary\n\n"
        f"- **Input:** `{input_path}`\n"
        f"- **Output:** `{output_path}`\n"
        f"- **Engine:** `{engine_name}`\n"
        f"- **Runtime:** `{runtime_sec:.6f}` sec\n"
        f"- **Audio duration:** `{'' if audio_duration_sec is None else f'{audio_duration_sec:.6f}'}` sec\n"
        f"- **RTF:** `{'' if rtf is None else f'{rtf:.6f}'}`\n"
        f"- **Warning:** `{warning}`\n"
        f"- **Error:** `{error}`\n\n"
        "## Plots\n\n"
        f"{plot_lines}\n"
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, markdown_path
