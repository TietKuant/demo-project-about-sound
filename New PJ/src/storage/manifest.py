"""Helpers for serializing minimal run manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_run_manifest(
    *,
    input_path: str,
    inferred_input_type: str,
    planned_output_paths: dict[str, str | None],
    selected_engine: str,
    stage_statuses: dict[str, str],
    dry_run: bool,
    final_status: str,
) -> dict[str, Any]:
    """Build a JSON-friendly run summary artifact."""
    return {
        "input_path": input_path,
        "inferred_input_type": inferred_input_type,
        "planned_output_paths": planned_output_paths,
        "selected_engine": selected_engine,
        "stage_statuses": stage_statuses,
        "dry_run": dry_run,
        "final_status": final_status,
    }


def write_manifest(manifest_path: str | Path, payload: dict[str, Any]) -> Path:
    """Write the run summary artifact to disk."""
    path = Path(manifest_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

