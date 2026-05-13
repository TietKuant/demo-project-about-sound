"""Lightweight checks for baseline output artifacts."""

from __future__ import annotations

from pathlib import Path


def output_exists(path: str | Path) -> bool:
    """Return True when a path exists on disk."""
    return Path(path).exists()


def output_is_non_empty(path: str | Path) -> bool:
    """Return True when a file exists and has non-zero size."""
    target = Path(path)
    return target.exists() and target.is_file() and target.stat().st_size > 0


def evaluate_output_artifact(path: str | Path) -> dict[str, bool]:
    """Run the minimal demo-only checks against an output artifact."""
    return {
        "output_exists": output_exists(path),
        "output_is_non_empty": output_is_non_empty(path),
    }


def evaluate_manifest(path: str | Path) -> dict[str, bool]:
    """Backward-compatible alias for older scaffold callers."""
    return evaluate_output_artifact(path)
