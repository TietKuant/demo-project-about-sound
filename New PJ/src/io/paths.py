"""Deterministic path helpers for dry-run pipeline planning."""

from __future__ import annotations

from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def validate_input_path(input_path: str | Path) -> Path:
    """Validate that the input path exists and points to a file."""
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")
    return path


def validate_output_dir(output_dir: str | Path) -> Path:
    """Return a normalized output directory path."""
    return Path(output_dir).expanduser().resolve()


def infer_input_type(input_path: str | Path) -> str:
    """Infer whether the file is audio or video based on extension."""
    suffix = Path(input_path).suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"Unsupported input file type: {suffix or '<no extension>'}")


def derive_output_mode(input_type: str) -> str:
    """Pick the default output mode for a given input type."""
    if input_type == "audio":
        return "audio"
    if input_type == "video":
        return "video"
    raise ValueError(f"Unsupported input type: {input_type}")


def derive_planned_output_path(input_path: str | Path, output_dir: str | Path, output_mode: str) -> Path:
    """Return the planned final media path without creating the file."""
    source = Path(input_path)
    target_dir = validate_output_dir(output_dir)
    if output_mode == "audio":
        return target_dir / f"{source.stem}.denoised.wav"
    if output_mode == "video":
        return target_dir / f"{source.stem}.denoised{source.suffix}"
    raise ValueError(f"Unsupported output mode: {output_mode}")


def derive_temp_audio_path(input_path: str | Path, output_dir: str | Path) -> Path:
    """Return the planned intermediate audio path in the sibling tmp directory."""
    source = Path(input_path)
    root_dir = validate_output_dir(output_dir).parent
    return root_dir / "tmp" / f"{source.stem}.prepared.wav"


def derive_manifest_path(input_path: str | Path, output_dir: str | Path) -> Path:
    """Return the manifest path for a dry-run pipeline execution."""
    source = Path(input_path)
    target_dir = validate_output_dir(output_dir)
    return target_dir / f"{source.stem}.manifest.json"
