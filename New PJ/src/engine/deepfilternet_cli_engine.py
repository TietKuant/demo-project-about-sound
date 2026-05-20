"""DeepFilterNet CLI engine adapter."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.engine.base import DenoiseEngine


@dataclass(slots=True)
class DeepFilterNetCliEngine(DenoiseEngine):
    """Denoise engine adapter backed by the deepFilter CLI command."""

    command_name: str = "deepFilter"
    command_path: str | None = None
    loaded: bool = False

    @property
    def name(self) -> str:
        return "deepfilternet"

    def load(self) -> None:
        """Validate that the DeepFilterNet CLI command is available."""
        command_path = shutil.which(self.command_name)
        if command_path is None:
            raise RuntimeError("DeepFilterNet CLI command not found on PATH: deepFilter")
        self.command_path = command_path
        self.loaded = True

    def denoise(self, input_audio_path: str | Path, output_audio_path: str | Path) -> Path:
        """Run deepFilter and copy its generated WAV to the requested output path."""
        if not self.loaded or self.command_path is None:
            raise RuntimeError("Engine must be loaded before denoise() is called.")

        source_audio = Path(input_audio_path).resolve()
        denoised_audio_path = Path(output_audio_path).resolve()
        output_dir = denoised_audio_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        pattern = f"{source_audio.stem}_DeepFilterNet*.wav"
        existing_matches = {path: path.stat().st_mtime for path in output_dir.glob(pattern)}

        command = [
            self.command_path,
            str(source_audio),
            "-o",
            str(output_dir),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "").strip() or "unknown deepFilter error"
            raise RuntimeError(f"deepFilter failed during denoise: {details}") from exc

        candidates = []
        for path in output_dir.glob(pattern):
            modified_at = path.stat().st_mtime
            if path not in existing_matches or modified_at > existing_matches[path]:
                candidates.append(path)

        if not candidates:
            raise RuntimeError(f"deepFilter did not produce an output matching: {pattern}")

        generated_path = max(candidates, key=lambda path: path.stat().st_mtime)
        shutil.copy2(generated_path, denoised_audio_path)
        return denoised_audio_path
