"""ffmpeg arnndn engine adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.engine.base import DenoiseEngine
from src.media.ffmpeg_wrapper import FFmpegWrapper


@dataclass(slots=True)
class FFmpegArnndnEngine(DenoiseEngine):
    """Denoise engine adapter backed by ffmpeg's arnndn filter."""

    ffmpeg_wrapper: FFmpegWrapper
    loaded: bool = False

    @property
    def name(self) -> str:
        return "ffmpeg-arnndn"

    def load(self) -> None:
        """Validate arnndn resources before running denoise."""
        self.ffmpeg_wrapper._build_arnndn_filter()
        self.loaded = True

    def denoise(self, input_audio_path: str | Path, output_audio_path: str | Path) -> Path:
        """Create a denoised WAV artifact at the requested output path."""
        if not self.loaded:
            raise RuntimeError("Engine must be loaded before denoise() is called.")

        source_audio = Path(input_audio_path).resolve()
        denoised_audio_path = Path(output_audio_path).resolve()
        denoised_audio_path.parent.mkdir(parents=True, exist_ok=True)
        arnndn_filter = self.ffmpeg_wrapper._build_arnndn_filter()
        command = [
            self.ffmpeg_wrapper.ffmpeg_binary,
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(source_audio),
            "-af",
            arnndn_filter,
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "48000",
            str(denoised_audio_path),
        ]
        self.ffmpeg_wrapper._run_ffmpeg(command, action="denoise_audio")
        return denoised_audio_path
