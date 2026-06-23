"""Thin wrapper that defines the media responsibilities of ffmpeg."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.io.paths import derive_planned_output_path, derive_temp_audio_path, infer_input_type


@dataclass(slots=True)
class MediaProbe:
    """Small metadata object returned by input probing."""

    input_path: Path
    input_type: str
    container_suffix: str


class FFmpegWrapper:
    """Execute local ffmpeg steps for probe, prepare, denoise, and export.

    This class owns the boundaries where ffmpeg integration lives:
    probing, audio preparation, cleaned audio export, and video remux.
    The current real denoise path uses ffmpeg's arnndn filter with a local model.
    """

    def __init__(
        self,
        ffmpeg_binary: str = "ffmpeg",
        arnndn_model_path: str | Path | None = None,
        arnndn_mix: float | None = None,
    ) -> None:
        self.ffmpeg_binary = ffmpeg_binary
        self.arnndn_model_path = (
            Path(arnndn_model_path)
            if arnndn_model_path is not None
            else Path(__file__).resolve().parents[2] / "models" / "arnndn" / "std.rnnn"
        )
        self.arnndn_mix = arnndn_mix

    def _run_ffmpeg(self, command: list[str], *, action: str) -> subprocess.CompletedProcess[str]:
        """Run an ffmpeg command and raise a clear error on failure."""
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required for media processing, but it was not found on PATH.") from exc
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "").strip() or "unknown ffmpeg error"
            raise RuntimeError(f"ffmpeg failed during {action}: {details}") from exc

    def _build_arnndn_filter(self) -> str:
        """Build the arnndn filter string for ffmpeg."""
        model_path = self.arnndn_model_path.resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"arnndn model file not found: {model_path}")

        escaped_model_path = model_path.as_posix().replace(":", "\\:")
        quoted_model_path = f"'{escaped_model_path}'"
        filter_parts = [f"arnndn=m={quoted_model_path}"]
        if self.arnndn_mix is not None:
            filter_parts.append(f"mix={self.arnndn_mix}")
        return ":".join(filter_parts)

    def probe_input(self, input_path: str | Path) -> MediaProbe:
        """Validate that ffmpeg can open the input and return minimal metadata."""
        path = Path(input_path).resolve()
        input_type = infer_input_type(path)
        command = [
            self.ffmpeg_binary,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ]
        self._run_ffmpeg(command, action="probe_input")
        return MediaProbe(
            input_path=path,
            input_type=input_type,
            container_suffix=path.suffix.lower(),
        )

    def prepare_audio(self, input_path: str | Path, output_dir: str | Path) -> Path:
        """Create a normalized working WAV file for the engine."""
        source = Path(input_path).resolve()
        prepared_audio_path = derive_temp_audio_path(source, output_dir)
        prepared_audio_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.ffmpeg_binary,
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(source),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "48000",
            str(prepared_audio_path),
        ]
        self._run_ffmpeg(command, action="prepare_audio")
        return prepared_audio_path

    def denoise_audio(self, prepared_audio_path: str | Path, input_path: str | Path, output_dir: str | Path) -> Path:
        """Create a real denoised WAV file using ffmpeg arnndn."""
        source_audio = Path(prepared_audio_path).resolve()
        denoised_audio_path = derive_planned_output_path(input_path, output_dir, output_mode="audio")
        denoised_audio_path.parent.mkdir(parents=True, exist_ok=True)
        arnndn_filter = self._build_arnndn_filter()
        command = [
            self.ffmpeg_binary,
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
        self._run_ffmpeg(command, action="denoise_audio")
        return denoised_audio_path

    def export_audio(
        self,
        source_audio_path: str | Path,
        input_path: str | Path,
        output_dir: str | Path,
    ) -> Path:
        """Write a real final audio artifact from the denoised WAV source."""
        source_audio = Path(source_audio_path).resolve()
        final_output_path = derive_planned_output_path(input_path, output_dir, output_mode="audio")
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        if source_audio == final_output_path:
            return final_output_path
        command = [
            self.ffmpeg_binary,
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(source_audio),
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(final_output_path),
        ]
        self._run_ffmpeg(command, action="export_audio")
        return final_output_path

    def apply_pitch_effect(
        self,
        prepared_audio_path: str | Path,
        output_path: str | Path,
        *,
        pitch_factor: float,
    ) -> Path:
        """Apply a duration-compensated pitch shift to normalized 48 kHz audio."""
        source_audio = Path(prepared_audio_path).resolve()
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        base_sample_rate = 48000
        shifted_sample_rate = int(round(base_sample_rate * pitch_factor))
        tempo_factor = 1.0 / pitch_factor
        filter_value = (
            f"asetrate={shifted_sample_rate},"
            f"aresample={base_sample_rate},"
            f"atempo={tempo_factor:.6f}"
        )
        command = [
            self.ffmpeg_binary,
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(source_audio),
            "-af",
            filter_value,
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(base_sample_rate),
            str(output),
        ]
        self._run_ffmpeg(command, action="apply_pitch_effect")
        return output

    def remux_video(
        self,
        input_video_path: str | Path,
        source_audio_path: str | Path,
        output_dir: str | Path,
    ) -> Path:
        """Write a real denoised video artifact by copying video and replacing audio."""
        input_video = Path(input_video_path).resolve()
        source_audio = Path(source_audio_path).resolve()
        final_output_path = derive_planned_output_path(input_video, output_dir, output_mode="video")
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_codec = "libopus" if final_output_path.suffix.lower() == ".webm" else "aac"
        command = [
            self.ffmpeg_binary,
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(input_video),
            "-i",
            str(source_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            audio_codec,
            "-shortest",
            str(final_output_path),
        ]
        self._run_ffmpeg(command, action="remux_video")
        return final_output_path
