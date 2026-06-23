"""Extract lightweight audio features from a real-sample manifest."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_COLUMNS = {
    "sample_id",
    "category",
    "input_path",
    "input_type",
    "language",
    "expected_task",
    "has_clean_reference",
    "notes",
}
FIELDNAMES = [
    "sample_id",
    "category",
    "input_path",
    "input_type",
    "expected_task",
    "status",
    "duration_sec",
    "rms_energy",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "notes",
    "error",
]
EPSILON = 1e-12
FRAME_LENGTH = 1024
HOP_LENGTH = 512


def _load_manifest(manifest_path: Path, limit: int | None) -> list[dict[str, str]]:
    with Path(manifest_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Audio feature manifest is missing required columns: {missing}")
        rows = list(reader)
    return rows if limit is None else rows[:limit]


def _resolve_input_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path)


def _duration_sec(input_path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(input_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def _decode_to_temp_wav(input_path: Path, temp_dir: Path) -> Path:
    wav_path = temp_dir / f"{input_path.stem}.mono.wav"
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(input_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip() or "unknown ffmpeg decode error"
        raise RuntimeError(f"ffmpeg failed during audio decode: {details}")
    return wav_path


def _read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if audio.size == 0:
        raise ValueError("Decoded audio is empty.")
    return audio, int(sample_rate)


def _audio_for_features(input_path: Path, temp_dir: Path) -> tuple[np.ndarray, int]:
    if input_path.suffix.lower() == ".wav":
        try:
            return _read_pcm_wav(input_path)
        except Exception:
            decoded_path = _decode_to_temp_wav(input_path, temp_dir)
            return _read_pcm_wav(decoded_path)
    decoded_path = _decode_to_temp_wav(input_path, temp_dir)
    return _read_pcm_wav(decoded_path)


def _frame_audio(audio: np.ndarray, frame_length: int = FRAME_LENGTH, hop_length: int = HOP_LENGTH) -> np.ndarray:
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))
    frame_count = 1 + max(0, (audio.size - frame_length) // hop_length)
    frame_starts = np.arange(frame_count)[:, None] * hop_length
    frame_offsets = np.arange(frame_length)[None, :]
    return audio[frame_starts + frame_offsets]


def _frame_zero_crossing_rates(frames: np.ndarray) -> np.ndarray:
    signs = np.signbit(frames)
    crossings = np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1)
    return crossings / max(frames.shape[1] - 1, 1)


def _features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    centered = audio - float(np.mean(audio))
    rms_energy = float(np.sqrt(np.mean(centered**2)))

    signs = np.signbit(centered)
    zero_crossings = np.count_nonzero(signs[1:] != signs[:-1])
    zero_crossing_rate = float(zero_crossings / max(centered.size - 1, 1))

    spectrum = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate)
    magnitude_sum = float(np.sum(spectrum))
    if magnitude_sum <= EPSILON:
        spectral_centroid_hz = 0.0
        spectral_bandwidth_hz = 0.0
    else:
        spectral_centroid_hz = float(np.sum(frequencies * spectrum) / magnitude_sum)
        spectral_bandwidth_hz = float(
            np.sqrt(np.sum(((frequencies - spectral_centroid_hz) ** 2) * spectrum) / magnitude_sum)
        )

    frames = _frame_audio(centered)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    frame_zcr = _frame_zero_crossing_rates(frames)
    windowed_frames = frames * np.hanning(frames.shape[1])
    frame_spectrum = np.abs(np.fft.rfft(windowed_frames, axis=1)) + EPSILON
    frame_frequencies = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)
    frame_magnitude = np.sum(frame_spectrum, axis=1)
    cumulative_magnitude = np.cumsum(frame_spectrum, axis=1)
    rolloff_threshold = 0.85 * frame_magnitude
    rolloff_indices = np.argmax(cumulative_magnitude >= rolloff_threshold[:, None], axis=1)
    spectral_rolloff_hz = float(np.mean(frame_frequencies[rolloff_indices]))
    spectral_flatness = float(np.mean(np.exp(np.mean(np.log(frame_spectrum), axis=1)) / np.mean(frame_spectrum, axis=1)))

    power_spectrum = frame_spectrum**2
    total_power = float(np.sum(power_spectrum))
    if total_power <= EPSILON:
        low_band_energy_ratio = 0.0
        mid_band_energy_ratio = 0.0
        high_band_energy_ratio = 0.0
    else:
        low_band_energy_ratio = float(np.sum(power_spectrum[:, frame_frequencies < 300.0]) / total_power)
        mid_band_energy_ratio = float(
            np.sum(power_spectrum[:, (frame_frequencies >= 300.0) & (frame_frequencies < 3400.0)]) / total_power
        )
        high_band_energy_ratio = float(np.sum(power_spectrum[:, frame_frequencies >= 3400.0]) / total_power)
    silence_threshold = max(float(np.max(frame_rms)) * 0.05, EPSILON)

    return {
        "rms_energy": rms_energy,
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_centroid_hz": spectral_centroid_hz,
        "spectral_bandwidth_hz": spectral_bandwidth_hz,
        "spectral_rolloff_hz": spectral_rolloff_hz,
        "spectral_flatness": spectral_flatness,
        "low_band_energy_ratio": low_band_energy_ratio,
        "mid_band_energy_ratio": mid_band_energy_ratio,
        "high_band_energy_ratio": high_band_energy_ratio,
        "rms_std": float(np.std(frame_rms)),
        "zcr_std": float(np.std(frame_zcr)),
        "silence_ratio": float(np.mean(frame_rms <= silence_threshold)),
    }


def _empty_row(manifest_row: dict[str, str], input_path: Path) -> dict[str, str]:
    return {
        "sample_id": manifest_row["sample_id"],
        "category": manifest_row["category"],
        "input_path": str(input_path),
        "input_type": manifest_row["input_type"],
        "expected_task": manifest_row["expected_task"],
        "status": "failed",
        "duration_sec": "",
        "rms_energy": "",
        "zero_crossing_rate": "",
        "spectral_centroid_hz": "",
        "spectral_bandwidth_hz": "",
        "notes": manifest_row["notes"],
        "error": "",
    }


def extract_audio_features(
    *,
    manifest_path: Path,
    output_path: Path,
    limit: int | None = None,
    skip_missing: bool = False,
) -> Path:
    """Extract feature rows from a Vietnamese real-sample manifest."""
    rows = _load_manifest(manifest_path, limit)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_rows: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="audio-feature-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for manifest_row in rows:
            input_path = _resolve_input_path(manifest_row["input_path"])
            row = _empty_row(manifest_row, input_path)

            if not input_path.exists():
                row["status"] = "skipped" if skip_missing else "failed"
                row["error"] = f"Input file not found: {input_path}"
                feature_rows.append(row)
                continue

            duration = _duration_sec(input_path)
            row["duration_sec"] = "" if duration is None else f"{duration:.6f}"
            try:
                audio, sample_rate = _audio_for_features(input_path, temp_dir)
                features = _features(audio, sample_rate)
                row.update(
                    {
                        "status": "success",
                        "rms_energy": f"{features['rms_energy']:.10f}",
                        "zero_crossing_rate": f"{features['zero_crossing_rate']:.10f}",
                        "spectral_centroid_hz": f"{features['spectral_centroid_hz']:.6f}",
                        "spectral_bandwidth_hz": f"{features['spectral_bandwidth_hz']:.6f}",
                    }
                )
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
            feature_rows.append(row)

    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(feature_rows)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the audio feature extraction CLI parser."""
    parser = argparse.ArgumentParser(description="Extract lightweight audio features from a manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-missing", action="store_true")
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        output_path = extract_audio_features(
            manifest_path=args.manifest,
            output_path=args.output,
            limit=args.limit,
            skip_missing=args.skip_missing,
        )
        print(f"Wrote audio feature CSV: {output_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
