"""File-safe waveform and spectrogram plots for restoration runs."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.eval.metrics import load_mono_audio


def _save_waveform(audio: np.ndarray, sample_rate: int, output_path: Path, title: str) -> Path:
    figure, axis = plt.subplots(figsize=(8, 3))
    try:
        times = np.arange(len(audio)) / sample_rate
        axis.plot(times, audio, linewidth=0.8)
        axis.set(title=title, xlabel="Time (sec)", ylabel="Amplitude")
        figure.tight_layout()
        figure.savefig(output_path, dpi=120)
    finally:
        plt.close(figure)
    return output_path


def _save_spectrogram(audio: np.ndarray, sample_rate: int, output_path: Path, title: str) -> Path:
    figure, axis = plt.subplots(figsize=(8, 3))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            axis.specgram(audio, Fs=sample_rate)
        axis.set(title=title, xlabel="Time (sec)", ylabel="Frequency (Hz)")
        figure.tight_layout()
        figure.savefig(output_path, dpi=120)
    finally:
        plt.close(figure)
    return output_path


def generate_restoration_plots(before_path: Path, after_path: Path, output_dir: Path) -> dict[str, Path]:
    """Generate before/after waveform and spectrogram PNG files."""
    report_dir = Path(output_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    before_audio, before_rate = load_mono_audio(before_path)
    after_audio, after_rate = load_mono_audio(after_path)

    outputs = {
        "waveform_before": report_dir / "waveform_before.png",
        "waveform_after": report_dir / "waveform_after.png",
        "spectrogram_before": report_dir / "spectrogram_before.png",
        "spectrogram_after": report_dir / "spectrogram_after.png",
    }
    _save_waveform(before_audio, before_rate, outputs["waveform_before"], "Waveform Before")
    _save_waveform(after_audio, after_rate, outputs["waveform_after"], "Waveform After")
    _save_spectrogram(before_audio, before_rate, outputs["spectrogram_before"], "Spectrogram Before")
    _save_spectrogram(after_audio, after_rate, outputs["spectrogram_after"], "Spectrogram After")
    return outputs
