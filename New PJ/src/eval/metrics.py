"""Small NumPy-based evaluation metrics for paired speech denoise outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


EPSILON = 1e-10


def _as_audio_array(signal: np.ndarray) -> np.ndarray:
    """Return a float audio array and reject empty inputs."""
    audio = np.asarray(signal, dtype=np.float64)
    if audio.size == 0:
        raise ValueError("Audio signal must not be empty.")
    return audio


def load_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file and return mono samples plus sample rate."""
    audio, sample_rate = sf.read(path)
    audio = _as_audio_array(audio)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, int(sample_rate)


def align_signals(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim two signals to their shared shortest length."""
    reference_audio = _as_audio_array(reference)
    estimate_audio = _as_audio_array(estimate)
    length = min(reference_audio.shape[0], estimate_audio.shape[0])
    if length == 0:
        raise ValueError("Aligned audio signals must not be empty.")
    return reference_audio[:length], estimate_audio[:length]


def snr_db(clean: np.ndarray, degraded: np.ndarray) -> float:
    """Return signal-to-noise ratio in dB against a clean reference."""
    clean_audio, degraded_audio = align_signals(clean, degraded)
    noise = clean_audio - degraded_audio
    signal_power = float(np.sum(clean_audio**2))
    noise_power = float(np.sum(noise**2))
    return float(10.0 * np.log10((signal_power + EPSILON) / (noise_power + EPSILON)))


def snr_improvement_db(clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray) -> float:
    """Return SNR improvement of enhanced audio over noisy audio."""
    return float(snr_db(clean, enhanced) - snr_db(clean, noisy))


def si_sdr_db(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """Return scale-invariant SDR in dB against a clean reference."""
    clean_audio, enhanced_audio = align_signals(clean, enhanced)
    clean_energy = float(np.sum(clean_audio**2))
    scale = float(np.sum(enhanced_audio * clean_audio) / (clean_energy + EPSILON))
    target = scale * clean_audio
    residual = enhanced_audio - target
    target_power = float(np.sum(target**2))
    residual_power = float(np.sum(residual**2))
    return float(10.0 * np.log10((target_power + EPSILON) / (residual_power + EPSILON)))
