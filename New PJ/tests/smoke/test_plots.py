"""Smoke tests for restoration plot generation."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import matplotlib.pyplot as plt

from src.eval.plots import generate_restoration_plots


class PlotsSmokeTests(unittest.TestCase):
    @staticmethod
    def _write_wav(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x01\x00\xff\xff" * 800)

    def test_generate_restoration_plots_writes_four_pngs_and_closes_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            before_path = root / "before.wav"
            after_path = root / "after.wav"
            output_dir = root / "report"
            self._write_wav(before_path)
            self._write_wav(after_path)
            figures_before = set(plt.get_fignums())

            outputs = generate_restoration_plots(before_path, after_path, output_dir)

            self.assertEqual(set(outputs), {"waveform_before", "waveform_after", "spectrogram_before", "spectrogram_after"})
            for output_path in outputs.values():
                self.assertTrue(output_path.exists())
                self.assertEqual(output_path.suffix, ".png")
            self.assertEqual(set(plt.get_fignums()), figures_before)


if __name__ == "__main__":
    unittest.main()
