"""Smoke tests for the minimal Gradio Fast Mode helper."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from app.gradio_app import restore_fast_mode
from src.api.contracts import DenoiseResult


class GradioAppSmokeTests(unittest.TestCase):
    @staticmethod
    def _write_wav(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x01\x00\xff\xff" * 800)

    def test_restore_fast_mode_routes_through_pipeline(self) -> None:
        captured_engine_names: list[str] = []

        def mock_pipeline(request: object) -> DenoiseResult:
            captured_engine_names.append(getattr(request, "engine_name"))
            output_path = Path(getattr(request, "output_dir")) / "sample.denoised.wav"
            self._write_wav(output_path)
            return DenoiseResult(
                status="completed_real",
                final_output_path=output_path,
                intermediate_audio_path=None,
                engine_name="deepfilternet",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "sample.wav"
            self._write_wav(input_path)

            with patch("app.gradio_app.run_pipeline", side_effect=mock_pipeline):
                original, restored, original_preview, restored_preview, plots, runtime_sec, duration_sec, rtf, error = restore_fast_mode(
                    input_path,
                    output_root=root / "outputs",
                )

        self.assertEqual(captured_engine_names, ["deepfilternet"])
        self.assertEqual(original, str(input_path.resolve()))
        self.assertTrue(restored)
        self.assertEqual(original_preview, str(input_path.resolve()))
        self.assertEqual(restored_preview, restored)
        self.assertEqual(len(plots), 4)
        self.assertTrue(runtime_sec)
        self.assertEqual(duration_sec, "0.100000")
        self.assertTrue(rtf)
        self.assertEqual(error, "")

    def test_restore_fast_mode_returns_error_when_pipeline_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "sample.wav"
            self._write_wav(input_path)

            with patch("app.gradio_app.run_pipeline", side_effect=RuntimeError("engine failed")):
                original, restored, original_preview, restored_preview, plots, runtime_sec, duration_sec, rtf, error = restore_fast_mode(
                    input_path,
                    output_root=root / "outputs",
                )

        self.assertEqual(original, str(input_path.resolve()))
        self.assertIsNone(restored)
        self.assertIsNone(original_preview)
        self.assertIsNone(restored_preview)
        self.assertEqual(plots, [])
        self.assertTrue(runtime_sec)
        self.assertEqual(duration_sec, "")
        self.assertEqual(rtf, "")
        self.assertIn("engine failed", error)

    def test_restore_fast_mode_prefers_intermediate_audio_for_before_plot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "sample.m4a"
            input_path.write_bytes(b"placeholder-audio-input")
            intermediate_path = root / "sample.prepared.wav"
            restored_path = root / "sample.denoised.wav"
            self._write_wav(intermediate_path)
            self._write_wav(restored_path)
            result = DenoiseResult(
                status="completed_real",
                final_output_path=restored_path,
                intermediate_audio_path=intermediate_path,
                engine_name="deepfilternet",
            )

            with patch("app.gradio_app.run_pipeline", return_value=result):
                with patch("app.gradio_app.generate_restoration_plots", return_value={}) as plots_mock:
                    restore_fast_mode(input_path, output_root=root / "outputs")

        plots_mock.assert_called_once()
        self.assertEqual(plots_mock.call_args.args[0], intermediate_path)
        self.assertEqual(plots_mock.call_args.args[1], restored_path)

    def test_restore_fast_mode_skips_plots_for_video_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "sample.mp4"
            input_path.write_bytes(b"placeholder-video-input")
            intermediate_path = root / "sample.prepared.wav"
            restored_path = root / "sample.denoised.mp4"
            self._write_wav(intermediate_path)
            restored_path.write_bytes(b"placeholder-video-output")
            result = DenoiseResult(
                status="completed_real",
                final_output_path=restored_path,
                intermediate_audio_path=intermediate_path,
                engine_name="deepfilternet",
            )

            with patch("app.gradio_app.run_pipeline", return_value=result):
                with patch("app.gradio_app.generate_restoration_plots") as plots_mock:
                    _, restored, original_preview, restored_preview, plots, _, duration_sec, rtf, warning = (
                        restore_fast_mode(input_path, output_root=root / "outputs")
                    )

        plots_mock.assert_not_called()
        self.assertEqual(restored, str(restored_path))
        self.assertIsNone(original_preview)
        self.assertIsNone(restored_preview)
        self.assertEqual(plots, [])
        self.assertEqual(duration_sec, "0.100000")
        self.assertTrue(rtf)
        self.assertIn("Plot generation skipped", warning)


if __name__ == "__main__":
    unittest.main()
