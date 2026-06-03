"""Smoke tests for adapting pipeline results to ProcessingResult."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.api.contracts import DenoiseResult
from src.pipeline.run_pipeline import pipeline_result_to_processing_result


class PipelineProcessingResultAdapterTests(unittest.TestCase):
    def test_audio_pipeline_result_becomes_restored_audio_artifact(self) -> None:
        result = DenoiseResult(
            status="completed_real",
            final_output_path=Path("outputs/sample.denoised.wav"),
            intermediate_audio_path=Path("outputs/sample.prepared.wav"),
            engine_name="deepfilternet",
        )

        processing_result = pipeline_result_to_processing_result(result, runtime_sec=1.5)

        self.assertEqual(processing_result.task_name, "clean_voice")
        self.assertEqual(processing_result.engine_name, "deepfilternet")
        self.assertEqual(processing_result.status, "success")
        self.assertEqual(processing_result.runtime_sec, 1.5)
        self.assertEqual(processing_result.primary_output_path, Path("outputs/sample.denoised.wav"))
        restored = processing_result.get_output("restored")
        self.assertIsNotNone(restored)
        self.assertEqual(restored.media_type, "audio")
        self.assertEqual(restored.role, "primary")

    def test_video_pipeline_result_becomes_restored_video_artifact(self) -> None:
        result = DenoiseResult(
            status="completed_real",
            final_output_path=Path("outputs/sample.denoised.mp4"),
            intermediate_audio_path=Path("outputs/sample.prepared.wav"),
            engine_name="deepfilternet",
        )

        processing_result = pipeline_result_to_processing_result(result)

        restored = processing_result.get_output("restored")
        self.assertIsNotNone(restored)
        self.assertEqual(restored.media_type, "video")
        self.assertEqual(processing_result.video_outputs, [restored])
        self.assertEqual(processing_result.audio_outputs, [])

    def test_missing_final_output_becomes_failed_processing_result(self) -> None:
        result = DenoiseResult(
            status="completed_real",
            final_output_path=None,
            intermediate_audio_path=Path("outputs/sample.prepared.wav"),
            engine_name="deepfilternet",
        )

        processing_result = pipeline_result_to_processing_result(result)

        self.assertEqual(processing_result.status, "failed")
        self.assertEqual(processing_result.outputs, [])
        self.assertIn("final output path", processing_result.error)

    def test_failed_pipeline_status_becomes_failed_processing_result(self) -> None:
        result = DenoiseResult(
            status="failed",
            final_output_path=Path("outputs/sample.denoised.wav"),
            intermediate_audio_path=None,
            engine_name="deepfilternet",
        )

        processing_result = pipeline_result_to_processing_result(result)

        self.assertEqual(processing_result.status, "failed")
        self.assertEqual(processing_result.outputs, [])
        self.assertIn("failed", processing_result.error)


if __name__ == "__main__":
    unittest.main()
