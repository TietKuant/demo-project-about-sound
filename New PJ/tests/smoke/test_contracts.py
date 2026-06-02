"""Smoke tests for processing output contracts."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.api.contracts import OutputArtifact, ProcessingResult


class ProcessingResultContractTests(unittest.TestCase):
    def test_single_restored_output_is_primary_artifact(self) -> None:
        restored_path = Path("outputs/restored.wav")
        result = ProcessingResult(
            task_name="clean_voice",
            engine_name="deepfilternet",
            status="success",
            runtime_sec=1.25,
            outputs=[
                OutputArtifact(
                    label="restored",
                    path=restored_path,
                    media_type="audio",
                )
            ],
        )

        self.assertEqual(result.primary_output_path, restored_path)
        self.assertEqual(result.get_output("restored").path, restored_path)
        self.assertEqual(result.audio_outputs, result.outputs)
        self.assertEqual(result.video_outputs, [])

    def test_multi_output_artifacts_support_label_lookup(self) -> None:
        vocals = OutputArtifact(
            label="vocals",
            path=Path("outputs/vocals.wav"),
            media_type="audio",
        )
        no_vocals = OutputArtifact(
            label="no_vocals",
            path=Path("outputs/no_vocals.wav"),
            media_type="audio",
            role="secondary",
        )
        result = ProcessingResult(
            task_name="extract_vocals",
            engine_name="demucs",
            status="success",
            runtime_sec=3.5,
            outputs=[vocals, no_vocals],
        )

        self.assertEqual(result.get_output("vocals"), vocals)
        self.assertEqual(result.get_output("no_vocals"), no_vocals)
        self.assertEqual(result.primary_output_path, vocals.path)
        self.assertEqual(result.audio_outputs, [vocals, no_vocals])

    def test_missing_output_label_returns_none(self) -> None:
        result = ProcessingResult(
            task_name="extract_vocals",
            engine_name="demucs",
            status="success",
            runtime_sec=None,
            outputs=[],
        )

        self.assertIsNone(result.get_output("vocals"))
        self.assertIsNone(result.primary_output_path)


if __name__ == "__main__":
    unittest.main()
