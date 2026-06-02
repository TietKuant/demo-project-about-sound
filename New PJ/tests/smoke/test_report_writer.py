"""Smoke tests for restore run report writing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.reporting.report_writer import write_restore_report


class ReportWriterSmokeTests(unittest.TestCase):
    def test_write_restore_report_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path, markdown_path = write_restore_report(
                report_dir=root / "report",
                input_path=root / "input.wav",
                output_path=root / "output.wav",
                engine_name="deepfilternet",
                runtime_sec=1.25,
                audio_duration_sec=2.5,
                rtf=0.5,
                plot_paths=["waveform_before.png", "waveform_after.png"],
                warning="",
                error="",
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(payload["engine_name"], "deepfilternet")
        self.assertEqual(payload["rtf"], 0.5)
        self.assertEqual(payload["plot_paths"], ["waveform_before.png", "waveform_after.png"])
        self.assertIn("Restore Run Summary", markdown)
        self.assertIn("deepfilternet", markdown)
        self.assertIn("waveform_before.png", markdown)


if __name__ == "__main__":
    unittest.main()
