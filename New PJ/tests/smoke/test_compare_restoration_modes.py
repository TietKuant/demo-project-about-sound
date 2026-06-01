"""Smoke tests for the restoration comparison workflow."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from scripts.compare_restoration_modes import DEFAULT_ENGINES, _parse_engines, compare_restoration_modes
from src.api.contracts import DenoiseResult


class CompareRestorationModesSmokeTests(unittest.TestCase):
    @staticmethod
    def _write_wav(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 1600)

    @staticmethod
    def _read_csv_rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as csv_file:
            return list(csv.DictReader(csv_file))

    def _mock_successful_pipeline(self, request: object) -> DenoiseResult:
        input_path = Path(getattr(request, "input_path"))
        output_dir = Path(getattr(request, "output_dir"))
        engine_name = getattr(request, "engine_name")
        output_path = output_dir / f"{input_path.stem}.denoised.wav"
        self._write_wav(output_path)
        return DenoiseResult(
            status="completed_real",
            final_output_path=output_path,
            intermediate_audio_path=None,
            engine_name=engine_name,
        )

    def test_default_engines_are_noisy_input_and_deepfilternet(self) -> None:
        self.assertEqual(_parse_engines(DEFAULT_ENGINES), ["noisy_input", "deepfilternet"])

    def test_noisy_input_writes_scoped_output_without_calling_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "sample.wav"
            output_root = root / "compare-runs"
            self._write_wav(input_path)

            with patch("scripts.compare_restoration_modes.run_pipeline") as pipeline_mock:
                run_dir = compare_restoration_modes(
                    input_path=input_path,
                    output_root=output_root,
                    engines=["noisy_input"],
                )
            csv_rows = self._read_csv_rows(run_dir / "summary.csv")
            json_rows = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            noisy_output_exists = Path(csv_rows[0]["output_path"]).exists()

        pipeline_mock.assert_not_called()
        self.assertEqual(run_dir.parent, output_root.resolve())
        self.assertTrue(run_dir.name.startswith("sample-"))
        self.assertEqual(csv_rows, json_rows)
        self.assertEqual(csv_rows[0]["engine"], "noisy_input")
        self.assertEqual(csv_rows[0]["status"], "success")
        self.assertEqual(csv_rows[0]["runtime_sec"], "0.000000")
        self.assertEqual(csv_rows[0]["rtf"], "0.000000")
        self.assertTrue(noisy_output_exists)
        self.assertTrue(Path(csv_rows[0]["output_path"]).is_relative_to(run_dir / "noisy_input"))

    def test_failed_engine_is_recorded_and_does_not_stop_workflow(self) -> None:
        calls: list[str] = []

        def mock_pipeline(request: object) -> DenoiseResult:
            engine_name = getattr(request, "engine_name")
            calls.append(engine_name)
            if engine_name == "deepfilternet":
                raise RuntimeError("engine failed")
            return self._mock_successful_pipeline(request)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "sample.wav"
            output_root = root / "compare-runs"
            self._write_wav(input_path)

            with patch("scripts.compare_restoration_modes.run_pipeline", side_effect=mock_pipeline):
                run_dir = compare_restoration_modes(
                    input_path=input_path,
                    output_root=output_root,
                    engines=["noisy_input", "deepfilternet", "ffmpeg-arnndn"],
                )
            csv_rows = self._read_csv_rows(run_dir / "summary.csv")
            json_rows = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            engine_dirs_exist = all(
                (run_dir / engine).is_dir() for engine in ("noisy_input", "deepfilternet", "ffmpeg-arnndn")
            )

        self.assertEqual(calls, ["deepfilternet", "ffmpeg-arnndn"])
        self.assertEqual([row["status"] for row in csv_rows], ["success", "failed", "success"])
        self.assertIn("engine failed", csv_rows[1]["error"])
        self.assertEqual(csv_rows, json_rows)
        self.assertTrue(engine_dirs_exist)
        self.assertTrue(Path(csv_rows[2]["output_path"]).is_relative_to(run_dir / "ffmpeg-arnndn"))


if __name__ == "__main__":
    unittest.main()
