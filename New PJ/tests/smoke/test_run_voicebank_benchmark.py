"""Smoke tests for the VoiceBank benchmark runner."""

from __future__ import annotations

import csv
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from src.api.contracts import DenoiseResult
from scripts.run_voicebank_benchmark import DEFAULT_ENGINES, _parse_engines, run_voicebank_benchmark


class RunVoiceBankBenchmarkSmokeTests(unittest.TestCase):
    @staticmethod
    def _write_wav(path: Path, samples: list[int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            frames = b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)
            wav_file.writeframes(frames)

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as csv_file:
            return list(csv.DictReader(csv_file))

    def _write_manifest(self, path: Path, pairs: list[tuple[str, Path, Path]]) -> None:
        lines = ["sample_id,noisy_path,clean_path,split"]
        for sample_id, noisy_path, clean_path in pairs:
            lines.append(f"{sample_id},{noisy_path},{clean_path},test")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _mock_successful_pipeline(self, request: object) -> DenoiseResult:
        input_path = Path(getattr(request, "input_path"))
        output_dir = Path(getattr(request, "output_dir"))
        engine_name = getattr(request, "engine_name")
        output_path = output_dir / f"{input_path.stem}.denoised.wav"
        self._write_wav(output_path, [1000, -1000, 500, -500])
        return DenoiseResult(
            status="completed_real",
            final_output_path=output_path,
            intermediate_audio_path=None,
            engine_name=engine_name,
        )

    def test_benchmark_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_path = root / "clean.wav"
            noisy_path = root / "noisy.wav"
            manifest_path = root / "manifest.csv"
            output_path = root / "benchmark.csv"
            self._write_wav(clean_path, [1000, -1000, 500, -500])
            self._write_wav(noisy_path, [800, -800, 400, -400])
            self._write_manifest(manifest_path, [("sample-1", noisy_path, clean_path)])

            with patch("scripts.run_voicebank_benchmark.run_pipeline", side_effect=self._mock_successful_pipeline):
                run_voicebank_benchmark(
                    manifest_path=manifest_path,
                    output_path=output_path,
                    output_root=root / "outputs",
                    engines=["ffmpeg-arnndn"],
                )
            rows = self._read_rows(output_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "sample-1")
        self.assertEqual(rows[0]["engine"], "ffmpeg-arnndn")
        self.assertEqual(rows[0]["status"], "success")
        self.assertTrue(rows[0]["enhanced_path"])
        self.assertTrue(rows[0]["audio_duration_sec"])
        self.assertTrue(rows[0]["rtf"])
        self.assertTrue(rows[0]["snr_noisy_db"])
        self.assertTrue(rows[0]["si_sdr_db"])

    def test_default_engines_include_noisy_input_first(self) -> None:
        self.assertEqual(
            _parse_engines(DEFAULT_ENGINES),
            ["noisy_input", "ffmpeg-arnndn", "noisereduce", "deepfilternet"],
        )

    def test_noisy_input_baseline_does_not_call_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_path = root / "clean.wav"
            noisy_path = root / "noisy.wav"
            manifest_path = root / "manifest.csv"
            output_path = root / "benchmark.csv"
            self._write_wav(clean_path, [1000, -1000, 500, -500])
            self._write_wav(noisy_path, [800, -800, 400, -400])
            self._write_manifest(manifest_path, [("sample-1", noisy_path, clean_path)])

            with patch("scripts.run_voicebank_benchmark.run_pipeline") as pipeline_mock:
                run_voicebank_benchmark(
                    manifest_path=manifest_path,
                    output_path=output_path,
                    output_root=root / "outputs",
                    engines=["noisy_input"],
                )
            rows = self._read_rows(output_path)

        pipeline_mock.assert_not_called()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["engine"], "noisy_input")
        self.assertEqual(rows[0]["status"], "success")
        self.assertEqual(rows[0]["runtime_sec"], "0.000000")
        self.assertTrue(rows[0]["audio_duration_sec"])
        self.assertEqual(rows[0]["rtf"], "0.000000")
        self.assertEqual(rows[0]["enhanced_path"], rows[0]["noisy_path"])
        self.assertEqual(rows[0]["snr_enhanced_db"], rows[0]["snr_noisy_db"])
        self.assertEqual(rows[0]["snr_improvement_db"], "0.000000")
        self.assertEqual(rows[0]["error"], "")

    def test_all_requested_engines_are_attempted_and_failures_do_not_stop_run(self) -> None:
        calls: list[str] = []

        def mock_pipeline(request: object) -> DenoiseResult:
            engine_name = getattr(request, "engine_name")
            calls.append(engine_name)
            if engine_name == "noisereduce":
                raise RuntimeError("engine failed")
            return self._mock_successful_pipeline(request)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_path = root / "clean.wav"
            noisy_path = root / "noisy.wav"
            manifest_path = root / "manifest.csv"
            output_path = root / "benchmark.csv"
            self._write_wav(clean_path, [1000, -1000, 500, -500])
            self._write_wav(noisy_path, [800, -800, 400, -400])
            self._write_manifest(manifest_path, [("sample-1", noisy_path, clean_path)])

            with patch("scripts.run_voicebank_benchmark.run_pipeline", side_effect=mock_pipeline):
                run_voicebank_benchmark(
                    manifest_path=manifest_path,
                    output_path=output_path,
                    output_root=root / "outputs",
                    engines=["noisy_input", "ffmpeg-arnndn", "noisereduce", "deepfilternet"],
                )
            rows = self._read_rows(output_path)

        self.assertEqual(calls, ["ffmpeg-arnndn", "noisereduce", "deepfilternet"])
        self.assertEqual([row["status"] for row in rows], ["success", "success", "failed", "success"])
        self.assertIn("engine failed", rows[2]["error"])
        self.assertTrue(rows[2]["runtime_sec"])
        self.assertTrue(rows[2]["rtf"])

    def test_limit_restricts_number_of_pairs(self) -> None:
        calls: list[str] = []

        def mock_pipeline(request: object) -> DenoiseResult:
            calls.append(Path(getattr(request, "input_path")).name)
            return self._mock_successful_pipeline(request)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_one = root / "clean-1.wav"
            noisy_one = root / "noisy-1.wav"
            clean_two = root / "clean-2.wav"
            noisy_two = root / "noisy-2.wav"
            manifest_path = root / "manifest.csv"
            output_path = root / "benchmark.csv"
            for path in (clean_one, noisy_one, clean_two, noisy_two):
                self._write_wav(path, [1000, -1000, 500, -500])
            self._write_manifest(
                manifest_path,
                [
                    ("sample-1", noisy_one, clean_one),
                    ("sample-2", noisy_two, clean_two),
                ],
            )

            with patch("scripts.run_voicebank_benchmark.run_pipeline", side_effect=mock_pipeline):
                run_voicebank_benchmark(
                    manifest_path=manifest_path,
                    output_path=output_path,
                    output_root=root / "outputs",
                    engines=["ffmpeg-arnndn"],
                    limit=1,
                )
            rows = self._read_rows(output_path)

        self.assertEqual(calls, ["noisy-1.wav"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "sample-1")


if __name__ == "__main__":
    unittest.main()
