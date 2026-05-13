"""Minimal smoke tests for imports and dry-run object construction."""

from __future__ import annotations

import json
import subprocess
import unittest
import uuid
import wave
from pathlib import Path
from unittest.mock import patch

from src.api.contracts import DenoiseRequest
from src.engine.deepfilternet_engine import DeepFilterNetEngine
from src.io.paths import derive_output_mode, infer_input_type
from src.media.ffmpeg_wrapper import FFmpegWrapper
from src.pipeline.run_pipeline import run_pipeline


class StructureSmokeTests(unittest.TestCase):
    @staticmethod
    def _write_sample_wav(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 1600)

    @staticmethod
    def _mock_ffmpeg_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if "-f" in command and "null" in command:
            return subprocess.CompletedProcess(command, 0, "", "")

        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".wav":
            StructureSmokeTests._write_sample_wav(output_path)
        else:
            output_path.write_bytes(b"placeholder-video-artifact")
        return subprocess.CompletedProcess(command, 0, "", "")

    def test_contract_can_be_constructed(self) -> None:
        request = DenoiseRequest(
            input_path=Path("samples/input_audio/example.wav"),
            output_dir=Path("outputs"),
            output_mode="audio",
            keep_intermediates=False,
        )
        self.assertEqual(request.output_mode, "audio")

    def test_engine_can_load_without_weights(self) -> None:
        engine = DeepFilterNetEngine()
        engine.load()
        self.assertTrue(engine.loaded)

    def test_input_type_inference_for_audio_extension(self) -> None:
        self.assertEqual(infer_input_type("example.wav"), "audio")
        self.assertEqual(derive_output_mode("video"), "video")

    def test_ffmpeg_wrapper_raises_clear_error_when_missing(self) -> None:
        wrapper = FFmpegWrapper()
        with patch("src.media.ffmpeg_wrapper.subprocess.run", side_effect=FileNotFoundError()):
            with self.assertRaisesRegex(RuntimeError, "ffmpeg is required"):
                wrapper.probe_input("example.wav")

    def test_ffmpeg_wrapper_raises_clear_error_when_model_missing(self) -> None:
        wrapper = FFmpegWrapper(arnndn_model_path=Path("models/arnndn/missing.rnnn"))
        with self.assertRaisesRegex(FileNotFoundError, "arnndn model file not found"):
            wrapper.denoise_audio("input.wav", "source.wav", "outputs")

    def test_ffmpeg_wrapper_formats_windows_arnndn_model_path(self) -> None:
        wrapper = FFmpegWrapper(arnndn_model_path=Path("models/arnndn/std.rnnn"))
        filter_value = wrapper._build_arnndn_filter()
        self.assertEqual(
            filter_value,
            "arnndn=m='C\\:/Users/ASUS/Desktop/New PJ/models/arnndn/std.rnnn'",
        )

    def test_audio_pipeline_writes_real_denoised_output_and_manifest(self) -> None:
        root = Path("tmp") / f"test-structure-audio-{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        input_path = root / "sample.wav"
        self._write_sample_wav(input_path)

        request = DenoiseRequest(
            input_path=input_path,
            output_dir=root / "outputs",
            output_mode="audio",
            keep_intermediates=False,
        )
        with patch("src.media.ffmpeg_wrapper.subprocess.run", side_effect=self._mock_ffmpeg_run):
            result = run_pipeline(request)

        self.assertEqual(result.status, "completed_real")
        self.assertIsNotNone(result.final_output_path)
        self.assertTrue(result.final_output_path.exists())
        self.assertIsNotNone(result.intermediate_audio_path)
        self.assertTrue(result.intermediate_audio_path.exists())
        self.assertEqual(result.final_output_path.name, "sample.denoised.wav")

        manifest_path = Path(result.run_summary["manifest_path"])
        self.assertTrue(manifest_path.exists())
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertFalse(payload["dry_run"])
        self.assertEqual(payload["selected_engine"], "ffmpeg-arnndn")
        self.assertIn("planned_output_paths", payload)
        self.assertEqual(payload["stage_statuses"]["prepare_audio"], "completed_real")
        self.assertEqual(payload["stage_statuses"]["denoise"], "completed_real")
        self.assertEqual(payload["stage_statuses"]["export_audio"], "completed_real")
        self.assertEqual(payload["final_status"], "completed_real")

    def test_video_pipeline_writes_real_denoised_remux_output_and_manifest(self) -> None:
        root = Path("tmp") / f"test-structure-video-{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        input_path = root / "sample.mp4"
        input_path.write_bytes(b"placeholder-video-input")

        request = DenoiseRequest(
            input_path=input_path,
            output_dir=root / "outputs",
            output_mode="video",
            keep_intermediates=False,
        )
        with patch("src.media.ffmpeg_wrapper.subprocess.run", side_effect=self._mock_ffmpeg_run):
            result = run_pipeline(request)

        self.assertEqual(result.status, "completed_real")
        self.assertIsNotNone(result.final_output_path)
        self.assertTrue(result.final_output_path.exists())
        self.assertEqual(result.final_output_path.name, "sample.denoised.mp4")

        manifest_path = Path(result.run_summary["manifest_path"])
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["stage_statuses"]["prepare_audio"], "completed_real")
        self.assertEqual(payload["stage_statuses"]["denoise"], "completed_real")
        self.assertEqual(payload["stage_statuses"]["remux_video"], "completed_real")
        self.assertEqual(payload["selected_engine"], "ffmpeg-arnndn")

if __name__ == "__main__":
    unittest.main()
