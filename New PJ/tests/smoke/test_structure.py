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
from src.engine.deepfilternet_cli_engine import DeepFilterNetCliEngine
from src.engine.deepfilternet_engine import DeepFilterNetEngine
from src.engine.ffmpeg_arnndn_engine import FFmpegArnndnEngine
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

    @staticmethod
    def _mock_noisereduce_denoise(input_audio_path: str | Path, output_audio_path: str | Path) -> Path:
        _ = Path(input_audio_path)
        output_path = Path(output_audio_path)
        StructureSmokeTests._write_sample_wav(output_path)
        return output_path

    @staticmethod
    def _mock_deepfilternet_denoise(input_audio_path: str | Path, output_audio_path: str | Path) -> Path:
        _ = Path(input_audio_path)
        output_path = Path(output_audio_path)
        StructureSmokeTests._write_sample_wav(output_path)
        return output_path

    def test_contract_can_be_constructed(self) -> None:
        request = DenoiseRequest(
            input_path=Path("samples/input_audio/example.wav"),
            output_dir=Path("outputs"),
            output_mode="audio",
            keep_intermediates=False,
        )
        self.assertEqual(request.output_mode, "audio")
        self.assertEqual(request.engine_name, "ffmpeg-arnndn")

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

    def test_ffmpeg_arnndn_engine_raises_clear_error_when_model_missing(self) -> None:
        wrapper = FFmpegWrapper(arnndn_model_path=Path("models/arnndn/missing.rnnn"))
        engine = FFmpegArnndnEngine(ffmpeg_wrapper=wrapper)
        with self.assertRaisesRegex(FileNotFoundError, "arnndn model file not found"):
            engine.load()

    def test_ffmpeg_wrapper_formats_resolved_arnndn_model_path(self) -> None:
        model_path = Path("models/arnndn/std.rnnn")
        wrapper = FFmpegWrapper(arnndn_model_path=model_path)
        filter_value = wrapper._build_arnndn_filter()
        expected_model_path = model_path.resolve().as_posix().replace(":", "\\:")
        self.assertTrue(filter_value.startswith("arnndn=m='"))
        self.assertIn(expected_model_path, filter_value)
        self.assertTrue(filter_value.endswith("'"))

    def test_deepfilternet_cli_engine_copies_generated_output_to_explicit_path(self) -> None:
        root = Path("tmp") / f"test-structure-deepfilter-engine-{uuid.uuid4().hex}"
        input_path = root / "sample.prepared.wav"
        output_path = root / "outputs" / "sample.denoised.wav"
        generated_path = output_path.parent / "sample.prepared_DeepFilterNet3.wav"
        self._write_sample_wav(input_path)

        def mock_deepfilter_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertEqual(command[0], "/usr/local/bin/deepFilter")
            self.assertEqual(command[1], str(input_path.resolve()))
            self.assertEqual(command[2], "-o")
            self.assertEqual(command[3], str(output_path.parent.resolve()))
            self._write_sample_wav(generated_path)
            return subprocess.CompletedProcess(command, 0, "", "")

        engine = DeepFilterNetCliEngine()
        with patch("src.engine.deepfilternet_cli_engine.shutil.which", return_value="/usr/local/bin/deepFilter"):
            engine.load()
        with patch("src.engine.deepfilternet_cli_engine.subprocess.run", side_effect=mock_deepfilter_run):
            result_path = engine.denoise(input_path, output_path)

        self.assertEqual(result_path, output_path.resolve())
        self.assertTrue(output_path.exists())

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

    def test_pipeline_routes_to_noisereduce_engine_when_requested(self) -> None:
        root = Path("tmp") / f"test-structure-noisereduce-{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        input_path = root / "sample.wav"
        self._write_sample_wav(input_path)

        request = DenoiseRequest(
            input_path=input_path,
            output_dir=root / "outputs",
            output_mode="audio",
            keep_intermediates=False,
            engine_name="noisereduce",
        )
        with patch("src.media.ffmpeg_wrapper.subprocess.run", side_effect=self._mock_ffmpeg_run):
            with patch("src.pipeline.run_pipeline.NoisereduceEngine.load", return_value=None):
                with patch(
                    "src.pipeline.run_pipeline.NoisereduceEngine.denoise",
                    side_effect=self._mock_noisereduce_denoise,
                ) as denoise_mock:
                    result = run_pipeline(request)

        self.assertEqual(result.engine_name, "noisereduce")
        self.assertEqual(result.final_output_path.name, "sample.denoised.wav")
        self.assertTrue(result.final_output_path.exists())
        denoise_mock.assert_called_once()

        manifest_path = Path(result.run_summary["manifest_path"])
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["selected_engine"], "noisereduce")
        self.assertEqual(payload["stage_statuses"]["denoise"], "completed_real")

    def test_pipeline_routes_to_deepfilternet_engine_when_requested(self) -> None:
        root = Path("tmp") / f"test-structure-deepfilternet-{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        input_path = root / "sample.wav"
        self._write_sample_wav(input_path)

        request = DenoiseRequest(
            input_path=input_path,
            output_dir=root / "outputs",
            output_mode="audio",
            keep_intermediates=False,
            engine_name="deepfilternet",
        )
        with patch("src.media.ffmpeg_wrapper.subprocess.run", side_effect=self._mock_ffmpeg_run):
            with patch("src.pipeline.run_pipeline.DeepFilterNetCliEngine.load", return_value=None):
                with patch(
                    "src.pipeline.run_pipeline.DeepFilterNetCliEngine.denoise",
                    side_effect=self._mock_deepfilternet_denoise,
                ) as denoise_mock:
                    result = run_pipeline(request)

        self.assertEqual(result.engine_name, "deepfilternet")
        self.assertEqual(result.final_output_path.name, "sample.denoised.wav")
        self.assertTrue(result.final_output_path.exists())
        denoise_mock.assert_called_once()

        manifest_path = Path(result.run_summary["manifest_path"])
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["selected_engine"], "deepfilternet")
        self.assertEqual(payload["stage_statuses"]["denoise"], "completed_real")

if __name__ == "__main__":
    unittest.main()
