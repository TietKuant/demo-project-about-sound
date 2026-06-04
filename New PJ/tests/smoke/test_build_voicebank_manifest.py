"""Smoke tests for the VoiceBank manifest builder script."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.build_voicebank_manifest import build_voicebank_manifest


class BuildVoiceBankManifestSmokeTests(unittest.TestCase):
    @staticmethod
    def _write_wav(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder-wav")

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as csv_file:
            return list(csv.DictReader(csv_file))

    def test_test_split_manifest_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "voicebank"
            output_path = Path(temp_dir) / "manifest.csv"
            self._write_wav(root / "noisy_testset_wav" / "p257_001.wav")
            self._write_wav(root / "clean_testset_wav" / "p257_001.wav")

            build_voicebank_manifest(voicebank_root=root, output_path=output_path, split="test", limit=20)
            rows = self._read_rows(output_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "p257_001")
        self.assertEqual(rows[0]["split"], "test")
        self.assertFalse(Path(rows[0]["noisy_path"]).is_absolute())
        self.assertFalse(Path(rows[0]["clean_path"]).is_absolute())
        self.assertNotIn("\\", rows[0]["noisy_path"])
        self.assertNotIn("\\", rows[0]["clean_path"])
        self.assertIn("/voicebank/noisy_testset_wav/", rows[0]["noisy_path"])
        self.assertIn("/voicebank/clean_testset_wav/", rows[0]["clean_path"])

    def test_train_split_manifest_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "voicebank"
            output_path = Path(temp_dir) / "manifest.csv"
            self._write_wav(root / "noisy_trainset_28spk_wav" / "p226_001.wav")
            self._write_wav(root / "clean_trainset_28spk_wav" / "p226_001.wav")

            build_voicebank_manifest(voicebank_root=root, output_path=output_path, split="train", limit=20)
            rows = self._read_rows(output_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "p226_001")
        self.assertEqual(rows[0]["split"], "train")

    def test_limit_is_applied_after_sorting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "voicebank"
            output_path = Path(temp_dir) / "manifest.csv"
            for name in ("p257_002.wav", "p257_001.wav", "p257_003.wav"):
                self._write_wav(root / "noisy_testset_wav" / name)
                self._write_wav(root / "clean_testset_wav" / name)

            build_voicebank_manifest(voicebank_root=root, output_path=output_path, split="test", limit=2)
            rows = self._read_rows(output_path)

        self.assertEqual([row["sample_id"] for row in rows], ["p257_001", "p257_002"])

    def test_clean_speech_manifest_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data" / "external" / "voicebank"
            output_path = Path(temp_dir) / "data" / "manifests" / "clean_speech.local.csv"
            self._write_wav(root / "clean_testset_wav" / "p257_001.wav")
            self._write_wav(root / "clean_testset_wav" / "p257_002.wav")

            build_voicebank_manifest(
                voicebank_root=root,
                output_path=output_path,
                split="test",
                limit=20,
                manifest_type="clean-speech",
            )
            rows = self._read_rows(output_path)

        self.assertEqual(list(rows[0].keys()), ["sample_id", "path", "split", "notes"])
        self.assertEqual([row["sample_id"] for row in rows], ["p257_001", "p257_002"])
        self.assertEqual(rows[0]["split"], "test")
        self.assertEqual(rows[0]["notes"], "VoiceBank-DEMAND clean speech source")
        self.assertFalse(Path(rows[0]["path"]).is_absolute())
        self.assertNotIn("\\", rows[0]["path"])
        self.assertEqual(rows[0]["path"], "../external/voicebank/clean_testset_wav/p257_001.wav")

    def test_clean_speech_mode_uses_clean_folder_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "voicebank"
            output_path = Path(temp_dir) / "clean_speech.csv"
            self._write_wav(root / "clean_trainset_28spk_wav" / "p226_001.wav")
            self._write_wav(root / "clean_trainset_28spk_wav" / "p226_002.wav")

            build_voicebank_manifest(
                voicebank_root=root,
                output_path=output_path,
                split="train",
                limit=20,
                manifest_type="clean-speech",
            )
            rows = self._read_rows(output_path)

        self.assertEqual([row["sample_id"] for row in rows], ["p226_001", "p226_002"])

    def test_clean_speech_limit_is_applied_after_sorting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data" / "external" / "voicebank"
            output_path = Path(temp_dir) / "data" / "manifests" / "clean_speech.local.csv"
            for name in ("p257_002.wav", "p257_001.wav", "p257_003.wav"):
                self._write_wav(root / "clean_testset_wav" / name)

            build_voicebank_manifest(
                voicebank_root=root,
                output_path=output_path,
                split="test",
                limit=2,
                manifest_type="clean-speech",
            )
            rows = self._read_rows(output_path)

        self.assertEqual([row["sample_id"] for row in rows], ["p257_001", "p257_002"])

    def test_missing_folders_raise_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "voicebank"
            output_path = Path(temp_dir) / "manifest.csv"

            with self.assertRaisesRegex(ValueError, "Expected noisy folder is missing"):
                build_voicebank_manifest(voicebank_root=root, output_path=output_path, split="test")

    def test_clean_speech_missing_clean_folder_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "voicebank"
            output_path = Path(temp_dir) / "clean_speech.csv"

            with self.assertRaisesRegex(ValueError, "Expected clean folder is missing"):
                build_voicebank_manifest(
                    voicebank_root=root,
                    output_path=output_path,
                    split="test",
                    manifest_type="clean-speech",
                )

    def test_unmatched_files_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "voicebank"
            output_path = Path(temp_dir) / "manifest.csv"
            self._write_wav(root / "noisy_testset_wav" / "p257_001.wav")
            self._write_wav(root / "noisy_testset_wav" / "p257_002.wav")
            self._write_wav(root / "clean_testset_wav" / "p257_001.wav")

            build_voicebank_manifest(voicebank_root=root, output_path=output_path, split="test", limit=20)
            rows = self._read_rows(output_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "p257_001")


if __name__ == "__main__":
    unittest.main()
