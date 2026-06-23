"""Smoke tests for VoiceBank manifest helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.datasets.voicebank_manifest import load_voicebank_manifest, validate_voicebank_pairs


class VoiceBankManifestSmokeTests(unittest.TestCase):
    def test_valid_manifest_loads_expected_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "manifest.csv"
            manifest_path.write_text(
                "sample_id,noisy_path,clean_path,split\n"
                "sample-1,noisy/sample-1.wav,clean/sample-1.wav,train\n"
                "sample-2,noisy/sample-2.wav,clean/sample-2.wav,test\n",
                encoding="utf-8",
            )

            pairs = load_voicebank_manifest(manifest_path, base_dir=root)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].sample_id, "sample-1")
        self.assertEqual(pairs[0].split, "train")
        self.assertEqual(pairs[0].noisy_path.name, "sample-1.wav")
        self.assertTrue(pairs[0].noisy_path.is_absolute())

    def test_missing_required_columns_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.csv"
            manifest_path.write_text(
                "sample_id,noisy_path,split\nsample-1,noisy.wav,train\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing required columns: clean_path"):
                load_voicebank_manifest(manifest_path)

    def test_empty_manifest_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.csv"
            manifest_path.write_text(
                "sample_id,noisy_path,clean_path,split\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "at least one row"):
                load_voicebank_manifest(manifest_path)

    def test_validate_voicebank_pairs_reports_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            noisy_path = root / "noisy.wav"
            noisy_path.write_bytes(b"placeholder")
            manifest_path = root / "manifest.csv"
            manifest_path.write_text(
                "sample_id,noisy_path,clean_path,split\n"
                "sample-1,noisy.wav,clean.wav,train\n",
                encoding="utf-8",
            )
            pairs = load_voicebank_manifest(manifest_path, base_dir=root)

            errors = validate_voicebank_pairs(pairs)

        self.assertEqual(len(errors), 1)
        self.assertIn("sample-1: missing clean file:", errors[0])


if __name__ == "__main__":
    unittest.main()
