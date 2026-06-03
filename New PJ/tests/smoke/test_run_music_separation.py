"""Smoke tests for the music separation workflow script."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

from scripts.run_music_separation import main, run_music_separation
from src.api.contracts import OutputArtifact, ProcessingResult


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _success_result(vocals_path: Path, no_vocals_path: Path) -> ProcessingResult:
    return ProcessingResult(
        task_name="extract_vocals",
        engine_name="demucs",
        status="success",
        runtime_sec=2.5,
        outputs=[
            OutputArtifact(label="vocals", path=vocals_path, media_type="audio", role="primary"),
            OutputArtifact(label="no_vocals", path=no_vocals_path, media_type="audio", role="secondary"),
        ],
    )


def test_music_separation_writes_success_summaries(tmp_path: Path) -> None:
    input_path = tmp_path / "Song.stem.mp4"
    output_root = tmp_path / "runs"
    input_path.write_bytes(b"music")

    def mock_separate(source: Path, run_dir: Path) -> ProcessingResult:
        vocals_path = run_dir / "htdemucs" / source.stem / "vocals.wav"
        no_vocals_path = run_dir / "htdemucs" / source.stem / "no_vocals.wav"
        vocals_path.parent.mkdir(parents=True, exist_ok=True)
        vocals_path.write_bytes(b"vocals")
        no_vocals_path.write_bytes(b"no-vocals")
        return _success_result(vocals_path, no_vocals_path)

    with patch("scripts.run_music_separation.DemucsCliEngine.separate", side_effect=mock_separate):
        run_dir = run_music_separation(input_path=input_path, output_root=output_root)

    csv_rows = _read_csv_rows(run_dir / "summary.csv")
    json_rows = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert csv_rows == json_rows
    assert csv_rows[0]["run_id"].startswith("Song.stem-")
    assert csv_rows[0]["task"] == "extract_vocals"
    assert csv_rows[0]["engine"] == "demucs"
    assert csv_rows[0]["status"] == "success"
    assert csv_rows[0]["runtime_sec"] == "2.500000"
    assert csv_rows[0]["vocals_path"].endswith("/vocals.wav")
    assert csv_rows[0]["no_vocals_path"].endswith("/no_vocals.wav")
    assert csv_rows[0]["error"] == ""


def test_music_separation_writes_failed_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "Song.stem.mp4"
    output_root = tmp_path / "runs"
    input_path.write_bytes(b"music")
    failed_result = ProcessingResult(
        task_name="extract_vocals",
        engine_name="demucs",
        status="failed",
        runtime_sec=0.25,
        outputs=[],
        error="demucs failed",
    )

    with patch("scripts.run_music_separation.DemucsCliEngine.separate", return_value=failed_result):
        run_dir = run_music_separation(input_path=input_path, output_root=output_root)

    row = _read_csv_rows(run_dir / "summary.csv")[0]
    assert row["status"] == "failed"
    assert row["runtime_sec"] == "0.250000"
    assert row["vocals_path"] == ""
    assert row["no_vocals_path"] == ""
    assert row["error"] == "demucs failed"


def test_music_separation_main_can_run_with_mocked_engine(tmp_path: Path, capsys: object) -> None:
    input_path = tmp_path / "Song.stem.mp4"
    output_root = tmp_path / "runs"
    input_path.write_bytes(b"music")

    def mock_separate(source: Path, run_dir: Path) -> ProcessingResult:
        return _success_result(run_dir / "vocals.wav", run_dir / "no_vocals.wav")

    argv = [
        "run_music_separation.py",
        "--input",
        str(input_path),
        "--output-root",
        str(output_root),
        "--demucs-python",
        "/isolated/bin/python",
    ]
    with patch("sys.argv", argv):
        with patch("scripts.run_music_separation.DemucsCliEngine.separate", side_effect=mock_separate):
            exit_code = main()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Wrote music separation run:" in captured.out
    assert list(output_root.glob("Song.stem-*/summary.csv"))
