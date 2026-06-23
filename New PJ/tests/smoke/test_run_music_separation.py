"""Smoke tests for the music separation workflow script."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

from scripts.run_music_separation import build_arg_parser, main, run_music_separation
from src.api.contracts import OutputArtifact, ProcessingResult
from src.router.task_registry import CLEAN_VOICE, EXTRACT_VOCALS, REMOVE_VOCALS


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
    assert csv_rows[0]["primary_output_label"] == "vocals"
    assert csv_rows[0]["primary_output_path"].endswith("/vocals.wav")
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
    assert row["primary_output_label"] == ""
    assert row["primary_output_path"] == ""
    assert row["vocals_path"] == ""
    assert row["no_vocals_path"] == ""
    assert row["error"] == "demucs failed"


def test_music_separation_remove_vocals_uses_no_vocals_as_primary_output(tmp_path: Path) -> None:
    input_path = tmp_path / "Song.stem.mp4"
    output_root = tmp_path / "runs"
    input_path.write_bytes(b"music")

    def mock_separate(source: Path, run_dir: Path) -> ProcessingResult:
        vocals_path = run_dir / "htdemucs" / source.stem / "vocals.wav"
        no_vocals_path = run_dir / "htdemucs" / source.stem / "no_vocals.wav"
        return _success_result(vocals_path, no_vocals_path)

    with patch("scripts.run_music_separation.DemucsCliEngine.separate", side_effect=mock_separate):
        run_dir = run_music_separation(
            input_path=input_path,
            output_root=output_root,
            task_name=REMOVE_VOCALS,
        )

    row = _read_csv_rows(run_dir / "summary.csv")[0]
    assert row["task"] == REMOVE_VOCALS
    assert row["primary_output_label"] == "no_vocals"
    assert row["primary_output_path"].endswith("/no_vocals.wav")
    assert row["vocals_path"].endswith("/vocals.wav")
    assert row["no_vocals_path"].endswith("/no_vocals.wav")


def test_music_separation_rejects_clean_voice_task(tmp_path: Path) -> None:
    input_path = tmp_path / "Song.stem.mp4"
    input_path.write_bytes(b"music")

    with patch("scripts.run_music_separation.DemucsCliEngine.separate") as separate_mock:
        try:
            run_music_separation(
                input_path=input_path,
                output_root=tmp_path / "runs",
                task_name=CLEAN_VOICE,
            )
        except ValueError as exc:
            assert "Unsupported music separation task: clean_voice" in str(exc)
        else:
            raise AssertionError("Expected clean_voice to be rejected")

    separate_mock.assert_not_called()


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
        "--task",
        REMOVE_VOCALS,
    ]
    with patch("sys.argv", argv):
        with patch("scripts.run_music_separation.DemucsCliEngine.separate", side_effect=mock_separate):
            exit_code = main()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Wrote music separation run:" in captured.out
    row = _read_csv_rows(next(output_root.glob("Song.stem-*/summary.csv")))[0]
    assert row["task"] == REMOVE_VOCALS
    assert row["primary_output_label"] == "no_vocals"


def test_music_separation_default_task_is_extract_vocals() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--input", "song.wav"])

    assert args.task == EXTRACT_VOCALS
