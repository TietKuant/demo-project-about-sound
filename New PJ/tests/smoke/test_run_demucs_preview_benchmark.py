"""Smoke tests for the Demucs MUSDB preview benchmark runner."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from scripts.run_demucs_preview_benchmark import run_demucs_preview_benchmark


def _write_manifest(path: Path, rows: list[tuple[str, Path]]) -> None:
    lines = ["track_id,split,input_path,dataset,source"]
    for track_id, input_path in rows:
        lines.append(f"{track_id},test,{input_path},MUSDB18-7-STEMS,sigsep-mus-db release preview")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_fake_outputs(command: list[str]) -> None:
    output_dir = Path(command[command.index("-o") + 1])
    model = command[command.index("-n") + 1]
    track_dir = output_dir / model / Path(command[-1]).stem
    track_dir.mkdir(parents=True, exist_ok=True)
    (track_dir / "vocals.wav").write_bytes(b"vocals")
    (track_dir / "no_vocals.wav").write_bytes(b"no-vocals")


def test_demucs_preview_benchmark_writes_success_summaries_and_expected_command(tmp_path: Path) -> None:
    input_path = tmp_path / "Alpha.stem.mp4"
    input_path.write_bytes(b"preview")
    manifest_path = tmp_path / "manifest.csv"
    output_root = tmp_path / "outputs"
    _write_manifest(manifest_path, [("Alpha", input_path)])
    commands: list[list[str]] = []

    def mock_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert kwargs == {"text": True, "capture_output": True, "check": False}
        _write_fake_outputs(command)
        return subprocess.CompletedProcess(command, 0, "completed", "")

    with patch("scripts.run_demucs_preview_benchmark.subprocess.run", side_effect=mock_run):
        run_demucs_preview_benchmark(
            manifest_path=manifest_path,
            output_root=output_root,
            demucs_python=Path("/isolated/bin/python"),
        )

    command = commands[0]
    assert command[:3] == ["/isolated/bin/python", "-m", "demucs"]
    assert command[command.index("--two-stems") + 1] == "vocals"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("-j") + 1] == "1"
    assert command[command.index("-n") + 1] == "htdemucs"

    csv_rows = _read_csv_rows(output_root / "summary.csv")
    json_rows = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert csv_rows == json_rows
    assert csv_rows[0]["status"] == "success"
    assert csv_rows[0]["returncode"] == "0"
    assert csv_rows[0]["vocals_path"].endswith("/vocals.wav")
    assert csv_rows[0]["no_vocals_path"].endswith("/no_vocals.wav")


def test_demucs_preview_benchmark_records_failed_returncode(tmp_path: Path) -> None:
    input_path = tmp_path / "Beta.stem.mp4"
    input_path.write_bytes(b"preview")
    manifest_path = tmp_path / "manifest.csv"
    output_root = tmp_path / "outputs"
    _write_manifest(manifest_path, [("Beta", input_path)])

    with patch(
        "scripts.run_demucs_preview_benchmark.subprocess.run",
        return_value=subprocess.CompletedProcess([], 7, "", "demucs failed"),
    ):
        run_demucs_preview_benchmark(manifest_path=manifest_path, output_root=output_root)

    row = _read_csv_rows(output_root / "summary.csv")[0]
    assert row["status"] == "failed"
    assert row["returncode"] == "7"
    assert "demucs failed" in row["error"]
    assert row["vocals_path"] == ""
    assert row["no_vocals_path"] == ""


def test_demucs_preview_benchmark_applies_limit(tmp_path: Path) -> None:
    first_input = tmp_path / "Alpha.stem.mp4"
    second_input = tmp_path / "Beta.stem.mp4"
    first_input.write_bytes(b"preview")
    second_input.write_bytes(b"preview")
    manifest_path = tmp_path / "manifest.csv"
    output_root = tmp_path / "outputs"
    _write_manifest(manifest_path, [("Alpha", first_input), ("Beta", second_input)])
    commands: list[list[str]] = []

    def mock_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        _write_fake_outputs(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    with patch("scripts.run_demucs_preview_benchmark.subprocess.run", side_effect=mock_run):
        run_demucs_preview_benchmark(manifest_path=manifest_path, output_root=output_root, limit=1)

    assert len(commands) == 1
    assert [row["track_id"] for row in _read_csv_rows(output_root / "summary.csv")] == ["Alpha"]
