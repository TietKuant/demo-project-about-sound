from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts.router_labeling_ui import create_app


@pytest.fixture
def labeling_ui(tmp_path: Path) -> tuple[TestClient, Path, list[dict[str, str]]]:
    first_media = tmp_path / "first.wav"
    second_media = tmp_path / "second.wav"
    first_media.write_bytes(b"first-audio")
    second_media.write_bytes(b"second-audio")
    rows = [
        {
            "sample_id": "sample-1",
            "path": str(first_media),
            "filename": "interview.wav",
            "router_label": "speech_noisy_general",
            "router_confidence": "0.82",
            "router_accepted": "true",
            "router_reason": "accepted_router_prediction",
            "human_label": "",
            "workflow_label": "",
            "notes": "listen for traffic",
        },
        {
            "sample_id": "sample-2",
            "path": str(second_media),
            "filename": "music.wav",
            "router_label": "music_with_vocals",
            "router_confidence": "0.91",
            "router_accepted": "true",
            "router_reason": "accepted_router_prediction",
            "human_label": "music_with_vocals",
            "workflow_label": "music_separation_package",
            "notes": "",
        },
    ]
    manifest_path = tmp_path / "router_human_labels.local.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return TestClient(create_app(manifest_path)), manifest_path, rows


def test_status_reports_labeling_counters(labeling_ui):
    client, _, _ = labeling_ui

    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json() == {
        "labeled": 1,
        "total": 2,
        "remaining": 1,
        "label_counts": {"music_with_vocals": 1},
    }


def test_index_contains_filename_and_label_buttons(labeling_ui):
    client, _, _ = labeling_ui

    response = client.get("/")

    assert response.status_code == 200
    assert "interview.wav" in response.text
    assert "speech_clean" in response.text
    assert "speech_noisy_general" in response.text
    assert "unknown_mixed" in response.text
    assert "1 labeled / 2 total" in response.text


def test_known_media_is_served(labeling_ui):
    client, _, _ = labeling_ui

    response = client.get("/media/sample-1")

    assert response.status_code == 200
    assert response.content == b"first-audio"


def test_label_update_writes_manifest_and_creates_backup(labeling_ui):
    client, manifest_path, original_rows = labeling_ui

    response = client.post(
        "/api/label/sample-1",
        json={
            "human_label": "speech_noisy_general",
            "notes": "confirmed road noise",
        },
    )

    assert response.status_code == 200
    assert response.json()["workflow_label"] == "speech_cleanup"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["human_label"] == "speech_noisy_general"
    assert rows[0]["workflow_label"] == "speech_cleanup"
    assert rows[0]["notes"] == "confirmed road noise"

    backup_path = Path(f"{manifest_path}.bak")
    assert backup_path.is_file()
    with backup_path.open(newline="", encoding="utf-8") as handle:
        backup_rows = list(csv.DictReader(handle))
    assert backup_rows == original_rows


def test_notes_update_writes_manifest_and_creates_backup(labeling_ui):
    client, manifest_path, _ = labeling_ui

    response = client.post(
        "/api/notes/sample-1",
        json={"notes": "new note"},
    )

    assert response.status_code == 200
    assert response.json()["notes"] == "new note"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["notes"] == "new note"
    assert Path(f"{manifest_path}.bak").is_file()


def test_unknown_sample_id_returns_404(labeling_ui):
    client, _, _ = labeling_ui

    assert client.get("/media/not-in-manifest").status_code == 404
    assert (
        client.post(
            "/api/label/not-in-manifest",
            json={"human_label": "speech_clean"},
        ).status_code
        == 404
    )
