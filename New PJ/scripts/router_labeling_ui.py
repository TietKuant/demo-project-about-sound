#!/usr/bin/env python3
"""Minimal local UI for labeling audio router manifest rows."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel


LABEL_WORKFLOWS = {
    "speech_clean": "no_process",
    "speech_noisy_general": "speech_cleanup",
    "speech_target_noise": "speech_cleanup",
    "music_with_vocals": "music_separation_package",
    "environment_only": "no_process",
    "unknown_mixed": "safe_abstain",
}
LABEL_SHORTCUTS = tuple(enumerate(LABEL_WORKFLOWS.items(), start=1))
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
EDITABLE_COLUMNS = ("human_label", "workflow_label", "notes")


class LabelUpdate(BaseModel):
    human_label: str
    notes: str | None = None


class NotesUpdate(BaseModel):
    notes: str


class ManifestSession:
    """In-memory manifest state with immediate durable CSV updates."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
        if not self.manifest_path.is_file():
            raise ValueError(f"Manifest is not a file: {self.manifest_path}")
        self.backup_path = Path(f"{self.manifest_path}.bak")
        self._backup_created = False
        self._lock = threading.Lock()
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.fieldnames = list(reader.fieldnames or [])
            if "sample_id" not in self.fieldnames or "path" not in self.fieldnames:
                raise ValueError("Manifest must contain sample_id and path columns.")
            self.rows = [dict(row) for row in reader]
        for column in EDITABLE_COLUMNS:
            if column not in self.fieldnames:
                self.fieldnames.append(column)
        self._row_indexes: dict[str, int] = {}
        self._media_paths: dict[str, Path] = {}
        for index, row in enumerate(self.rows):
            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id or sample_id in self._row_indexes:
                raise ValueError("Manifest sample_id values must be non-empty and unique.")
            self._row_indexes[sample_id] = index
            raw_path = Path(row.get("path") or "").expanduser()
            if not raw_path.is_absolute():
                raw_path = self.manifest_path.parent / raw_path
            media_path = raw_path.resolve(strict=False)
            if media_path.is_file():
                self._media_paths[sample_id] = media_path

    def status(self) -> dict[str, Any]:
        labeled_rows = [
            row for row in self.rows if (row.get("human_label") or "").strip()
        ]
        counts = Counter(
            (row.get("human_label") or "").strip() for row in labeled_rows
        )
        total = len(self.rows)
        labeled = len(labeled_rows)
        return {
            "labeled": labeled,
            "total": total,
            "remaining": total - labeled,
            "label_counts": dict(sorted(counts.items())),
        }

    def row_at(self, index: int) -> tuple[int, dict[str, str]]:
        if not self.rows:
            raise HTTPException(status_code=404, detail="Manifest has no rows.")
        bounded_index = min(max(index, 0), len(self.rows) - 1)
        return bounded_index, self.rows[bounded_index]

    def media_path(self, sample_id: str) -> Path:
        path = self._media_paths.get(sample_id)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="Unknown sample_id.")
        return path

    def next_unlabeled_index(self, current_index: int) -> int:
        if not self.rows:
            return 0
        for offset in range(1, len(self.rows) + 1):
            index = (current_index + offset) % len(self.rows)
            if not (self.rows[index].get("human_label") or "").strip():
                return index
        return min(max(current_index, 0), len(self.rows) - 1)

    def update_label(
        self,
        sample_id: str,
        human_label: str,
        notes: str | None,
    ) -> dict[str, Any]:
        workflow_label = LABEL_WORKFLOWS.get(human_label)
        if workflow_label is None:
            raise HTTPException(status_code=422, detail="Unsupported human_label.")
        with self._lock:
            index = self._index_for(sample_id)
            row = self.rows[index]
            row["human_label"] = human_label
            row["workflow_label"] = workflow_label
            if notes is not None:
                row["notes"] = notes
            self._write()
            return {
                "sample_id": sample_id,
                "human_label": human_label,
                "workflow_label": workflow_label,
                "next_index": self.next_unlabeled_index(index),
                "status": self.status(),
            }

    def update_notes(self, sample_id: str, notes: str) -> dict[str, Any]:
        with self._lock:
            index = self._index_for(sample_id)
            self.rows[index]["notes"] = notes
            self._write()
            return {
                "sample_id": sample_id,
                "notes": notes,
                "status": self.status(),
            }

    def _index_for(self, sample_id: str) -> int:
        try:
            return self._row_indexes[sample_id]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown sample_id.") from exc

    def _write(self) -> None:
        if not self._backup_created:
            shutil.copy2(self.manifest_path, self.backup_path)
            self._backup_created = True
        temporary_path = self.manifest_path.with_name(
            f".{self.manifest_path.name}.tmp"
        )
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=self.fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(self.rows)
        temporary_path.replace(self.manifest_path)


def _display(value: object) -> str:
    text = "" if value is None else str(value)
    return html.escape(text) if text else "—"


def _render_page(session: ManifestSession, index: int) -> str:
    if not session.rows:
        return """<!doctype html><html><body><h1>Router labeling</h1>
<p>The manifest contains no rows.</p></body></html>"""
    index, row = session.row_at(index)
    status = session.status()
    sample_id = row["sample_id"]
    media_path = session._media_paths.get(sample_id)
    media_tag = "<p class='missing'>Media file is missing.</p>"
    if media_path is not None:
        tag = "video" if media_path.suffix.lower() in VIDEO_SUFFIXES else "audio"
        media_tag = (
            f"<{tag} controls preload='metadata' src='/media/"
            f"{html.escape(sample_id, quote=True)}'></{tag}>"
        )
    label_buttons = "\n".join(
        (
            f"<button class='label-button' data-label='{label}' "
            f"onclick=\"saveLabel('{label}')\">"
            f"<strong>{shortcut}</strong> {_display(label)}"
            f"<span>→ {_display(workflow)}</span></button>"
        )
        for shortcut, (label, workflow) in LABEL_SHORTCUTS
    )
    counts = " · ".join(
        f"{html.escape(label)}: {count}"
        for label, count in status["label_counts"].items()
    ) or "No labels saved yet"
    previous_index = max(index - 1, 0)
    next_index = min(index + 1, len(session.rows) - 1)
    next_unlabeled = session.next_unlabeled_index(index)
    page_data = json.dumps(
        {
            "sampleId": sample_id,
            "index": index,
            "previousIndex": previous_index,
            "nextIndex": next_index,
            "nextUnlabeledIndex": next_unlabeled,
        }
    ).replace("</", "<\\/")
    notes = html.escape(row.get("notes") or "")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Router labeling</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; background: #121417; color: #edf0f2; }}
    main {{ max-width: 1050px; margin: 0 auto; padding: 24px; }}
    header, section {{ background: #1d2126; border: 1px solid #343b43;
      border-radius: 10px; padding: 18px; margin-bottom: 14px; }}
    h1 {{ margin: 0 0 6px; font-size: 22px; }}
    .counter {{ color: #9fe870; font-weight: 700; }}
    .counts, .muted, dt {{ color: #a9b1ba; }}
    dl {{ display: grid; grid-template-columns: 170px 1fr; gap: 7px 14px; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    audio, video {{ width: 100%; max-height: 420px; }}
    .labels {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 9px; }}
    button {{ border: 1px solid #4a535e; background: #282e35; color: #fff;
      border-radius: 7px; padding: 11px 13px; cursor: pointer; text-align: left; }}
    button:hover {{ background: #343c45; }}
    .label-button span {{ display: block; color: #9fe870; margin: 4px 0 0 24px; }}
    .nav {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    textarea {{ box-sizing: border-box; width: 100%; min-height: 90px;
      background: #111418; color: #fff; border: 1px solid #4a535e;
      border-radius: 7px; padding: 10px; }}
    .message {{ min-height: 22px; color: #9fe870; }}
    @media (max-width: 700px) {{
      .labels {{ grid-template-columns: 1fr; }}
      dl {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Router labeling</h1>
    <div class="counter">{status['labeled']} labeled / {status['total']} total /
      {status['remaining']} remaining</div>
    <div class="counts">{html.escape(counts)}</div>
  </header>
  <section>
    <h2>{_display(row.get('filename') or Path(row.get('path') or '').name)}</h2>
    {media_tag}
    <dl>
      <dt>Path</dt><dd>{_display(row.get('path'))}</dd>
      <dt>Router label</dt><dd>{_display(row.get('router_label'))}</dd>
      <dt>Router confidence</dt><dd>{_display(row.get('router_confidence'))}</dd>
      <dt>Router accepted</dt><dd>{_display(row.get('router_accepted'))}</dd>
      <dt>Router reason</dt><dd>{_display(row.get('router_reason'))}</dd>
      <dt>Current human label</dt><dd>{_display(row.get('human_label'))}</dd>
      <dt>Current workflow</dt><dd>{_display(row.get('workflow_label'))}</dd>
    </dl>
  </section>
  <section>
    <h2>Decision</h2>
    <div class="labels">{label_buttons}</div>
    <p class="muted">Keyboard shortcuts 1–6 apply a label and advance.</p>
  </section>
  <section>
    <label for="notes"><strong>Notes</strong></label>
    <textarea id="notes">{notes}</textarea>
    <div class="nav">
      <button onclick="goTo(page.previousIndex)">Previous</button>
      <button onclick="goTo(page.nextIndex)">Next</button>
      <button onclick="goTo(page.nextUnlabeledIndex)">Next unlabeled</button>
      <button onclick="goTo(page.nextIndex)">Skip</button>
      <button onclick="saveNotes()">Save notes</button>
    </div>
    <p id="message" class="message"></p>
  </section>
</main>
<script>
const page = {page_data};
const shortcuts = {json.dumps({str(number): label for number, (label, _) in LABEL_SHORTCUTS})};
function goTo(index) {{ window.location.href = '/?index=' + index; }}
async function request(url, body) {{
  const response = await fetch(url, {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(body)
  }});
  if (!response.ok) throw new Error((await response.json()).detail || 'Save failed');
  return response.json();
}}
async function saveLabel(label) {{
  try {{
    const result = await request('/api/label/' + encodeURIComponent(page.sampleId), {{
      human_label: label,
      notes: document.getElementById('notes').value
    }});
    goTo(result.next_index);
  }} catch (error) {{ document.getElementById('message').textContent = error.message; }}
}}
async function saveNotes() {{
  try {{
    await request('/api/notes/' + encodeURIComponent(page.sampleId), {{
      notes: document.getElementById('notes').value
    }});
    document.getElementById('message').textContent = 'Notes saved.';
  }} catch (error) {{ document.getElementById('message').textContent = error.message; }}
}}
document.addEventListener('keydown', event => {{
  if (event.target.matches('textarea, input, select')) return;
  if (shortcuts[event.key]) {{ event.preventDefault(); saveLabel(shortcuts[event.key]); }}
}});
</script>
</body>
</html>"""


def create_app(manifest_path: str | Path) -> FastAPI:
    session = ManifestSession(manifest_path)
    app = FastAPI(title="Router Labeling UI")
    app.state.manifest_session = session

    @app.get("/", response_class=HTMLResponse)
    def index(index: int = Query(default=0)) -> str:
        return _render_page(session, index)

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return session.status()

    @app.get("/media/{sample_id}")
    def media(sample_id: str) -> FileResponse:
        return FileResponse(session.media_path(sample_id))

    @app.post("/api/label/{sample_id}")
    def update_label(sample_id: str, update: LabelUpdate) -> dict[str, Any]:
        return session.update_label(sample_id, update.human_label, update.notes)

    @app.post("/api/notes/{sample_id}")
    def update_notes(sample_id: str, update: NotesUpdate) -> dict[str, Any]:
        return session.update_notes(sample_id, update.notes)

    return app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local router labeling UI.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8010, type=int)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    import uvicorn

    uvicorn.run(create_app(args.manifest), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
