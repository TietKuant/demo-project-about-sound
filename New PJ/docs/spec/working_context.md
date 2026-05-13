# Working Context

## Project Goal

- Build a minimal local demo that accepts one short audio or video file, prepares audio, denoises it, and outputs cleaned audio or a remuxed video.

## Scope Constraints

- Offline only.
- Single-file CLI flow only.
- No UI.
- No HTTP service.
- No realtime logic.
- No training or fine-tuning.
- Keep architecture-first module boundaries intact.

## Already Completed

- Docs and architecture notes are in place.
- Python project scaffold and module layout exist.
- CLI pipeline exists.
- `ffmpeg` probe and audio preparation are implemented.
- Real denoise path currently uses `ffmpeg` `arnndn` with `models/arnndn/std.rnnn`.
- Audio output and video remux output are real artifacts.
- Manifest/run-summary generation exists.
- Smoke tests cover core scaffold behavior.

## Not Yet Completed

- DeepFilterNet integration is still not implemented.
- No engine selection layer.
- No cleanup policy for intermediates yet.
- No broader format-compatibility hardening.
- End-to-end local validation exists for basic audio and video cases, but not yet with meaningful noisy demo samples.

## Current Active Phase

- Phase 2B: real local denoise path using `ffmpeg` `arnndn`, while keeping the DeepFilterNet adapter in place as future architecture.

## Scope-Creep Rules

- Do not change docs unless explicitly requested.
- Do not add UI, HTTP, or service infrastructure.
- Do not add realtime or streaming behavior.
- Do not add training code or model-management complexity.
- Do not replace the current architecture with a shortcut implementation.
- Keep changes small, explicit, and local to the current phase goal.
