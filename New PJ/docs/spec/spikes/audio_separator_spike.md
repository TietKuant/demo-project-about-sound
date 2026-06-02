# Audio Separator Spike

## Status

Preflight only. Installation and model execution have not started.

The pip dry-run could not reach PyPI from the sandbox because DNS resolution for `pypi.org` failed. Dependency expansion is therefore unknown. Stop before installation.

## Intended Package

- Package: `audio-separator[cpu]`
- Project: `python-audio-separator`
- Purpose: bounded spike for the MVP music-separation task card.
- Expected CLI: `audio-separator`

## Isolated Environment

- Environment path: `.venv-audio-separator/`
- Python: `3.11.15`
- Pip: `26.1`
- Main project environment `.venv/` must not be modified.

## Proposed Install Command

```bash
.venv-audio-separator/bin/pip install 'audio-separator[cpu]'
```

## Preflight Commands

```bash
.venv-audio-separator/bin/pip index versions audio-separator
.venv-audio-separator/bin/pip install --dry-run 'audio-separator[cpu]'
```

## Pip Dry-Run Result

- Result: blocked.
- Cause: sandbox DNS failure while resolving `pypi.org`.
- Pip reported no matching distribution only because package index access failed.
- Dependency list could not be inspected.

## Expected Dependencies

The exact dependency set is unknown until the dry-run succeeds.

Expected categories based on the official project installation path:

- CPU inference runtime such as ONNX Runtime.
- Audio processing dependencies.
- FFmpeg available on the local machine.
- Package CLI and pretrained-model handling dependencies.

Do not approve installation until pip prints the resolved package plan.

## Expected First-Run Behavior

- The CLI may download a pretrained separation model on first use.
- The model cache location and download size must be recorded before execution.
- Do not download models during preflight.

## Expected Test Input

- Local bounded input: `samples/input_audio/demo_real.wav`
- Limitation: this repository currently has no dedicated music sample. This file validates install and runtime wiring only, not separation quality.

## Expected Outputs

The first successful run should produce two user-visible WAV artifacts:

- `vocals.wav`
- `instrumental.wav`

If the selected model uses different output names, record the actual paths and map them before any integration work.

## Runtime Measurement Plan

1. Record input path and audio duration.
2. Run one CLI separation command.
3. Measure wall-clock runtime.
4. Record output paths and file sizes.
5. Confirm both outputs are playable WAV files.
6. Record model name, model cache path, and any downloaded file size.

## Stop Rules

- Do not install if the dry-run shows obviously huge or uncontrolled dependency changes.
- Do not install into `.venv/`.
- Do not edit `pyproject.toml`.
- Do not integrate into the app.
- Do not change engine contracts.
- Do not add UI.
- Do not download models during preflight.
- If installation or the first bounded run fails, record the error and stop.
