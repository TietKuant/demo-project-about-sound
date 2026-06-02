# Demucs Spike

## Status

Preflight only. No installation has started.

## Purpose

Demucs is the fallback music / vocal separation engine candidate after `audio-separator[cpu]` failed during local installation.

Target task cards:

- Extract Vocals.
- Remove Vocals.
- Split Stems.

## Intended Package And Command

- Package: `demucs`
- Proposed install command:

```bash
.venv-demucs/bin/pip install demucs
```

- Proposed CLI check:

```bash
.venv-demucs/bin/python -m demucs --help
```

## Isolated Environment Plan

- Environment path: `.venv-demucs/`
- Do not modify `.venv/`.
- Do not edit `pyproject.toml`.
- Add `.venv-demucs/` to git ignore rules before creating the environment.

## Expected Input

- Use one bounded local audio / music sample.
- If no legal music sample exists, obtain a short legal or royalty-free sample before quality testing.
- `samples/input_audio/demo_real.wav` may validate CLI execution only. It does not validate separation quality.

## Expected Outputs

Expected user-visible artifacts:

- `vocals.wav` or equivalent vocals stem.
- `no_vocals.wav`, accompaniment, or other stems depending on model output.

Record actual output names before any integration work.

## Runtime Measurement Plan

1. Record input path and duration.
2. Measure wall-clock runtime for one bounded CLI run.
3. Record output paths and file sizes.
4. Confirm outputs are playable.

## Model Download And Cache Risk

- Demucs may download pretrained model weights on first run.
- Record the cache path before any real run if visible.
- Record downloaded model file size before allowing the run to continue if visible.

## Stop Rules

- Do not install into `.venv/`.
- Do not edit `pyproject.toml`.
- Do not integrate into the app.
- Do not change the `EngineResult` / `OutputArtifact` contract yet.
- Do not add UI.
- Stop if installation requires uncontrolled dependency changes.
- Stop if model download is unexpectedly huge.
- Stop if the first bounded run fails, and record the error.
