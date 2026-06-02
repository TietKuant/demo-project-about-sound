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

## Install Dry-Run Result

- Command:
  `.venv-demucs/bin/pip install --dry-run demucs`

- Result:
  Passed.

- Observed package version:
  `demucs-4.0.1`

- Available versions observed:
  `4.0.1`, `4.0.0`, `3.0.6`, `3.0.5`, `3.0.4`, `3.0.3`, `3.0.2`, `3.0.1`, `3.0.0`, `2.0.3`, `2.0.2`, `2.0.1`, `2.0.0`, `0.0.2`, `0.0.1`

- Planned major dependencies:
  - `torch-2.2.2`
  - `torchaudio-2.2.2`
  - `openunmix-1.3.0`
  - `dora_search-0.1.12`
  - `julius-0.2.7`
  - `lameenc-1.8.2`
  - `einops-0.8.2`
  - `numpy-2.4.6`

- Risk assessment:
  The dry-run did not show the `llvmlite` / CMake build failure seen in the `audio-separator[cpu]` spike.
  The main expected risk is PyTorch/Torchaudio install size and first-run pretrained model download.

- Decision:
  Demucs is acceptable to attempt installation in `.venv-demucs/`.
  Do not install into `.venv/`.
  Do not edit `pyproject.toml`.
  Do not integrate into the app during installation.

## Install Attempt 1 Result

- Command:
  `.venv-demucs/bin/pip install demucs`

- Result:
  Passed.

- Installed package:
  `demucs-4.0.1`

- Major installed dependencies observed:
  - `torch-2.2.2`
  - `torchaudio-2.2.2`
  - `openunmix-1.3.0`
  - `dora-search-0.1.12`
  - `numpy-2.4.6`

- Import check:
  `import demucs` passed.

- CLI help check:
  `.venv-demucs/bin/python -m demucs --help` printed the expected CLI help, including output directory, device, segment, jobs, and `--two-stems`.

- Important warning:
  The CLI help command emitted a NumPy/PyTorch compatibility warning:
  `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.6`
  and PyTorch reported `Failed to initialize NumPy: _ARRAY_API not found`.

- Decision:
  Treat installation as a partial pass.
  Do not run a model yet.
  Pin NumPy to `<2` inside `.venv-demucs/` before the first bounded separation run.
  Do not modify the main `.venv/`.
  Do not edit `pyproject.toml`.
  Do not integrate into the app.
