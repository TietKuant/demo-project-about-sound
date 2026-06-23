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

## NumPy Compatibility Fix Result

- Command:
  `.venv-demucs/bin/pip install 'numpy<2'`

- Result:
  Passed.

- Change:
  `numpy-2.4.6` was replaced with `numpy-1.26.4`.

- Verification:
  `import numpy`, `import torch`, and `import demucs` passed.
  `.venv-demucs/bin/python -m demucs --help` printed the expected CLI help without the previous NumPy `_ARRAY_API` warning.

- Decision:
  The Demucs isolated environment is now ready for one bounded separation run.
  Do not integrate into the app yet.
  Do not change the main `.venv/`.
  Do not edit `pyproject.toml`.

## Bounded Run Attempt 1 Result

- Command 1:
  `.venv-demucs/bin/python -m demucs --two-stems vocals --device cpu -j 1 --segment 7 -o outputs/spikes/demucs-smoke samples/input_audio/demo_real.wav`

- Command 2:
  `.venv-demucs/bin/python -m demucs --two-stems vocals --device cpu -j 1 -o outputs/spikes/demucs-smoke samples/input_audio/demo_real.wav`

- Result:
  Failed.

- Model download:
  The default `htdemucs` model downloaded successfully to the local torch cache.
  Observed download size: `80.2M`.

- Direct failure:
  Both runs failed inside Demucs `htdemucs.py` / `hdemucs.py` with `AssertionError` during `pad1d(...)`.

- Interpretation:
  This does not invalidate Demucs installation.
  Demucs installed, imported, printed CLI help, downloaded the pretrained model, and reached model inference.
  The current test input `samples/input_audio/demo_real.wav` is not a valid music-separation quality test input and appears unsuitable for this bounded run.

- Decision:
  Stop using `samples/input_audio/demo_real.wav` for Demucs execution tests.
  Prepare a short legal music sample before the next bounded run.
  The next Demucs test input should be a 10-30 second music clip, preferably stereo WAV, with vocals/instrumental content.
  If no legal music sample is available, generate a simple synthetic music-like smoke sample only to test artifact generation, not separation quality.

## Bounded Run Attempt 2 Result

- Input:
  `samples/input_audio/music_smoke.wav`

- Input note:
  This is a synthetic stereo smoke sample. It is suitable for validating Demucs execution and output artifact generation only. It is not suitable for separation-quality evaluation.

- Command:
  `.venv-demucs/bin/python -m demucs --two-stems vocals --device cpu -j 1 -o outputs/spikes/demucs-smoke samples/input_audio/music_smoke.wav`

- Result:
  Passed.

- Runtime:
  Approximately `19.630` seconds wall-clock.

- Output artifacts:
  - `outputs/spikes/demucs-smoke/htdemucs/music_smoke/vocals.wav`
  - `outputs/spikes/demucs-smoke/htdemucs/music_smoke/no_vocals.wav`

- Decision:
  Demucs passes the local artifact-generation smoke test.
  It is acceptable as the fallback music/vocal separation engine candidate.
  Before app integration, run one legal real music sample or MUSDB sample to validate task suitability.
  Do not commit generated output files.
  Do not commit the synthetic smoke sample unless the project explicitly decides to keep generated smoke fixtures.

## Bounded Run Attempt 3 Result — MUSDB18 Preview

- Input:
  `data/external/musdb18-preview/test/Cristina Vane - So Easy.stem.mp4`

- Input source:
  MUSDB18 7-second preview dataset downloaded from the official `sigsep-mus-db` release.

- Command:
  `.venv-demucs/bin/python -m demucs --two-stems vocals --device cpu -j 1 -o outputs/spikes/demucs-musdb-preview "data/external/musdb18-preview/test/Cristina Vane - So Easy.stem.mp4"`

- Result:
  Passed.

- Runtime:
  Approximately `18.676` seconds wall-clock.

- Output artifacts:
  - `outputs/spikes/demucs-musdb-preview/htdemucs/Cristina Vane - So Easy.stem/vocals.wav`
  - `outputs/spikes/demucs-musdb-preview/htdemucs/Cristina Vane - So Easy.stem/no_vocals.wav`

- Decision:
  Demucs passes the domain-valid MUSDB18 preview separation smoke test.
  Demucs is accepted as the fallback music/vocal separation engine candidate for the MVP.
  Next implementation phase should design `EngineResult` / `OutputArtifact` V2 before app integration.
  Do not commit generated output files or external dataset files.
