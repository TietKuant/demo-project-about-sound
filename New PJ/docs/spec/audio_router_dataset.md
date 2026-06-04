# Audio Router Dataset

## Purpose

This dataset layer prepares training data for a future ML audio router.
The router should predict the broad content type of an input file before a task is selected.
It is not integrated into the app yet, but the project now includes baseline training and standalone inference for this router.

## Router Labels

The initial router labels are:

- `speech_noise`: noisy speech examples from the target-noise suppression dataset.
- `music`: music or music-video examples from MUSDB18 preview.
- `environment_noise`: non-speech environmental sound examples from ESC-50 or UrbanSound8K.

These labels describe input content type. They are not the same as final user intent.

## Conservative Task Mapping

Task mapping must remain separate from router prediction.
For example, a `music` prediction may suggest `extract_vocals`, but the user may still choose another supported task.
A `speech_noise` prediction may suggest `clean_voice` or the experimental `target_noise_suppression` task, but the app must avoid claiming automatic intent detection is solved.

## Source Datasets

The builder can use any subset of these local sources:

- `target_noise_v1`: uses `mixed_path` as `speech_noise`.
- `musdb18_preview`: scans `.stem.mp4` files as `music`.
- `ESC-50`: reads `meta/esc50.csv` and maps rows to `environment_noise`.
- `UrbanSound8K`: reads metadata and maps rows to `environment_noise`.

The output manifest stores paths relative to the manifest CSV location when possible.
Generated local manifests should normally use a `.local.csv` suffix and must not be committed.

## Output Columns

```text
sample_id,input_path,router_label,source,split,notes
```

The manifest is a structured dataset for router training or analysis.
It does not replace the custom target-noise suppressor dataset, which is paired waveform data for denoising.

## Current Scope

This stage adds the audio router dataset manifest layer and a baseline router training script.
It does not add app UI changes or task-runner integration yet.

## S9A-2 Baseline Training

S9A-2 adds a baseline training script for the future router.
The script trains a small PyTorch classifier from lightweight audio features:

- duration,
- RMS energy,
- zero crossing rate,
- spectral centroid,
- spectral bandwidth.

The router predicts content type only: `speech_noise`, `music`, or `environment_noise`.
It does not decide final user intent by itself.
The current feature set is intentionally small and limited, so results should be treated as a baseline rather than a final analyzer.
Feature extraction can fail for unreadable or unsupported media rows; the trainer writes `feature_rows.csv` and records manifest/success/failed row counts in `metrics.json`.

App integration is future work.

## Router Inference

The trained checkpoint can be tested with `scripts/run_audio_router.py`.
The script loads `checkpoint.pt`, extracts the same lightweight feature set from one audio/video input, normalizes the features using saved training statistics, and writes a JSON summary with:

- predicted content label,
- confidence,
- per-label probabilities,
- extracted feature values.

This predicts broad content type only.
It does not decide final user intent and does not run a processing engine.
App integration is the next step after standalone validation.
