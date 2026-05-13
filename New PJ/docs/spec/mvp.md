# MVP Spec: Offline Audio/Video Denoise Demo

## Summary
The MVP is a single-run, local workflow for denoising one short audio or video file at a time. It is intentionally narrow: one default denoise engine, one CLI-first execution path, local files only, and simple artifacts that prove the architecture works.

## In Scope
- accept one short audio or video file from the local filesystem
- inspect the input and determine whether audio extraction is required
- extract or normalize audio with `ffmpeg` when needed
- run a default pretrained denoise engine offline
- export cleaned audio
- remux cleaned audio back into the original video container when requested
- return a small result contract and persist predictable artifacts

## Out of Scope
- UI or browser workflow
- realtime processing
- training, tuning, or model selection UX
- batch execution
- background workers or queues
- objective quality scoring promises
- network service deployment

## End-to-End Flow
1. `src/api` receives a `DenoiseRequest` from the CLI layer.
2. `src/io` validates paths, checks that the file exists, and derives output locations.
3. `src/media` probes the input. If the input is video, it extracts audio to a temporary working file. If needed, it converts audio to the engine's expected format.
4. `src/engine` loads the default `DeepFilterNet` adapter and denoises the prepared audio.
5. `src/media` writes cleaned audio and optionally remuxes it back into the original video container without altering the video stream.
6. `src/eval` performs lightweight post-run checks.
7. `src/storage` finalizes output and temporary artifact handling.
8. `src/api` returns a `DenoiseResult` with final paths and a concise run summary.

## Boundary Contracts

### `DenoiseRequest`
Conceptual request shape for the CLI-first boundary:
- `input_path`
- `output_dir`
- `output_mode`: `audio` or `video`
- `keep_intermediates`: default `false`

### `DenoiseResult`
Conceptual result shape returned at the boundary:
- `status`
- `final_output_path`
- `intermediate_audio_path` optional
- `engine_name`
- `run_summary`

## Module Responsibilities

### `src/api`
- Define the boundary-contract module for the CLI-first flow.
- Parse or map operator intent into `DenoiseRequest`.
- Return `DenoiseResult`.
- Remain transport-agnostic in MVP. It is not a real HTTP service.

### `src/io`
- Validate local input paths and output directories.
- Classify file intent at a basic level for downstream handling.
- Generate deterministic output names and working locations.

### `src/media`
- Probe media metadata.
- Extract audio from video inputs.
- Convert audio into an engine-friendly format when required.
- Export cleaned audio.
- Remux cleaned audio back into the original video container while preserving the original video stream.

### `src/engine`
- Encapsulate the default pretrained denoise engine.
- Expose a minimal load-and-denoise adapter contract.
- Hide engine-specific preprocessing or invocation details from the pipeline.

### `src/pipeline`
- Orchestrate the stages in the correct order.
- Propagate clear stage-level failures.
- Coordinate audio-only and video-remux branches through one consistent run path.

### `src/eval`
- Perform minimal demo-only checks after processing.
- Confirm that expected outputs exist and basic media expectations are met.
- Contribute lightweight summary data only; no advanced scoring.

### `src/storage`
- Define temp and output path conventions.
- Manage intermediate file retention based on `keep_intermediates`.
- Keep `tmp/` and `outputs/` responsibilities distinct.

## Supported Input/Output Cases

### Audio Input
- Input: supported local audio file
- Processing: optional normalization, then denoise
- Output: cleaned audio written to `outputs/`

### Video Input
- Input: supported local video file with an audio track
- Processing: extract audio, denoise, remux
- Output: video written to `outputs/` with the original video stream preserved and cleaned audio substituted

## Operational Constraints
- local filesystem inputs only
- short files only, typically under 1-2 minutes for MVP planning
- one default engine only
- synchronous single-run execution
- offline inference path
- failure stops the run at the current stage with a clear error

## Artifact Policy
- `outputs/` stores final cleaned audio or remuxed video artifacts.
- `tmp/` stores extracted audio, normalized working audio, and other intermediates.
- A minimal run summary may be stored with the final artifact or alongside intermediates, as long as the location is documented consistently.

## Failure Expectations
- unsupported or unreadable files fail before denoise begins
- missing `ffmpeg` fails at media preparation with a clear setup/runtime error
- missing engine dependency or weights fails before or during engine execution with a clear setup/runtime error
- remux failure does not silently discard the cleaned audio artifact if it was already produced

## MVP Verification Scenarios
- audio input with a supported format produces cleaned audio in `outputs/`
- video input with a supported format produces remuxed video while preserving the original video stream
- invalid or unsupported input fails before denoising with a clear error
- missing engine dependency or missing `ffmpeg` fails with a clear setup/runtime error
- pipeline cleans up or isolates intermediates in `tmp/` according to `keep_intermediates`
- future smoke checks belong under `tests/smoke`
