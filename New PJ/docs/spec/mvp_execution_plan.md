# MVP Execution Plan

## Relationship To The Master Roadmap

- `docs/spec/master_product_roadmap.md` defines the product vision and architecture rules.
- `docs/spec/mvp_execution_plan.md` defines the practical build order for implementation.

The execution plan does not replace the master roadmap. It turns the roadmap into a focused MVP sequence.

## MVP Vertical Slice

Build one complete user flow:

`upload audio/video -> run Fast Mode -> listen before/after -> export output -> show runtime/RTF -> show basic spectrogram/report later`

Fast Mode uses DeepFilterNet through the existing pipeline.

## Phase 0: Lock Current State

### Goal

Record the current working system before UI work begins.

### Tickets

- Save the master product roadmap.
- Record the LavaSR spike as deferred after its dependency build failure.
- Confirm the existing engine adapters and benchmark scripts are committed.
- Confirm the test suite passes.

### Done Criteria

- Roadmap documents are stored under `docs/spec/`.
- LavaSR status is recorded as deferred.
- Current tests pass.

## Phase 1: Benchmark Smoke Locked

### Goal

Keep one repeatable evaluation smoke benchmark for regression checks.

### Tickets

- Confirm the VoiceBank subset manifest loads correctly.
- Run the benchmark smoke workflow with `noisy_input`, `ffmpeg-arnndn`, `noisereduce`, and `deepfilternet`.
- Confirm CSV output includes metrics, runtime, audio duration, and RTF.
- Keep the benchmark runner separate from UI code.

### Done Criteria

- Benchmark smoke run completes.
- CSV contains one row per sample and selected engine.
- `noisy_input` is present as the no-processing baseline.
- Failed engine runs are recorded without stopping the benchmark.

## Phase 2: Minimal Compare Workflow

### Goal

Run the current restoration paths on one file and collect outputs for direct listening comparison.

### Tickets

- Add a compare workflow that reuses the existing pipeline.
- Run `noisy_input`, Fast Mode, and the current comparison engines on one prepared input.
- Record output paths, status, errors, runtime, and RTF.
- Keep generated files isolated under a compare-run output directory.

### Done Criteria

- One command produces comparison outputs for one input file.
- One failed engine does not stop other runs.
- Results identify each engine output clearly.

## Phase 3: Minimal Gradio UI

### Goal

Provide a Restore tab for the Fast Mode vertical slice.

### Tickets

- Add audio/video upload.
- Run Fast Mode through the existing pipeline.
- Add before/after audio playback.
- Add output export.
- Show runtime and RTF.
- Add queue and concurrency controls for long-running tasks.

### Done Criteria

- User can upload audio or video.
- User can run Fast Mode.
- User can listen before and after processing.
- User can export the result.
- Runtime and RTF are visible.
- UI does not call engine adapters directly.

## Phase 4: Video Support And Visualization

### Goal

Complete the demo workflow for video and add basic visual evidence.

### Tickets

- Confirm video remux works from the UI flow.
- Add basic spectrogram generation for before/after audio.
- Generate plots with Matplotlib using file-safe non-interactive rendering.
- Close Matplotlib figures after saving.
- Add a small run report view.

### Done Criteria

- Video upload produces an exportable remuxed video.
- Before/after spectrogram images are generated safely.
- Report view shows engine, runtime, RTF, and output path.

## Phase 5: Report And Demo Package

### Goal

Prepare a stable graduation-project demo and evaluation package.

### Tickets

- Run the agreed VoiceBank benchmark size.
- Summarize results by engine.
- Prepare demo inputs for audio and video.
- Record known limitations.
- Confirm clean setup and demo steps.

### Done Criteria

- Benchmark CSV and summary are available.
- Demo runs from a documented sequence.
- Audio and video examples are ready.
- Limitations are stated directly.
- Tests pass.

## Implementation Guardrails

- UI must call the pipeline, not engine adapters directly.
- All engines operate on prepared WAV input.
- Do not use MP3 intermediates.
- Heavy engines must stay isolated behind adapters and isolated environments.
- Studio engine bake-off is MVP-plus and must not block the Fast Mode UI.
- Matplotlib plots must use non-interactive file-safe generation and close figures.
- Gradio long-running tasks must use queue and concurrency controls.
- Signal alignment must use `scipy.signal.correlate` or FFT-based correlation, not Python loops.

## Source-Separation Boundary

Demucs and `python-audio-separator` are music / source-separation tools. They are not core speech restoration engines.

They are later add-ons only. They are not part of the MVP core.

## Immediate Next Order

1. Save the roadmap.
2. Record LavaSR as deferred.
3. Confirm benchmark smoke.
4. Implement the minimal compare workflow.
5. Implement the minimal Gradio Restore tab.
6. Add visualization and report output.
