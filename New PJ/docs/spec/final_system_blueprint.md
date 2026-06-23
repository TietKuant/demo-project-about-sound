# Final System Blueprint

## 1. Product Goal

Build a Speech Restoration Studio for audio and video files.

The product restores speech clarity when recordings contain background noise or degraded voice quality.

The product is:

- focused on speech restoration
- usable with audio and video input
- built around tested restoration modes

The product is not:

- a general all-sound cleaner
- a source-separation product
- only a benchmark CLI

## 2. Required Product Modes

### Fast Mode

- Engine: DeepFilterNet.
- Purpose: practical speech enhancement with acceptable runtime.

### Studio / Restoration Mode

- Engine: not fixed yet.
- Selection method: local bake-off against restoration candidates.
- Purpose: stronger restoration quality when runtime is less important.

### Compare Mode

- Compare `noisy_input`, Fast Mode, and Studio Mode outputs.
- Provide direct listening comparison for one input file.

### Benchmark Mode

- Run VoiceBank-DEMAND paired speech evaluation.
- Export CSV results and a summary report.
- Include objective metrics and RTF.

## 3. Engine Selection Gate

An engine enters the app only if it:

1. Installs successfully in an isolated environment.
2. Provides a usable CLI or Python API.
3. Runs one local WAV file.
4. Produces a playable output file.
5. Has acceptable runtime for its intended mode.
6. Has a clear license and source repository.

If any required check fails, defer or reject the engine before app integration.

## 4. Current Engine Decisions

- DeepFilterNet: keep as Fast Mode.
- `noisereduce`: keep as DSP baseline.
- `ffmpeg-arnndn`: keep as FFmpeg / RNN baseline.
- LavaSR: deferred after local dependency build failure.
- VoiceRestore: next spike candidate.
- VoiceFixer: later spike.
- Resemble Enhance: later spike.
- `python-audio-separator`: reject as a core engine. Consider only as an optional music / vocal add-on.
- ClearerVoice-Studio: not MVP. Consider for future evaluation or toolkit use.

## 5. Architecture

- `src/engine`: engine adapters behind stable denoise contracts.
- `src/pipeline`: media validation, preparation, engine execution, audio export, and video remux.
- `src/eval`: evaluation metrics and paired-speech analysis helpers.
- `scripts`: benchmark, bake-off, and comparison workflows.
- App UI: build only after the Studio Mode engine decision.

## 6. Immediate Roadmap

1. Record the LavaSR spike result.
2. Lock this system blueprint.
3. Spike VoiceRestore next.
4. If VoiceRestore fails the engine selection gate, spike VoiceFixer.
5. Choose the Studio Mode engine from the bake-off results.
6. Build the UI after the engine decision.
