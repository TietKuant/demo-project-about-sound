# Engine Checkpoint

## Current branch
- feature/engine-layer

## Completed
- Wrapped existing ffmpeg arnndn denoise flow into `FFmpegArnndnEngine`.
- Added `NoisereduceEngine`.
- Added `engine_name` to `DenoiseRequest`.
- Added CLI option `--engine`.
- Current selectable engines:
  - `ffmpeg-arnndn`
  - `noisereduce`

## Local environment tested
- Python 3.11.15 via Homebrew
- FFmpeg 8.1.1 via Homebrew
- noisereduce 3.0.3
- soundfile 0.13.1
- torch 2.2.2
- torchaudio 2.2.2
- deepfilternet installed, command available as `deepFilter`

## Manual tests passed
- `python -m pytest` -> 9 passed
- `noisereduce` processed:
  - `samples/input_audio/demo_real.wav`
  - `samples/real_audio/indoor_fan.m4a`
  - `samples/input_video/demo_video.mp4`
- DeepFilterNet CLI processed:
  - input: `tmp/deepfilter-test/indoor_fan_48k.wav`
  - output: `tmp/deepfilter-test/indoor_fan_48k_DeepFilterNet3.wav`

## Important notes
- `samples/input_audio/demo.wav` is not a valid WAV file. It is ASCII text and should not be used for audio tests.
- `demo_real.wav` is a valid WAV file.
- DeepFilterNet CLI command on this machine is `deepFilter`, not `deep-filter`.
- DeepFilterNet requires PyTorch.
- DeepFilterNet model was loaded from local cache:
  `~/Library/Caches/DeepFilterNet/DeepFilterNet3`
- DeepFilterNet ran on CPU.
- `noisereduce` may emit a runtime warning for very short or quiet files, but the pipeline can still complete.

## Current MVP engine roadmap
1. `ffmpeg-arnndn` - existing fallback engine.
2. `noisereduce` - traditional spectral-gating baseline.
3. `deepfilternet` - pretrained ML speech enhancement engine.

## Not in MVP
- UI
- classifier / PANNs / YAMNet
- source separation
- training / fine-tuning
- ClearerVoice
- AudioSep

## DeepFilterNet pipeline integration
- Added `DeepFilterNetCliEngine`.
- Engine name: `deepfilternet`.
- Uses CLI command: `deepFilter`.
- Pipeline manual test passed:
  `python -m src.pipeline.run_pipeline samples/real_audio/indoor_fan.m4a --output-dir tmp/manual-runs --engine deepfilternet`
- Output:
  `tmp/manual-runs/indoor_fan.denoised.wav`
- Unit/smoke tests after integration:
  `python -m pytest` -> 11 passed.

## DeepFilterNet pipeline integration
- Added `DeepFilterNetCliEngine`.
- Engine name: `deepfilternet`.
- Uses CLI command: `deepFilter`.
- Pipeline manual test passed:
  `python -m src.pipeline.run_pipeline samples/real_audio/indoor_fan.m4a --output-dir tmp/manual-runs --engine deepfilternet`
- Output:
  `tmp/manual-runs/indoor_fan.denoised.wav`
- Unit/smoke tests after integration:
  `python -m pytest` -> 11 passed.
