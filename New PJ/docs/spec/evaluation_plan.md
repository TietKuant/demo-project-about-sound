# Evaluation Plan

## Problem Scope

This project evaluates single-channel speech denoising / speech enhancement for local audio and video files.

- Input: audio or video containing speech with background noise.
- Output: clearer speech with reduced background noise.
- Focus: practical offline comparison for a graduation project, not production-grade benchmarking.

## Engines To Evaluate

- `ffmpeg-arnndn`: fallback RNN denoise filter available through `ffmpeg`.
- `noisereduce`: traditional spectral-gating baseline.
- `deepfilternet`: pretrained ML speech enhancement engine.

## Dataset Strategy

- Main benchmark: VoiceBank-DEMAND / Valentini clean-noisy paired speech subset.
- Real local samples: demo and listening comparison only.
- Do not use AudioSet, MUSDB18, or UrbanSound8K as the MVP benchmark.

The benchmark should use paired clean/noisy speech so intrusive metrics can compare each enhanced output against a clean reference.

## Metrics

- STOI: speech intelligibility estimate.
- SI-SDR: signal quality / distortion ratio.
- SNR improvement: noise reduction relative to the noisy input.
- PESQ: optional later, because setup/licensing can be more involved.

Do not compute intrusive metrics on real local samples unless a clean reference is available.

## Benchmark Output

The benchmark should produce:

- CSV with `sample_id`, `engine`, `runtime_sec`, metrics, `output_path`, and `status/error`.
- Summary markdown/table comparing engines across the benchmark subset.
- Optional spectrograms later for visual explanation.

## MVP Done Criteria

- Load at least 20 clean/noisy pairs.
- Run all 3 engines on each pair.
- Export benchmark CSV.
- Produce summary table.
- Tests pass.

## Out Of Scope For Now

- New engines.
- UI.
- Training or fine-tuning.
- Classifier/router.
- Source separation.
