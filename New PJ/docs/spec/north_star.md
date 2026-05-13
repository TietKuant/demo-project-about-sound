# North Star: Minimal Offline Denoise Demo

## Problem Statement
The project needs a realistic demo that shows end-to-end audio or video noise reduction without committing to a large product surface. The demo should prove that a local file can move through intake, media preparation, denoising, and export in a way that is easy to explain to stakeholders and cheap to implement.

## Demo Goal
Produce a local, offline workflow that accepts one short audio or video file, runs a pretrained denoise engine, and writes a usable output artifact:
- audio input becomes cleaned audio
- video input becomes a remuxed video that keeps the original video stream and swaps in cleaned audio

The demo is architecture-first. It should make the system shape obvious before any production concerns are added.

## North-Star User Journey
`input file -> probe/extract -> denoise -> export/remux`

The intended operator experience is:
1. Point a local CLI command at a short audio or video file.
2. Validate the file and inspect basic media metadata.
3. If the input is video, extract and normalize its audio with `ffmpeg`.
4. Run the default pretrained denoise engine offline.
5. Write cleaned audio to `outputs/`.
6. If the input was video and `output_mode=video`, remux the cleaned audio back into the original container while preserving the original video stream.
7. Return a small run summary with output paths and stage status.

## Success Criteria
The proposal/demo is successful if it:
- runs locally with no network requirement during inference
- handles one short file per run
- produces an audible path to improvement for noisy short audio or video content
- preserves the original video stream when remuxing video output
- keeps intermediate files isolated from final outputs
- exposes a simple boundary contract that can later support a local API wrapper without changing the core pipeline

## Design Principles
- Keep one default path. Use one denoise engine, one primary workflow, and a narrow set of input assumptions.
- Separate orchestration from media work. `src/pipeline` coordinates the run, while `src/media` owns `ffmpeg` interactions.
- Treat `src/api` as a contract boundary only. It defines request and result shapes for the CLI flow, not a real HTTP service.
- Keep evaluation demo-focused. `src/eval` only performs lightweight post-run checks and summary generation.
- Prefer explainability over feature breadth. The repository should read like a small system with clear seams, not a feature-complete product.

## Non-Goals
This demo does not include:
- a graphical UI
- realtime or streaming denoise
- microphone/live capture
- model training or fine-tuning
- cloud deployment or multi-user service concerns
- batch processing
- speaker separation, source separation, dereverb, or restoration beyond basic denoise
- advanced metrics dashboards or benchmarking infrastructure

## Default Technical Assumptions
- Primary operator surface is a local CLI.
- `DeepFilterNet` is the default pretrained denoise engine behind an adapter in `src/engine`.
- `ffmpeg` is the media tool for probing, extracting, converting, and remuxing.
- Inputs come from the local filesystem only.
- Outputs are stored locally in `outputs/`, with temporary artifacts in `tmp/`.
