# Product Direction V2

This document is the project source of truth for the next implementation phase.

## 1. Final Vietnamese Title

**Xây dựng ứng dụng xử lý âm thanh trong tệp audio và video bằng kỹ thuật học máy**

## 2. Final English Title

**ML-Based Audio Processing Application for Audio and Video Files**

## 3. Architecture Name

**Engine-Routed ML Audio Processing Box**

## 4. Product Definition

The user uploads a raw audio or video file and selects an audio-processing task. The system prepares WAV audio, routes the task to a suitable ML engine, and exports processed audio or remuxed video.

## 5. Detailed Processing Workflow

1. Input validation.
2. Media probing.
3. Task-aware audio preparation.
4. User task selection.
5. Engine routing.
6. Engine processing.
7. Output artifact handling.
8. Evaluation and reporting.

### Task-Aware Audio Preparation

- Speech cleanup uses 48 kHz mono PCM WAV.
- Music / vocal separation preserves stereo when possible.
- Future event-detection models may require 16 kHz mono audio.
- Do not use MP3 intermediates.
- For video input, process the audio stream and remux the result into video when needed.

## 6. MVP Task Cards

### Clean Voice / Reduce Speech Noise

- Engine: DeepFilterNet.
- Output: restored WAV or remuxed video.

### Extract Vocal / Remove Vocal / Split Stems

- Engine: `python-audio-separator`.
- Output: vocals, instrumental, or multi-stem WAV files.

### Baseline / Evaluation

- `noisy_input` and `noisereduce` are baselines only.
- They are not product engines.

## 7. MVP Datasets

- VoiceBank-DEMAND for speech enhancement evaluation.
- MUSDB18 / MUSDB18-HQ sample for vocal and music separation evaluation.

## 8. EngineResult And OutputArtifact Requirement

The engine layer must support:

- Single-output engines: `restored.wav`.
- Multi-output engines: `vocals.wav`, `instrumental.wav`, `drums.wav`, `bass.wav`, `other.wav`.

The next engine contract must represent output artifacts explicitly instead of assuming one denoised WAV output.

## 9. Local App Architecture

- Media preparation layer.
- Task selection layer.
- Engine router.
- Engine adapters.
- Output artifact layer.
- Evaluation layer.

## 10. Kaggle / External Source Intake Policy

Kaggle can provide datasets, notebooks, and models, but each source type must be classified separately.

- Kaggle datasets are used only when mapped to a project task.
- Kaggle notebooks are implementation references, not automatic academic sources.
- Kaggle models are accepted only when their input, output, task, license, and runtime are clear.
- A model enters the project only through a spike with one local input file, one expected output, runtime measurement, and failure log.

Examples:

- Google SoundStream on Kaggle is a real model but not a current MVP engine. It reconstructs audio from mel-spectrograms for music generation, not speech cleanup or vocal separation.
- ESC-50 and UrbanSound8K are useful later for event detection / routing. They are not suitable for MVP speech-enhancement or music-separation metrics.

## 11. Engine Discovery Workstream

The project will actively scan GitHub, Hugging Face, Kaggle Models, Kaggle notebooks, Kaggle datasets, Papers With Code, arXiv, and official project pages for end-to-end audio-processing engines.

### Candidate Source Types

- GitHub repositories for runnable engines.
- Hugging Face models and Spaces for pretrained models.
- Kaggle models when task, input, output, license, and runtime are clear.
- Kaggle notebooks as implementation references only.
- Kaggle datasets only when mapped to a task.
- Papers and arXiv for understanding model category and limitations.

### Engine Intake Gate

A candidate engine can enter the spike queue only if it:

1. Maps to a defined task card.
2. Has clear input and output.
3. Has pretrained weights or requires no training.
4. Can run locally or in an isolated environment.
5. Has an acceptable license.
6. Produces a user-visible output artifact, unless it is an analyzer / router.
7. Has a bounded spike: one input file, one expected output, runtime measured, and failure logged.

### Current Candidate Queue

#### MVP Candidate

- DeepFilterNet.
- `python-audio-separator`.

#### MVP-Plus

- Demucs / HTDemucs direct CLI if `python-audio-separator` fails.
- YAMNet / EfficientAT for analyzer / router.
- VoiceFixer / VoiceRestore for degraded speech restoration.

#### Future

- AudioSep / LASS for query-based target-sound separation.
- Target-speaker extraction models.

#### Rejected / Non-MVP Examples

- Google SoundStream on Kaggle: not a current cleanup or separation task.
- Spleeter: dependency risk.
- ClearerVoice-Studio: powerful but too monolithic and heavy for the MVP.

## 12. Immediate Implementation Sequence

1. Freeze UI and report polish.
2. Spike `python-audio-separator` in an isolated environment.
3. Design `EngineResult` / `OutputArtifact` contract V2.
4. Integrate Music Separation Mode only if the spike passes.
5. Add MUSDB sample evaluation.

## 13. Explicit Non-Goals

- No FastAPI, Celery, or Redis production backend now.
- No microservices claim in the MVP.
- No universal sound-removal claim.
- No AudioSep or LASS in the MVP.
- No target-speaker extraction in the MVP.
- No new UI polish before the second engine spike.

## 14. Active Source Of Truth Policy

This file is the active source of truth. Older blueprint and spec files are historical references unless their rules are explicitly copied here.

## 15. Thesis Contribution Statement

Đề tài xây dựng một ứng dụng xử lý âm thanh trong tệp audio và video theo kiến trúc định tuyến engine, cho phép lựa chọn tác vụ và sử dụng mô hình học máy phù hợp. Hệ thống tập trung vào hai bài toán MVP có thể kiểm chứng: tăng cường tiếng nói và tách thành phần âm nhạc, đồng thời xuất kết quả dưới dạng audio hoặc video đã ghép lại.
