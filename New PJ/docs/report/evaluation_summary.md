# Evaluation Summary Report

## Project Title

**English:** ML-Based Audio Processing Application for Audio and Video Files

**Vietnamese:** Xây dựng ứng dụng xử lý âm thanh trong tệp audio và video bằng kỹ thuật học máy

## Architecture Summary

The system is an engine-routed ML audio processing box. A user selects a task, the task registry defines the supported workflow, the workflow runner calls the correct engine adapter, and the system writes output artifacts plus summary reports.

```text
user task -> task registry -> workflow runner -> engine adapter -> output artifacts/report
```

This design keeps task selection separate from engine execution. It also allows speech restoration and music/vocal separation to share the same project structure without claiming one universal audio-cleaning model.

## Supported MVP Tasks

| Task | Engine | Output |
| --- | --- | --- |
| `clean_voice` | DeepFilterNet | restored audio or remuxed video |
| `extract_vocals` | Demucs | `vocals.wav` |
| `remove_vocals` | Demucs | `no_vocals.wav` |

## Unified Task Runner Evidence

The unified CLI entrypoint is:

```text
scripts/run_audio_task.py
```

It supports all MVP user tasks:

- `clean_voice`
- `extract_vocals`
- `remove_vocals`

The runner routes `clean_voice` through the existing media pipeline and routes music separation tasks through the Demucs workflow. Each run writes `summary.csv` and `summary.json` with task, engine, status, runtime, input type, and primary output path.

## MUSDB Preview Benchmark

This benchmark validates the music/vocal separation path on MUSDB18 preview samples. It confirms that Demucs runs successfully and produces the expected multi-output artifacts.

| Track | Status | Runtime (sec) | Outputs |
| --- | --- | ---: | --- |
| AM Contra - Heart Peripheral | success | 19.721144 | `vocals.wav`, `no_vocals.wav` |
| Al James - Schoolboy Facination | success | 17.843943 | `vocals.wav`, `no_vocals.wav` |
| Angels In Amplifiers - I'm Alright | success | 16.037222 | `vocals.wav`, `no_vocals.wav` |

Average runtime: approximately **17.87 seconds**.

## Vietnamese Real Sample Suite

This suite validates the system on local private Vietnamese audio/video samples. These samples are not committed to git.

| Sample | Condition | Status | Duration (sec) | Runtime (sec) | RTF | Output |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `vi_quiet_001` | quiet speech | success | 9.194667 | 13.944466 | 1.516582 | `denoised.wav` |
| `vi_fan_001` | fan noise | success | 12.608000 | 12.390375 | 0.982739 | `denoised.wav` |
| `vi_traffic_001` | traffic noise | success | 11.840000 | 13.691291 | 1.156359 | `denoised.wav` |
| `vi_cafe_001` | cafe/room noise | success | 10.730667 | 12.602057 | 1.174396 | `denoised.wav` |
| `vi_phone_video_001` | phone video | success | 9.483333 | 26.726173 | 2.818226 | `denoised.mov` |

## Evaluation Interpretation

The real Vietnamese samples validate pipeline compatibility, task routing, runtime reporting, RTF reporting, and output artifact generation on realistic local inputs.

They do not prove objective speech quality improvement because the samples do not have clean reference signals. Intrusive metrics such as SI-SDR, STOI, PESQ, and SNR improvement are reserved for paired datasets where a clean reference exists.

## Limitations

- The project does not train models from scratch.
- The MVP depends on pretrained engines.
- The system does not claim universal sound removal.
- Target speaker extraction is future work.
- Query-based target sound removal is future work.
- The analyzer is currently rule-based and contract-level, not a deployed ML analyzer.

## Next Technical Improvements

- Add audio feature extraction / EDA: RMS, ZCR, spectral centroid, and spectral bandwidth.
- Add task recommendation with a real analyzer model later.
- Add an optional multi-engine graph: vocal extraction -> speech cleanup.
