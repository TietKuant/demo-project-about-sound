# Audio Router Real-Audio Validation

This document records the real-audio validation protocol for Audio Router V4. It is an aggregate report template and does not include private raw audio contents.

## Scope

- Validation set: 14 local real-audio files.
- Reported grouping: expected group labels only.
- Private audio/video filenames and contents are not part of the committed report.

## Expected Groups

The local validation set may be summarized by groups such as:

- `speech_clean`
- `speech_noisy_general`
- `speech_target_noise`
- `music_with_vocals`
- `environment_only`
- `hard_cases`

## Current Finding

Audio Router V4 is an experimental baseline. Controlled source-disjoint MVP evaluation was useful, but real-audio validation exposed domain shift and overconfident misclassification.

Runtime was made conservative by using a default confidence threshold of `0.90` and explicit accepted/abstain decision fields. This reduces unsafe automatic routing, but it does not fix model generalization.

## Export Command

Use the exporter to merge batch CSV rows with per-file JSON summaries:

```bash
python scripts/export_audio_router_real_validation_report.py \
  --batch-csv outputs/audio-router-real-validation/router_batch.csv \
  --summaries-dir outputs/audio-router-real-validation/router_summaries \
  --output-csv outputs/audio-router-real-validation/merged_feature_report.csv \
  --output-md docs/results/audio_router_real_audio_validation.md
```

## Interpretation Rules

- Do not claim V4 is production-ready.
- Do not claim fully automatic task routing.
- Report accepted predictions separately from rejected predictions.
- Count dangerous accepted routes explicitly.
- Treat the router as analyzer evidence unless the V5 evaluation improves real-audio generalization.
