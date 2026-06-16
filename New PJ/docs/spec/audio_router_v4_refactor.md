# Audio Router V4 Refactor Plan

## Purpose

The audio router should use mathematical audio features and ML to detect audio patterns, then recommend the correct processing algorithm:

- DeepFilterNet for speech enhancement.
- Demucs for music source separation.
- Custom target-noise suppressor for speech with selected target noise.

Router v3 is useful as a baseline, but it is not enough as an academic router. It reports high internal test accuracy, but the label design is too coarse and the dataset sources are likely confounded with labels.

## Current Router V3 Audit

Router v3 manifest:

| Item | Value |
|---|---:|
| Total rows | 1544 |
| speech_noise | 700 |
| music | 144 |
| environment_noise | 700 |
| test_accuracy | 0.9409 |

Sources:

| Source | Rows |
|---|---:|
| target_noise_v1 | 234 |
| voicebank_clean | 233 |
| voicebank_noisy | 233 |
| musdb18_preview | 144 |
| esc50 | 350 |
| urbansound8k | 350 |

Current features:

- duration_sec
- rms_energy
- zero_crossing_rate
- spectral_centroid_hz
- spectral_bandwidth_hz
- spectral_rolloff_hz
- spectral_flatness
- low_band_energy_ratio
- mid_band_energy_ratio
- high_band_energy_ratio
- rms_std
- zcr_std
- silence_ratio

## Problem Statement

Router v3 proves the project can build a feature matrix, train a classifier, save a checkpoint, and run inference. That is report-worthy as a baseline.

However, router v3 is not yet strong enough to drive task selection in the app. The current labels are broad:

- speech_noise
- music
- environment_noise

These labels do not distinguish clean speech, general noisy speech, target-noise speech, vocal music, instrumental music, and mixed or unknown content. A router with these labels can appear accurate on internal tests while still failing realistic inputs.

## Source-Label Confounding Risk

The main risk is source-label confounding. The model may learn dataset identity instead of acoustic meaning.

Examples:

- VoiceBank files may have one recording style and map mostly to speech.
- MUSDB preview files may have one container/production style and map to music.
- ESC-50 and UrbanSound8K files may have short event-style clips and map to environment noise.
- target_noise_v1 is synthetic and may have mixing artifacts that identify it.

If train/test rows from the same source distribution are split randomly, the reported accuracy can be high even when the model does not generalize to real Vietnamese recordings.

Router v4 must use group-disjoint and source-balanced evaluation to reduce this risk.

## Content Classification vs Processor Routing

Content classification answers:

> What kind of audio content is this?

Processor routing answers:

> Which processing algorithm should be recommended or blocked for this user goal?

These are not the same.

For example:

- A file can contain speech and music at the same time.
- A music video can contain vocals, background music, and environmental noise.
- Speech with traffic noise should usually route to speech enhancement, not music separation.
- Target-noise suppression should only be recommended when the input is compatible with the custom target-noise model and a checkpoint is available.

Router v4 should predict content taxonomy first. A separate conservative mapping should convert taxonomy into route and engine targets.

## Terminology

Router v4 uses four separate fields:

- content_label: acoustic taxonomy label.
- route_target: task-level decision using repo task IDs or no_process/manual_required/abstain/out_of_scope.
- engine_target: model/processor family.
- candidate_tasks: list of task IDs when user choice is required.

Controlled route_target values only:

- clean_voice
- target_noise_suppression
- extract_vocals
- remove_vocals
- no_process
- manual_required
- abstain
- out_of_scope

fallback_route_target uses the same enum as route_target when present. Use an empty/null value when no fallback exists. fallback_engine_target is empty/null when no fallback exists.

## Taxonomy V4

Router v4 content labels:

- speech_clean
- speech_noisy_general
- speech_target_noise
- music_with_vocals
- music_instrumental
- environment_only
- unknown_mixed

speech_target_noise means only the project-supported target noise set: dog_bark, car_horn, siren. It does not imply universal target-noise detection.

## Taxonomy To Routing Mapping

| content_label | route_target | engine_target | candidate_tasks | fallback_route_target | fallback_engine_target | conditions | notes |
|---|---|---|---|---|---|---|---|
| speech_clean | no_process | none | clean_voice |  |  | none | clean_voice is optional; clean speech should not be over-processed. |
| speech_noisy_general | clean_voice | deepfilternet | clean_voice |  |  | speech-dominant input | main speech enhancement route. |
| speech_target_noise | target_noise_suppression | target_noise_suppressor | target_noise_suppression | clean_voice | deepfilternet | audio-only input, checkpoint available, target_noise_label in dog_bark/car_horn/siren | project-supported target-noise only; not universal noise removal. |
| music_with_vocals | manual_required | demucs | extract_vocals; remove_vocals |  |  | music mixture with likely vocals | user goal decides whether to keep vocals or accompaniment. |
| music_instrumental | no_process | none | extract_vocals; remove_vocals |  |  | no vocal evidence | Demucs can remain available as manual optional processing, but no automatic processing is needed. |
| environment_only | out_of_scope | none |  |  |  | no speech/music target | MVP does not claim universal environmental sound removal. |
| unknown_mixed | abstain | none |  | manual_required |  | low confidence or conflicting evidence | do not auto-route. |

The router should support abstain/low-confidence behavior. It must not force a processor when confidence is low or when the taxonomy does not map cleanly to an MVP processor.

## Multi-label Evidence And Content Label Collapse Rule

contains_* fields are multi-label evidence. content_label is the single training target for the v4 baseline.

Collapse priority:

1. If evidence is low-confidence or conflicting beyond rules, use unknown_mixed.
2. If contains_speech and contains_target_noise with supported target_noise_label, use speech_target_noise.
3. If contains_speech and noisy but no supported target noise, use speech_noisy_general.
4. If contains_speech and not noisy and not music, use speech_clean.
5. If contains_music and contains_vocals, use music_with_vocals.
6. If contains_music and not contains_vocals, use music_instrumental.
7. If only environment/no speech/no music, use environment_only.
8. Otherwise, use unknown_mixed.

This is a baseline collapse rule. Ambiguous speech+music mixtures should go to unknown_mixed unless there is a documented dominant-source policy.

## Dataset Source Mapping

| Source | V4 content_label |
|---|---|
| VoiceBank clean | speech_clean |
| VoiceBank noisy | speech_noisy_general |
| target_noise_v1 | speech_target_noise |
| MUSDB mixture | music_with_vocals |
| MUSDB accompaniment/no_vocals if available | music_instrumental |
| ESC-50 / UrbanSound8K non-target | environment_only |
| Vietnamese private samples | holdout only |

Vietnamese private samples must not be used for training by default. They should be used as a separate real-world holdout to report generalization behavior honestly.

## Manifest V4 Schema

Required columns:

```text
sample_id,
input_path,
source_dataset,
split,
split_key,
group_id,
duration_sec,
sample_rate,
channels,
taxonomy_version,
original_sample_rate,
normalized_sample_rate,
normalization_version,
analysis_window_sec,
source_fold,
original_split,
content_label,
route_target,
engine_target,
candidate_tasks,
fallback_route_target,
fallback_engine_target,
dominant_source_policy,
label_method,
label_confidence,
contains_speech,
contains_music,
contains_vocals,
contains_environment_noise,
contains_target_noise,
target_noise_label,
snr_db,
is_synthetic,
clean_source_id,
noise_source_id,
mix_seed,
language,
domain,
checksum,
features_version,
event_count,
event_time_spans,
feature_status,
notes
```

Field intent:

- content_label is the router taxonomy label.
- route_target is the controlled task-level decision.
- engine_target is the candidate processor family.
- candidate_tasks lists task IDs when user choice is required.
- fallback_route_target is the controlled fallback task-level decision.
- fallback_engine_target is the fallback processor family.
- taxonomy_version records the taxonomy used to collapse labels.
- original_sample_rate records the source sample rate before normalization.
- normalized_sample_rate records the sample rate used for router feature extraction.
- normalization_version records the preprocessing policy.
- analysis_window_sec records the analysis window used for aggregation.
- source_fold and original_split preserve upstream dataset split metadata.
- dominant_source_policy records any policy used for ambiguous mixed content.
- split_key and group_id support group-disjoint splitting.
- label_method records whether the label came from dataset metadata, synthetic generation, manual annotation, or weak labeling.
- label_confidence records how reliable the label is.
- contains_* fields allow multi-label analysis even when content_label is single-label.
- checksum helps detect changed files.
- features_version tracks the feature extraction version used for training.
- event_count and event_time_spans record event annotations when available.
- feature_status records feature extraction quality or failure state.

## Split And Leakage Policy

Use split-then-mix for synthetic mixtures.

Leakage rules:

- For target_noise_v1, clean_source_id must not cross train/val/test.
- For target_noise_v1, noise_source_id must not cross train/val/test.
- ESC-50 must respect original folds.
- UrbanSound8K must respect original folds.
- MUSDB must respect track-level train/test split.
- VoiceBank should use speaker-disjoint splits where possible.
- Add CI tests that assert no forbidden group/source overlap across splits.

## Normalization And Anti-Fingerprint Policy

Router features must reduce dataset fingerprints:

- Decode to a controlled audio format before feature extraction.
- Use a fixed target sample rate for router features.
- Use consistent mono/stereo policy.
- Use loudness or amplitude normalization.
- Use fixed analysis windows or segment-level aggregation.
- Treat duration_sec as metadata/QC, not as a default training feature unless duration has been normalized.
- Treat original_sample_rate, codec/container, and raw duration as possible dataset fingerprints.

## Feature V4 Plan

Keep current features:

- rms_energy
- zero_crossing_rate
- spectral_centroid_hz
- spectral_bandwidth_hz
- spectral_rolloff_hz
- spectral_flatness
- low_band_energy_ratio
- mid_band_energy_ratio
- high_band_energy_ratio
- rms_std
- zcr_std
- silence_ratio

Add features:

- mfcc_01_mean ... mfcc_13_mean
- mfcc_01_std ... mfcc_13_std
- chroma_01_mean ... chroma_12_mean
- chroma_01_std ... chroma_12_std
- spectral_contrast_01_mean ... spectral_contrast_07_mean
- spectral_contrast_01_std ... spectral_contrast_07_std
- onset_strength_mean
- onset_strength_std
- snr_estimate_db
- vad_speech_ratio
- rms_p95
- rms_max
- zcr_p95
- spectral_centroid_p95
- spectral_bandwidth_p95
- spectral_flatness_p95
- onset_strength_p95
- frame_energy_spike_ratio
- mel-spectrogram metadata for future CNN

duration_sec belongs to metadata/QC, not the default training feature set.

Feature v4 should use flattened CSV columns, not variable-length arrays.

snr_estimate_db and vad_speech_ratio are required schema fields. If estimation fails, record feature_status instead of silently dropping them.

The first v4 implementation should still be lightweight and reproducible. If a feature requires a heavier dependency, it must be justified and isolated.

## Evaluation Plan

Router v4 evaluation must include:

- macro-F1
- per-class F1
- confusion matrix
- validation split for abstain threshold selection
- group-disjoint split
- source-balanced holdout
- coverage-vs-macro-F1 reporting
- per-source metrics to detect source-label confounding
- dataset-identity probe: train/evaluate a classifier predicting source_dataset from features to detect confounding
- leave-one-source-out or source-heldout evaluation where feasible
- bootstrap confidence intervals for macro-F1/per-class F1 when sample counts are small
- calibration/ECE or reliability check for abstain confidence
- per-target-noise breakdown for dog_bark, car_horn, siren
- leakage tests in CI
- Vietnamese real holdout reported separately
- abstain/low-confidence behavior

Accuracy alone is not enough. Macro-F1 and per-class F1 are required because the taxonomy may be imbalanced.

Group-disjoint split rules:

- Do not let near-duplicate clips or stems from the same source leak across train/test.
- Use group_id for tracks, speakers, source recordings, or synthetic clean/noise origins.
- Keep Vietnamese real samples outside training and report them separately.

Source-balanced holdout rules:

- Each major source should contribute to evaluation when possible.
- No single source should dominate the test score.
- Report failures by source as well as by content_label.

Confounding checks:

- Run a dataset-identity probe that predicts source_dataset from the same features.
- If source_dataset is easy to predict, report that router labels may be source-confounded.
- Use leave-one-source-out or source-heldout evaluation where feasible.
- Report per-source metrics.

Abstain behavior:

- The router should return low-confidence/unknown_mixed when uncertain.
- Choose the abstain threshold on a validation split, not on the final test split.
- Report coverage-vs-macro-F1 so the report shows the tradeoff between routing more files and making fewer wrong routes.
- Check calibration/ECE or another reliability measure for abstain confidence.
- Abstaining is better than routing to the wrong processor.
- The app should treat router output as evidence, not as automatic authority.

Target-noise evaluation:

- Report dog_bark, car_horn, and siren separately.
- Do not average target-noise behavior into one number without the per-class breakdown.

## Downstream Routing Utility

Router v4 baseline is evaluated by classification metrics.

A separate controlled subset should evaluate whether the selected route improves downstream output. Do not use a single universal oracle metric across DeepFilterNet, Demucs, and target-noise suppressor.

Use task-appropriate metrics where references exist:

- speech enhancement: speech metrics on paired clean/noisy speech when clean reference exists.
- source separation: separation metrics on music stems when reference stems exist.
- target-noise suppression: paired synthetic target-noise metrics and per-target-noise breakdown.

Report this as routing utility, not as a replacement for taxonomy labels.

## Known Remaining Risk

Router v4 may still be source-confounded if each class is dominated by one dataset family.

The project must report this honestly. Source diversity and source-heldout tests are required before claiming acoustic meaning.

## Frozen Processor Policy

DeepFilterNet and Demucs remain frozen processors in this sprint.

The project wraps and evaluates them. It does not fine-tune DeepFilterNet or Demucs. Any future training work should be limited to project-owned models such as the custom target-noise suppressor or the router classifier.

## Before Modifying The Gradio App Again

Do not change the Gradio app again until these are done:

1. Build the router v4 manifest schema.
2. Generate a small v4 manifest from existing local datasets.
3. Extract v4 features with a recorded features_version.
4. Train a router v4 baseline.
5. Evaluate with macro-F1, per-class F1, confusion matrix, group-disjoint split, and source-balanced holdout.
6. Run Vietnamese private samples as holdout only.
7. Decide an abstain threshold.
8. Document the router’s known failure cases.
9. Keep the UI contract: The app must not auto-run router decisions. Router v4 provides analyzer evidence and a suggested route. The user or planner must confirm execution before a processor runs.

Only after that should the app display router v4 results or use them as planning evidence.
