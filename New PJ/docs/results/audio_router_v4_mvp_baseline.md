# Audio Router V4 MVP Baseline

This is a 4-class MVP router baseline, not the full future router. It evaluates whether the current lightweight acoustic-feature classifier can predict a reduced Audio Router V4 content taxonomy before any app integration.

## Manifest Summary

Rows: 2144

Split:

| Split | Rows |
|---|---:|
| train | 1700 |
| test | 444 |

Source dataset:

| Source dataset | Rows |
|---|---:|
| esc50 | 600 |
| musdb18_preview | 144 |
| target_noise_v2_source_disjoint | 400 |
| urbansound8k | 600 |
| voicebank_clean | 400 |

Content label:

| Content label | Rows |
|---|---:|
| environment_only | 1200 |
| music_with_vocals | 144 |
| speech_clean | 400 |
| speech_target_noise | 400 |

## Route And Engine Summary

Route target:

| Route target | Rows |
|---|---:|
| manual_required | 144 |
| no_process | 400 |
| out_of_scope | 1200 |
| target_noise_suppression | 400 |

Engine target:

| Engine target | Rows |
|---|---:|
| demucs | 144 |
| none | 1600 |
| target_noise_suppressor | 400 |

## Final Metrics

| Metric | Value |
|---|---:|
| accuracy | 0.7905405405405406 |
| macro_f1 | 0.7792304443064266 |
| weighted_f1 | 0.7861506423579432 |

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| environment_only | 0.7427385892116183 | 0.9675675675675676 | 0.8403755868544602 | 185 |
| music_with_vocals | 1.0 | 0.9 | 0.9473684210526316 | 50 |
| speech_clean | 0.4583333333333333 | 0.9565217391304348 | 0.619718309859155 | 23 |
| speech_target_noise | 0.9545454545454546 | 0.5645161290322581 | 0.7094594594594595 | 186 |

## Source-Level Metrics

| Source dataset | Test rows | Accuracy |
|---|---:|---:|
| esc50 | 126 | 0.9603174603174603 |
| musdb18_preview | 50 | 0.9 |
| target_noise_v2_source_disjoint | 186 | 0.5645161290322581 |
| urbansound8k | 59 | 0.9830508474576272 |
| voicebank_clean | 23 | 0.9565217391304348 |

## Commands Used

High-level experiment sequence:

1. Build V3 router manifest with `target_noise_v2_source_disjoint`.
2. Convert V3 router manifest to V4.
3. Filter the V4 MVP manifest.
4. Train the router baseline.

## Interpretation

The MVP router is suitable as a reproducible baseline because every retained content label has both train and test coverage. The strongest classes are `music_with_vocals` and `environment_only`. The `speech_target_noise` class has high precision but moderate recall, which means the classifier is conservative: when it predicts target-noise, it is usually correct, but it still misses a substantial portion of target-noise examples. Therefore, this result supports a baseline routing experiment, not a production-grade automatic router.

## Generated Artifacts

These artifacts were generated locally and are not necessarily committed:

- `data/manifests/audio_router.source_disjoint.final.local.csv`
- `data/manifests/audio_router.v4.source_disjoint.final.local.csv`
- `data/manifests/audio_router.v4.mvp.final.local.csv`
- `outputs/audio-router-v4-mvp-final-eval/metrics.json`

## Coverage Validation

All retained MVP content labels have both train and test coverage:

- `environment_only`: train/test
- `music_with_vocals`: train/test
- `speech_clean`: train/test
- `speech_target_noise`: train/test

## Limitations

- `speech_noisy_general` is excluded because current data lacks valid train coverage for that label.
- `speech_target_noise` recall is moderate, so this MVP should be described as a baseline classifier, not a production-grade router.
- The app/analyzer should not be described as a fully automatic ML router unless this trained model is explicitly integrated.
