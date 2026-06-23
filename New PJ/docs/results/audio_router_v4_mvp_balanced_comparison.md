# Audio Router V4 MVP Balanced Loss Comparison

This experiment compares the original unweighted Audio Router V4 MVP baseline with a class-balanced loss variant. The purpose is to improve `speech_target_noise` recall while preserving macro-level performance.

## Experiment Setup

| Run | Class weighting | Epochs | Learning rate | Seed |
|---|---|---:|---:|---:|
| none | none | 80 | 0.01 | 42 |
| balanced | balanced | 80 | 0.01 | 42 |

Balanced class weights were computed from train rows only using `total_train_rows / (num_classes * train_count_for_class)`.

## Balanced Class Weights

| Label | Weight |
|---|---:|
| environment_only | 0.4187 |
| music_with_vocals | 4.5213 |
| speech_clean | 1.1273 |
| speech_target_noise | 1.9860 |

## Overall Metrics

| Run | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| none | 0.7905 | 0.7792 | 0.7862 |
| balanced | 0.8739 | 0.8597 | 0.8763 |

## Per-Class Metrics

| Run | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| none | environment_only | 0.7427 | 0.9676 | 0.8404 | 185 |
| none | music_with_vocals | 1.0000 | 0.9000 | 0.9474 | 50 |
| none | speech_clean | 0.4583 | 0.9565 | 0.6197 | 23 |
| none | speech_target_noise | 0.9545 | 0.5645 | 0.7095 | 186 |
| balanced | environment_only | 0.9290 | 0.8486 | 0.8870 | 185 |
| balanced | music_with_vocals | 0.9434 | 1.0000 | 0.9709 | 50 |
| balanced | speech_clean | 0.5789 | 0.9565 | 0.7213 | 23 |
| balanced | speech_target_noise | 0.8641 | 0.8548 | 0.8595 | 186 |

## Target-Noise Error Reduction

| Run | Correct target-noise | Predicted environment_only | Predicted speech_clean | Predicted music_with_vocals |
|---|---:|---:|---:|---:|
| none | 105 | 58 | 23 | 0 |
| balanced | 159 | 12 | 14 | 1 |

## Interpretation

The balanced-loss variant substantially improves the main weakness of the original MVP baseline. `speech_target_noise` recall increases from 0.5645 to 0.8548, and its F1 score increases from 0.7095 to 0.8595. The number of target-noise samples misclassified as `environment_only` drops from 58 to 12.

Although `speech_target_noise` precision decreases from 0.9545 to 0.8641, the trade-off is favorable because macro-F1 also improves from 0.7792 to 0.8597. This indicates that the improvement is not only a target-class recall gain but a stronger overall MVP router baseline.

This result should be described as an improved lightweight-feature baseline, not as a production-grade automatic router. The app/analyzer should only be described as an ML router after this trained model is explicitly integrated.

## Local Artifacts

- `outputs/audio-router-v4-mvp-final-eval-none/metrics.json`
- `outputs/audio-router-v4-mvp-final-eval-none/test_predictions.csv`
- `outputs/audio-router-v4-mvp-final-eval-balanced/metrics.json`
- `outputs/audio-router-v4-mvp-final-eval-balanced/test_predictions.csv`
