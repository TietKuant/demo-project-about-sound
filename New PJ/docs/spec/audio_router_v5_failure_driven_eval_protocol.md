# Audio Router V5 Failure-Driven Evaluation Protocol

## Core Framing

Audio Router V5 is a calibrated abstaining audio router. Its purpose is safe route recommendation under uncertainty, not maximum closed-set classification accuracy.

Accuracy and F1 remain useful, but they are secondary. Core evaluation must include calibration, risk-coverage, accepted accuracy, abstention behavior, and dangerous-route count.

## Type-1 Audit-Only Signals

The following fields are used only for dataset and evaluation audits:

- `source_corpus`
- `recording_id`, speaker id, or equivalent group id
- noise source corpus
- synthetic versus real status
- clean-source by noise-source pair matrix
- group leakage checks

These signals must not be fed into the runtime router. A model that relies on them may learn corpus fingerprints instead of acoustic semantics.

## Type-2 Runtime-Safety Gates

Runtime safety should use signals that are independent from the main classifier where practical:

- independent VAD or speech-presence signal;
- music-likelihood signal;
- silence, too-short, and corrupt-input checks;
- target-noise cue when a reliable detector is available.

If a safety gate and classifier disagree, the route must become `manual_required`. A classifier score alone must not override a failed safety gate.

## Pre-Registered Probes And Reactions

| Probe | Failure mode tested | Required reaction |
| --- | --- | --- |
| Corpus identity probe | Corpus fingerprinting | If accuracy is much above chance, report fingerprint risk, do not trust controlled metrics alone, and improve normalization, splits, or augmentation. |
| Synthetic-versus-real probe | Synthetic mixing signature | If separation is easy, do not claim the synthetic class generalizes. Depend on real holdout evidence and add realism augmentation or real noisy speech. |
| Pair matrix coverage | Diagonal clean/noise pairing shortcut | If cross-combinations are missing, regenerate data using an explicit cross-product or balanced pairing planner. |
| Group-disjoint check | Speaker, fold, song, or recording leakage | If leakage exists, rebuild splits before tuning models or thresholds. |
| Dangerous-route count | Unsafe automatic processing | If dangerous routes exceed zero on the red-team holdout, increase abstention, require manual confirmation, or add an independent gate. |

## Dataset Integrity Gates

Every training or evaluation manifest must pass:

- source coverage by split;
- clean-source by noise-source pair matrix coverage;
- group-disjoint checks for speaker, recording, fold, song, and noise source where applicable;
- target leakage checks;
- audio health checks for existence, decoding, duration, sample rate, channels, silence, and clipping;
- an explicit warning when validation data is not representative of deployment inputs.

Dataset generation must stop when required source groups are missing or split leakage is detected.

## Red-Team Holdout Policy

The red-team holdout must be frozen, versioned, and excluded from training, feature selection, model selection, and threshold tuning.

It must include should-abstain cases and realistic Vietnamese examples:

- clean speech;
- speech with fan noise;
- speech in cafe or room noise;
- speech with traffic noise;
- speech with supported target noise;
- music with vocals;
- instrumental music;
- speech over music, television, or radio;
- environment-only audio;
- silence, very short clips, corrupt files, and other out-of-distribution inputs.

Each item needs provenance and license or privacy notes. Private recordings and copyrighted media must not be committed. Small holdout size must be reported honestly and must not be presented as broad population evidence.

## Calibration And Abstention

Expected Calibration Error (ECE) and risk-coverage reporting are required metrics.

Report at least:

- coverage at each confidence threshold;
- accepted accuracy;
- accepted macro-F1 where meaningful;
- rejection rate;
- dangerous accepted routes;
- calibration error;
- per-class and per-source accepted/rejected counts.

Threshold tuning requires a representative development or calibration set. Thresholds must not be tuned on the frozen red-team holdout or a source-confounded validation set.

## Failure-Driven Evaluation Sequence

1. Validate manifest integrity and group disjointness.
2. Run source and synthetic-fingerprint probes.
3. Evaluate normal and source-disjoint test splits.
4. Tune calibration and abstention only on the designated calibration set.
5. Freeze the threshold and model.
6. Run the frozen red-team holdout once for the reported experiment.
7. Record dangerous routes and abstention failures before considering app integration.

## Time-Boxed Final-Year Scope

The minimal defensible deliverable is:

- the V4 negative baseline and documented real-audio failure;
- the V5 operational label policy;
- dataset integrity and audit harnesses;
- a frozen red-team holdout;
- a conservative abstaining prototype with calibration reporting.

Large-scale real noisy-speech collection, extensive realism augmentation, and production-grade generalization may remain future work. The project should prefer a defensible abstaining system over an overclaimed automatic router.
