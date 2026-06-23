# Audio Analyzer MVP

## 1. Purpose

The Audio Analyzer is the academic decision layer of the system. It converts raw audio into interpretable acoustic evidence, then uses that evidence to suggest a suitable processing task.

The analyzer/router does not perform restoration or separation by itself. Processors execute tasks such as speech enhancement or source separation. The analyzer supports task selection by reporting features, classifier output, confidence, and a practical recommendation.

## 2. Problem framing

The routing problem is:

Given an input audio signal `x(t)`, determine whether the signal is closer to:

- noisy speech
- music / vocal music
- environment noise / uncertain

Then map that interpretation to the most suitable processor.

The supported processing problems are different:

- Speech enhancement: `x(t) = speech + noise`; the objective is to recover clearer speech.
- Music source separation: `x(t) = vocals + accompaniment`; the objective is to estimate separated stems.
- Target-noise suppression: `x(t) = speech + known target noise classes`; the objective is to reconstruct cleaner speech for supported target-noise examples.

These problems should not be treated as one generic "remove sound" task. Each has different assumptions, outputs, and evaluation methods.

## 3. Input normalization

Video input is first decoded to its audio stream. Audio input is loaded directly when possible.

For lightweight feature extraction, non-WAV inputs are decoded through `ffmpeg` to mono 16 kHz PCM. WAV inputs may be read directly when possible; if direct reading fails, they fall back to the decode path. Directly readable WAV files are converted to mono when needed, but they are not necessarily resampled.

Therefore, the safe academic claim is that the analyzer uses a consistent lightweight analysis path for decoded media, not that every possible input is always resampled in the same way. Features are then computed from the waveform and spectrum to make input characteristics more comparable across formats such as WAV, M4A, MP3, MP4, and MOV.

This analyzer signal is separate from the processing pipeline requirements. Processors may use different internal sample rates or formats.

## 4. Feature groups and interpretation

The current feature extraction layer is implemented in `scripts/extract_audio_features.py`.

The extractor computes the feature groups below. Different consumers may use or display a subset: for example, the demo UI may show a compact feature table, while the router checkpoint stores the feature columns used for classification.

Energy group:

- `rms_energy`
- `rms_std`
- `silence_ratio`

Interpretation: overall signal strength, frame-level energy variation, and the amount of low-energy or silent regions.

Temporal group:

- `zero_crossing_rate`
- `zcr_std`

Interpretation: sign-change rate and its variation. These are useful as weak proxies for noisy, high-frequency, or transient content.

Spectral shape group:

- `spectral_centroid_hz`
- `spectral_bandwidth_hz`
- `spectral_rolloff_hz`
- `spectral_flatness`

Interpretation: frequency center, spectral spread, high-frequency reach, and noise-like versus tonal spectral behavior.

Band-energy group:

- `low_band_energy_ratio`
- `mid_band_energy_ratio`
- `high_band_energy_ratio`

Interpretation: distribution of energy across low-frequency, speech-range, and high-frequency bands.

No single feature can perfectly detect speech, music, vocals, or environmental noise. These features are weak evidence individually, but they become more useful when interpreted together or passed into a classifier.

## 5. Router classifier

The current baseline router is a lightweight classifier trained on extracted feature vectors.

- Input: lightweight acoustic feature vector.
- Model: small MLP classifier.
- Output: `predicted_label`, confidence, and class probabilities.
- Labels: `speech_noise`, `music`, `environment_noise`.
- Evaluation: accuracy and confusion matrix on a held-out split.

This classifier is a baseline router, not a universal audio understanding model. Its output should be treated as evidence for routing, especially because real-world audio may differ from the training sources.

## 6. Routing policy

MVP routing table:

| Router label | Suggested processor | Algorithm | Reason |
| --- | --- | --- | --- |
| `speech_noise` | `clean_voice` | DeepFilterNet | The input is speech-like or noisy-speech-like. Speech enhancement is suitable because the objective is to suppress background noise while preserving speech. |
| `music` | `extract_vocals` by default; `remove_vocals` as alternative | Demucs | The input is music-like. Source separation is suitable because vocals and accompaniment are structured sources, not simple noise. |
| `environment_noise` | no automatic processor by default | none | Environment noise alone does not define a speech enhancement or music separation objective. The system should avoid automatic processing unless the user manually selects a task. |

Target-noise suppression policy:

Do not auto-suggest `target_noise_suppression` from generic `speech_noise` alone. It should require stronger evidence that the input matches project-supported target-noise speech cases, such as dataset metadata, a target-noise label, or a future target-noise classifier.

For the current MVP, `target_noise_suppression` is experimental/manual. The generic `speech_noise` router label should not be treated as equivalent to project-supported target-noise classes such as dog bark, car horn, or siren.

## 7. Confidence policy

The router confidence should be used conservatively:

- High confidence: show the suggested task.
- Low confidence: show no automatic recommendation.
- Router disabled or failed: fall back safely and do not pretend the model made a decision.
- Manual override: the user can still choose a task directly.

Any numeric confidence threshold is an implementation parameter and should be tuned with validation data, not assumed to be universally correct.

## 8. What the app should display

The academic display order should be:

1. Input metadata
2. Extracted features
3. Feature interpretation
4. Router label, confidence, and probabilities
5. Suggested processor
6. Algorithm/principle
7. Reason
8. Limitation or caution

The main explanation should use user-facing language. It should avoid exposing raw internal names such as `blocked_reasons` as the primary explanation.

## 9. Limitations

- Lightweight features are not enough for perfect speech, music, vocal, or environmental-noise detection.
- Current router labels are coarse.
- Target-noise auto-routing is not fully supported yet.
- Metrics for arbitrary user input are limited because clean/reference signals are often unavailable.
- Source separation quality should be evaluated with reference-based metrics only when reference stems exist.
- The router may fail on real-world domain-shifted samples, so it should support recommendations rather than force automatic execution.

## 10. Future work

- CNN on mel-spectrograms for a stronger router.
- Singing voice detection.
- Target-noise event classifier.
- Better SNR/noise-floor estimator.
- Analyzer metrics dashboard.
- PESQ/STOI for speech enhancement when clean reference exists.
- SDR/SI-SDR for source separation when reference stems exist.
