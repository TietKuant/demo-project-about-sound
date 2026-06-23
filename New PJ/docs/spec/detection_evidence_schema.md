# Detection Evidence Dataset v1

## Purpose

Detection Evidence Dataset v1 is the reusable data layer between media ingestion and router/fusion model development:

`Audio/video → signal features + detector scores + labels → ML router/fusion`

The dataset stores human or source-provided labels, audio-derived signal features, and optional router/speech-gate outputs in one row per sample. It is evidence for training, evaluation, error analysis, and future fusion experiments. Building this dataset does not train a model or change production routing.

File names, paths, dataset names, IDs, split fields, and notes are metadata only. They must not be used as ML input features. Runtime decisions and trained models must use audio-derived features and detector outputs, never filename or path parsing.

## Columns

### Identity, provenance, and labels

| Column | Meaning |
| --- | --- |
| `sample_id` | Stable sample identifier from the source manifest. |
| `input_path` | Resolved media path; metadata only. |
| `source_dataset` | Source collection name; metadata only. |
| `split` | Train/validation/test assignment. |
| `group_id` | Group-disjoint split identity such as speaker, track, or source recording. |
| `content_label` | Single router taxonomy target. |
| `workflow_label` | Expected controller workflow. |
| `contains_speech` | Multi-label speech evidence. |
| `contains_music` | Multi-label music evidence. |
| `contains_target_noise` | Whether supported target noise is present. |
| `target_noise_label` | Target event type when known. |
| `snr_db` | Signal-to-noise ratio when known. |
| `is_synthetic` | Whether the sample was generated or mixed. |
| `clean_source_id` | Clean source lineage identifier. |
| `noise_source_id` | Noise source lineage identifier. |

### Extraction status

| Column | Meaning |
| --- | --- |
| `status` | `success`, `failed`, or `skipped`. |
| `error` | Extraction or inference error text. |

### Audio-derived signal features

| Column | Meaning |
| --- | --- |
| `duration_sec` | Decoded audio duration. |
| `rms_energy` | Global RMS energy. |
| `zero_crossing_rate` | Global zero-crossing rate. |
| `spectral_centroid_hz` | Spectral centroid. |
| `spectral_bandwidth_hz` | Spectral bandwidth. |
| `spectral_rolloff_hz` | Mean frame spectral rolloff. |
| `spectral_flatness` | Mean frame spectral flatness. |
| `low_band_energy_ratio` | Energy below 300 Hz. |
| `mid_band_energy_ratio` | Energy from 300–3400 Hz. |
| `high_band_energy_ratio` | Energy at or above 3400 Hz. |
| `rms_std` | Frame RMS standard deviation. |
| `zcr_std` | Frame zero-crossing-rate standard deviation. |
| `silence_ratio` | Fraction of low-energy frames. |

These values come from `scripts.extract_audio_features._audio_for_features` and `_features`.

### Router evidence

| Column | Meaning |
| --- | --- |
| `router_label` | Top 5-class router label. |
| `router_confidence` | Top router probability. |
| `router_accepted` | Whether router confidence passed its threshold. |
| `router_decision_reason` | Router/guard decision reason. |
| `router_prob_environment_only` | Probability for `environment_only`. |
| `router_prob_music_with_vocals` | Probability for `music_with_vocals`. |
| `router_prob_speech_clean` | Probability for `speech_clean`. |
| `router_prob_speech_noisy_general` | Probability for `speech_noisy_general`. |
| `router_prob_speech_target_noise` | Probability for `speech_target_noise`. |

### Speech-present gate evidence

| Column | Meaning |
| --- | --- |
| `speech_gate_label` | `speech_present` or `non_speech`. |
| `speech_gate_confidence` | Top speech-gate probability. |
| `speech_gate_prob_non_speech` | Probability for `non_speech`. |
| `speech_gate_prob_speech_present` | Probability for `speech_present`. |

### Ranking and review

| Column | Meaning |
| --- | --- |
| `top_label` | Highest router-probability label. |
| `top_score` | Highest router probability. |
| `second_label` | Second-highest router-probability label. |
| `second_score` | Second-highest router probability. |
| `score_margin` | `top_score - second_score`. |
| `needs_review` | True for failed/skipped rows, accepted label conflicts, margin below 0.15, or router confidence below the configured threshold. |

