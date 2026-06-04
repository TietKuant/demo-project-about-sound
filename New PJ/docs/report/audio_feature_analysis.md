# Audio Feature Analysis Report

## Purpose

This layer makes the project more ML/data-centric by analyzing input audio characteristics before or alongside engine execution. It gives the system a simple feature matrix for dataset characterization and future task-routing work, instead of treating the project only as an engine wrapper.

## Feature Meanings

- **RMS Energy:** overall signal energy / loudness proxy.
- **Zero Crossing Rate:** signal sign-change rate, often higher for noisy or high-frequency content.
- **Spectral Centroid:** frequency "center of mass"; higher values usually mean brighter or noisier high-frequency content.
- **Spectral Bandwidth:** spread of frequency content.

## Vietnamese Real Sample Feature Matrix

| Sample | Condition | Duration (sec) | RMS Energy | ZCR | Spectral Centroid (Hz) | Spectral Bandwidth (Hz) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vi_quiet_001` | quiet speech | 9.194667 | 0.0555342661 | 0.0172118999 | 1758.675904 | 2974.033162 |
| `vi_fan_001` | fan noise | 12.608000 | 0.0347115837 | 0.0295786068 | 2379.094471 | 3299.411481 |
| `vi_traffic_001` | traffic noise | 11.840000 | 0.0488170959 | 0.0266298368 | 1787.871480 | 2627.745238 |
| `vi_cafe_001` | cafe/room noise | 10.730667 | 0.1322347753 | 0.0289750253 | 1925.217918 | 2627.352057 |
| `vi_phone_video_001` | phone video | 9.483333 | 0.0229896345 | 0.0196657265 | 1414.321546 | 2365.434499 |

## Interpretation

- The cafe/room-noise sample has the highest RMS energy, which indicates the strongest overall signal level among the five samples.
- The fan-noise sample has the highest zero crossing rate and spectral centroid, matching the expectation that fan noise can introduce high-frequency, steady background content.
- The phone-video sample has the lowest RMS energy and the lowest spectral centroid, suggesting a quieter and less bright recording.
- The quiet-speech sample has the lowest zero crossing rate.

## Limitations

- These are descriptive signal features, not objective quality metrics.
- They do not replace SI-SDR, STOI, or PESQ when clean references are available.
- They do not prove speech enhancement quality.
- They are used for dataset characterization and future routing/analyzer work.

## Relationship To Future Work

These features can support future rule-based or ML-based task recommendation. The current MVP still uses explicit user-selected tasks.
