# Current System Capability

## Status Summary

The project currently implements an engine-routed ML audio processing architecture with two validated processing workflows.

## Supported MVP Tasks

| Task | Engine | Input Type | Output Artifacts | Status |
| --- | --- | --- | --- | --- |
| clean_voice | DeepFilterNet | speech audio/video | restored audio/video | implemented |
| extract_vocals | Demucs | music audio/video | vocals.wav, no_vocals.wav | implemented |
| remove_vocals | Demucs | music audio/video | no_vocals.wav, vocals.wav | implemented |

## Dataset Evidence

### Speech Enhancement

VoiceBank-DEMAND is used for paired speech enhancement evaluation.

### Music / Vocal Separation

MUSDB18 7-second preview is used for music/vocal separation validation.

Latest local benchmark:

| Track | Dataset | Status | Runtime Sec | Outputs |
| --- | --- | --- | ---: | --- |
| AM Contra - Heart Peripheral | MUSDB18-7-STEMS | success | 19.721144 | vocals.wav, no_vocals.wav |
| Al James - Schoolboy Facination | MUSDB18-7-STEMS | success | 17.843943 | vocals.wav, no_vocals.wav |
| Angels In Amplifiers - I'm Alright | MUSDB18-7-STEMS | success | 16.037222 | vocals.wav, no_vocals.wav |

Average runtime for the three preview samples: approximately 17.87 seconds.

## Vietnamese Real Sample Protocol

A Vietnamese real sample protocol exists to validate real-world Vietnamese audio/video inputs without making false objective metric claims when clean references are unavailable.

## Current Limitations

- The system does not train new models.
- The system does not claim universal sound removal.
- DeepFilterNet is used for speech cleanup, not music separation.
- Demucs is used for music/vocal separation, not speech denoising.
- Intrusive metrics such as SI-SDR, STOI, PESQ, and SNR improvement require clean references.
- Vietnamese real sample evaluation still needs actual local samples.
