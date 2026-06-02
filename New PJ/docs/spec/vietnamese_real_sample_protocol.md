# Vietnamese Real Sample Evaluation Protocol

## Purpose

Validate the application on realistic Vietnamese audio and video inputs outside VoiceBank-DEMAND and MUSDB18.

## Why This Protocol Exists

VoiceBank-DEMAND is useful for paired speech-enhancement metrics because it provides clean references. It is not a Vietnamese dataset and does not cover all real-world phone, room, street, cafe, or video-recording conditions.

Vietnamese real samples provide practical validation for local usage. They do not replace paired benchmark datasets.

## Sample Categories

- `vietnamese_speech_quiet`
- `vietnamese_speech_fan`
- `vietnamese_speech_traffic`
- `vietnamese_speech_cafe`
- `vietnamese_phone_video`
- `vietnamese_music_or_music_video_optional`

## Minimum Sample Set

Use at least 5 samples:

- At least 3 audio files.
- At least 2 video files when possible.

## Recommended Duration

Use samples between 10 and 60 seconds.

## Allowed Input Formats

- `wav`
- `mp3`
- `m4a`
- `flac`
- `mp4`
- `mov`
- `mkv`

## Evaluation Tasks

- Run Clean Voice / Reduce Speech Noise for speech samples.
- Run Extract Vocal / Remove Vocal only for music or video samples that contain music.
- Run Analyzer / Recommendation later if implemented.

## Allowed Metrics Without A Clean Reference

- `runtime_sec`
- `audio_duration_sec`
- RTF
- `output_exists`
- Before / after listening note.
- Waveform / spectrogram comparison.
- Warning / error.

## Forbidden Metrics Without A Clean Reference

- SI-SDR.
- STOI.
- PESQ.
- SNR improvement.

Do not report objective quality improvement when no clean reference exists.

## Privacy And License Rules

- Do not commit real user recordings.
- Do not commit copyrighted music or video.
- Store local samples under `data/private/` or `data/external/`.
- Keep `data/private/` and `data/external/` ignored by git.
- Commit a manifest only when paths are examples or non-sensitive.

## Naming Convention

Store private local files using:

```text
data/private/vietnamese-real/<category>/<sample_id>.<ext>
```

## Pass / Fail Criteria

A sample passes practical validation only when:

- The pipeline does not crash.
- The output artifact exists.
- Runtime and RTF are recorded.
- A limitation note is recorded.
- The report does not make a false objective-quality claim without a clean reference.
