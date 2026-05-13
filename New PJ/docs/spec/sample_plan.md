# Sample Data Plan

## Goal
- Use small, meaningful demo/testing samples to validate current denoise behavior on realistic inputs.
- Do not use this set to claim broad audio-quality performance.

## Sample Group 1: Synthetic Sanity Samples
- Purpose: confirm the pipeline works end to end on controlled inputs where the noise source is known.
- How to obtain it: generate short clean tones or spoken phrases locally, then mix in simple synthetic noise such as hiss, hum, or background white noise with `ffmpeg`.
- What success looks like: the pipeline runs reliably, produces output artifacts, and the output is obviously different from the noisy input in a predictable way.

## Sample Group 2: Small Real Noisy Audio Samples
- Purpose: validate whether the current denoise path is useful on realistic short audio clips.
- How to obtain it: collect a few short local recordings with common background noise such as fan noise, room noise, or street noise, and keep each clip under about 1-2 minutes.
- What success looks like: speech or primary content is easier to hear after processing, obvious artifacts are limited, and the sample is good enough for demo discussion.

## Sample Group 3: Small Real Noisy Video Samples
- Purpose: validate the video branch, especially extraction, denoise, and remux with the original video stream preserved.
- How to obtain it: use one or two short locally available videos with audible background noise and a clear main subject, keeping duration under about 1-2 minutes.
- What success looks like: the remuxed video plays correctly, the video stream is preserved, and the processed audio is meaningfully cleaner than the original.

## Recommended MVP Demo Set
- 2 synthetic samples:
  - short spoken phrase with added white noise
  - short spoken phrase with added low hum or fan-like noise
- 2 real audio samples:
  - short indoor noisy voice recording
  - short outdoor or street-noise voice recording
- 1 real video sample:
  - short handheld clip with visible speaker and steady background noise

## Practical Notes
- Prefer WAV for synthetic and audio-only sanity checks.
- Keep filenames explicit and stable so outputs are easy to compare.
- Store only a very small set at first; expand only if a concrete gap appears in the current phase.
