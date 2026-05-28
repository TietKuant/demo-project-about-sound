# Dataset Plan

## Main Benchmark Dataset

Use the VoiceBank-DEMAND / Valentini noisy speech database as the main MVP benchmark.

- It provides clean/noisy paired speech.
- It is suitable for objective metrics because each noisy sample has a clean reference.
- It matches the project focus: single-channel speech denoising / speech enhancement.

## Dataset Usage

Start small and practical:

- Use a subset of 20-50 clean/noisy pairs.
- Do not commit dataset files to git.
- Store the local dataset under `data/raw/voicebank/`.
- Create a manifest CSV later, for example: `data/manifests/voicebank_subset.csv`.

## Real Local Samples

Use real local samples only for demo and manual listening:

- `samples/real_audio/*.m4a`
- `samples/input_video/*.mp4`

Do not compute intrusive metrics on these samples unless a clean reference is available.

## Out Of Scope

- AudioSet.
- MUSDB18.
- UrbanSound8K as the MVP benchmark.
- Training or fine-tuning.
- Source separation datasets.

## Next Planned Deliverables

- Dataset folder structure.
- Manifest CSV format.
- Loader/check script.
- Benchmark runner later.
