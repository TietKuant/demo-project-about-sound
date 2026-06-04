# Dataset Source Audit

## Purpose

Before using public audio datasets for training or evaluation, the project needs a small validation layer.
Dataset manifests can contain broken paths, unsupported labels, unreadable media, very short clips, very long clips, or duplicate rows.

The validator reduces manual checking and helps prevent broken training data from entering future dataset builds.

## Source Policy

`jim-schwoebel/voice_datasets` and Twine are useful catalog/list sources. They are not treated as training datasets by themselves.

Actual audio data should come from official dataset sources or clearly licensed mirrors.

For target-noise suppression:

- start with UrbanSound8K or ESC-50 for labeled noise/event classes,
- use VoiceBank-DEMAND or another clean speech subset for clean speech,
- keep raw downloaded data outside git,
- commit only small example manifests with fake or non-sensitive paths.

## Path Convention

Relative paths are resolved relative to the manifest CSV file location.

If a manifest is stored under:

```text
data/manifests/
```

then paths pointing to local dataset files under `data/external/` should start with:

```text
../external/
```

Example:

```text
../external/urbansound8k/audio/fold1/dog_001.wav
../external/voicebank/clean_trainset_28spk_wav/p226_001.wav
```

Local manifests may use absolute paths when needed, but `.local.csv` files with private or machine-specific paths must not be committed.

## Why Validation Is Needed

Public audio datasets are not automatically ready for this project. Before building mixtures, each source manifest should be checked for:

- file existence,
- duplicate paths,
- label scope,
- readable audio metadata,
- duration limits,
- sample rate metadata,
- channel metadata.

This is especially important before creating mixed training data, because one bad source manifest can produce many bad generated examples.

## Validator Output

The validator writes:

```text
<output-root>/manifest_audit.csv
<output-root>/manifest_audit_summary.json
```

The CSV records one row per manifest entry with status and error details.
The summary JSON records total valid/invalid counts, duplicate path count, label counts, and status counts.

## Current Scope

This sprint validates manifests only.

It does not:

- download datasets,
- commit raw audio,
- generate mixed training audio,
- train a model,
- add a target-noise suppression engine,
- change the app UI.

## Intended Flow

1. Download a public dataset locally from its official source.
2. Create a clean speech or noise source manifest.
3. Run `scripts/validate_audio_dataset_manifest.py`.
4. Fix invalid rows or labels.
5. Use only validated manifests as input to the target-noise suppression dataset builder.
