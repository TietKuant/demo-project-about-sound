# Target Noise Suppression Dataset Layer

## Purpose

This dataset layer prepares structured evidence for a future custom `target_noise_suppression` engine.
It does not train a model yet and does not add a new engine to the app.

The goal is to make the project more ML/data-centric. Instead of only running pretrained engines, the project can describe a supervised audio dataset where the input is noisy speech and the target is clean speech.

## Audio Versus Tabular ML

In a tabular house-price project, each CSV row contains features such as area, rooms, location, and a target price.

In audio suppression, the raw signal cannot fit naturally into one small CSV row. The waveform lives in audio files, while the CSV stores paths and labels. The manifest is still the structured ML dataset because each row defines:

- the clean speech source,
- the labeled target noise source,
- the generated mixed noisy speech,
- the clean target output,
- the noise class,
- the SNR used to create the mixture,
- the dataset split.

## Dataset Formula

Clean speech + labeled target noise = mixed noisy speech.

For supervised training or evaluation later:

- input = `mixed_path`
- target = `target_path` / `clean_path`
- class label = `noise_label`

The target remains clean speech. The model should learn to suppress the selected noise class while preserving speech.

## Manifest Columns

The generated manifest uses:

```text
sample_id,clean_path,noise_path,mixed_path,target_path,noise_label,snr_db,split
```

`noise_label` is used for class scope, filtering, and later analysis. Example labels include `dog_bark`, `car_horn`, and `siren`.

## Local Data Rules

Raw speech, raw noise, and generated mixed audio must not be committed to git.
Use local ignored folders such as:

```text
data/private/
data/external/
outputs/
tmp/
```

Only example manifests with fake or non-sensitive relative paths should be committed.

## Builder Inputs

Clean speech manifest:

```text
sample_id,path,split,notes
```

Noise manifest:

```text
noise_id,path,noise_label,source,notes
```

The builder filters noise rows by selected classes, generates mixed audio paths, writes the dataset manifest, and writes a summary JSON.

## Current Mixing Method

The current builder applies approximate noise amplitude scaling from `snr_db` before mixing:

```text
noise_volume = 10 ** (-snr_db / 20)
```

Examples:

- `snr_db = 0` uses noise volume `1.0`
- `snr_db = 5` uses noise volume about `0.562`
- `snr_db = -5` uses noise volume about `1.778`

This is a simple amplitude-scaling approximation. It does not compute true measured SNR after mixing yet.

Future improvement should normalize clean and noise RMS before mixing, then verify the measured SNR of the generated mixed file.

## Current Scope

This sprint adds only the dataset and manifest preparation layer.

It does not:

- train a target-noise model,
- add a new app task,
- add a new engine,
- modify current DeepFilterNet or Demucs behavior,
- claim target sound removal is solved.

## Future Task

Future task name: `target_noise_suppression`.

That future task can use this dataset layer to train or evaluate a custom model for selected target noise classes.
