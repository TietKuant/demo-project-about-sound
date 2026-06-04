# UrbanSound8K Noise Manifest Preparation

## Purpose

This script converts UrbanSound8K metadata into the project noise manifest format used by the target-noise suppression dataset layer.

It does not decode audio, validate media, generate mixtures, train a model, or add a new engine.

## Initial Noise Classes

The first target-noise suppression classes are:

- `dog_bark`
- `car_horn`
- `siren`

These are selected because they are concrete target noise events and are useful for demonstrating class-scoped suppression data.

## Input

UrbanSound8K metadata is expected to contain:

```text
slice_file_name,fold,class
```

The audio root should point to the local UrbanSound8K audio folder that contains fold folders:

```text
UrbanSound8K/audio/fold1/
UrbanSound8K/audio/fold2/
...
```

## Output

The script writes the project noise manifest columns:

```text
noise_id,path,noise_label,source,notes
```

The script fails fast if no selected classes are provided or if the selected classes do not match any UrbanSound8K metadata rows.

Typical local output:

```text
data/manifests/noise_sources.local.csv
```

`.local.csv` files may contain local machine paths or local dataset assumptions and must not be committed.

## Path Convention

Paths are written relative to the output manifest location.

If the output manifest is under:

```text
data/manifests/
```

and UrbanSound8K is stored under:

```text
data/external/UrbanSound8K/
```

then output paths should look like:

```text
../external/UrbanSound8K/audio/fold1/7061-6-0-0.wav
```

## Required Follow-Up Validation

After creating the manifest, run:

```text
python scripts/validate_audio_dataset_manifest.py \
  --manifest data/manifests/noise_sources.local.csv \
  --path-column path \
  --label-column noise_label \
  --allowed-labels dog_bark car_horn siren \
  --output-root outputs/audits/urbansound8k-noise
```

Only validated rows should be used by:

```text
scripts/build_target_noise_suppression_dataset.py
```

## Current Scope

This sprint adds metadata conversion only.

It does not:

- download UrbanSound8K,
- commit UrbanSound8K audio,
- commit `.local.csv` manifests,
- generate mixed training audio,
- train a target-noise model.
