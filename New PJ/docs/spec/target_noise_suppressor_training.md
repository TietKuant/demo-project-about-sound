# Target Noise Suppressor Training

## Purpose

This is the first custom-trained baseline model in the project.

It trains a small waveform denoising model for the future `target_noise_suppression` task. The model is intentionally simple so it can run on CPU and provide an initial machine-learning baseline before larger models are considered.

## Dataset

The training script uses the synthetic paired dataset produced by the target-noise suppression dataset builder.

Expected manifest:

```text
target_noise_suppression.csv
```

Required columns:

```text
mixed_path,target_path,split,noise_label,snr_db
```

Training meaning:

- input = `mixed_path`
- target = `target_path`
- split = train/test selection
- `noise_label` and `snr_db` are retained for dataset scope and analysis

The current local dataset version is `target-noise-v1`.

## Model

The baseline model is `TinyWaveformDenoiser`.

It uses small 1D convolution layers:

```text
[batch, 1, samples] -> Conv1d -> ReLU -> Conv1d -> ReLU -> Conv1d -> [batch, 1, samples]
```

It does not use pretrained weights.
It does not use DeepFilterNet, Demucs, noisereduce, librosa, or sklearn.

## Training Script

Entrypoint:

```text
scripts/train_target_noise_suppressor.py
```

The script:

- reads `target_noise_suppression.csv`,
- loads mixed and target WAV files with `soundfile`,
- converts audio to mono,
- resamples with `torchaudio` when needed,
- crops or pads fixed-length segments,
- trains with L1 waveform loss,
- evaluates mixed-input baseline error versus model-output error.

## Outputs

The script writes:

```text
<output-dir>/checkpoint.pt
<output-dir>/config.json
<output-dir>/metrics.json
<output-dir>/loss_curve.csv
```

These are generated artifacts and should not be committed.

## Metrics

Current metrics include:

- `train_loss`
- `test_loss`
- `baseline_mixed_l1`
- `model_output_l1`
- `test_mse`

`baseline_mixed_l1` compares the mixed noisy waveform to the clean target.
`model_output_l1` compares the model output to the clean target.

This gives a simple before/after training signal, not a final speech-quality evaluation.

## Current Scope

This sprint adds training only.

It does not:

- integrate the model into the app,
- add an inference engine,
- change `scripts/run_audio_task.py`,
- modify the Gradio UI,
- train a final thesis model,
- commit checkpoints or datasets.

## Future Work

The next sprint can add:

- standalone inference script,
- checkpoint loading,
- output WAV export,
- evaluation against held-out target-noise samples,
- optional app/runner integration after the model proves useful.
