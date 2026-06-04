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

## Inference

Trained checkpoints can be used by:

```text
scripts/run_target_noise_suppressor.py
```

The inference script loads `checkpoint.pt`, reads an input audio file, runs the model, and writes a generated enhanced WAV file.

If a clean `--target` file is provided, the script also writes before/after comparison metrics:

- `baseline_mixed_l1`
- `model_output_l1`
- `output_mse`

The standalone inference script is also reused by the experimental unified runner integration. The task is exposed through the runner as `target_noise_suppression`, but it remains experimental and requires an explicit checkpoint path.

## Aggregate Evaluation

One inference sample is not enough to decide whether the custom checkpoint is useful. A single example can be worse even when the model improves aggregate held-out performance.

Aggregate evaluation uses:

```text
scripts/evaluate_target_noise_suppressor.py
```

The evaluator loads one checkpoint, runs inference in memory across multiple held-out manifest rows, and writes:

```text
<output-dir>/evaluation_summary.json
<output-dir>/per_sample_metrics.csv
```

It reports:

- `improvement_rate`
- `mean_baseline_mixed_l1`
- `mean_model_output_l1`
- `mean_output_mse`
- `relative_l1_improvement`
- grouped diagnostics by `noise_label`
- grouped diagnostics by `snr_db`

This aggregate result should decide whether the custom model is ready for runner or UI integration.

Grouped diagnostics help explain where the model works and where it fails. For example, the model may improve `dog_bark` samples but worsen `siren` samples, or it may behave differently at low SNR and high SNR.

The evaluator also reports `error_power_improvement_db`, an auxiliary waveform diagnostic:

```text
10 * log10(baseline_error_power / model_error_power)
```

Positive values mean the model reduced squared error power relative to the mixed input. This is useful for diagnostics, but waveform L1 remains the primary baseline metric for this sprint.

## Experimental Runner Integration

The unified runner can run the experimental task:

```text
target_noise_suppression
```

The checkpoint must be supplied explicitly. Use either:

```text
--target-noise-checkpoint /path/to/checkpoint.pt
```

or:

```text
TARGET_NOISE_SUPPRESSOR_CHECKPOINT=/path/to/checkpoint.pt
```

The source code must not hardcode local checkpoint paths.

This runner integration is audio-only for now. Video inputs fail clearly because this sprint does not add audio extraction/remuxing for the custom model.

This task is experimental. It demonstrates a custom-trained target-noise baseline, not a final speech enhancement quality claim.

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

This implementation adds a custom training baseline, standalone inference, aggregate evaluation, and experimental unified-runner integration.

It does not:

- claim final speech-enhancement quality,
- support video input for the custom target-noise task,
- train a final thesis-grade model,
- commit checkpoints, generated audio, or datasets.

## Future Work

The next sprint can add:

- a reusable engine adapter if the runner integration grows,
- richer audio-quality metrics beyond waveform L1/MSE,
- a trained ML router for task recommendation,
- stronger app UI messaging around the experimental checkpoint requirement.
