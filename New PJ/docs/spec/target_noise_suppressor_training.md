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

This is a standalone inference script only. It is not integrated into the app UI or `scripts/run_audio_task.py` yet.

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

This aggregate result should decide whether the custom model is ready for runner or UI integration.

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

This implementation adds a custom training baseline, a standalone inference script, and aggregate checkpoint evaluation.

It does not:

- integrate the model into the app,
- integrate the model into `scripts/run_audio_task.py`,
- modify the Gradio UI,
- train a final thesis-grade model,
- commit checkpoints, generated audio, or datasets.

## Future Work

The next sprint can add:

- a reusable engine adapter,
- integration with `scripts/run_audio_task.py`,
- richer audio-quality metrics beyond waveform L1/MSE,
- optional app UI integration after the standalone model path proves useful.
