# Evaluation Protocol

## Scope

This benchmark evaluates single-channel speech denoising / speech enhancement.

- Use VoiceBank-DEMAND clean/noisy paired test data for objective evaluation.
- Use real local samples only for demo and manual listening.
- Do not report intrusive metrics for real local samples without a clean reference.

## Required Benchmark Anchors

- `noisy_input`: lower-bound no-processing baseline.
- Clean reference: target / upper reference for comparison, not an engine output.
- Enhanced outputs: results produced by each selected denoise engine.

The baseline is required so reported results show whether an engine improves over the original noisy input.

## Current Metrics

- SI-SDR.
- SNR improvement.
- `runtime_sec`.
- RTF: processing runtime divided by audio duration.

## Known Limitations

- SNR can be misleading when an engine changes signal gain.
- Signal alignment is not yet implemented. Timing shifts may affect intrusive metrics.
- STOI is not yet implemented.
- PESQ is not yet implemented.
- PESQ requires correct sample-rate handling, commonly 16 kHz for wideband evaluation.
- The current 20-pair manifest is for development and smoke benchmarking. It is not sufficient for final thesis results.

## Next Required Metric Improvements

1. Add signal alignment before intrusive metrics.
2. Add STOI.
3. Add PESQ if setup is stable.
4. Run a larger benchmark with 100 pairs.
5. Run the full 824-pair VoiceBank-DEMAND test set if runtime permits.

## Final Evaluation Done Criteria

- Benchmark includes the `noisy_input` baseline.
- Benchmark runs all selected engines.
- CSV contains metrics and RTF.
- Results are summarized by engine.
- Final reported results do not rely only on 20 cherry-picked samples.
