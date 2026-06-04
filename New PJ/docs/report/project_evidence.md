# Project Evidence Pack

> Internal technical evidence for report writing. This file records reproducible project status, metrics, and demo checks. It is not the final thesis report.

Generated at: Fri Jun  5 04:17:25 +07 2026

## 1. Git State
```text
7cbdfad Ignore local runtime artifacts
f0345a4 Remove generated outputs and caches from tracking
d749953 Add experimental target noise suppression runner task
96795d0 Add target noise suppressor evaluation diagnostics
0fd4a12 Add target noise suppressor aggregate evaluation
a499fe4 Add target noise suppressor inference script
2ece78c Add target noise suppressor training baseline
d6e7927 Limit pytest collection to project tests
21cceea Ignore generated audit outputs
3d64b85 Add VoiceBank clean speech manifest mode
7b4ab78 Add UrbanSound8K noise manifest builder
c1c5251 Add audio dataset manifest validator
ee63527 Add target noise suppression dataset layer
77d1f6e Add preflight analyzer to demo UI
2302820 Add minimal audio task demo UI
```

## 2. Test Status
```text
................................................................... [ 52%]
............................................................             [100%]
127 passed, 5 subtests passed in 7.07s
```

## 3. Supported Tasks
```text
clean_voice | engine=deepfilternet | input_kind=audio_or_video | outputs=['restored']
target_noise_suppression | engine=target_noise_suppressor | input_kind=audio_only | outputs=['enhanced']
extract_vocals | engine=demucs | input_kind=audio_or_video_with_music | outputs=['vocals', 'no_vocals']
remove_vocals | engine=demucs | input_kind=audio_or_video_with_music | outputs=['no_vocals', 'vocals']
```

## 4. Dataset Summary: target-noise-v1
```json
{
  "total_rows": 1200,
  "counts_by_split": {
    "test": 199,
    "train": 1001
  },
  "counts_by_noise_label": {
    "car_horn": 208,
    "dog_bark": 524,
    "siren": 468
  },
  "snr_db_values": [
    -5.0,
    0.0,
    5.0
  ],
  "selected_classes": [
    "car_horn",
    "dog_bark",
    "siren"
  ]
}

```

## 5. Training Metrics: target-noise-suppressor-v2-fulltrain
```json
{
  "train_loss": 0.027042508562589503,
  "last_epoch_train_loss": 0.026900339183154758,
  "test_loss": 0.03871467362284361,
  "baseline_mixed_l1": 0.05408296082636819,
  "model_output_l1": 0.03871467362284361,
  "test_mse": 0.0038685599498768995,
  "train_rows": 1001,
  "test_rows": 199
}

```

## 6. Training Loss Curve
```csv
epoch,train_loss
1,0.03009544
2,0.02791430
3,0.02752117
4,0.02744587
5,0.02690034
```

## 7. Aggregate Evaluation Diagnostics
```json
{
  "total_samples": 199,
  "improved_samples": 133,
  "worsened_samples": 66,
  "improvement_rate": 0.6683417085427136,
  "mean_baseline_mixed_l1": 0.0464695202853313,
  "mean_model_output_l1": 0.030507797272361102,
  "mean_output_mse": 0.0027138649046669094,
  "mean_error_power_improvement_db": 0.9571163079112757,
  "relative_l1_improvement": 0.3434880092362116,
  "split": "test",
  "checkpoint_path": "outputs/model-runs/target-noise-suppressor-v2-fulltrain/checkpoint.pt",
  "manifest_path": "outputs/target-noise-v1/manifests/target_noise_suppression.csv",
  "by_noise_label": {
    "car_horn": {
      "total_samples": 34,
      "improved_samples": 21,
      "worsened_samples": 13,
      "improvement_rate": 0.6176470588235294,
      "mean_baseline_mixed_l1": 0.04263067040386993,
      "mean_model_output_l1": 0.03493735999526346,
      "mean_output_mse": 0.00334449959165581,
      "relative_l1_improvement": 0.18046421357493092
    },
    "dog_bark": {
      "total_samples": 73,
      "improved_samples": 50,
      "worsened_samples": 23,
      "improvement_rate": 0.684931506849315,
      "mean_baseline_mixed_l1": 0.040566951262506284,
      "mean_model_output_l1": 0.02801447657689656,
      "mean_output_mse": 0.0024480932042934,
      "relative_l1_improvement": 0.30942612878112086
    },
    "siren": {
      "total_samples": 92,
      "improved_samples": 62,
      "worsened_samples": 30,
      "improvement_rate": 0.6739130434782609,
      "mean_baseline_mixed_l1": 0.05257178587919992,
      "mean_model_output_l1": 0.030849180730950575,
      "mean_output_mse": 0.002691688326076078,
      "relative_l1_improvement": 0.4131989200093717
    }
  },
  "by_snr_db": {
    "-5": {
      "total_samples": 59,
      "improved_samples": 42,
      "worsened_samples": 17,
      "improvement_rate": 0.711864406779661,
      "mean_baseline_mixed_l1": 0.0734246166154616,
      "mean_model_output_l1": 0.03934755823496035,
      "mean_output_mse": 0.00394639806817996,
      "relative_l1_improvement": 0.4641094492732479
    },
    "0": {
      "total_samples": 71,
      "improved_samples": 44,
      "worsened_samples": 27,
      "improvement_rate": 0.6197183098591549,
      "mean_baseline_mixed_l1": 0.040519859529064284,
      "mean_model_output_l1": 0.028444445114845122,
      "mean_output_mse": 0.002341486865320099,
      "relative_l1_improvement": 0.29801224768702983
    },
    "5": {
      "total_samples": 69,
      "improved_samples": 47,
      "worsened_samples": 22,
      "improvement_rate": 0.6811594202898551,
      "mean_baseline_mixed_l1": 0.02954307434645117,
      "mean_model_output_l1": 0.025072320553379646,
      "mean_output_mse": 0.0020431313415705836,
      "relative_l1_improvement": 0.1513300119223566
    }
  }
}

```

## 8. Experimental Runner Real Check
Summary path: outputs/audio-task-runs-realcheck/target_noise_suppression/00010_p232_071_siren_159752-8-2-0_snr5-763f518b/summary.csv

```csv
run_id,task,engine,input_path,input_type,status,runtime_sec,primary_output_path,error
00010_p232_071_siren_159752-8-2-0_snr5-763f518b,target_noise_suppression,target_noise_suppressor,outputs/target-noise-v1/mixed/test/00010_p232_071_siren_159752-8-2-0_snr5.wav,audio,success,1.409698,outputs/audio-task-runs-realcheck/target_noise_suppression/00010_p232_071_siren_159752-8-2-0_snr5-763f518b/00010_p232_071_siren_159752-8-2-0_snr5.target_noise_suppressed.wav,
```

## 9. Current Scope Notes
- The app supports a unified audio task demo.
- Supported tasks are clean_voice, target_noise_suppression, extract_vocals, and remove_vocals.
- clean_voice uses DeepFilterNet path through the existing pipeline.
- extract_vocals/remove_vocals use Demucs-based music separation.
- target_noise_suppression is an experimental custom-trained baseline.
- The custom model was trained on synthetic paired data: VoiceBank clean speech mixed with UrbanSound8K target noise.
- The first target-noise classes are dog_bark, car_horn, and siren.
- The analyzer/preflight recommendation is feature-based and intent-assisted, not yet a trained ML router.
- The project should not claim final production-quality speech enhancement.
