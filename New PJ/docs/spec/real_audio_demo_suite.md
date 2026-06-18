# Real Audio Demo Suite

## Purpose

The real-audio demo suite validates that the existing processing workflows can handle local audio or video inputs end to end. It is product and demonstration evidence, not a model-training pipeline.

The suite runs the existing unified task runner for:

- `clean_voice`
- `extract_vocals`
- `remove_vocals`
- `target_noise_suppression`

It can also run an `auto` case through the conservative audio router.

## Manifest

Input CSV columns:

```text
input_path,case_id,expected_task,description,notes
```

Optional listening-evaluation columns:

```text
human_rating,human_notes,expected_auto_behavior,acceptable_routes,dangerous_routes,objective_metric_name,objective_score_before,objective_score_after
```

`expected_task` must be one of the four supported tasks or `auto`.

Manual tasks are the reliable path. A manual task is executed directly and is not blocked by router output.

For auto cases, write `expected_auto_behavior`, `acceptable_routes`, and `dangerous_routes` before running the suite. Route lists use semicolon-separated values. The report marks the result as `pass`, `dangerous_failure`, `unexpected_route`, or `unspecified`.

After router inference, the suite checks the route target, recommended task, and selected-task candidate against `dangerous_routes` before engine execution. A match is blocked as `manual_required`, no engine runs, and the expectation result is `dangerous_failure`. Manual `expected_task` cases bypass this auto-only safety gate.

The first real run must include at least one environment-only or silence auto case.

## Conservative Auto Behavior

The auto router is advisory:

- rejected, low-confidence, or `manual_required` decisions are recorded as abstained;
- no engine is forced after abstention;
- accepted `no_process` and `out_of_scope` decisions are recorded directly and do not run an engine;
- `target_noise_suppression` is manual-only in auto mode because the custom processor is experimental;
- only an accepted, supported, non-manual route may execute a task.

This behavior preserves the router as evidence rather than treating it as production authority.

## Outputs

Each `case_id` receives a local output folder containing:

- task-generated audio or video artifacts when processing succeeds;
- `case_summary.json`;
- `router_summary.json` for auto cases when router inference runs.

The suite root contains:

- `report.csv`;
- `report.md`.

The report records status, selected task, router evidence, outputs, abstentions, and errors. One failed case does not stop the remaining cases.

For each case, the suite also records:

- best-effort input duration, sample rate, and channel count;
- wall-clock processing time;
- optional `human_rating` and `human_notes` for listening-based evaluation.

Metadata is read with `ffprobe` when available, with a WAV-header fallback. Metadata failure does not fail processing.

## Human Rating Rubric

### `clean_voice`

1. Worse or unusable.
2. Less noise but strong artifacts or speech damage.
3. Little or no clear improvement.
4. Clearer speech with acceptable artifacts.
5. Clearly improved speech with minimal artifacts.

### Demucs Separation

1. Unusable separation.
2. Output exists but has heavy bleed or artifacts.
3. Partially useful.
4. Good separation with some artifacts.
5. Strong, usable separation.

### `target_noise_suppression`

1. Unusable or speech destroyed.
2. Target reduced but speech badly damaged.
3. Partial target reduction.
4. Useful reduction with acceptable speech quality.
5. Strong target reduction with speech preserved.

### Auto Cases

For auto cases, `human_rating` evaluates routing safety rather than audio quality:

1. Dangerous wrong route.
2. Wrong route but not destructive.
3. Abstained but could have recommended.
4. Acceptable conservative decision.
5. Correct safe recommendation or correct abstention.

The `objective_metric_*` fields are optional placeholders. DNSMOS or another objective metric may be entered later when available; this suite does not install or compute heavy metrics.

## Usage

```bash
python scripts/run_real_audio_demo_suite.py \
  --manifest data/manifests/real_audio_demo.local.csv \
  --output-root outputs/real-audio-demo-suite
```

Optional runtime arguments provide router, Demucs, and target-noise checkpoint paths.

## Data And Artifact Policy

Real recordings, copyrighted media, processed outputs, and local manifests containing private paths must not be committed. Keep them under ignored local data and output directories.

The suite demonstrates actual input/output behavior and supports listening-based comparison. It does not replace paired objective evaluation or prove general model quality.
