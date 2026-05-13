# Open Questions

These items remain intentionally unresolved after the MVP scope is fixed. They matter for implementation quality, packaging, or the first extension after the demo, but they do not block writing the initial architecture-first docs.

## Runtime And Packaging
- Should the first implementation target a Python-only local runtime, or should the repository treat the runtime choice as an implementation detail until code starts?
- How should environment setup be presented for the demo: manual dependency installation, a single bootstrap script, or container-based execution?

## `ffmpeg` Dependency Strategy
- Is `ffmpeg` expected to be preinstalled on the operator machine, bundled separately, or checked at startup with setup guidance?
- Do we want to pin a minimum `ffmpeg` version in the first implementation docs once code begins?

## Media Format Boundaries
- Which exact audio and video containers/codecs should count as supported in v1?
- Should the first version aggressively normalize all extracted audio to one working format such as mono WAV, or only convert when the engine requires it?
- How should the pipeline behave when a video file has multiple audio tracks or no usable audio track?

## Engine Weights And Licensing
- Where should pretrained weights live for local runs: a cache location outside the repo, a `data/` subdirectory, or user-managed paths?
- What license and redistribution constraints apply to the chosen pretrained model and its weights?
- Do we need a documented offline-first path for obtaining weights before the first demo run?

## Minimal Evaluation Scope
- What exact checks belong in `src/eval` for v1: file existence only, metadata sanity checks, duration comparison, or a tiny human-review checklist?
- Should the demo explicitly report when evaluation is qualitative only so users do not infer benchmark-grade claims?

## Future Shape Of `src/api`
- Should `src/api` remain an internal boundary-contract module permanently, or should it later become the source of a local HTTP wrapper?
- If a local service is added later, should the request/result schema remain identical to the CLI contract or branch into separate transport-specific models?
