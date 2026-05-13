# ADR-001: MVP Scope For Offline Denoise Demo

- Status: Accepted
- Date: 2026-04-01

## Context
The repository is architecture-first and currently contains empty module directories and placeholder documentation. The immediate need is a realistic demo proposal, not a full product. The system must show a credible path from local noisy media input to cleaned output while keeping implementation cost low and system boundaries clear.

The requested repository shape already centers on these modules:
- `src/io`
- `src/media`
- `src/engine`
- `src/pipeline`
- `src/eval`
- `src/storage`
- `src/api`

The user also constrained scope aggressively:
- no UI
- no realtime flow
- no training
- no advanced features
- docs first

## Decision
Build the MVP as a minimal, offline, CLI-first denoise workflow around:
- `ffmpeg` for media probing, extraction, conversion, and remux
- one pretrained denoise engine adapter in `src/engine`
- `DeepFilterNet` as the default engine for the first implementation path
- `src/api` as a boundary-contract module only, not a real HTTP service
- `src/eval` as a minimal, demo-only post-run validation layer

The MVP handles one short local file per run:
- audio input -> cleaned audio output
- video input -> audio extraction, denoise, remuxed video output

## Consequences

### Positive
- Delivery risk is low because the system follows one narrow path.
- The architecture is easy to explain because each module has one primary responsibility.
- Local-only execution makes the demo independent from service infrastructure.
- `src/api` can later support another interface without forcing core-pipeline changes.
- Minimal evaluation keeps the first implementation focused on proving the flow works.

### Negative
- Format support will be intentionally narrow at first.
- Quality perception depends heavily on one default engine and the sample inputs.
- The project will not answer product questions about scale, latency, or multi-user access.
- Minimal evaluation means output quality claims must remain cautious and qualitative.

## Rejected Alternatives

### UI-First Application
Rejected because it adds presentation work before the pipeline and contracts are stable.

### Realtime Pipeline
Rejected because it introduces latency, chunking, buffering, and device concerns that are outside the proposal/demo goal.

### Multi-Engine Benchmarking
Rejected because it broadens the decision surface before the first end-to-end path exists.

### Training Or Custom Models
Rejected because data, compute, and evaluation complexity would dominate the project.

### Cloud Or Network Service
Rejected because it adds deployment, auth, and operational concerns unrelated to the first local demo.

## Follow-On Implication
The next implementation step should create only the minimum contracts and orchestration needed to support the documented single-file offline path. Any additional interface or quality feature should be justified after the demo flow works end to end.
