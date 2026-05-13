# Architecture

## High-Level Component View

```mermaid
flowchart LR
    operator[CLI Operator]
    input[(Local Input File)]
    outputs[(outputs/)]
    temp[(tmp/)]
    ffmpeg[ffmpeg]
    weights[(Pretrained Model Weights)]

    subgraph app[Offline Denoise Demo]
        api[src/api<br/>Boundary contracts]
        io[src/io<br/>Path validation and naming]
        pipeline[src/pipeline<br/>Run orchestration]
        media[src/media<br/>Probe, extract, convert, remux]
        engine[src/engine<br/>DeepFilterNet adapter]
        eval[src/eval<br/>Demo-only checks]
        storage[src/storage<br/>Temp/output conventions]
    end

    operator --> api
    input --> io
    api --> pipeline
    io --> pipeline
    pipeline --> media
    pipeline --> engine
    pipeline --> eval
    pipeline --> storage
    media <--> ffmpeg
    engine --> weights
    storage --> temp
    storage --> outputs
    media --> storage
    engine --> storage
    eval --> storage
```

## Processing Sequence

```mermaid
sequenceDiagram
    actor Operator
    participant API as src/api
    participant IO as src/io
    participant Pipeline as src/pipeline
    participant Media as src/media
    participant Engine as src/engine
    participant Eval as src/eval
    participant Storage as src/storage
    participant FFmpeg as ffmpeg

    Operator->>API: submit DenoiseRequest
    API->>IO: validate paths and derive output names
    IO-->>API: validated input context
    API->>Pipeline: start run
    Pipeline->>Media: probe input

    alt Audio input
        Media-->>Pipeline: prepared audio path
    else Video input
        Media->>FFmpeg: extract and normalize audio
        FFmpeg-->>Media: working audio file
        Media-->>Pipeline: prepared audio path
        Note over Media,FFmpeg: Video-remux branch keeps the original video stream and swaps in cleaned audio later.
    end

    Pipeline->>Engine: load default engine and denoise
    Engine-->>Pipeline: cleaned audio path
    Pipeline->>Media: export audio or remux video
    Media->>FFmpeg: write final media artifact
    FFmpeg-->>Media: final artifact path
    Pipeline->>Eval: run minimal post-run checks
    Eval-->>Pipeline: run summary
    Pipeline->>Storage: finalize outputs and temp retention
    Storage-->>API: final paths and summary
    API-->>Operator: DenoiseResult
```

## Notes
- `src/api` is a contract boundary for the CLI flow, not a network service in MVP.
- `src/eval` is limited to lightweight demo checks such as output existence and basic sanity validation.
- `src/media` is the only module that talks directly to `ffmpeg`.
- `src/storage` is responsible for keeping `outputs/` and `tmp/` distinct.
