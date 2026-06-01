"""Run a small VoiceBank benchmark across selected denoise engines."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from src.api.contracts import DenoiseRequest
from src.datasets.voicebank_manifest import load_voicebank_manifest, validate_voicebank_pairs
from src.eval.metrics import load_mono_audio, si_sdr_db as compute_si_sdr_db
from src.eval.metrics import snr_db as compute_snr_db
from src.eval.metrics import snr_improvement_db as compute_snr_improvement_db
from src.io.paths import derive_output_mode, infer_input_type
from src.pipeline.run_pipeline import run_pipeline


DEFAULT_ENGINES = "noisy_input,ffmpeg-arnndn,noisereduce,deepfilternet"
FIELDNAMES = [
    "sample_id",
    "engine",
    "status",
    "error",
    "runtime_sec",
    "noisy_path",
    "clean_path",
    "enhanced_path",
    "snr_noisy_db",
    "snr_enhanced_db",
    "snr_improvement_db",
    "si_sdr_db",
]


def _parse_engines(value: str) -> list[str]:
    engines = [engine.strip() for engine in value.split(",") if engine.strip()]
    if not engines:
        raise ValueError("At least one engine must be provided.")
    return engines


def _empty_metric_row(sample_id: str, engine: str, noisy_path: Path, clean_path: Path) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "engine": engine,
        "status": "failed",
        "error": "",
        "runtime_sec": "",
        "noisy_path": str(noisy_path),
        "clean_path": str(clean_path),
        "enhanced_path": "",
        "snr_noisy_db": "",
        "snr_enhanced_db": "",
        "snr_improvement_db": "",
        "si_sdr_db": "",
    }


def run_voicebank_benchmark(
    *,
    manifest_path: Path,
    output_path: Path,
    output_root: Path,
    engines: list[str],
    limit: int | None = None,
) -> Path:
    """Run selected engines on a VoiceBank manifest and write metric CSV rows."""
    pairs = load_voicebank_manifest(manifest_path)
    validation_errors = validate_voicebank_pairs(pairs)
    if validation_errors:
        raise ValueError("VoiceBank manifest has missing files: " + "; ".join(validation_errors))

    if limit is not None:
        pairs = pairs[:limit]

    rows: list[dict[str, str]] = []
    for pair in pairs:
        clean_audio, _ = load_mono_audio(pair.clean_path)
        noisy_audio, _ = load_mono_audio(pair.noisy_path)
        snr_noisy_value = compute_snr_db(clean_audio, noisy_audio)
        output_mode = derive_output_mode(infer_input_type(pair.noisy_path))

        for engine in engines:
            row = _empty_metric_row(pair.sample_id, engine, pair.noisy_path, pair.clean_path)
            if engine == "noisy_input":
                row.update(
                    {
                        "status": "success",
                        "runtime_sec": "0.000000",
                        "enhanced_path": str(pair.noisy_path),
                        "snr_noisy_db": f"{snr_noisy_value:.6f}",
                        "snr_enhanced_db": f"{snr_noisy_value:.6f}",
                        "snr_improvement_db": "0.000000",
                        "si_sdr_db": f"{compute_si_sdr_db(clean_audio, noisy_audio):.6f}",
                    }
                )
                rows.append(row)
                continue

            started_at = time.perf_counter()
            try:
                result = run_pipeline(
                    DenoiseRequest(
                        input_path=pair.noisy_path,
                        output_dir=Path(output_root) / pair.sample_id / engine,
                        output_mode=output_mode,
                        engine_name=engine,
                    )
                )
                runtime_sec = time.perf_counter() - started_at
                if result.final_output_path is None:
                    raise RuntimeError("Pipeline did not return a final output path.")

                enhanced_audio, _ = load_mono_audio(result.final_output_path)
                snr_enhanced_value = compute_snr_db(clean_audio, enhanced_audio)
                row.update(
                    {
                        "status": "success",
                        "runtime_sec": f"{runtime_sec:.6f}",
                        "enhanced_path": str(result.final_output_path),
                        "snr_noisy_db": f"{snr_noisy_value:.6f}",
                        "snr_enhanced_db": f"{snr_enhanced_value:.6f}",
                        "snr_improvement_db": f"{compute_snr_improvement_db(clean_audio, noisy_audio, enhanced_audio):.6f}",
                        "si_sdr_db": f"{compute_si_sdr_db(clean_audio, enhanced_audio):.6f}",
                    }
                )
            except Exception as exc:
                row["runtime_sec"] = f"{time.perf_counter() - started_at:.6f}"
                row["error"] = str(exc)
            rows.append(row)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return output.resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the benchmark CLI parser."""
    parser = argparse.ArgumentParser(description="Run VoiceBank denoise benchmark.")
    parser.add_argument("--manifest", default=Path("data/manifests/voicebank_subset.csv"), type=Path)
    parser.add_argument("--output", default=Path("outputs/reports/voicebank_benchmark.csv"), type=Path)
    parser.add_argument("--output-root", default=Path("outputs/benchmarks/voicebank"), type=Path)
    parser.add_argument("--engines", default=DEFAULT_ENGINES)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        output_path = run_voicebank_benchmark(
            manifest_path=args.manifest,
            output_path=args.output,
            output_root=args.output_root,
            engines=_parse_engines(args.engines),
            limit=args.limit,
        )
        print(f"Wrote VoiceBank benchmark CSV: {output_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
