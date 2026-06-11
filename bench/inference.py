# SPDX-License-Identifier: Apache-2.0
"""Inference-path benchmarks (performance budget).

The default no-argument mode keeps the lightweight commitment
microbenchmarks used by CI regression checks. Release mode benchmarks the
actual ``geno-lewm-score`` command and writes the machine-readable
``efficiency_report.json`` required by the first paper/demo package.

Usage::

    python -m bench.inference                  # default: 200 iters, write results
    python -m bench.inference --iters 50       # quick smoke
    python -m bench.inference --no-write       # report-only
    python -m bench.inference --release-efficiency \
      --model-dir model --vcf input.vcf --fasta ref.fa \
      --variant 1:10:A:T --window ACGT... --output-json efficiency_report.json

Result files land at ``bench/results/<machine>/inference.<workload>.json``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from bench._harness import (
    DEFAULT_RESULTS_DIR,
    BenchResult,
    current_commit,
    machine_id,
    report_to_stdout,
    time_callable,
    write_result,
)
from geno_lewm.action.spec import EditSpec
from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import (
    DtypeConfig,
    Manifest,
    PoolingConfig,
    ReceiptOutput,
    compute_input_commitment,
    compute_output_commitment,
    load_manifest,
    sha256_bytes,
    sha256_file,
)
from tools.release.efficiency_report import (
    GENERATED_BY as EFFICIENCY_REPORT_GENERATED_BY,
    parse_efficiency_report,
)

DEFAULT_RELEASE_SAMPLES: Final = 100
DEFAULT_RELEASE_WARMUP_BATCHES: Final = 10


def _bench_input_commitment(iters: int, warmup: int) -> BenchResult:
    window = "ACGT" * 1024  # 4 kB synthetic window
    edit = EditSpec(chrom="1", pos=1000, ref="A", alt="T")
    pooling = PoolingConfig(
        state_layer=20, pool_type="centered_mean", pool_radius=8, normalize=True
    )
    dtype = DtypeConfig(encoder_dtype="bf16", predictor_dtype="bf16")

    def workload() -> None:
        compute_input_commitment(
            reference_window=window,
            edit_spec=edit,
            pooling_config=pooling,
            dtype_config=dtype,
        )

    return time_callable(
        "inference.input_commitment",
        workload,
        iters=iters,
        warmup=warmup,
        dtype="bf16",
        extra={"window_bytes": str(len(window))},
    )


def _bench_output_commitment(iters: int, warmup: int) -> BenchResult:
    output = ReceiptOutput(
        sigma_raw=0.7321,
        sigma_calibrated=0.812,
        bucket_id="coding.missense",
        confidence=0.94,
        low_confidence=False,
    )

    def workload() -> None:
        compute_output_commitment(output)

    return time_callable(
        "inference.output_commitment",
        workload,
        iters=iters,
        warmup=warmup,
        dtype="bf16",
    )


@dataclass(frozen=True, slots=True)
class ReleaseEfficiencyRequest:
    """Inputs for a release-grade inference efficiency benchmark."""

    model_dir: Path
    vcf: Path
    fasta: Path
    output_json: Path
    variant: str
    window: str
    window_start_bp: int = 0
    backend: str = "auto"
    batch_size: int = 64
    samples: int = DEFAULT_RELEASE_SAMPLES
    warmup_batches: int = DEFAULT_RELEASE_WARMUP_BATCHES
    command: str = "geno-lewm-score"
    commit_sha: str | None = None
    dataset_snapshot: str | None = None
    hardware: str | None = None
    peak_memory_bytes: int | None = None
    allow_fixture_manifest: bool = False


def write_release_efficiency_report(
    request: ReleaseEfficiencyRequest,
    *,
    runner: object | None = None,
    memory_probe: object | None = None,
) -> Path:
    """Run the score command benchmark and write ``efficiency_report.json``."""
    payload = build_release_efficiency_payload(
        request,
        runner=runner,
        memory_probe=memory_probe,
    )
    report = parse_efficiency_report(payload)
    request.output_json.parent.mkdir(parents=True, exist_ok=True)
    request.output_json.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return request.output_json


def build_release_efficiency_payload(
    request: ReleaseEfficiencyRequest,
    *,
    runner: object | None = None,
    memory_probe: object | None = None,
) -> dict[str, object]:
    """Return a normalized efficiency payload from repeated score runs."""
    _validate_release_request(request)
    manifest = load_manifest(request.model_dir / "manifest.json")
    if not request.allow_fixture_manifest and _looks_like_fixture_manifest(manifest):
        raise InputError("fixture/test manifests cannot back a release efficiency benchmark")
    commit = request.commit_sha or current_commit()
    if commit == "unknown":
        raise InputError("commit SHA could not be resolved; pass --commit-sha")
    dataset_snapshot = request.dataset_snapshot or _dataset_snapshot_from_manifest(manifest)
    run = _default_runner if runner is None else runner
    probe = _peak_child_rss_bytes if memory_probe is None else memory_probe
    with tempfile.TemporaryDirectory(prefix="geno-lewm-bench-inference-") as tmp:
        output_root = Path(tmp)
        single_command = _single_variant_command(request)
        batch_command = _batch_command(request, output_root / "batch.scores.jsonl")
        single_samples = _time_command_samples(
            single_command,
            samples=request.samples,
            warmup=request.warmup_batches,
            runner=run,
            memory_probe=probe,
        )
        batch_samples = _time_command_samples(
            batch_command,
            samples=request.samples,
            warmup=request.warmup_batches,
            runner=run,
            memory_probe=probe,
        )
    variant_count = _count_vcf_alternates(request.vcf)
    single_latency_ms = _median(single_samples.elapsed_ns) / 1_000_000.0
    median_batch_seconds = _median(batch_samples.elapsed_ns) / 1_000_000_000.0
    throughput = variant_count / median_batch_seconds
    measured_peak = max(
        (*single_samples.peak_memory_bytes, *batch_samples.peak_memory_bytes), default=0
    )
    peak_memory = request.peak_memory_bytes or measured_peak
    if peak_memory <= 0:
        raise InputError(
            "peak memory could not be measured",
            remediation="rerun on a platform that exposes child RSS or pass --peak-memory-bytes",
        )
    return {
        "schema_version": "1.0.0",
        "generated_by": EFFICIENCY_REPORT_GENERATED_BY,
        "generated_at": _utc_now(),
        "model_id": manifest.model_id(),
        "model_release": manifest.release_id,
        "dataset_snapshot": dataset_snapshot,
        "commit": commit,
        "command": _benchmark_command(request),
        "hardware": request.hardware or f"{machine_id()} ({platform.platform(terse=True)})",
        "runtime": f"Python {platform.python_version()}; backend={request.backend}",
        "warmup_batches": request.warmup_batches,
        "samples": request.samples,
        "measurements": {
            "single_variant_latency_ms": single_latency_ms,
            "batched_throughput_variants_per_s": throughput,
            "peak_memory_bytes": peak_memory,
        },
        "inputs": _input_identities(request, manifest),
        "limitations": [
            "Subprocess wall-clock timing includes CLI startup and artifact loading overhead.",
            "Batched throughput is computed from scored VCF alternate rows per completed command.",
            "Peak memory is best-effort child-process RSS from the benchmark host.",
        ],
    }


@dataclass(frozen=True, slots=True)
class _TimedSamples:
    elapsed_ns: tuple[int, ...]
    peak_memory_bytes: tuple[int, ...]


def _time_command_samples(
    command: tuple[str, ...],
    *,
    samples: int,
    warmup: int,
    runner: object,
    memory_probe: object,
) -> _TimedSamples:
    elapsed: list[int] = []
    peaks: list[int] = []
    for index in range(warmup + samples):
        before = _call_memory_probe(memory_probe)
        start = time.perf_counter_ns()
        completed = _call_runner(runner, command)
        elapsed_ns = time.perf_counter_ns() - start
        peak = max(before, _call_memory_probe(memory_probe))
        if completed.returncode != 0:
            raise InputError(
                "score benchmark command failed",
                details={
                    "command": list(command),
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
            )
        if index >= warmup:
            elapsed.append(elapsed_ns)
            peaks.append(peak)
    return _TimedSamples(elapsed_ns=tuple(elapsed), peak_memory_bytes=tuple(peaks))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.inference",
        description="Inference-path benchmarks and release efficiency report generation.",
    )
    parser.add_argument(
        "--release-efficiency",
        action="store_true",
        help="run real geno-lewm-score subprocess benchmarks and write efficiency_report.json",
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--vcf", type=Path)
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--variant")
    parser.add_argument("--window")
    parser.add_argument("--window-start-bp", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=DEFAULT_RELEASE_SAMPLES)
    parser.add_argument("--commit-sha")
    parser.add_argument("--dataset-snapshot")
    parser.add_argument(
        "--hardware",
        help=(
            "hardware label to record in release efficiency evidence; defaults to "
            "the benchmark host identity"
        ),
    )
    parser.add_argument("--peak-memory-bytes", type=int)
    parser.add_argument(
        "--allow-fixture-manifest",
        action="store_true",
        help="allow fixture/test manifests for local benchmark smoke tests only",
    )
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--no-write", action="store_true", help="report-only; do not persist")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="results directory root (default: bench/results)",
    )
    args = parser.parse_args(argv)

    if args.release_efficiency:
        request = ReleaseEfficiencyRequest(
            model_dir=_required_path("--model-dir", args.model_dir),
            vcf=_required_path("--vcf", args.vcf),
            fasta=_required_path("--fasta", args.fasta),
            output_json=_required_path("--output-json", args.output_json),
            variant=_required_text("--variant", args.variant),
            window=_required_text("--window", args.window),
            window_start_bp=args.window_start_bp,
            backend=args.backend,
            batch_size=args.batch_size,
            samples=args.samples,
            warmup_batches=args.warmup,
            commit_sha=args.commit_sha,
            dataset_snapshot=args.dataset_snapshot,
            hardware=args.hardware,
            peak_memory_bytes=args.peak_memory_bytes,
            allow_fixture_manifest=args.allow_fixture_manifest,
        )
        try:
            path = write_release_efficiency_report(request)
        except GenoLeWMError as exc:
            sys.stderr.write(f"error: {exc.message or str(exc)}\n")
            if exc.details:
                sys.stderr.write(
                    json.dumps({"details": exc.details}, indent=2, sort_keys=True) + "\n"
                )
            if exc.remediation:
                sys.stderr.write(f"remediation: {exc.remediation}\n")
            return exit_code_for(exc)
        print(f"wrote {path}")
        return 0

    results = [
        _bench_input_commitment(args.iters, args.warmup),
        _bench_output_commitment(args.iters, args.warmup),
    ]
    for r in results:
        report_to_stdout(r)
        if not args.no_write:
            path = write_result(r, out_dir=args.out_dir)
            print(f"  wrote {path}")
    # Round-trip JSON to confirm serialisation is well-formed.
    for r in results:
        json.dumps(r.to_json())
    return 0


def _validate_release_request(request: ReleaseEfficiencyRequest) -> None:
    for field, path in (
        ("model_dir", request.model_dir),
        ("vcf", request.vcf),
        ("fasta", request.fasta),
    ):
        if not path.exists():
            raise InputError(f"{field} does not exist", details={"path": str(path)})
    if not request.model_dir.is_dir():
        raise InputError("model_dir must be a directory", details={"path": str(request.model_dir)})
    if request.samples <= 0:
        raise InputError("samples must be a positive integer")
    if request.warmup_batches < 0:
        raise InputError("warmup must be a non-negative integer")
    if request.batch_size <= 0:
        raise InputError("batch_size must be a positive integer")
    if (
        not isinstance(request.window_start_bp, int)
        or isinstance(request.window_start_bp, bool)
        or request.window_start_bp < 0
    ):
        raise InputError("window_start_bp must be a non-negative integer")
    if request.peak_memory_bytes is not None and request.peak_memory_bytes <= 0:
        raise InputError("peak_memory_bytes must be a positive integer when supplied")


def _single_variant_command(request: ReleaseEfficiencyRequest) -> tuple[str, ...]:
    return (
        request.command,
        "--quiet",
        "--no-banner",
        "--model-dir",
        str(request.model_dir),
        "--backend",
        request.backend,
        "--variant",
        request.variant,
        "--window",
        request.window,
        "--window-start-bp",
        str(request.window_start_bp),
        "--no-receipt",
    )


def _batch_command(request: ReleaseEfficiencyRequest, output: Path) -> tuple[str, ...]:
    return (
        request.command,
        "--quiet",
        "--no-banner",
        "--model-dir",
        str(request.model_dir),
        "--backend",
        request.backend,
        "--vcf",
        str(request.vcf),
        "--fasta",
        str(request.fasta),
        "--output",
        str(output),
        "--batch-size",
        str(request.batch_size),
        "--no-progress",
        "--no-receipt",
    )


def _benchmark_command(request: ReleaseEfficiencyRequest) -> list[str]:
    command = [
        "python",
        "-m",
        "bench.inference",
        "--release-efficiency",
        "--model-dir",
        str(request.model_dir),
        "--vcf",
        str(request.vcf),
        "--fasta",
        str(request.fasta),
        "--variant",
        request.variant,
        "--window",
        "<redacted-inline-window>",
        "--window-start-bp",
        str(request.window_start_bp),
        "--output-json",
        str(request.output_json),
        "--backend",
        request.backend,
        "--batch-size",
        str(request.batch_size),
        "--samples",
        str(request.samples),
        "--warmup",
        str(request.warmup_batches),
    ]
    if request.hardware is not None:
        command.extend(("--hardware", request.hardware))
    return command


def _input_identities(request: ReleaseEfficiencyRequest, manifest: Manifest) -> dict[str, object]:
    checkpoint = request.model_dir / manifest.predictor.file
    window_bytes = request.window.encode("ascii")
    return {
        "model_manifest": _file_identity(
            request.model_dir / "manifest.json",
            portable_path="model/manifest.json",
        ),
        "checkpoint": _file_identity(
            checkpoint,
            portable_path=(Path("model") / manifest.predictor.file).as_posix(),
        ),
        "vcf": _file_identity(
            request.vcf,
            portable_path=(Path("benchmark_inputs") / request.vcf.name).as_posix(),
        ),
        "fasta": _file_identity(
            request.fasta,
            portable_path=(Path("benchmark_inputs") / request.fasta.name).as_posix(),
        ),
        "single_window": {
            "path": "inline:single_window",
            "sha256": sha256_bytes(window_bytes),
            "size_bytes": len(window_bytes),
        },
    }


def _file_identity(path: Path, *, portable_path: str) -> dict[str, object]:
    return {
        "path": portable_path,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _dataset_snapshot_from_manifest(manifest: Manifest) -> str:
    value = manifest.training.data_snapshot.get("snapshot")
    if isinstance(value, str) and value.strip():
        return value.strip()
    for candidate in manifest.training.data_snapshot.values():
        if candidate.strip():
            return candidate.strip()
    raise InputError("dataset snapshot is not recorded; pass --dataset-snapshot")


def _looks_like_fixture_manifest(manifest: Manifest) -> bool:
    parts = [
        manifest.release_id,
        manifest.model_version,
        *manifest.training.data_snapshot.keys(),
        *manifest.training.data_snapshot.values(),
    ]
    text = " ".join(parts).lower()
    return any(token in text for token in ("fixture", "dummy", "test"))


def _count_vcf_alternates(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.startswith("#") or not raw.strip():
                continue
            columns = raw.rstrip("\n").split("\t")
            if len(columns) < 5:
                raise InputError("VCF row has fewer than five columns", details={"path": str(path)})
            alts = [alt for alt in columns[4].split(",") if alt and alt != "."]
            count += len(alts)
    if count <= 0:
        raise InputError("VCF contains no alternate alleles", details={"path": str(path)})
    return count


def _median(values: Sequence[int]) -> int:
    if not values:
        raise InputError("cannot compute median over empty samples")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return round((ordered[mid - 1] + ordered[mid]) / 2)


def _default_runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _call_runner(runner: object, command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    if not callable(runner):
        raise TypeError("runner must be callable")
    completed = runner(command)
    if not isinstance(completed, subprocess.CompletedProcess):
        raise TypeError("runner must return subprocess.CompletedProcess")
    return completed


def _peak_child_rss_bytes() -> int:
    try:
        import resource
    except ModuleNotFoundError:
        return 0

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    max_rss = int(usage.ru_maxrss)
    if max_rss <= 0:
        return 0
    if platform.system() == "Darwin":
        return max_rss
    return max_rss * 1024


def _call_memory_probe(probe: object) -> int:
    if not callable(probe):
        raise TypeError("memory_probe must be callable")
    value = probe()
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("memory_probe must return an integer byte count")
    return value


def _required_path(name: str, value: Path | None) -> Path:
    if value is None:
        raise InputError(f"release efficiency benchmark requires {name}")
    return value


def _required_text(name: str, value: str | None) -> str:
    if value is None or not value.strip():
        raise InputError(f"release efficiency benchmark requires {name}")
    return value.strip()


def _utc_now() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
