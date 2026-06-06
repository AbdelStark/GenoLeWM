# Benchmark harness

Performance benchmarks for GenoLeWM, defined by
[RFC-0016 §3.4](../rfcs/0016-performance-budget.md). The harness is
stdlib-only; per-target benchmark scripts may use heavier dependencies,
but the shared library at [`_harness.py`](_harness.py) does not.

## Layout

| File | What it times |
|------|---------------|
| [`inference.py`](inference.py) | Commitment microbenchmarks by default; release mode benchmarks the real `geno-lewm-score` command and writes `efficiency_report.json`. |
| [`training.py`](training.py) | `apply_edit` / `apply_edits` batches (data-prep hot path). Full training-step lands with [#44](https://github.com/AbdelStark/GenoLeWM/issues/44). |
| [`rollout.py`](rollout.py) | AR rollout speed report comparing cached `ARPredictor` rollout with naive repeated one-step `Predictor.forward` calls for the RFC-0004 K=5/K=20 targets. |
| [`planning.py`](planning.py) | Placeholder for the CEM solver ([#59](https://github.com/AbdelStark/GenoLeWM/issues/59) / [#60](https://github.com/AbdelStark/GenoLeWM/issues/60) / [#61](https://github.com/AbdelStark/GenoLeWM/issues/61)). |
| [`profile.py`](profile.py) | Canonical profiler invocations (py-spy, cProfile, tracemalloc, torch.profiler). |
| [`_harness.py`](_harness.py) | Shared library: `BenchResult`, `time_callable`, `write_result`, `machine_id`. |

## Result schema

Each script persists JSON to `bench/results/<machine>/<benchmark>.json`:

```json
{
  "schema_version": "1.0.0",
  "name": "inference.input_commitment",
  "iters": 200,
  "warmup": 20,
  "samples_ns": [...],
  "median_ns": 8421,
  "p25_ns": 8302,
  "p75_ns": 8590,
  "iqr_ns": 288,
  "metadata": {
    "commit": "abc1234",
    "timestamp": "2026-05-21T18:00:00+00:00",
    "machine": "M3-Max-laptop",
    "python_version": "3.12.5",
    "platform": "macOS-14.5-arm64",
    "dtype": "bf16",
    "extra": {"window_bytes": "4096"}
  }
}
```

`machine` is sanitised so it is safe as a directory name. Override
with `GENO_LEWM_BENCH_MACHINE=...` so multiple CI runners write to
distinct sub-directories.

## Running

```bash
# Default — 200 iters with warmup, persist results
python -m bench.inference
python -m bench.training
uv run --extra train python -m bench.rollout
python -m bench.planning

# Quick smoke without persistence
python -m bench.inference --iters 50 --no-write
uv run --extra train python -m bench.rollout \
  --k 5 --k 20 --iters 10 --warmup 2 --no-write

# Release efficiency evidence: median single-variant latency, batched VCF
# throughput, and child-process peak RSS from the real score command.
python -m bench.inference --release-efficiency \
  --model-dir model \
  --vcf demo/input.vcf \
  --fasta demo/ref.fa \
  --variant 1:10:A:T \
  --window ACGT... \
  --output-json model/efficiency_report.json

# Profile under cProfile
python -m bench.profile --run-cprofile-on bench.inference
```

`bench.rollout` records the normalized command in `rollout.ar_speed.json`
so release-readiness reports can bind the exact benchmark invocation.
`bench.rollout --require-targets` exits non-zero unless every requested
horizon meets the RFC-0004 speedup target (`>=2x` at K=5 and `>=5x` at
K=20). Use that flag for rollout-performance gates once the attention
KV-cache implementation is expected to satisfy the target.

## Regression detection

[`tools/ci/perf_regression.py`](../tools/ci/perf_regression.py)
compares the current result set against a committed baseline and fails
if any tracked benchmark regresses by more than the configured
threshold (default 5 %, RFC-0016 §3.7). The detector tolerates
placeholder entries (`iters=0`).
