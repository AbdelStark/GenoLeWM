# RFC-0016: Performance budget and profiling discipline

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-05-20
- **Depends on:** RFC-0007, RFC-0010, RFC-0015
- **Supersedes:** —
- **Implementation status:** Not started

---

## 1. Summary

Performance is a public commitment. This RFC fixes the reference
machines, the per-class latency / throughput / memory targets, the
quantization-quality budget, the profiling tools, and the regression
process. Numbers are in [`docs/spec/08-performance-budget.md`](../docs/spec/08-performance-budget.md);
this RFC defines the mechanism.

## 2. Motivation

The freedom-tech promise (on-device personal-genome inference) only
matters if the model is fast enough on consumer hardware. The variant
scoring claim only matters if the latency is materially better than
Carbon's zero-shot path. Without a budget, every "make it work" PR drifts
toward worse numbers. With one, regressions are observable and the
project cannot ship a misbehaving release.

## 3. Specification

### 3.1 Reference machines

- **Server**: 1× H100 80 GB.
- **Workstation**: 1× RTX 4090 24 GB.
- **Laptop**: Apple M3 Max 64 GB.
- **CPU-only laptop**: 8-core x86-64, 32 GB.

CI per-PR benchmarks run on a hosted GPU runner. Release-candidate
benchmarks run on the four dedicated machines.

### 3.2 Targets

The full table is in
[`docs/spec/08-performance-budget.md`](../docs/spec/08-performance-budget.md).
Load-bearing numbers:

- Single-variant warm scoring on M3 Max: < 200 ms.
- 100 k-variant VCF on M3 Max: < 30 min.
- Phase-1 baseline training on a single H100: < 24 h.
- Predictor + Carbon-500M at int4 on M3 Max: < 8 GB unified memory.

### 3.3 Quantization budget

| Level | Max AUROC drop (ClinVar coding) | Max rollout cosine drop |
|-------|-------------------------------:|------------------------:|
| int8 predictor | 1.0 | 0.02 |
| int4 Carbon + int8 predictor | 2.0 | 0.05 |

A quantized checkpoint is shippable only if it stays within these
budgets on the documented benchmark machine.

### 3.4 Benchmark harness

`bench/` directory with:

- `bench/inference.py`: latency / throughput / memory benchmarks.
- `bench/training.py`: training-step timing on synthetic data.
- `bench/planning.py`: CEM-call latency.
- `bench/profile.py`: profiler entry points (py-spy, torch.profiler).

Each benchmark writes a structured result to `bench/results/<machine>/<benchmark>.json`
with median, IQR, and metadata (commit, hardware, torch version, dtype).

### 3.5 Microbenchmarks

`pytest-benchmark` over hot functions. Run nightly; historical results
persisted; > 5% regression opens a tracking issue.

### 3.6 Profiling tools

| Concern | Tool |
|---------|------|
| Python CPU | py-spy (sampled), cProfile (deep) |
| GPU | torch.profiler, NVIDIA Nsight Systems |
| Memory | tracemalloc, nvidia-smi, Metal System Trace |
| Microbench | pytest-benchmark |

The tool list is canonical for v1; alternatives are allowed but the
canonical set is what CI invokes.

### 3.7 Regression process

1. Nightly benchmark detects > 5% regression on a tracked metric.
2. CI opens a tracking issue (label: `type:perf`, `priority:p1`).
3. Engineer bisects to the culprit commit.
4. Outcomes:
   - revert,
   - fix-forward,
   - accept (requires RFC amendment + CHANGELOG entry under "Changed").

### 3.8 What is *not* a perf gate

- Microbench results below the 5% threshold.
- One-off CI flakes (median over 5 runs is the deciding number).
- Memory under bf16 mode when the change moves data to int8 with
  quality compensation; the relevant comparison is "same dtype, prior
  commit."

## 4. Rationale and alternatives

### 4.1 Why budget rather than aspirational targets?

Aspirational targets do not block releases. A budget that blocks releases
is the only mechanism that survives contributor churn.

### 4.2 Why include Apple Silicon as a first-class reference machine?

The freedom-tech audience skews Mac. Treating M3 Max as a third-tier
optional target would let the on-device claim quietly rot.

### 4.3 Why not use Triton / TVM / a custom kernel from day one?

Custom kernels make portability harder. The benchmark targets are
achievable with stock PyTorch on CUDA and Core ML / MLX on Apple. We
will revisit if a kernel is the only path to a target; not pre-optimize.

### 4.4 Why a 5% regression threshold?

CI runners are shared and noisy. 5% is empirically above the noise floor
on the GitHub-hosted GPU tier. The threshold is reviewable; lowering it
requires switching to dedicated runners.

## 5. Unresolved questions

- Whether to add iOS / iPad performance targets. Tracked as OQ-PERF-1.
- Whether to ship a `geno-lewm-bench` CLI exposing the harness publicly.
  Tracked as OQ-PERF-2.
- STARK proof-time targets for Phase 4 once provers stabilize. Tracked
  as OQ-PERF-3.
- Whether to publish historical benchmark dashboards (extra ops burden).

## 6. Future work

- Continuous benchmark visualization (e.g., codspeed.io or a local
  dashboard).
- Per-PR performance smoke that runs a 30-second subset on the hosted
  runner; signal-noise still acceptable.
- Memory-allocator instrumentation to attribute regressions to specific
  call sites.

## 7. Changelog

- 2026-05-20 — Initial draft.
