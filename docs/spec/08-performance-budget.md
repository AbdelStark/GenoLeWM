# 08 — Performance budget

- Status: Authoritative for v0.1
- Companion RFCs: [RFC-0007](../rfcs/0007-evaluation-suite.md),
  [RFC-0010](../rfcs/0010-on-device-personal-genome-deployment.md),
  [RFC-0016](../rfcs/0016-performance-budget.md)

Performance is part of the public release contract. Numbers here are
v0.1 release gates, not current measured results: a release that misses
any target is not shippable as v0.1 unless the regression is documented
and accepted via an explicit RFC amendment. Until
`efficiency_report.json` is generated from the first public checkpoint,
dataset, and demo artifact set, public docs and papers must describe
these numbers as budgets or targets only.

## Reference machines

| Class | Spec | Where used |
|-------|------|-----------|
| **Server** | 1× NVIDIA H100 80 GB SXM5, 256 GB RAM, NVMe SSD | training, batched inference |
| **Workstation** | 1× NVIDIA RTX 4090 24 GB, 64 GB RAM, NVMe SSD | researcher local |
| **Laptop** | Apple M3 Max, 64 GB unified memory, NVMe SSD | on-device target |
| **CPU-only laptop** | 8-core x86-64, 32 GB RAM, NVMe SSD | accessibility fallback |

Per-PR checks may run deterministic microbenchmarks where suitable
hardware is available. Release-candidate benchmarks must run on a
dedicated reference machine per class and record explicit limitations
when a class is unavailable.

## Inference latency (warm cache)

| Metric | H100 | RTX 4090 | M3 Max | CPU-only |
|--------|-----:|---------:|-------:|---------:|
| Single-variant scoring (warm) | < 5 ms | < 20 ms | < 200 ms | < 1.5 s |
| Single-variant scoring (cold; Carbon call) | < 50 ms | < 100 ms | < 800 ms | < 6 s |
| Predictor-only forward pass (bf16) | < 1 ms | < 3 ms | < 25 ms | < 200 ms |
| Action-encoder forward pass | < 0.2 ms | < 0.5 ms | < 5 ms | < 30 ms |

Cold path includes one Carbon-500M call on the edited window; the warm
path assumes both windows are in the LRU.

## Inference throughput (batched)

| Metric (batch=256) | H100 | RTX 4090 | M3 Max |
|--------------------|-----:|---------:|-------:|
| Variants per second | > 5,000 | > 1,000 | > 100 |
| Tokens per second (Carbon-500M, bf16) | > 1.0 M | > 250 k | > 30 k |

Batched throughput is measured after warm-up (10 batches discarded);
reported value is the median across 100 batches.

## Memory footprint

| Metric | H100 | RTX 4090 | M3 Max |
|--------|-----:|---------:|-------:|
| Predictor only (loaded) | < 200 MB | < 200 MB | < 200 MB |
| Predictor + Carbon-500M (bf16) | < 3 GB | < 3 GB | < 8 GB |
| Predictor + Carbon-500M (int4) | n/a | n/a | < 1 GB |
| Process RSS at idle (no model) | < 200 MB | < 200 MB | < 300 MB |
| Process RSS during VCF scoring (b=64) | < 4 GB | < 4 GB | < 10 GB |

Peak memory is measured via `psutil` (RSS) and `nvidia-smi` (GPU).
Apple Silicon uses unified memory; reported value is the larger of CPU
RSS and Metal peak.

Release candidates record the actual first-paper single-variant latency,
batched throughput, and peak-memory run as
`efficiency_report.json`, generated with
`python -m bench.inference --release-efficiency ... --output-json ...`.
That artifact must include the benchmark command, hardware/runtime notes,
input identities, sample count, warm-up count, and limitations; prose in
the README or paper draft must not substitute for the machine-readable
report.

## Disk footprint

| Item | Size |
|------|------|
| GenoLeWM artifact at bf16 (predictor + action encoder + calibration) | < 120 MB |
| GenoLeWM artifact at int8 | < 40 MB |
| Carbon-500M at bf16 (Hub download) | ~1 GB |
| Reference embedding cache (default config, full human genome) | < 5 GB |
| Sum of model + cache + tokenizer + receipts (typical user) | < 5 GB |

## Whole-VCF scoring

| Operation | H100 | RTX 4090 | M3 Max |
|-----------|-----:|---------:|-------:|
| 100k-variant VCF scoring | < 1 min | < 5 min | < 30 min |
| Whole-genome reference encoding | < 30 min | < 2 h | < 8 h |

These targets assume the reference embedding cache is empty at the start
(cold path). With a warm cache, throughput approaches the per-variant
predictor throughput.

## Training compute budget

| Phase | Tokens (target embeddings) | Wall clock on H100 |
|-------|---------------------------:|-------------------:|
| Phase 1 baseline | ~10 B | ~24 h |
| Phase 1 full | ~50 B | ~5 d |
| Phase 2 (LoRA + full edit suite) | ~100 B | ~10 d |

The Phase 1 baseline must complete in under 24 hours on a single H100; an
implementation that breaks this budget is not shippable.

## Planning latency

CEM with `n_iterations=5`, `n_samples=1024`, `K=5` (default config):

| Metric | H100 | M3 Max |
|--------|-----:|-------:|
| Planning call (end-to-end) | < 1 s | < 30 s |
| Predictor calls per planning call | 25,600 | 25,600 |

Planning with larger `n_samples` or `K` scales linearly in predictor
calls.

## Quantization quality budget

A quantized variant is shippable only when:

| Quantization | Max AUROC drop on ClinVar coding | Max rollout cosine drop |
|--------------|---------------------------------:|------------------------:|
| `int8` (predictor only) | 1.0 | 0.02 |
| `int4` (Carbon Q4_K_M + predictor int8) | 2.0 | 0.05 |

Drops are measured against the bf16 reference checkpoint on the same data.

## Profiling discipline

- **CPU profiling:** `py-spy` (sampled) for CLI commands; `cProfile` for
  small reproducers.
- **GPU profiling:** `torch.profiler` for training; NVIDIA Nsight Systems
  for ad-hoc deep dives.
- **Memory profiling:** `tracemalloc` for Python; `nvidia-smi`/Metal
  System Trace for device memory.
- **Microbenchmarks:** `pytest-benchmark` for hot functions; results
  persisted in `bench/results/`.

After the first measured release baseline exists, every performance
regression > 5% on any benchmark in this section is treated as a defect.
CI should not auto-fail on noisy shared-runner perf changes, but release
automation or a scheduled benchmark job must open a tracking issue when
a confirmed regression is detected.

## Performance regression process

1. Nightly benchmark detects a > 5% regression.
2. Tracking issue opened with the suspect range of commits.
3. Bisect to the culprit commit.
4. Either revert, fix, or formally accept the regression with an RFC
   amendment + CHANGELOG entry.

## Invariants

| ID | Invariant | Enforced by |
|----|-----------|-------------|
| INV-PERF-1 | The Phase-1 baseline run completes within 24 hours on a single H100 | release-gate check |
| INV-PERF-2 | Predictor + Carbon-500M run within 8 GB unified memory on M3 Max at int4 | release-gate check |
| INV-PERF-3 | Single-variant warm-cache scoring on M3 Max ≤ 200 ms | `bench.inference --release-efficiency`; `tools.release.efficiency_report` |
| INV-PERF-4 | No benchmark in this document regresses > 5% silently between releases | nightly job |
| INV-PERF-5 | Memory targets are measured peak-RSS, not steady-state | benchmark harness |

## Open questions

| ID | Question | Owner | Target |
|----|----------|-------|--------|
| OQ-PERF-1 | Whether to add iOS / iPad performance targets in v0.2 | core | v0.2 |
| OQ-PERF-2 | Whether to ship a `geno-lewm-bench` command exposing the benchmark harness publicly | core | v0.2 |
| OQ-PERF-3 | What public latency target to require for the real terminal demo | core | first demo release |
