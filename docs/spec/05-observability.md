# 05 — Observability

- Status: Authoritative for v0.1
- Companion RFC: [RFC-0013](../../rfcs/0013-observability.md)

GenoLeWM's observability surface has three layers — logging, metrics,
tracing — plus a strict redaction filter that runs over all of them.
Logs and metrics are structured; nothing is free-form. The redaction
filter is the single defense against personal-data exfiltration, and
its tests are non-negotiable.

## Principles

1. **Structured by default.** Every event is a key-value record. Free-text
   `print` is forbidden in `geno_lewm/`.
2. **Redaction by default.** The logger receives no user variants, no
   personal genome data, no FASTA bytes. The redaction filter rejects any
   field that fails its allowlist.
3. **Stable keys.** Event names and metric names are part of the public
   contract. Renaming an event name is a MAJOR change.
4. **Local-first.** Default sink is the local filesystem; remote sinks
   (wandb, OpenTelemetry collectors) are opt-in.
5. **Cost-bounded.** The observability layer must not dominate training
   compute. Sampling and rate-limiting are explicit.

## Logging

### Format

JSON Lines (`.jsonl`), one record per line, UTF-8.

```json
{
  "ts": "2026-05-20T15:42:01.123Z",
  "severity": "info",
  "event": "training.step",
  "run_id": "run-2026-05-20-h100-0",
  "step": 12345,
  "epoch": 2,
  "phase": "phase1",
  "duration_ms": 412,
  "data": { ... event-specific fields ... }
}
```

### Required fields on every record

| Field | Type | Notes |
|-------|------|-------|
| `ts` | string | ISO-8601 UTC, millisecond resolution |
| `severity` | enum | `debug` \| `info` \| `warn` \| `error` |
| `event` | string | dotted-lowercase event name |
| `run_id` | string | random or human-supplied per process |
| `data` | object | event-specific structured fields |

Optional but standardized:

| Field | Type | Notes |
|-------|------|-------|
| `step`, `epoch`, `phase` | ints / string | training loop |
| `duration_ms` | int | for spans |
| `trace_id`, `span_id` | string | OpenTelemetry-compatible 16/8-byte hex |
| `error_code` | string | only on `severity=error` records |

### Event registry

Event names are registered in `geno_lewm/observability.py::EVENTS`.
The registry is the source of truth and is regenerated into
[`docs/api/log-events.md`](../api/log-events.md) on each release.

Canonical events for v0.1:

| Event | Severity | When emitted |
|-------|----------|--------------|
| `training.run.start` | info | trainer initialized |
| `training.run.end` | info | trainer exited |
| `training.step` | debug | every step |
| `training.epoch.end` | info | every epoch |
| `training.checkpoint.write` | info | checkpoint saved |
| `training.collapse.alert` | warn | one of the alert criteria tripped |
| `training.metric` | info | scalar metric logged |
| `eval.run.start` | info | benchmark started |
| `eval.run.end` | info | benchmark finished |
| `eval.regression` | error | smoke eval regression |
| `data.cache.hit` | debug | cache hit |
| `data.cache.miss` | debug | cache miss |
| `data.shard.write` | info | new shard written |
| `inference.score.start` | debug | scoring call entered |
| `inference.score.end` | debug | scoring call returned |
| `inference.batch.end` | info | batch finished |
| `inference.network.blocked` | error | fail-closed guard tripped |
| `attestation.receipt.write` | info | receipt written |
| `attestation.verify.start` | info | verifier started |
| `attestation.verify.end` | info | verifier returned |
| `attestation.verify.mismatch` | error | a hash check failed |
| `error` | error | any `GenoLeWMError` raised inside a logged span |

### Severities

- `debug`: high-volume; off by default in release builds.
- `info`: durable record of normal operation.
- `warn`: degraded behavior; investigation suggested.
- `error`: typed failure; always accompanied by `error_code`.

### Sinks

Default: local filesystem at `${GENO_LEWM_LOG_DIR}/{run_id}.jsonl`.

Optional sinks:

- **wandb**: training metrics + summary events. Opt-in via `--wandb-project`.
- **stderr**: human-readable rendering for interactive use. Filtered by
  severity.
- **OpenTelemetry**: OTLP/gRPC export if `OTEL_EXPORTER_OTLP_ENDPOINT`
  is set. Disabled by default.

There is no default cloud sink. Telemetry is never enabled by default.

### Redaction

The redaction filter operates on the `data` field of every record. Rules:

1. **Allowlist of keys**: keys not in the per-event allowlist are dropped
   with a `redacted_keys` counter increment.
2. **Type allowlist**: nested values may be scalars, lists of scalars, or
   shallow dicts of scalars. Bytes objects and tensors are always dropped.
3. **Pattern filters**: any string field whose value matches a DNA-string
   pattern (`^[ACGTN]{20,}$`) is dropped.
4. **Personal-data fields**: explicit deny list for `vcf_content`,
   `genotype`, `sample_id`, `user_email`, etc.

The redaction filter has unit tests for every event in the registry and
property tests over random payloads.

## Metrics

### Format

Counter / gauge / histogram, with stable names. Metrics names are
dotted-lowercase, follow Prometheus naming conventions, and are
registered in `geno_lewm/observability.py::METRICS`.

### Canonical metrics

| Name | Kind | Unit | Notes |
|------|------|------|-------|
| `geno_lewm.training.step.duration` | histogram | ms | per training step |
| `geno_lewm.training.loss.pred` | gauge | unitless | last logged pred loss |
| `geno_lewm.training.loss.reg` | gauge | unitless | last logged reg loss (Phase 2 active, Phase 1 monitoring) |
| `geno_lewm.training.collapse.alert` | counter | events | total alerts during run |
| `geno_lewm.data.cache.hit` | counter | ops | per-call increment |
| `geno_lewm.data.cache.miss` | counter | ops | per-call increment |
| `geno_lewm.data.encode.duration` | histogram | ms | per-window encoder call |
| `geno_lewm.inference.score.duration` | histogram | ms | per variant |
| `geno_lewm.inference.batch.throughput` | gauge | variants/s | last batched scoring |
| `geno_lewm.inference.memory.peak_bytes` | gauge | bytes | RSS peak per process |
| `geno_lewm.planning.cem.iter.duration` | histogram | ms | per CEM iteration |
| `geno_lewm.planning.cem.calls` | counter | predictor calls | per planning run |
| `geno_lewm.attestation.verify.duration` | histogram | ms | per verification |
| `geno_lewm.observability.redacted_keys` | counter | keys | total dropped by filter |
| `geno_lewm.errors.raised` | counter | events | tagged with `code` |

### Exporters

- **Prometheus textfile**: written to `${GENO_LEWM_LOG_DIR}/metrics.prom`
  every `metrics_interval_s` (default 30 s).
- **OpenTelemetry OTLP/gRPC**: opt-in via env var.
- **wandb**: training metrics auto-forwarded when `--wandb-project` set.

## Tracing

Optional in v1; OpenTelemetry-compatible.

- Spans wrap top-level operations (`training.step`, `inference.score`,
  `planning.cem.iter`).
- Spans are emitted only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
- Trace IDs propagate into logs via `trace_id` / `span_id` fields.

## Sampling and rate limits

- Per-step training records are sampled at `1 / log_every` steps (default
  `log_every = 100`). Other event types are unsampled.
- Cache hit/miss debug records are sampled at `1 / 1000` by default.
- Metric flushes are not sampled.

## Configuration

The observability layer is configured via:

| Env var | Default | Meaning |
|---------|---------|---------|
| `GENO_LEWM_LOG_DIR` | `~/.geno-lewm/logs` | local log root |
| `GENO_LEWM_LOG_LEVEL` | `info` | global log severity threshold |
| `GENO_LEWM_LOG_FORMAT` | `jsonl` | `jsonl` \| `pretty` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | enables tracing if set |
| `GENO_LEWM_METRICS_INTERVAL_S` | `30` | flush interval |
| `GENO_LEWM_REDACTION_STRICT` | `1` | force-fail on disallowed payload (recommended) |

CLI flags override env vars; env vars override defaults.

## Privacy contract

- The default log root excludes any personal-data fields. The runtime
  fails closed on `GENO_LEWM_REDACTION_STRICT=1` if a redaction rule is
  bypassed.
- Crash logs sanitize stack traces of locals before writing; only model
  identifiers and stack frames are preserved.
- Per-variant scoring **never** logs the variant bases; it logs only the
  variant id (hash of chrom+pos+ref+alt) and the categorical outputs.

Failures of the privacy contract are bugs and surface as
`InternalError.InvariantViolation`.

## Invariants

| ID | Invariant | Enforced by |
|----|-----------|-------------|
| INV-OBS-1 | Every logged event has a name registered in `EVENTS` | linter rule |
| INV-OBS-2 | Every metric name is registered in `METRICS` | linter rule |
| INV-OBS-3 | The redaction filter is the only path from `logger.<level>(...)` to a sink | call-graph check |
| INV-OBS-4 | DNA strings ≥ 20 bp never appear in any log record | property test in CI |
| INV-OBS-5 | Telemetry is opt-in; no observability sink phones home without an env var | integration test |
| INV-OBS-6 | A crash before logger init still produces a sanitized minimal record | wrapper in `__main__` |

## Open questions

| ID | Question | Owner | Target |
|----|----------|-------|--------|
| OQ-OBS-1 | Whether to add an `aggregation.window_s` knob for batched metric exports | core | v0.2 |
| OQ-OBS-2 | Whether the redaction filter should also strip `clinvar_id` (it can be re-personalized via lookup) | core | before v0.2 |
| OQ-OBS-3 | Whether tracing should be on by default for CLI commands (off currently) | core | v0.3 |
