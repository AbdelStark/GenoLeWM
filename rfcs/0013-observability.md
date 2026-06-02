# RFC-0013: Observability — logging, metrics, redaction

- **Status:** Accepted
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-06-02
- **Depends on:** RFC-0001, RFC-0012
- **Supersedes:** —
- **Implementation status:** Implemented for the v1 local/opt-in sink
  surface: structured JSONL logging, event and metric registries, strict
  redaction, Prometheus textfile metrics, trace-context propagation,
  opt-in wandb sink, and event/metric registry linting are present with
  unit, integration, and lint coverage. OpenTelemetry export remains a
  future optional sink.

---

## 1. Summary

This RFC specifies GenoLeWM's observability layer: the structured log
format, the canonical event and metric registries, the optional tracing
surface, the sink set (local filesystem default; wandb / OpenTelemetry
opt-in), and — load-bearing for the project's privacy posture — the
redaction filter that blocks personal genome data from ever leaving a
record. Observability is opt-in for remote sinks, mandatory in shape for
local records, and fully redacted by default.

## 2. Motivation

The project's threat model
([`docs/spec/06-security.md`](../docs/spec/06-security.md)) treats personal
variant data as the most sensitive consumer-AI category. The single most
likely exfiltration channel is the observability surface: a stray
`logger.info(f"variant={variant!r}")` ships a person's DNA into a wandb
dashboard. The redaction filter is the structural defense against that
class of bug, and the entire observability layer is designed so it is the
only path from `logger.<level>(...)` to a sink.

A consistent log format and a registered event namespace also let
downstream tooling (dashboards, log parsers, paper-replication scripts)
treat GenoLeWM logs as data.

## 3. Specification

The full contract is in [`docs/spec/05-observability.md`](../docs/spec/05-observability.md).
This RFC locks the load-bearing decisions.

### 3.1 Logging format

JSON Lines, one record per line, UTF-8.
Required fields: `ts`, `severity`, `event`, `run_id`, `data`.
Severity enum: `debug | info | warn | error`.

### 3.2 Event registry

`EVENTS` is a tuple of `(name, severity, summary)` in
`geno_lewm/observability.py`. Event names are dotted-lowercase and stable
across MINORs; renaming is a MAJOR change.

A linter rule walks every call to `logger.<level>(event=...)` and
asserts the name is registered.

### 3.3 Metrics

Counter / gauge / histogram with stable names registered in `METRICS`.
Prometheus naming conventions (lowercase, underscored, unit suffixes).
Local sink: textfile collector at `${GENO_LEWM_LOG_DIR}/metrics.prom`,
flushed every `metrics_interval_s` (default 30 s).

### 3.4 Tracing

Optional OpenTelemetry-compatible spans. Disabled unless
`OTEL_EXPORTER_OTLP_ENDPOINT` is set. Trace IDs propagate into log
records via `trace_id` / `span_id`.

### 3.5 Redaction filter

Single filter at the logger boundary. Rules in priority order:

1. Per-event allowlist on `data` keys. Unallowlisted keys are dropped
   and counted via `geno_lewm.observability.redacted_keys`.
2. Type allowlist for values: scalars, lists of scalars, shallow dicts
   of scalars. Bytes / tensors / arrays dropped.
3. Pattern filter on string fields: any value matching
   `^[ACGTNacgtn]{20,}$` is dropped regardless of key.
4. Explicit deny-list: `vcf_content`, `genotype`, `sample_id`,
   `user_email`, `email`, `phone`, `address`, `dob`, `birthdate`, etc.

`GENO_LEWM_REDACTION_STRICT=1` (default) causes redaction-rule violations
to raise `InternalError.InvariantViolation` rather than silently drop.
This is the default because silent drops are an exfiltration risk
disguised as observability noise.

### 3.6 Sinks

| Sink | Default | Activation |
|------|--------:|------------|
| Local filesystem (JSONL) | on | always |
| stderr (pretty) | on | TTY detected |
| wandb | off | `--wandb-project` or `WANDB_PROJECT` env |
| OpenTelemetry OTLP | off | `OTEL_EXPORTER_OTLP_ENDPOINT` env |

There is no default cloud sink. No sink can be enabled by a config file
the user did not edit; every remote sink is opt-in.

### 3.7 Sampling

- Per-step training records: `1 / log_every` (default 100).
- Cache hit/miss debug records: `1 / 1000`.
- Metric flushes: unsampled.

### 3.8 Configuration

Env vars in `docs/spec/05-observability.md`. CLI flags override env vars
which override defaults. The config schema is documented in
[RFC-0017 §3.3](0017-configuration-system.md#33-namespaces).

## 4. Rationale and alternatives

### 4.1 Why JSONL and not OTel-only?

JSONL is the lowest-friction format for both human and machine. OTel
adds a runtime dependency that we want opt-in only. JSONL files survive
crashes, replays, and offline workflows in a way that an OTel collector
endpoint does not.

### 4.2 Why a single redaction filter rather than per-call-site care?

Per-call-site care does not survive contributor churn. A single
filter is a structural defense; a code-review rule is a process defense
that decays.

### 4.3 Why `GENO_LEWM_REDACTION_STRICT=1` by default?

Failing closed on observability is the only sane default for a
personal-health tool. Users who explicitly want lossy redaction (in
debugging contexts) can set `=0` and accept the responsibility.

### 4.4 Why not Prometheus client library by default?

The Prometheus client adds a runtime dependency and an HTTP scrape
endpoint that is undesirable for on-device users. The textfile collector
is sufficient and is local-only.

### 4.5 Why are event and metric names public?

Downstream dashboards and paper replications grep for them. Treating
them as private API would let us rename freely but break every user's
analysis pipeline. The stability cost is worth it.

## 5. Unresolved questions

- Whether to add a per-call structured-data type validation that rejects
  unallowlisted nested types at logger entry (currently dropped silently).
  Tracked as OQ-OBS-2.
- Whether tracing should be on by default for CLI commands. Currently
  off; tracked as OQ-OBS-3.
- Whether the prom textfile path should be configurable per-run rather
  than per-environment.

## 6. Future work

- Auto-generated registries (`EVENTS`, `METRICS`) into
  [`docs/api/log-events.md`](../docs/api/log-events.md) and
  [`docs/api/metrics.md`](../docs/api/metrics.md).
- A `geno-lewm logs explain <event-name>` CLI for fast diagnostic
  reference.
- An audit-log mode that records every observability emission to a
  signed append-only log, intended for reproducibility workflows where the
  log itself is part of the artifact.

## 7. Changelog

- 2026-06-02 — Accepted after the v1 observability surface landed in
  code and tests; OpenTelemetry export remains future optional work.
- 2026-05-20 — Initial draft.
