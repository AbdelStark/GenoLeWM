# 04 — Error model

- Status: Authoritative for v0.1
- Companion RFC: [RFC-0012](../rfcs/0012-error-taxonomy.md)

The error model is the contract on how subsystems fail. The whole of
GenoLeWM raises typed exceptions from a single hierarchy rooted at
`geno_lewm.errors.GenoLeWMError`. No subsystem swallows exceptions, no
public function returns `None` on failure, and no subsystem stringly-
types its errors.

## Hierarchy

```
Exception
└── GenoLeWMError                    # root; all GenoLeWM exceptions inherit
    ├── ConfigError                  # bad configuration or missing field
    │   ├── SchemaCompatError        # on-disk schema MAJOR mismatch
    │   └── MissingConfigError       # required config field absent
    │
    ├── InputError                   # caller-supplied input invalid
    │   ├── InvalidEditError         # EditSpec invariants violated
    │   ├── UnsupportedEditError     # edit type / length not in v1 scope
    │   ├── WindowMismatchError      # window ref bases ≠ EditSpec.ref
    │   ├── OverlappingEditsError    # haplotype edits overlap
    │   ├── OutOfWindowError         # rel_pos outside window
    │   └── VcfParseError            # malformed VCF / FASTA
    │
    ├── ResourceError                # capacity, IO, network
    │   ├── CacheCorruptError        # cache shard fails integrity check
    │   ├── DiskFullError            # storage exhausted during write
    │   ├── OutOfMemoryError         # explicit reraise of CUDA OOM with context
    │   ├── ModelNotFoundError       # checkpoint missing / not downloadable
    │   ├── RuntimeSetupError        # first-run network step failed
    │   └── NetworkCallProhibitedError  # runtime fail-closed (RFC-0010 §3.7)
    │
    ├── TrainingError                # training-loop-specific failures
    │   ├── CollapseDetectedError    # collapse alert criteria tripped
    │   ├── NaNLossError             # loss became NaN / Inf
    │   └── DataLoaderError          # data pipeline failure
    │
    ├── EvalError                    # evaluation harness failures
    │   ├── EvalDatasetError         # benchmark data load failed
    │   └── EvalRegressionError      # smoke-eval gate failed
    │
    ├── DeployError                  # export / runtime failures
    │   ├── ExportFormatError        # ONNX / Core ML / GGUF conversion failed
    │   ├── QuantizationError        # int8/int4 calibration failed
    │   └── BackendUnsupportedError  # backend not available on host
    │
    ├── AttestationError             # verifiable-inference failures
    │   ├── ManifestHashMismatchError    # manifest content != stated model_id
    │   ├── InputCommitmentMismatchError # recomputed commitment != receipt
    │   ├── OutputCommitmentMismatchError# bit-mismatch on re-run
    │   ├── AttestationKindUnsupportedError  # verifier doesn't know kind
    │   └── ReceiptSchemaError       # receipt JSON invalid
    │
    └── InternalError                # bugs we caught; should never surface
        ├── InvariantViolation       # an `INV-*` invariant was breached
        └── UnreachableError         # control flow reached "unreachable" branch
```

Every leaf class is concrete and named. Adding a new leaf is a MINOR
change; removing or renaming one is a MAJOR change.

## Error payloads

Every `GenoLeWMError` exposes a structured payload:

```python
class GenoLeWMError(Exception):
    code: str                       # stable error code, e.g., "INPUT.INVALID_EDIT"
    message: str                    # human-readable
    details: dict[str, object]      # JSON-serializable structured fields
    remediation: str | None         # actionable hint when one exists

    def to_dict(self) -> dict[str, object]: ...
    def to_json(self) -> str: ...
```

Error codes are dotted-uppercase, prefixed by the top-level category, and
documented in the registry. Examples:

- `INPUT.INVALID_EDIT`
- `INPUT.UNSUPPORTED_EDIT`
- `INPUT.WINDOW_MISMATCH`
- `RESOURCE.CACHE_CORRUPT`
- `RESOURCE.NETWORK_PROHIBITED`
- `TRAINING.COLLAPSE_DETECTED`
- `DEPLOY.QUANTIZATION_FAILED`
- `ATTESTATION.MANIFEST_HASH_MISMATCH`

Codes are part of the public surface. Renaming a code is a MAJOR change.

## Error code registry

The registry lives at `geno_lewm/errors.py::ERROR_CODES` and is the source
of truth. A linter rule enforces that every code raised at runtime is
registered. The registry is regenerated into
[`docs/api/error-codes.md`](../api/error-codes.md) on each release.

## Raise vs return discipline

| Situation | Mechanism |
|-----------|-----------|
| Caller-supplied data fails a documented invariant | raise typed `InputError` subclass |
| Expected absence (cache miss, optional field) | return `None` or sentinel; document in API |
| Resource exhaustion (memory, disk, network) | raise typed `ResourceError` subclass |
| Internal invariant violation | raise `InvariantViolation`; log at ERROR; never silent |
| Verifier discovers a mismatch | raise typed `AttestationError` subclass |
| Training instability (NaN, collapse) | raise typed `TrainingError`; trainer can opt to catch |
| CLI top-level | catch `GenoLeWMError`, exit non-zero, print `code` + `message` |

The CLI catches at exactly one place: `geno_lewm/cli/_dispatch.py`. Library
callers see the full traceback unless they explicitly catch.

## Exit codes

CLI exit codes:

| Code | Meaning |
|------|---------|
| 0 | success |
| 1 | uncategorized failure (a bug in CLI dispatch; should be rare) |
| 2 | `InputError` family |
| 3 | `ConfigError` family |
| 4 | `ResourceError` family |
| 5 | `TrainingError` family |
| 6 | `EvalError` family |
| 7 | `DeployError` family |
| 8 | `AttestationError` family |
| 9 | `InternalError` family (please file a bug) |
| 130 | SIGINT |

Tooling that wraps the CLI relies on these codes; bumping them is a
MAJOR change.

## Failure modes by subsystem

The table is exhaustive for v0.1. Adding a new mode requires an RFC
update and a `code` registration.

| Subsystem | Failure mode | Detection | Raised as | Recovery |
|-----------|--------------|-----------|-----------|----------|
| EditSpec | malformed bases | constructor invariant | `InvalidEditError` | caller fixes input |
| EditSpec | length > V1_MAX_LEN | length check | `UnsupportedEditError` | defer to v2 / SV adapter |
| Apply | ref-bases mismatch | string compare | `WindowMismatchError` | re-fetch window or fix EditSpec |
| Haplotype | overlapping edits | interval check | `OverlappingEditsError` | caller decomposes |
| Encoder | model not on Hub | HTTP 404 | `ModelNotFoundError` | check `--model-id`, network |
| Encoder | OOM on long window | torch OOM | `OutOfMemoryError` | use smaller window |
| Cache | shard truncated | length / CRC | `CacheCorruptError` | `geno-lewm-cache-windows --repair` |
| Predictor | NaN/Inf in loss | per-step check | `NaNLossError` | restart from last checkpoint |
| Trainer | collapse criteria tripped | monitoring hooks | `CollapseDetectedError` | trainer stops; alert in wandb |
| Trainer | corrupt batch | dataloader exception | `DataLoaderError` | log + skip; abort if rate > 1% |
| Eval | benchmark file absent | filesystem | `EvalDatasetError` | `geno-lewm-prepare-*` |
| Eval | smoke-eval delta > threshold | comparison | `EvalRegressionError` | block PR |
| Export | unsupported op for target | converter | `ExportFormatError` | use alternative target |
| Export | int8 calibration failed | activation stats | `QuantizationError` | re-run with more calibration data |
| Runtime | wrong backend at load | capability probe | `BackendUnsupportedError` | reload with `backend="auto"` |
| Runtime | post-setup network call attempted | URL hook | `NetworkCallProhibitedError` | bug — file a report |
| Receipt | malformed JSON | schema check | `ReceiptSchemaError` | verifier rejects |
| Verifier | manifest hash mismatch | hash compare | `ManifestHashMismatchError` | verifier rejects |
| Verifier | input commitment mismatch | hash compare | `InputCommitmentMismatchError` | verifier rejects |
| Verifier | output commitment mismatch | hash compare | `OutputCommitmentMismatchError` | verifier rejects |
| Internal | invariant breach | runtime assert | `InvariantViolation` | bug |

## Logging error events

Every raised `GenoLeWMError` emits a structured log event:

```
{"event": "error", "code": <code>, "message": <message>, "details": {...},
 "remediation": <remediation>, "ts": <iso-8601>}
```

See [`05-observability.md`](05-observability.md) for the log format. Error
events are tagged with `severity=error`. `InternalError` events
additionally include a stack trace and the file/line of the raise.

## Translation to receipts

The receipt format (RFC-0011 §3.3) does not include error details; a
failed inference does not produce a receipt. Partial-failure semantics
(e.g., per-variant errors during VCF scoring) are recorded in an adjacent
`.errors.jsonl` file with one error record per failed variant.

## Invariants

| ID | Invariant | Enforced by |
|----|-----------|-------------|
| INV-ERR-1 | Every raised exception in `geno_lewm/` is a `GenoLeWMError` subclass | linter rule |
| INV-ERR-2 | Every `raise` statement uses a code registered in `ERROR_CODES` | linter rule |
| INV-ERR-3 | No public function returns `None` to signal failure | type checker |
| INV-ERR-4 | Test suite asserts both the exception class and the `code` for every documented failure mode | `tests/unit/test_errors.py` |
| INV-ERR-5 | Error messages do not embed user data; structured `details` carry it | redaction filter in `observability.py` |

## Open questions

| ID | Question | Owner | Target |
|----|----------|-------|--------|
| OQ-ERR-1 | Whether to add `cause`-chain inspection helpers for `__cause__` reading | core | v0.2 |
| OQ-ERR-2 | Whether to expose typed `Result[T, E]` returns at API boundaries (Rust-style) | core | v0.3 |
| OQ-ERR-3 | Whether to integrate with `sentry-sdk` or similar — only acceptable if redaction-by-default is non-negotiable | core | post-v1 |
