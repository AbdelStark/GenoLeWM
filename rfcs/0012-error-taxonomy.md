# RFC-0012: Error taxonomy and exception hierarchy

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-06-11
- **Depends on:** RFC-0001
- **Supersedes:** —
- **Implementation status:** Implemented for the v1 typed exception
  surface: stable error-code registry, typed subclasses, structured
  details/remediation payloads, CLI exit-code mapping, and docs-build
  generation of `api/error-codes.md` exist with unit, API, lint, and
  docs-generation coverage. Error-code lookup CLI and programmatic
  remediation helpers remain future work.

---

## 1. Summary

This RFC fixes the cross-cutting exception hierarchy used by every
GenoLeWM subsystem, the error-code registry, the structured payload
format, the CLI exit-code mapping, and the discipline around raising vs
returning. The goal is a single mechanical contract for failure
reporting so that callers (library, CLI, downstream tools) can route on
type and code without reading docstrings.

## 2. Motivation

Without a shared error taxonomy, every subsystem invents its own
exceptions, the CLI's error messages become inconsistent, and downstream
tooling cannot distinguish "input malformed" from "model corrupt" from
"network refused." The taxonomy also gives the receipt format
([RFC-0011](0011-artifact-provenance-receipts.md)) a stable vocabulary
for documenting failure modes, and the observability layer
([RFC-0013](0013-observability.md)) a stable `code` field for filtering.

## 3. Specification

### 3.1 Hierarchy

The full hierarchy is in [`docs/spec/04-error-model.md`](../docs/spec/04-error-model.md).
The root is `geno_lewm.errors.GenoLeWMError`. Top-level branches:

- `ConfigError`
- `InputError`
- `ResourceError`
- `TrainingError`
- `EvalError`
- `DeployError`
- `ProvenanceError`
- `InternalError`

Each branch has named leaves enumerated in the spec document. Adding a
leaf is a MINOR change; renaming or removing one is a MAJOR change.

### 3.2 Payload

```python
class GenoLeWMError(Exception):
    code: str
    message: str
    details: dict[str, object]
    remediation: str | None

    def __init__(self,
                 message: str,
                 *,
                 code: str,
                 details: dict[str, object] | None = None,
                 remediation: str | None = None) -> None: ...

    def to_dict(self) -> dict[str, object]: ...
    def to_json(self) -> str: ...
```

`code` strings are dotted-uppercase, category-prefixed, and registered in
`geno_lewm/errors.py::ERROR_CODES`. Examples in
[`docs/spec/04-error-model.md`](../docs/spec/04-error-model.md).

### 3.3 Registry

`ERROR_CODES` is a tuple of `(code, exception_class, summary)` tuples.
The linter rule `ast/check_error_codes.py` walks every `raise` statement
in `geno_lewm/` and confirms:

- The raised expression is a `GenoLeWMError` subclass.
- The `code` keyword argument is a literal string registered in `ERROR_CODES`.
- The exception class matches the second element of the registered tuple.

Violations fail CI.

### 3.4 CLI exit codes

| Code | Category |
|------|----------|
| 0 | success |
| 2 | `InputError` |
| 3 | `ConfigError` |
| 4 | `ResourceError` |
| 5 | `TrainingError` |
| 6 | `EvalError` |
| 7 | `DeployError` |
| 8 | `ProvenanceError` |
| 9 | `InternalError` |
| 1, 130 | reserved |

CLI dispatcher: `geno_lewm/cli/_dispatch.py::main` catches `GenoLeWMError`
at exactly one place, prints a one-line summary plus the structured
payload, and exits with the mapped code.

### 3.5 Raise discipline

- **Validation failure** of caller-supplied input: raise typed
  `InputError` subclass with a `code` that names the rule.
- **Expected absence** (cache miss, optional field): return `None` or a
  sentinel; document in the API. Never raise.
- **Resource exhaustion**: raise typed `ResourceError` subclass with a
  `details` payload naming the resource and the budget exceeded.
- **Internal invariant violation**: raise `InvariantViolation`; log at
  ERROR; never silent.

### 3.6 Translation to logs and receipts

Every raised exception inside a logged span emits an
`{event: "error", code, message, details, remediation, ts}` log record.
The receipt format does not include errors; partial-failure scoring (one
variant of many) records per-variant errors in an adjacent
`.errors.jsonl` file.

### 3.7 Module shape

A single file `geno_lewm/errors.py` defines every class and the registry.
No submodules. This is intentional: callers can do `from geno_lewm.errors
import *` and trust the full surface is visible.

## 4. Rationale and alternatives

### 4.1 Why a single file?

Splitting the hierarchy into per-subsystem modules would force callers
to know which subsystem raised what before importing the right module —
exactly the discovery problem the taxonomy is meant to solve. A single
file is the simplest contract.

### 4.2 Why stringly-named codes rather than enums?

Codes are part of the stable external contract and need to be greppable
across logs, receipts, and CHANGELOGs. Enums add a Python layer that
serializes back to the same string anyway; the simpler approach is
just to use the string and lint that it is registered.

### 4.3 Why not adopt a third-party error library?

`pydantic.ValidationError`, `attrs.exceptions`, `tenacity` retry errors
— each addresses a slice but none cover the cross-cutting hierarchy we
need. A bespoke 200-line `errors.py` is cheaper than aligning with one
of those libraries.

### 4.4 Why typed exceptions over `Result[T, E]`?

Python's exception ergonomics are the language's preferred style, and the
public surface is more readable with `raise InvalidEditError(...)` than
with `return Err(InvalidEdit(...))`. We considered a `Result`-style API
([OQ-ERR-2](../docs/spec/04-error-model.md#open-questions)) for future
versions; v1 ships typed exceptions.

## 5. Unresolved questions

- Whether to expose typed `Result[T, E]` returns at API boundaries in
  a later major version. Tracked as OQ-ERR-2.
- Whether to integrate `sentry-sdk` or similar for upload of crash
  diagnostics. Only acceptable if the redaction filter is enforced; see
  OQ-ERR-3.
- Whether to provide `__cause__`-chain helpers for downstream
  diagnostics. Tracked as OQ-ERR-1.

## 6. Future work

- A `geno-lewm errors lookup CODE` CLI for fast diagnostic lookup.
- Programmatic remediation: each leaf may expose a `try_fix()` method
  that attempts a documented recovery (e.g., `CacheCorruptError.try_fix()`
  calls `cache-windows --repair`).

## 7. Changelog

- 2026-06-11 — Updated implementation status for generated error-code
  docs and kept lookup/remediation helpers as future work.
- 2026-06-02 — Updated implementation status for the v1 typed error
  hierarchy and CLI exit-code mapping.
- 2026-05-20 — Initial draft.
