# SPDX-License-Identifier: Apache-2.0
"""GenoLeWM exception hierarchy and error-code registry.

This module is the single source of truth for the runtime error model.
Every other subsystem raises typed exceptions from this hierarchy.

Discipline:

- Caller-supplied invalid data       -> ``InputError`` family
- Misconfiguration / missing fields  -> ``ConfigError`` family
- Capacity, IO, network              -> ``ResourceError`` family
- Training-loop instability          -> ``TrainingError`` family
- Eval harness failures              -> ``EvalError`` family
- Export / runtime backend failures  -> ``DeployError`` family
- Receipt/hash consistency failure   -> ``ProvenanceError`` family
- Bugs we caught                     -> ``InternalError`` family
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "ERROR_CODES",
    "BackendUnsupportedError",
    "CacheCorruptError",
    "CacheKeyAlreadyIndexedError",
    "CollapseDetectedError",
    "ConfigError",
    "DataLoaderError",
    "DeployError",
    "DiskFullError",
    "ErrorCodeEntry",
    "EvalDatasetError",
    "EvalError",
    "EvalRegressionError",
    "ExportFormatError",
    "GenoLeWMError",
    "InputCommitmentMismatchError",
    "InputError",
    "InternalError",
    "InvalidEditError",
    "InvariantViolation",
    "ManifestHashMismatchError",
    "MissingConfigError",
    "ModelNotFoundError",
    "NaNLossError",
    "NetworkCallProhibitedError",
    "OutOfMemoryError",
    "OutOfWindowError",
    "OutputCommitmentMismatchError",
    "OverlappingEditsError",
    "ProvenanceError",
    "ProvenanceKindUnsupportedError",
    "QuantizationError",
    "ReceiptSchemaError",
    "ResourceError",
    "RuntimeSetupError",
    "SchemaCompatError",
    "SnapshotLineageError",
    "TrainingError",
    "UnknownTopLevelKeyError",
    "UnreachableError",
    "UnsupportedEditError",
    "VcfParseError",
    "WindowMismatchError",
    "exit_code_for",
]


# ---------------------------------------------------------------------------
# Root


class GenoLeWMError(Exception):
    """Root of the GenoLeWM exception hierarchy.

    Every typed exception in this package inherits from ``GenoLeWMError``.
    Subclasses must set a ``code`` class attribute that matches an entry
    in :data:`ERROR_CODES`.
    """

    #: Stable dotted-uppercase error code (e.g. ``"INPUT.INVALID_EDIT"``).
    #: Subclasses MUST override.
    code: str = "INTERNAL.GENO_LEWM_ERROR"

    def __init__(
        self,
        message: str = "",
        *,
        details: Mapping[str, Any] | None = None,
        remediation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details) if details else {}
        self.remediation = remediation

    def to_dict(self) -> dict[str, Any]:
        """Return the structured payload as a plain dict.

        The output is JSON-serializable provided ``details`` values are
        JSON-native. Keys mirror the structured-log ``error`` event so
        the CLI dispatcher and observability layer can pass it through
        without translation.
        """
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "remediation": self.remediation,
        }

    def to_json(self) -> str:
        """Return the structured payload as a JSON string.

        Adds a UTC ``ts`` field so log sinks receive a self-contained
        record.
        """
        payload = self.to_dict()
        payload["ts"] = datetime.now(tz=timezone.utc).isoformat()
        return json.dumps(payload, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Config


class ConfigError(GenoLeWMError):
    """Configuration is malformed or incompatible."""

    code = "CONFIG.GENERIC"


class SchemaCompatError(ConfigError):
    """An on-disk artifact's schema MAJOR version is incompatible."""

    code = "CONFIG.SCHEMA_INCOMPAT"


class MissingConfigError(ConfigError):
    """A required configuration field is absent."""

    code = "CONFIG.MISSING_FIELD"


class UnknownTopLevelKeyError(ConfigError):
    """A configuration payload contained a top-level key not in the schema."""

    code = "CONFIG.UNKNOWN_TOP_LEVEL_KEY"


# ---------------------------------------------------------------------------
# Input


class InputError(GenoLeWMError):
    """Caller-supplied input violates a documented invariant."""

    code = "INPUT.GENERIC"


class InvalidEditError(InputError):
    """An ``EditSpec`` fails one of its constructor invariants."""

    code = "INPUT.INVALID_EDIT"


class UnsupportedEditError(InputError):
    """An edit's type or length is outside the v1 scope (edit contract)."""

    code = "INPUT.UNSUPPORTED_EDIT"


class WindowMismatchError(InputError):
    """Window reference bases do not match ``EditSpec.ref`` at ``rel_pos``."""

    code = "INPUT.WINDOW_MISMATCH"


class OverlappingEditsError(InputError):
    """Two or more edits in a haplotype overlap in genomic coordinates."""

    code = "INPUT.OVERLAPPING_EDITS"


class OutOfWindowError(InputError):
    """An edit's ``rel_pos`` falls outside the encoder window."""

    code = "INPUT.OUT_OF_WINDOW"


class VcfParseError(InputError):
    """A VCF or FASTA input is malformed."""

    code = "INPUT.VCF_PARSE"


# ---------------------------------------------------------------------------
# Resource


class ResourceError(GenoLeWMError):
    """Capacity, IO, or network failure."""

    code = "RESOURCE.GENERIC"


class CacheCorruptError(ResourceError):
    """A cache shard failed an integrity check."""

    code = "RESOURCE.CACHE_CORRUPT"


class CacheKeyAlreadyIndexedError(CacheCorruptError):
    """A serialized cache publication lost the logical-key reservation race."""

    code = "RESOURCE.CACHE_KEY_ALREADY_INDEXED"


class DiskFullError(ResourceError):
    """Storage was exhausted during a write."""

    code = "RESOURCE.DISK_FULL"


class OutOfMemoryError(ResourceError):
    """Re-raise of CUDA / host OOM with attached GenoLeWM context.

    Intentionally shadows the builtin ``OutOfMemoryError`` only inside
    this module's namespace; downstream code that needs the builtin can
    use ``builtins.OutOfMemoryError``.
    """

    code = "RESOURCE.OOM"


class ModelNotFoundError(ResourceError):
    """A model checkpoint is missing locally and cannot be downloaded."""

    code = "RESOURCE.MODEL_NOT_FOUND"


class RuntimeSetupError(ResourceError):
    """First-run network setup step (download / verification) failed."""

    code = "RESOURCE.RUNTIME_SETUP"


class NetworkCallProhibitedError(ResourceError):
    """A post-setup network call was attempted under fail-closed policy.

    See runtime contract ("on-device fail-closed").
    """

    code = "RESOURCE.NETWORK_PROHIBITED"


# ---------------------------------------------------------------------------
# Training


class TrainingError(GenoLeWMError):
    """Training-loop-specific failure."""

    code = "TRAINING.GENERIC"


class CollapseDetectedError(TrainingError):
    """A representation-collapse alert tripped (training contract)."""

    code = "TRAINING.COLLAPSE_DETECTED"


class NaNLossError(TrainingError):
    """Loss became NaN or Inf."""

    code = "TRAINING.NAN_LOSS"


class DataLoaderError(TrainingError):
    """Data pipeline raised an exception the trainer cannot recover from."""

    code = "TRAINING.DATALOADER"


# ---------------------------------------------------------------------------
# Eval


class EvalError(GenoLeWMError):
    """Evaluation harness failure."""

    code = "EVAL.GENERIC"


class EvalDatasetError(EvalError):
    """A benchmark file or dataset could not be loaded."""

    code = "EVAL.DATASET"


class EvalRegressionError(EvalError):
    """A smoke-eval gate threshold was breached."""

    code = "EVAL.REGRESSION"


# ---------------------------------------------------------------------------
# Deploy


class DeployError(GenoLeWMError):
    """Export or runtime backend failure."""

    code = "DEPLOY.GENERIC"


class ExportFormatError(DeployError):
    """Conversion to ONNX, Core ML, or GGUF failed."""

    code = "DEPLOY.EXPORT_FORMAT"


class QuantizationError(DeployError):
    """int8 or int4 calibration failed."""

    code = "DEPLOY.QUANTIZATION_FAILED"


class BackendUnsupportedError(DeployError):
    """Requested runtime backend is not available on the host."""

    code = "DEPLOY.BACKEND_UNSUPPORTED"


# ---------------------------------------------------------------------------
# Provenance


class ProvenanceError(GenoLeWMError):
    """Receipt or artifact-provenance failure (artifact-provenance contract)."""

    code = "PROVENANCE.GENERIC"


class ManifestHashMismatchError(ProvenanceError):
    """Recomputed manifest hash does not match the stated ``model_id``."""

    code = "PROVENANCE.MANIFEST_HASH_MISMATCH"


class InputCommitmentMismatchError(ProvenanceError):
    """Recomputed input commitment does not match the receipt."""

    code = "PROVENANCE.INPUT_COMMITMENT_MISMATCH"


class OutputCommitmentMismatchError(ProvenanceError):
    """Recomputed output bytes do not match the receipt."""

    code = "PROVENANCE.OUTPUT_COMMITMENT_MISMATCH"


class ProvenanceKindUnsupportedError(ProvenanceError):
    """Verifier does not understand the receipt provenance kind."""

    code = "PROVENANCE.KIND_UNSUPPORTED"


class ReceiptSchemaError(ProvenanceError):
    """Receipt JSON failed schema validation."""

    code = "PROVENANCE.RECEIPT_SCHEMA"


class SnapshotLineageError(ProvenanceError, ValueError):
    """Snapshot-lineage evidence violates its closed semantic contract."""

    code = "PROVENANCE.SNAPSHOT_LINEAGE"


# ---------------------------------------------------------------------------
# Internal


class InternalError(GenoLeWMError):
    """A bug we caught; should never surface to end users."""

    code = "INTERNAL.GENERIC"


class InvariantViolation(InternalError):
    """A runtime invariant marked ``INV-*`` was breached."""

    code = "INTERNAL.INVARIANT_VIOLATION"


class UnreachableError(InternalError):
    """Control flow reached a branch marked unreachable."""

    code = "INTERNAL.UNREACHABLE"


# ---------------------------------------------------------------------------
# Registry


class ErrorCodeEntry:
    """A single immutable row in the error-code registry."""

    __slots__ = ("code", "exception_class", "summary")

    def __init__(self, code: str, exception_class: type[GenoLeWMError], summary: str) -> None:
        self.code = code
        self.exception_class = exception_class
        self.summary = summary

    def __iter__(self) -> Iterator[Any]:
        # Allow tuple-style unpacking ``code, cls, summary = entry``.
        yield self.code
        yield self.exception_class
        yield self.summary

    def __repr__(self) -> str:
        return (
            f"ErrorCodeEntry(code={self.code!r}, "
            f"exception_class={self.exception_class.__name__}, "
            f"summary={self.summary!r})"
        )


#: Source of truth for stable error codes. Codes are part of the public
#: surface; renaming one is a MAJOR change. New codes are MINOR additions
#: and require a corresponding leaf class.
ERROR_CODES: tuple[ErrorCodeEntry, ...] = (
    # Config
    ErrorCodeEntry("CONFIG.GENERIC", ConfigError, "Configuration error"),
    ErrorCodeEntry("CONFIG.SCHEMA_INCOMPAT", SchemaCompatError, "On-disk schema MAJOR mismatch"),
    ErrorCodeEntry("CONFIG.MISSING_FIELD", MissingConfigError, "Required config field absent"),
    ErrorCodeEntry(
        "CONFIG.UNKNOWN_TOP_LEVEL_KEY",
        UnknownTopLevelKeyError,
        "Top-level config key is not in the schema",
    ),
    # Input
    ErrorCodeEntry("INPUT.GENERIC", InputError, "Caller-supplied input invalid"),
    ErrorCodeEntry("INPUT.INVALID_EDIT", InvalidEditError, "EditSpec invariants violated"),
    ErrorCodeEntry(
        "INPUT.UNSUPPORTED_EDIT", UnsupportedEditError, "Edit type or length out of scope"
    ),
    ErrorCodeEntry(
        "INPUT.WINDOW_MISMATCH", WindowMismatchError, "Window ref bases differ from EditSpec.ref"
    ),
    ErrorCodeEntry("INPUT.OVERLAPPING_EDITS", OverlappingEditsError, "Haplotype edits overlap"),
    ErrorCodeEntry("INPUT.OUT_OF_WINDOW", OutOfWindowError, "rel_pos outside window"),
    ErrorCodeEntry("INPUT.VCF_PARSE", VcfParseError, "Malformed VCF or FASTA"),
    # Resource
    ErrorCodeEntry("RESOURCE.GENERIC", ResourceError, "Capacity, IO, or network failure"),
    ErrorCodeEntry(
        "RESOURCE.CACHE_CORRUPT", CacheCorruptError, "Cache shard failed integrity check"
    ),
    ErrorCodeEntry(
        "RESOURCE.CACHE_KEY_ALREADY_INDEXED",
        CacheKeyAlreadyIndexedError,
        "Cache logical key was published by another writer",
    ),
    ErrorCodeEntry("RESOURCE.DISK_FULL", DiskFullError, "Storage exhausted during write"),
    ErrorCodeEntry("RESOURCE.OOM", OutOfMemoryError, "CUDA or host OOM with context"),
    ErrorCodeEntry(
        "RESOURCE.MODEL_NOT_FOUND", ModelNotFoundError, "Checkpoint missing or not downloadable"
    ),
    ErrorCodeEntry("RESOURCE.RUNTIME_SETUP", RuntimeSetupError, "First-run network setup failed"),
    ErrorCodeEntry(
        "RESOURCE.NETWORK_PROHIBITED",
        NetworkCallProhibitedError,
        "Runtime fail-closed: post-setup network call attempted",
    ),
    # Training
    ErrorCodeEntry("TRAINING.GENERIC", TrainingError, "Training-loop failure"),
    ErrorCodeEntry("TRAINING.COLLAPSE_DETECTED", CollapseDetectedError, "Collapse alert tripped"),
    ErrorCodeEntry("TRAINING.NAN_LOSS", NaNLossError, "Loss became NaN or Inf"),
    ErrorCodeEntry("TRAINING.DATALOADER", DataLoaderError, "Data pipeline failure"),
    # Eval
    ErrorCodeEntry("EVAL.GENERIC", EvalError, "Evaluation harness failure"),
    ErrorCodeEntry("EVAL.DATASET", EvalDatasetError, "Benchmark dataset load failed"),
    ErrorCodeEntry("EVAL.REGRESSION", EvalRegressionError, "Smoke-eval gate failed"),
    # Deploy
    ErrorCodeEntry("DEPLOY.GENERIC", DeployError, "Export or runtime backend failure"),
    ErrorCodeEntry(
        "DEPLOY.EXPORT_FORMAT", ExportFormatError, "ONNX/Core ML/GGUF conversion failed"
    ),
    ErrorCodeEntry("DEPLOY.QUANTIZATION_FAILED", QuantizationError, "int8/int4 calibration failed"),
    ErrorCodeEntry(
        "DEPLOY.BACKEND_UNSUPPORTED", BackendUnsupportedError, "Backend unavailable on host"
    ),
    # Provenance
    ErrorCodeEntry("PROVENANCE.GENERIC", ProvenanceError, "Receipt/provenance failure"),
    ErrorCodeEntry(
        "PROVENANCE.MANIFEST_HASH_MISMATCH",
        ManifestHashMismatchError,
        "Manifest content != stated model_id",
    ),
    ErrorCodeEntry(
        "PROVENANCE.INPUT_COMMITMENT_MISMATCH",
        InputCommitmentMismatchError,
        "Recomputed input commitment != receipt",
    ),
    ErrorCodeEntry(
        "PROVENANCE.OUTPUT_COMMITMENT_MISMATCH",
        OutputCommitmentMismatchError,
        "Bit-mismatch on output re-run",
    ),
    ErrorCodeEntry(
        "PROVENANCE.KIND_UNSUPPORTED",
        ProvenanceKindUnsupportedError,
        "Verifier does not understand receipt provenance kind",
    ),
    ErrorCodeEntry("PROVENANCE.RECEIPT_SCHEMA", ReceiptSchemaError, "Receipt JSON invalid"),
    ErrorCodeEntry(
        "PROVENANCE.SNAPSHOT_LINEAGE",
        SnapshotLineageError,
        "Snapshot-lineage evidence violates its closed semantic contract",
    ),
    # Internal
    ErrorCodeEntry("INTERNAL.GENERIC", InternalError, "Internal error"),
    ErrorCodeEntry(
        "INTERNAL.INVARIANT_VIOLATION", InvariantViolation, "An INV-* invariant was breached"
    ),
    ErrorCodeEntry(
        "INTERNAL.UNREACHABLE", UnreachableError, "Control reached an unreachable branch"
    ),
)


# Registry self-consistency: build once at import so that linter and tests
# can rely on it without rescanning the module.
_CODE_TO_ENTRY: dict[str, ErrorCodeEntry] = {entry.code: entry for entry in ERROR_CODES}
if len(_CODE_TO_ENTRY) != len(ERROR_CODES):  # pragma: no cover - tested explicitly
    raise InvariantViolation(
        "Duplicate code in ERROR_CODES",
        details={"len_registry": len(ERROR_CODES), "len_unique_codes": len(_CODE_TO_ENTRY)},
    )


# ---------------------------------------------------------------------------
# Exit codes consumed by the CLI dispatcher. Tooling that wraps the CLI
# relies on these values; changing any value is a breaking public contract.
_EXIT_CODE_BY_FAMILY: tuple[tuple[type[GenoLeWMError], int], ...] = (
    # Order matters: most specific first. ``InternalError`` is a
    # ``GenoLeWMError`` subclass, so it must be checked before the generic
    # fallback at the bottom of ``exit_code_for``.
    (InputError, 2),
    (ConfigError, 3),
    (ResourceError, 4),
    (TrainingError, 5),
    (EvalError, 6),
    (DeployError, 7),
    (ProvenanceError, 8),
    (InternalError, 9),
)


def exit_code_for(exc: BaseException) -> int:
    """Return the CLI exit code for ``exc``.

    Used by ``geno_lewm.cli._dispatch``. Non-GenoLeWM exceptions map to
    exit code 1 ("uncategorized failure"); ``KeyboardInterrupt`` maps to
    130.
    """
    if isinstance(exc, KeyboardInterrupt):
        return 130
    if not isinstance(exc, GenoLeWMError):
        return 1
    for family, code in _EXIT_CODE_BY_FAMILY:
        if isinstance(exc, family):
            return code
    return 1
