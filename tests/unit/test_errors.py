"""Tests for ``geno_lewm.errors``.

These tests pin the contract documented in ``docs/spec/04-error-model.md``:
every leaf class exists, is registered, carries a registered ``code`` prefix,
and round-trips through ``to_dict`` / ``to_json``.
"""

from __future__ import annotations

import json

import pytest

from geno_lewm import errors as err

# Every leaf class declared in the spec hierarchy. The order matches
# docs/spec/04-error-model.md so the test diff stays grep-friendly.
LEAF_CLASSES: tuple[tuple[type[err.GenoLeWMError], str], ...] = (
    # Config
    (err.SchemaCompatError, "CONFIG.SCHEMA_INCOMPAT"),
    (err.MissingConfigError, "CONFIG.MISSING_FIELD"),
    # Input
    (err.InvalidEditError, "INPUT.INVALID_EDIT"),
    (err.UnsupportedEditError, "INPUT.UNSUPPORTED_EDIT"),
    (err.WindowMismatchError, "INPUT.WINDOW_MISMATCH"),
    (err.OverlappingEditsError, "INPUT.OVERLAPPING_EDITS"),
    (err.OutOfWindowError, "INPUT.OUT_OF_WINDOW"),
    (err.VcfParseError, "INPUT.VCF_PARSE"),
    # Resource
    (err.CacheCorruptError, "RESOURCE.CACHE_CORRUPT"),
    (err.DiskFullError, "RESOURCE.DISK_FULL"),
    (err.OutOfMemoryError, "RESOURCE.OOM"),
    (err.ModelNotFoundError, "RESOURCE.MODEL_NOT_FOUND"),
    (err.RuntimeSetupError, "RESOURCE.RUNTIME_SETUP"),
    (err.NetworkCallProhibitedError, "RESOURCE.NETWORK_PROHIBITED"),
    # Training
    (err.CollapseDetectedError, "TRAINING.COLLAPSE_DETECTED"),
    (err.NaNLossError, "TRAINING.NAN_LOSS"),
    (err.DataLoaderError, "TRAINING.DATALOADER"),
    # Eval
    (err.EvalDatasetError, "EVAL.DATASET"),
    (err.EvalRegressionError, "EVAL.REGRESSION"),
    # Deploy
    (err.ExportFormatError, "DEPLOY.EXPORT_FORMAT"),
    (err.QuantizationError, "DEPLOY.QUANTIZATION_FAILED"),
    (err.BackendUnsupportedError, "DEPLOY.BACKEND_UNSUPPORTED"),
    # Attestation
    (err.ManifestHashMismatchError, "ATTESTATION.MANIFEST_HASH_MISMATCH"),
    (err.InputCommitmentMismatchError, "ATTESTATION.INPUT_COMMITMENT_MISMATCH"),
    (err.OutputCommitmentMismatchError, "ATTESTATION.OUTPUT_COMMITMENT_MISMATCH"),
    (err.AttestationKindUnsupportedError, "ATTESTATION.KIND_UNSUPPORTED"),
    (err.ReceiptSchemaError, "ATTESTATION.RECEIPT_SCHEMA"),
    # Internal
    (err.InvariantViolation, "INTERNAL.INVARIANT_VIOLATION"),
    (err.UnreachableError, "INTERNAL.UNREACHABLE"),
)


@pytest.mark.parametrize(("cls", "code"), LEAF_CLASSES)
def test_leaf_class_has_registered_code(cls: type[err.GenoLeWMError], code: str) -> None:
    assert cls.code == code
    registered = {entry.code for entry in err.ERROR_CODES}
    assert code in registered, f"{cls.__name__} code {code!r} missing from ERROR_CODES"


@pytest.mark.parametrize(("cls", "code"), LEAF_CLASSES)
def test_leaf_inherits_from_root(cls: type[err.GenoLeWMError], code: str) -> None:
    del code  # pytest parametrize ties the tuples; only ``cls`` is asserted here.
    assert issubclass(cls, err.GenoLeWMError)


@pytest.mark.parametrize(("cls", "code"), LEAF_CLASSES)
def test_to_dict_round_trips(cls: type[err.GenoLeWMError], code: str) -> None:
    exc = cls(
        "human message",
        details={"k": 1, "nested": [1, 2, 3]},
        remediation="do the thing",
    )
    payload = exc.to_dict()
    assert payload == {
        "code": code,
        "message": "human message",
        "details": {"k": 1, "nested": [1, 2, 3]},
        "remediation": "do the thing",
    }


@pytest.mark.parametrize(("cls", "code"), LEAF_CLASSES)
def test_to_json_includes_ts(cls: type[err.GenoLeWMError], code: str) -> None:
    exc = cls("m", details={"x": 1})
    decoded = json.loads(exc.to_json())
    assert decoded["code"] == code
    assert decoded["message"] == "m"
    assert decoded["details"] == {"x": 1}
    # ``to_json`` MUST timestamp the payload so log sinks receive a
    # self-contained record. ISO-8601 ends with timezone offset or 'Z'.
    assert "ts" in decoded
    assert "T" in decoded["ts"]


def test_registry_codes_are_unique() -> None:
    codes = [entry.code for entry in err.ERROR_CODES]
    assert len(codes) == len(set(codes))


def test_registry_codes_have_dotted_uppercase_prefix() -> None:
    valid_prefixes = {
        "CONFIG",
        "INPUT",
        "RESOURCE",
        "TRAINING",
        "EVAL",
        "DEPLOY",
        "ATTESTATION",
        "INTERNAL",
    }
    for entry in err.ERROR_CODES:
        prefix, _, _ = entry.code.partition(".")
        assert prefix in valid_prefixes, f"Bad code prefix in {entry.code}"
        assert entry.code == entry.code.upper(), f"Code must be uppercase: {entry.code}"


def test_registry_entry_tuple_unpacking() -> None:
    entry = err.ERROR_CODES[0]
    code, cls, summary = entry
    assert code == entry.code
    assert cls is entry.exception_class
    assert summary == entry.summary


def test_default_payload_is_empty() -> None:
    exc = err.InvalidEditError("hi")
    assert exc.details == {}
    assert exc.remediation is None
    assert exc.message == "hi"


def test_details_are_copied_not_aliased() -> None:
    src = {"a": 1}
    exc = err.InvalidEditError("m", details=src)
    src["a"] = 2
    assert exc.details == {"a": 1}


def test_exit_code_mapping_matches_spec() -> None:
    assert err.exit_code_for(err.InvalidEditError("x")) == 2
    assert err.exit_code_for(err.MissingConfigError("x")) == 3
    assert err.exit_code_for(err.CacheCorruptError("x")) == 4
    assert err.exit_code_for(err.NaNLossError("x")) == 5
    assert err.exit_code_for(err.EvalRegressionError("x")) == 6
    assert err.exit_code_for(err.ExportFormatError("x")) == 7
    assert err.exit_code_for(err.ManifestHashMismatchError("x")) == 8
    assert err.exit_code_for(err.InvariantViolation("x")) == 9
    assert err.exit_code_for(KeyboardInterrupt()) == 130
    assert err.exit_code_for(RuntimeError("not ours")) == 1


def test_root_error_has_default_code() -> None:
    # The root class still carries a registered-shape code so a stray
    # ``raise GenoLeWMError(...)`` is at least well-typed for logging.
    assert err.GenoLeWMError.code.startswith("INTERNAL.")


def test_all_exported_names_resolve() -> None:
    for name in err.__all__:
        assert hasattr(err, name), f"__all__ lists missing attribute: {name}"


def test_each_leaf_is_raisable_and_catchable() -> None:
    for cls, _code in LEAF_CLASSES:
        with pytest.raises(err.GenoLeWMError) as ei:
            raise cls("boom", details={"why": "test"})
        assert ei.value.message == "boom"
        assert ei.value.details == {"why": "test"}
