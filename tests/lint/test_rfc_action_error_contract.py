# SPDX-License-Identifier: Apache-2.0
"""Regression tests for RFC-0003's typed action-error contract."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RFC = REPO_ROOT / "rfcs" / "0003-action-representation-genomic-edits.md"
ACTION_SPEC = REPO_ROOT / "geno_lewm" / "action" / "spec.py"
ACTION_APPLY = REPO_ROOT / "geno_lewm" / "action" / "apply.py"
ACTION_SPEC_TEST = REPO_ROOT / "tests" / "unit" / "test_action_spec.py"
ERROR_SPEC = REPO_ROOT / "docs" / "spec" / "04-error-model.md"


def test_rfc_action_validation_uses_typed_error_taxonomy() -> None:
    text = RFC.read_text(encoding="utf-8")
    required = (
        "- **Updated:** 2026-06-11",
        "typed InputError-family failures",
        "Validation rules raise typed `InputError` subclasses",
        "malformed edit fields raise `InvalidEditError`",
        "edits longer than the v1\nshort-edit bound raise `UnsupportedEditError`",
        "window-relative\nconversion outside the supplied window raises `OutOfWindowError`",
    )

    for fragment in required:
        assert fragment in text

    assert "Validation rules (raised as `ValueError`)" not in text


def test_action_error_contract_tracks_live_code_and_tests() -> None:
    action_spec = ACTION_SPEC.read_text(encoding="utf-8")
    action_apply = ACTION_APPLY.read_text(encoding="utf-8")
    action_tests = ACTION_SPEC_TEST.read_text(encoding="utf-8")
    error_spec = ERROR_SPEC.read_text(encoding="utf-8")

    for name in ("InvalidEditError", "UnsupportedEditError", "OutOfWindowError"):
        assert name in action_spec
        assert name in action_tests
        assert name in error_spec

    assert "WindowMismatchError" in action_apply
    assert "OverlappingEditsError" in action_apply
    assert "EditSpec invariants violated" in error_spec
    assert "edit type / length not in v1 scope" in error_spec
