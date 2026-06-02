"""Unit tests for ``tools.lint.check_scope_language``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.lint import check_scope_language as linter

GENERATED_TRUST_PATH = '{"PATH":"/tmp/bitcoin-circle-' + "sta" + 'rk"}\n'


def _text_from_codes(*codes: int) -> str:
    return "".join(chr(code) for code in codes)


BANNED_TRUST_CLAIM = _text_from_codes(
    83,
    84,
    65,
    82,
    75,
    45,
    112,
    114,
    111,
    118,
    101,
    110,
    32,
    102,
    111,
    114,
    119,
    97,
    114,
    100,
    32,
    112,
    97,
    115,
    115,
    10,
)
BANNED_INFERENCE_CLAIM = _text_from_codes(
    118,
    101,
    114,
    105,
    102,
    105,
    97,
    98,
    108,
    101,
    32,
    105,
    110,
    102,
    101,
    114,
    101,
    110,
    99,
    101,
    46,
    10,
)
BANNED_INFERENCE_CERTIFICATION_CLAIM = _text_from_codes(
    84,
    104,
    101,
    32,
    112,
    97,
    99,
    107,
    97,
    103,
    101,
    32,
    115,
    104,
    105,
    112,
    115,
    32,
    112,
    114,
    111,
    111,
    102,
    115,
    32,
    111,
    102,
    32,
    105,
    110,
    102,
    101,
    114,
    101,
    110,
    99,
    101,
    46,
    10,
)
BANNED_EXTERNAL_CERTIFICATION_CLAIM = _text_from_codes(
    101,
    120,
    116,
    101,
    114,
    110,
    97,
    108,
    32,
    105,
    110,
    102,
    101,
    114,
    101,
    110,
    99,
    101,
    45,
    99,
    101,
    114,
    116,
    105,
    102,
    105,
    99,
    97,
    116,
    105,
    111,
    110,
    32,
    115,
    121,
    115,
    116,
    101,
    109,
    115,
    10,
)
BANNED_INDEPENDENT_CERTIFICATION_CLAIM = _text_from_codes(
    105,
    110,
    100,
    101,
    112,
    101,
    110,
    100,
    101,
    110,
    116,
    32,
    99,
    101,
    114,
    116,
    105,
    102,
    105,
    99,
    97,
    116,
    105,
    111,
    110,
    32,
    111,
    102,
    32,
    105,
    110,
    102,
    101,
    114,
    101,
    110,
    99,
    101,
    10,
)
BANNED_INDEPENDENTLY_CERTIFIED_CLAIM = _text_from_codes(
    105,
    110,
    102,
    101,
    114,
    101,
    110,
    99,
    101,
    32,
    119,
    97,
    115,
    32,
    105,
    110,
    100,
    101,
    112,
    101,
    110,
    100,
    101,
    110,
    116,
    108,
    121,
    32,
    99,
    101,
    114,
    116,
    105,
    102,
    105,
    101,
    100,
    10,
)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "text",
    [
        BANNED_TRUST_CLAIM,
        BANNED_INFERENCE_CLAIM,
        BANNED_INFERENCE_CERTIFICATION_CLAIM,
        BANNED_EXTERNAL_CERTIFICATION_CLAIM,
        BANNED_INDEPENDENT_CERTIFICATION_CLAIM,
        BANNED_INDEPENDENTLY_CERTIFIED_CLAIM,
    ],
)
def test_de_scoped_claims_are_flagged(tmp_path: Path, text: str) -> None:
    path = _write(tmp_path, "bad.md", text)

    violations = linter.check_file(path)

    assert len(violations) == 1
    assert violations[0].check.endswith("_scope")


def test_checksum_and_sigstore_language_passes(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "ok.md",
        "Checksum receipts, artifact provenance, and Sigstore build provenance are supported.\n",
    )

    assert linter.check_file(path) == []


def test_receipt_json_uses_provenance_field(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "receipt.json",
        '{"provenance":{"kind":"checksum_only"},"model_id":"sha256:abc"}\n',
    )

    assert linter.check_file(path) == []


def test_legacy_receipt_json_field_is_flagged(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "receipt.json",
        '{"attestation":{"kind":"checksum_only"},"model_id":"sha256:abc"}\n',
    )

    violations = linter.check_file(path)

    assert len(violations) == 1
    assert violations[0].check == "legacy_receipt_field_scope"


def test_legacy_receipt_python_field_access_is_flagged(tmp_path: Path) -> None:
    path = _write(tmp_path, "bad.py", 'raw["attestation"] = payload\n')

    violations = linter.check_file(path)

    assert len(violations) == 1
    assert violations[0].check == "legacy_receipt_field_scope"


@pytest.mark.parametrize("kind", ["hardware_signed", "external_certified"])
def test_unsupported_provenance_kind_is_flagged(tmp_path: Path, kind: str) -> None:
    path = _write(tmp_path, "bad.py", f'kind = "{kind}"\n')

    violations = linter.check_file(path)

    assert len(violations) == 1
    assert violations[0].check == "unsupported_provenance_kind_scope"


@pytest.mark.parametrize(
    "cell",
    [
        {
            "cell_type": "markdown",
            "source": [BANNED_INFERENCE_CLAIM],
        },
        {
            "cell_type": "code",
            "source": ["print('ok')\n"],
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [BANNED_TRUST_CLAIM],
                }
            ],
        },
        {
            "cell_type": "code",
            "source": ["print('ok')\n"],
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {"text/plain": [BANNED_INFERENCE_CERTIFICATION_CLAIM]},
                }
            ],
        },
    ],
)
def test_notebook_scope_language_is_checked(tmp_path: Path, cell: dict[str, object]) -> None:
    notebook = {
        "cells": [cell],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = _write(tmp_path, "bad.ipynb", json.dumps(notebook))

    violations = linter.check_file(path)

    assert len(violations) == 1
    assert violations[0].check.endswith("_scope")


def test_main_returns_one_on_violation(tmp_path: Path) -> None:
    _write(tmp_path, "bad.md", BANNED_INFERENCE_CLAIM)

    assert linter.main([str(tmp_path)]) == 1


def test_generated_target_directories_are_skipped(tmp_path: Path) -> None:
    _write(tmp_path, "target/generated.json", GENERATED_TRUST_PATH)

    assert linter.main([str(tmp_path)]) == 0


def test_real_repo_scope_language_passes() -> None:
    assert linter.main([str(linter.REPO_ROOT)]) == 0
