# SPDX-License-Identifier: Apache-2.0
"""Integration smoke for the committed fixture corpus (testing contract).

Confirms that the on-disk fixtures under ``tests/fixtures/`` parse
cleanly through the corresponding public dataclasses. This is the
smallest non-trivial integration test: file system → loader → typed
provenance API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.provenance import RECEIPT_SCHEMA_VERSION, Receipt, read_receipt
from tests.fixtures import load_json


def test_sample_window_is_acgt(fixtures_dir: Path) -> None:
    body = (fixtures_dir / "sample_window.fa").read_text(encoding="utf-8")
    lines = [line for line in body.splitlines() if not line.startswith(">")]
    sequence = "".join(lines)
    assert sequence, "fixture window is empty"
    assert set(sequence) <= set("ACGTN"), f"non-ACGTN bases in fixture: {set(sequence)}"


def test_sample_receipt_round_trips_through_read_receipt(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    raw = load_json("sample_receipt.json")
    # Canonicalize and write to disk so ``read_receipt`` exercises the
    # JSON loader + schema validation.
    target = tmp_path / "sample_receipt.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    receipt = read_receipt(target)
    assert isinstance(receipt, Receipt)
    assert receipt.schema_version == RECEIPT_SCHEMA_VERSION
    assert receipt.output.bucket_id == "coding.missense"


@pytest.mark.parametrize("fixture_name", ["sample_receipt.json"])
def test_fixture_files_are_well_formed_json(fixture_name: str) -> None:
    """Every JSON fixture must parse cleanly."""
    load_json(fixture_name)  # raises on bad JSON
