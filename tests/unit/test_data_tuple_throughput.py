"""Tests for release tuple throughput checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.test_training_preflight import _write_release_dataset
from tools.data.tuple_throughput import GENERATED_BY, SCHEMA_VERSION, measure_tuple_throughput


def test_measure_tuple_throughput_uses_release_dataset_windows(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)

    payload = measure_tuple_throughput(dataset_dir=dataset_dir, samples=8)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["generated_by"] == GENERATED_BY
    assert payload["dataset_snapshot_id"] == "geno-lewm-data-v0.1.0-r1"
    assert payload["dataset_manifest"]["path"] == "dataset_manifest.json"
    assert payload["dataset_manifest"]["sha256"].startswith("sha256:")
    assert payload["dataset_manifest"]["size_bytes"] > 0
    assert payload["seed"] == 0
    assert payload["samples"] == 8
    assert payload["windows"] == 1
    assert payload["gnomad_edits"] == 3
    assert payload["tuples_per_second"] > 0
