"""Tests for release tuple throughput checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.test_training_preflight import _write_release_dataset
from tools.data.tuple_throughput import GENERATED_BY, measure_tuple_throughput


def test_measure_tuple_throughput_uses_release_dataset_windows(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)

    payload = measure_tuple_throughput(dataset_dir=dataset_dir, samples=8)

    assert payload["generated_by"] == GENERATED_BY
    assert payload["samples"] == 8
    assert payload["windows"] == 1
    assert payload["gnomad_edits"] == 3
    assert payload["tuples_per_second"] > 0
