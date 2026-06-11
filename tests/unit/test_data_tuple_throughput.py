"""Tests for release tuple throughput checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.provenance import sha256_file
from tests.unit.test_training_preflight import _write_release_dataset
from tools.data.tuple_throughput import GENERATED_BY, main, measure_tuple_throughput


def test_measure_tuple_throughput_uses_release_dataset_windows(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)

    payload = measure_tuple_throughput(
        dataset_dir=dataset_dir,
        samples=8,
        seed=7,
        min_tuples_per_second=0.01,
    )

    assert payload["generated_by"] == GENERATED_BY
    assert payload["dataset_snapshot_id"] == "geno-lewm-data-v0.1.0-r1"
    assert payload["dataset_manifest"] == {
        "path": "dataset_manifest.json",
        "sha256": sha256_file(dataset_dir / "dataset_manifest.json"),
        "size_bytes": (dataset_dir / "dataset_manifest.json").stat().st_size,
    }
    assert payload["seed"] == 7
    assert payload["requested_samples"] == 8
    assert payload["samples"] == 8
    assert payload["min_tuples_per_second"] == 0.01
    assert payload["passed_min_tuples_per_second"] is True
    assert payload["windows"] == 1
    assert payload["gnomad_edits"] == 3
    assert payload["tuples_per_second"] > 0


def test_tuple_throughput_main_writes_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    output = tmp_path / "reports" / "tuple_throughput_report.json"

    rc = main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--samples",
            "4",
            "--seed",
            "11",
            "--min-tuples-per-second",
            "0.01",
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert output.is_file()
    stdout_payload = json.loads(captured.out)
    report_payload = json.loads(output.read_text(encoding="utf-8"))
    assert report_payload == stdout_payload
    assert report_payload["samples"] == 4
    assert report_payload["seed"] == 11
    assert report_payload["passed_min_tuples_per_second"] is True
