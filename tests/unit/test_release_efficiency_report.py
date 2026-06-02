"""Tests for the release efficiency report validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from geno_lewm.provenance import sha256_bytes
from tools.release.efficiency_report import (
    GENERATED_BY,
    REPORT_NAME,
    main,
    parse_efficiency_report,
)


def test_parse_efficiency_report_accepts_measured_payload() -> None:
    report = parse_efficiency_report(_payload())

    assert report.generated_by == GENERATED_BY
    assert report.model_id.startswith("sha256:")
    assert report.samples == 32
    assert report.measurements.single_variant_latency_ms == 12.5
    assert report.measurements.batched_throughput_variants_per_s == 512.0
    assert report.measurements.peak_memory_bytes == 123456789
    assert dict(report.inputs)["checkpoint"].path == "model/predictor.safetensors"


def test_efficiency_report_main_writes_normalized_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_json = tmp_path / "raw_efficiency.json"
    output = tmp_path / REPORT_NAME
    input_json.write_text(json.dumps(_payload()), encoding="utf-8")

    rc = main(["--input-json", str(input_json), "--output", str(output)])
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out == f"wrote {output}\n"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["generated_by"] == GENERATED_BY
    assert payload["measurements"]["peak_memory_bytes"] == 123456789


def test_parse_efficiency_report_rejects_placeholder_text() -> None:
    payload = _payload()
    payload["hardware"] = "TODO"

    with pytest.raises(InputError, match="placeholder text is not allowed"):
        parse_efficiency_report(payload)


def test_parse_efficiency_report_rejects_negative_samples() -> None:
    payload = _payload()
    payload["samples"] = -1

    with pytest.raises(InputError, match="samples must be a positive integer"):
        parse_efficiency_report(payload)


def test_parse_efficiency_report_rejects_missing_measurement() -> None:
    payload = _payload()
    payload["measurements"].pop("single_variant_latency_ms")

    with pytest.raises(InputError, match="single_variant_latency_ms"):
        parse_efficiency_report(payload)


def test_parse_efficiency_report_rejects_missing_generator() -> None:
    payload = _payload()
    payload.pop("generated_by")

    with pytest.raises(InputError, match="generated_by"):
        parse_efficiency_report(payload)


@pytest.mark.parametrize(
    "path",
    [
        "/Users/example/private/input.vcf",
        "../private/input.vcf",
        "https://example.test/input.vcf",
    ],
)
def test_parse_efficiency_report_rejects_private_or_remote_input_paths(path: str) -> None:
    payload = _payload()
    payload["inputs"]["checkpoint"]["path"] = path

    with pytest.raises(InputError, match="input paths must be package-relative"):
        parse_efficiency_report(payload)


def test_parse_efficiency_report_rejects_malformed_inline_input_path() -> None:
    payload = _payload()
    payload["inputs"]["checkpoint"]["path"] = "inline:../secret"

    with pytest.raises(InputError, match="inline input paths"):
        parse_efficiency_report(payload)


def _payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "generated_by": GENERATED_BY,
        "generated_at": "2026-06-01T00:00:00Z",
        "model_id": sha256_bytes(b"model"),
        "model_release": "geno-lewm-v0.1.0-r1",
        "dataset_snapshot": "geno-lewm-data-v0.1.0-r1",
        "commit": "abcdef1234567890",
        "command": [
            "python",
            "-m",
            "bench.inference",
            "--model-dir",
            "model",
            "--batch-size",
            "64",
        ],
        "hardware": "Apple M3 Max CPU",
        "runtime": "Python 3.12, torch CPU backend",
        "warmup_batches": 2,
        "samples": 32,
        "measurements": {
            "single_variant_latency_ms": 12.5,
            "batched_throughput_variants_per_s": 512.0,
            "peak_memory_bytes": 123456789,
        },
        "inputs": {
            "checkpoint": {
                "path": "model/predictor.safetensors",
                "sha256": sha256_bytes(b"checkpoint"),
                "size_bytes": 12,
            },
            "dataset_manifest": {
                "path": "dataset/dataset_manifest.json",
                "sha256": sha256_bytes(b"dataset"),
                "size_bytes": 34,
            },
        },
        "limitations": [
            "Single local hardware profile; no cross-platform timing claim is made.",
            "The run records inference efficiency only, not training throughput.",
        ],
    }
