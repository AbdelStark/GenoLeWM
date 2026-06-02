# SPDX-License-Identifier: Apache-2.0
"""Unit tests for RFC-0009 surprise scoring orchestration."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from geno_lewm import EditSpec
from geno_lewm._artifact_sources import SCORE_JSONL_GENERATED_BY, SCORE_JSONL_SCHEMA_VERSION
from geno_lewm.errors import InputError, VcfParseError
from geno_lewm.surprise import CalibrationBucket, CalibrationTable, SurpriseResult
from geno_lewm.surprise.score import score_variant, score_vcf


class FakeEncoder:
    def encode(self, window: str, *, edit_locus: int | None = None) -> tuple[float, ...]:
        del edit_locus
        denom = float(len(window))
        return tuple(window.count(base) / denom for base in "ACGT")


class FakeActionEncoder:
    def __call__(self, edits: object) -> tuple[float, ...]:
        del edits
        return (1.0,)


class EchoPredictor:
    def __call__(self, state: object, action: object) -> object:
        del action
        return state


class BadShapePredictor:
    def __call__(self, state: object, action: object) -> object:
        del state, action
        return (1.0, 2.0, 3.0)


def test_score_variant_computes_raw_and_calibrated_surprise() -> None:
    result = score_variant(
        EditSpec(chrom="1", pos=1, ref="A", alt="T"),
        FakeEncoder(),
        FakeActionEncoder(),
        EchoPredictor(),
        _calibration(),
        reference_window="ACGT",
        region="missense_variant",
    )

    assert result.bucket_id == "coding_missense|mid|none"
    assert result.sigma_raw == pytest.approx(math.sqrt(0.125))
    assert result.sigma_calibrated == pytest.approx(math.sqrt(0.125))
    assert result.confidence == 1.0
    assert result.low_confidence is False
    assert result.to_dict()["sigma_raw"] == result.sigma_raw


def test_score_variant_supports_parent_bucket_backoff() -> None:
    result = score_variant(
        EditSpec(chrom="1", pos=1, ref="A", alt="T"),
        FakeEncoder(),
        FakeActionEncoder(),
        EchoPredictor(),
        CalibrationTable(
            buckets=(
                CalibrationBucket(
                    bucket_id="coding_missense|mid",
                    n_calibration=1_000,
                    cdf=(0.0, 1.0),
                    sigma_grid=(0.0, 1.0),
                ),
                CalibrationBucket(
                    bucket_id="coding_missense|mid|none",
                    n_calibration=10,
                    cdf=(0.0, 1.0),
                    sigma_grid=(0.0, 1.0),
                    back_off_to="coding_missense|mid",
                ),
            )
        ),
        reference_window="ACGT",
        region="missense_variant",
    )

    assert result.bucket_id == "coding_missense|mid"


def test_score_variant_rejects_invalid_aggregation_and_output_shape() -> None:
    with pytest.raises(InputError, match="aggregation"):
        score_variant(
            EditSpec(chrom="1", pos=1, ref="A", alt="T"),
            FakeEncoder(),
            FakeActionEncoder(),
            EchoPredictor(),
            _calibration(),
            reference_window="ACGT",
            aggregation="sum",
        )

    with pytest.raises(InputError, match="output length"):
        score_variant(
            EditSpec(chrom="1", pos=1, ref="A", alt="T"),
            FakeEncoder(),
            FakeActionEncoder(),
            BadShapePredictor(),
            _calibration(),
            reference_window="ACGT",
            region="missense_variant",
        )


def test_score_vcf_writes_jsonl_for_explicit_reference_windows(tmp_path: Path) -> None:
    vcf_path = tmp_path / "input.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t1\t.\tA\tT\t.\tPASS\t.\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "scores.jsonl"

    written = score_vcf(
        vcf_path,
        FakeEncoder(),
        FakeActionEncoder(),
        EchoPredictor(),
        _calibration(),
        output_path,
        reference_windows={"1:1:A:T": "ACGT"},
        region="missense_variant",
    )

    assert written == output_path
    [payload] = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert payload["schema_version"] == SCORE_JSONL_SCHEMA_VERSION
    assert payload["generated_by"] == SCORE_JSONL_GENERATED_BY
    assert payload["chrom"] == "1"
    assert payload["pos"] == 1
    assert payload["ref"] == "A"
    assert payload["alt"] == "T"
    assert payload["bucket_id"] == "coding_missense|mid|none"


def test_score_vcf_extracts_reference_windows_from_fasta(tmp_path: Path) -> None:
    sequence = "ACGT" * 1024
    fasta_path = tmp_path / "reference.fa"
    fasta_path.write_text(f">chr1 reference contig\n{sequence}\n", encoding="utf-8")
    vcf_path = tmp_path / "input.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t1\t.\tA\tT\t.\tPASS\t.\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "scores.jsonl"

    written = score_vcf(
        vcf_path,
        FakeEncoder(),
        FakeActionEncoder(),
        EchoPredictor(),
        _calibration(),
        output_path,
        reference_fasta=fasta_path,
        window_bp=4096,
        region="missense_variant",
    )

    assert written == output_path
    [payload] = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert payload["generated_by"] == SCORE_JSONL_GENERATED_BY
    assert payload["chrom"] == "1"
    assert payload["ref"] == "A"
    assert payload["alt"] == "T"
    assert payload["bucket_id"] == "coding_missense|mid|none"
    assert payload["sigma_raw"] > 0.0


def test_score_vcf_rejects_fasta_reference_mismatch(tmp_path: Path) -> None:
    fasta_path = tmp_path / "reference.fa"
    fasta_path.write_text(f">1\n{'ACGT' * 1024}\n", encoding="utf-8")
    vcf_path = tmp_path / "input.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t1\t.\tC\tT\t.\tPASS\t.\n",
        encoding="utf-8",
    )

    with pytest.raises(VcfParseError, match="do not match VCF REF"):
        score_vcf(
            vcf_path,
            FakeEncoder(),
            FakeActionEncoder(),
            EchoPredictor(),
            _calibration(),
            tmp_path / "scores.jsonl",
            reference_fasta=fasta_path,
            window_bp=4096,
            region="missense_variant",
        )


def test_score_vcf_requires_explicit_windows_and_scoreable_rows(tmp_path: Path) -> None:
    vcf_path = tmp_path / "input.vcf"
    vcf_path.write_text("##fileformat=VCFv4.2\n", encoding="utf-8")

    with pytest.raises(InputError, match="reference_windows or reference_fasta"):
        score_vcf(
            vcf_path,
            FakeEncoder(),
            FakeActionEncoder(),
            EchoPredictor(),
            _calibration(),
            tmp_path / "scores.jsonl",
        )

    with pytest.raises(VcfParseError, match="no scoreable"):
        score_vcf(
            vcf_path,
            FakeEncoder(),
            FakeActionEncoder(),
            EchoPredictor(),
            _calibration(),
            tmp_path / "scores.jsonl",
            reference_windows={"1": "ACGT"},
        )


def test_surprise_result_validates_public_contract() -> None:
    with pytest.raises(InputError, match="sigma_raw"):
        SurpriseResult(-1.0, 0.5, "bucket", 1.0, False)
    with pytest.raises(InputError, match="sigma_calibrated"):
        SurpriseResult(0.0, 1.5, "bucket", 1.0, False)
    with pytest.raises(InputError, match="bucket_id"):
        SurpriseResult(0.0, 0.5, "", 1.0, False)
    with pytest.raises(InputError, match="low_confidence"):
        SurpriseResult(0.0, 0.5, "bucket", 1.0, 0)  # type: ignore[arg-type]


def _calibration() -> CalibrationTable:
    return CalibrationTable(
        buckets=(
            CalibrationBucket(
                bucket_id="coding_missense|mid|none",
                n_calibration=1_000,
                cdf=(0.0, 0.5, 1.0),
                sigma_grid=(0.0, 0.5, 1.0),
            ),
        )
    )
