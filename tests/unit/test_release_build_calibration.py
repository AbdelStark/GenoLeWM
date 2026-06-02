# SPDX-License-Identifier: Apache-2.0
"""Tests for calibration-table generation (#169)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm import EditSpec
from geno_lewm.errors import InputError, ModelNotFoundError
from geno_lewm.surprise.calibration import CalibrationExample
from geno_lewm.surprise.score import build_calibration_examples_from_vcf, raw_surprise_example
from tests.unit.test_surprise_score import EchoPredictor, FakeActionEncoder, FakeEncoder

_VCF = (
    "##fileformat=VCFv4.2\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    "1\t1\t.\tA\tT\t.\tPASS\t.\n"
)


def test_raw_surprise_example_returns_bucket_and_raw_sigma() -> None:
    example = raw_surprise_example(
        EditSpec(chrom="1", pos=1, ref="A", alt="T"),
        FakeEncoder(),
        FakeActionEncoder(),
        EchoPredictor(),
        reference_window="ACGT",
        region="missense_variant",
    )
    assert isinstance(example, CalibrationExample)
    assert example.bucket_id == "coding_missense|mid|none"
    assert example.sigma_raw >= 0.0


def test_build_calibration_examples_from_vcf_scores_each_variant(tmp_path: Path) -> None:
    vcf = tmp_path / "background.vcf"
    vcf.write_text(_VCF, encoding="utf-8")

    examples = build_calibration_examples_from_vcf(
        vcf,
        FakeEncoder(),
        FakeActionEncoder(),
        EchoPredictor(),
        reference_windows={"1:1:A:T": "ACGT"},
        region="missense_variant",
    )
    assert len(examples) == 1
    assert all(isinstance(item, CalibrationExample) for item in examples)


def test_build_calibration_examples_requires_window_source(tmp_path: Path) -> None:
    vcf = tmp_path / "background.vcf"
    vcf.write_text(_VCF, encoding="utf-8")
    with pytest.raises(InputError, match="reference_windows or reference_fasta"):
        build_calibration_examples_from_vcf(
            vcf, FakeEncoder(), FakeActionEncoder(), EchoPredictor()
        )


def test_load_scorer_modules_requires_manifest(tmp_path: Path) -> None:
    from geno_lewm.deploy import load_scorer_modules

    with pytest.raises(ModelNotFoundError, match=r"manifest\.json"):
        load_scorer_modules(tmp_path)


def test_build_calibration_tool_writes_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pyarrow")
    from tools.release import build_calibration as tool

    monkeypatch.setattr(
        tool,
        "load_scorer_modules",
        lambda model_dir: (FakeEncoder(), FakeActionEncoder(), EchoPredictor()),
    )
    sequence = "ACGT" * 1024
    fasta = tmp_path / "reference.fa"
    fasta.write_text(f">chr1 reference\n{sequence}\n", encoding="utf-8")
    vcf = tmp_path / "background.vcf"
    vcf.write_text(_VCF, encoding="utf-8")
    output = tmp_path / "model" / "calibration.parquet"

    # A single-variant fixture is intentionally sparse; build_calibration_table
    # warns about backoff to the global bucket. Real runs use thousands.
    with pytest.warns(RuntimeWarning, match="sparse"):
        summary = tool.build_calibration(
            model_dir=tmp_path / "model",
            vcf=vcf,
            fasta=fasta,
            output=output,
            window_bp=4096,
        )
    assert output.is_file()
    assert summary["examples"] == 1
    assert summary["buckets"] >= 1


def test_build_calibration_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("pyarrow")
    from tools.release import build_calibration as tool

    monkeypatch.setattr(
        tool,
        "load_scorer_modules",
        lambda model_dir: (FakeEncoder(), FakeActionEncoder(), EchoPredictor()),
    )
    sequence = "ACGT" * 1024
    fasta = tmp_path / "reference.fa"
    fasta.write_text(f">chr1 reference\n{sequence}\n", encoding="utf-8")
    vcf = tmp_path / "background.vcf"
    vcf.write_text(_VCF, encoding="utf-8")
    output = tmp_path / "model" / "calibration.parquet"

    with pytest.warns(RuntimeWarning, match="sparse"):
        rc = tool.main(
            [
                "--model-dir",
                str(tmp_path / "model"),
                "--vcf",
                str(vcf),
                "--fasta",
                str(fasta),
                "--output",
                str(output),
                "--window-bp",
                "4096",
            ]
        )
    assert rc == 0
    assert output.is_file()
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["examples"] == 1
