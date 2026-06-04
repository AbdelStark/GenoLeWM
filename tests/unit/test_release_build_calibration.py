# SPDX-License-Identifier: Apache-2.0
"""Tests for calibration-table generation (#169)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm import EditSpec
from geno_lewm.errors import InputError, ModelNotFoundError
from geno_lewm.provenance import (
    SCHEMA_VERSION,
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    load_manifest,
    sha256_bytes,
    sha256_file,
    write_manifest,
)
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
    model_dir = _write_model_dir(tmp_path / "model")
    sequence = "ACGT" * 1024
    fasta = tmp_path / "reference.fa"
    fasta.write_text(f">chr1 reference\n{sequence}\n", encoding="utf-8")
    vcf = tmp_path / "background.vcf"
    vcf.write_text(_VCF, encoding="utf-8")
    output = model_dir / "calibration.parquet"
    summary_json = model_dir / "calibration_report.json"
    manifest = load_manifest(model_dir / "manifest.json")

    # A single-variant fixture is intentionally sparse; build_calibration_table
    # warns about backoff to the global bucket. Real runs use thousands.
    with pytest.warns(RuntimeWarning, match="sparse"):
        summary = tool.build_calibration(
            model_dir=model_dir,
            vcf=vcf,
            fasta=fasta,
            output=output,
            summary_json=summary_json,
            window_bp=4096,
        )
    assert output.is_file()
    assert summary_json.is_file()
    assert summary["model_id"] == manifest.model_id()
    assert summary["model_manifest"]["sha256"] == sha256_file(model_dir / "manifest.json")
    assert summary["inputs"]["vcf"]["sha256"] == sha256_file(vcf)
    assert summary["inputs"]["fasta"]["sha256"] == sha256_file(fasta)
    assert summary["calibration_artifact"]["path"] == "calibration.parquet"
    assert summary["calibration_artifact"]["sha256"] == sha256_file(output)
    assert summary["examples"] == 1
    assert summary["buckets"] >= 1
    serialized = summary_json.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized


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
    model_dir = _write_model_dir(tmp_path / "model")
    sequence = "ACGT" * 1024
    fasta = tmp_path / "reference.fa"
    fasta.write_text(f">chr1 reference\n{sequence}\n", encoding="utf-8")
    vcf = tmp_path / "background.vcf"
    vcf.write_text(_VCF, encoding="utf-8")
    output = model_dir / "calibration.parquet"
    summary_json = model_dir / "calibration_report.json"

    with pytest.warns(RuntimeWarning, match="sparse"):
        rc = tool.main(
            [
                "--model-dir",
                str(model_dir),
                "--vcf",
                str(vcf),
                "--fasta",
                str(fasta),
                "--output",
                str(output),
                "--summary-json",
                str(summary_json),
                "--window-bp",
                "4096",
            ]
        )
    assert rc == 0
    assert output.is_file()
    assert summary_json.is_file()
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["examples"] == 1
    assert payload["generated_by"] == tool.GENERATED_BY


def test_build_calibration_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    from tools.release import build_calibration as tool

    with pytest.raises(SystemExit) as excinfo:
        tool.main(["--help"])

    assert excinfo.value.code == 0
    assert "--summary-json" in capsys.readouterr().out


def _write_model_dir(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "predictor.safetensors").write_bytes(b"predictor")
    (root / "action_encoder.safetensors").write_bytes(b"action")
    (root / "calibration.parquet").write_bytes(b"placeholder-calibration")
    (root / "train_config.yaml").write_text("seed: 0\n", encoding="utf-8")
    (root / "eval_report.md").write_text("# eval\n", encoding="utf-8")
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        model_name="geno-lewm",
        model_version="0.1.0",
        release_id="geno-lewm-v0.1.0-r1",
        encoder=ManifestEncoder(
            id="HuggingFaceBio/Carbon-500M",
            revision="main",
            hash=sha256_bytes(b"encoder"),
        ),
        predictor=ManifestArtifact(
            file="predictor.safetensors",
            hash=sha256_file(root / "predictor.safetensors"),
            dtype="bf16",
        ),
        action_encoder=ManifestArtifact(
            file="action_encoder.safetensors",
            hash=sha256_file(root / "action_encoder.safetensors"),
            dtype="bf16",
        ),
        calibration=ManifestArtifact(
            file="calibration.parquet",
            hash=sha256_file(root / "calibration.parquet"),
            version="1.0.0",
        ),
        training=ManifestTraining(
            config_file="train_config.yaml",
            hash=sha256_file(root / "train_config.yaml"),
            data_snapshot={"snapshot": "geno-lewm-data-v0.1.0-r1"},
        ),
        eval=ManifestArtifact(file="eval_report.md", hash=sha256_file(root / "eval_report.md")),
    )
    write_manifest(manifest, root / "manifest.json")
    return root
