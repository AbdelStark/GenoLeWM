"""Tests for ``geno-lewm-carbon-baseline``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm._artifact_sources import CARBON_ZERO_SHOT_GENERATED_BY
from geno_lewm.cli import _dispatch, carbon_baseline as cli


def test_carbon_baseline_requires_inputs(capsys: pytest.CaptureFixture[str]) -> None:
    rc = _dispatch.run_app(cli.app, argv=["--quiet", "--no-banner"])
    captured = capsys.readouterr()

    assert rc == 2
    assert "requires --vcf" in captured.err


def test_carbon_baseline_writes_score_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    vcf, fasta = _write_variant_inputs(tmp_path)
    model_dir = tmp_path / "carbon"
    model_dir.mkdir()
    output = tmp_path / "carbon_zero_shot_scores.jsonl"
    metadata = tmp_path / "carbon_zero_shot_summary.json"
    cache = tmp_path / "carbon_zero_shot_logp_cache.jsonl"
    calls: dict[str, object] = {}

    def fake_loader(
        model_path: Path,
        *,
        revision: str,
        dtype: str,
        device: str | None,
        trust_remote_code: bool,
        local_files_only: bool,
    ) -> object:
        calls.update(
            {
                "model_path": model_path,
                "revision": revision,
                "dtype": dtype,
                "device": device,
                "trust_remote_code": trust_remote_code,
                "local_files_only": local_files_only,
            }
        )
        return lambda sequence: float(sequence.count("A"))

    monkeypatch.setattr(cli, "load_carbon_logp_scorer", fake_loader)

    rc = _dispatch.run_app(
        cli.app,
        argv=[
            "--quiet",
            "--no-banner",
            "--vcf",
            str(vcf),
            "--fasta",
            str(fasta),
            "--carbon-model-dir",
            str(model_dir),
            "--carbon-revision",
            "main@abc123",
            "--dtype",
            "fp32",
            "--window-bp",
            "4096",
            "--output-scores",
            str(output),
            "--metadata-output",
            str(metadata),
            "--artifact-root",
            str(tmp_path),
            "--logp-cache-jsonl",
            str(cache),
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    summary = json.loads(captured.out)
    assert summary["records"] == 1
    assert summary["score_field"] == "carbon_zero_shot_score"
    assert summary["carbon_model"] == "carbon"
    assert summary["vcf"] == "variants.vcf"
    assert summary["fasta"] == "ref.fa"
    assert summary["output_scores"] == "carbon_zero_shot_scores.jsonl"
    assert summary["logp_cache"] == "carbon_zero_shot_logp_cache.jsonl"
    assert calls == {
        "model_path": model_dir,
        "revision": "main@abc123",
        "dtype": "fp32",
        "device": None,
        "trust_remote_code": False,
        "local_files_only": True,
    }
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["generated_by"] == CARBON_ZERO_SHOT_GENERATED_BY
    assert row["carbon_zero_shot_score"] == 1.0
    metadata_payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert metadata_payload["records"] == 1
    assert metadata_payload["carbon_model"] == "carbon"
    assert metadata_payload["output_scores"] == "carbon_zero_shot_scores.jsonl"


def test_carbon_baseline_rejects_release_metadata_paths_outside_artifact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    release_root = tmp_path / "release"
    outside = tmp_path / "outside"
    release_root.mkdir()
    outside.mkdir()
    vcf, fasta = _write_variant_inputs(release_root)
    model_dir = release_root / "carbon"
    model_dir.mkdir()
    output = outside / "carbon_zero_shot_scores.jsonl"

    def fake_loader(
        model_path: Path,
        *,
        revision: str,
        dtype: str,
        device: str | None,
        trust_remote_code: bool,
        local_files_only: bool,
    ) -> object:
        del model_path, revision, dtype, device, trust_remote_code, local_files_only
        return lambda sequence: float(sequence.count("A"))

    monkeypatch.setattr(cli, "load_carbon_logp_scorer", fake_loader)

    rc = _dispatch.run_app(
        cli.app,
        argv=[
            "--quiet",
            "--no-banner",
            "--vcf",
            str(vcf),
            "--fasta",
            str(fasta),
            "--carbon-model-dir",
            str(model_dir),
            "--output-scores",
            str(output),
            "--artifact-root",
            str(release_root),
        ],
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert "metadata paths must stay inside --artifact-root" in captured.err
    assert not output.exists()


def _write_variant_inputs(tmp_path: Path) -> tuple[Path, Path]:
    fasta = tmp_path / "ref.fa"
    fasta.write_text(">1\nACGTACGT\n", encoding="utf-8")
    vcf = tmp_path / "variants.vcf"
    vcf.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.3",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
                "1\t1\t.\tA\tT\t.\tPASS\t.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return vcf, fasta
