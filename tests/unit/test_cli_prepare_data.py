"""CLI tests for local dataset preparation commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.cli import _dispatch, prepare_clinvar, prepare_gnomad
from geno_lewm.provenance import sha256_file


def test_prepare_gnomad_requires_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = _dispatch.run_app(prepare_gnomad.app, argv=["--quiet", "--no-banner"])
    captured = capsys.readouterr()

    assert rc == 2
    assert "prepare-gnomad requires --input-vcf" in captured.err
    assert "research tool" not in captured.err


def test_prepare_gnomad_cli_writes_json_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("pyarrow")
    vcf_path = _write_gnomad_vcf(tmp_path / "gnomad.vcf")

    rc = _dispatch.run_app(
        prepare_gnomad.app,
        argv=[
            "--quiet",
            "--no-banner",
            "--input-vcf",
            str(vcf_path),
            "--output",
            str(tmp_path),
            "--release",
            "v4.1",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    output_path = tmp_path / "gnomad" / "v4.1" / "variants.parquet"
    assert payload["records_written"] == 1
    assert payload["output_path"] == str(output_path)
    assert payload["command"].startswith("geno-lewm-prepare-gnomad --input-vcf ")
    assert payload["input_vcf"] == {
        "path": str(vcf_path),
        "sha256": sha256_file(vcf_path),
        "size_bytes": vcf_path.stat().st_size,
    }
    assert payload["output_parquet"] == {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
    }
    assert payload["runtime"]["elapsed_seconds"] >= 0.0
    assert (
        payload["runtime"]["process_peak_rss_bytes"] is None
        or payload["runtime"]["process_peak_rss_bytes"] > 0
    )
    assert "ru_maxrss" in payload["runtime"]["peak_memory_note"]


def test_prepare_clinvar_requires_release_after_input_and_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    vcf_path = _write_clinvar_vcf(tmp_path / "clinvar.vcf")
    rc = _dispatch.run_app(
        prepare_clinvar.app,
        argv=["--quiet", "--no-banner", "--input-vcf", str(vcf_path), "--output", str(tmp_path)],
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert "prepare-clinvar requires --release" in captured.err
    assert "research tool" not in captured.err


def test_prepare_clinvar_cli_writes_json_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("pyarrow")
    vcf_path = _write_clinvar_vcf(tmp_path / "clinvar.vcf")

    rc = _dispatch.run_app(
        prepare_clinvar.app,
        argv=[
            "--quiet",
            "--no-banner",
            "--input-vcf",
            str(vcf_path),
            "--output",
            str(tmp_path),
            "--release",
            "2026-04-15",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    output_path = tmp_path / "clinvar" / "2026-04-15" / "variants.parquet"
    assert payload["records_written"] == 1
    assert payload["output_path"] == str(output_path)
    assert payload["command"].startswith("geno-lewm-prepare-clinvar --input-vcf ")
    assert payload["input_vcf"] == {
        "path": str(vcf_path),
        "sha256": sha256_file(vcf_path),
        "size_bytes": vcf_path.stat().st_size,
    }
    assert payload["output_parquet"] == {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
    }
    assert payload["runtime"]["elapsed_seconds"] >= 0.0
    assert (
        payload["runtime"]["process_peak_rss_bytes"] is None
        or payload["runtime"]["process_peak_rss_bytes"] > 0
    )
    assert "ru_maxrss" in payload["runtime"]["peak_memory_note"]


def _write_gnomad_vcf(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
                "1\t10\trs1\tA\tC\t.\tPASS\tAF=0.02;AF_afr=0.03",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_clinvar_vcf(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
                "1\t100\t101\tA\tG\t.\t.\tCLNSIG=Pathogenic;ALLELEID=111",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path
