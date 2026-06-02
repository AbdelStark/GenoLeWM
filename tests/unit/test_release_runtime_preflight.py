"""Tests for the terminal-demo runtime preflight report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.test_release_model_package import _write_model_inputs
from tools.release.model_package import build_model_package
from tools.release.runtime_preflight import (
    DependencyProbe,
    RuntimePreflightReport,
    RuntimePreflightRequest,
    build_runtime_preflight_report,
    build_terminal_demo_command,
    main,
    write_runtime_preflight_report,
)


def test_runtime_preflight_accepts_release_artifacts(tmp_path: Path) -> None:
    model_dir, vcf, fasta = _write_inputs(tmp_path)
    request = RuntimePreflightRequest(
        model_dir=model_dir,
        vcf=vcf,
        fasta=fasta,
        output_dir=tmp_path / "demo",
        backend="cpu",
    )

    report = build_runtime_preflight_report(
        request,
        generated_at="2026-06-01T12:00:00Z",
        dependency_probe=_available_dependency,
    )

    assert report.ok is True
    assert report.model_id is not None
    assert report.selected_backend == "cpu"
    assert report.network_guard["ok"] is True
    assert all(probe.available for probe in report.dependencies)
    assert any(artifact["path"].endswith("model_card.md") for artifact in report.artifacts)
    assert "--receipt" in build_terminal_demo_command(request)


def test_runtime_preflight_reports_missing_inputs_and_dependencies(tmp_path: Path) -> None:
    model_dir, _vcf, _fasta = _write_inputs(tmp_path)
    request = RuntimePreflightRequest(
        model_dir=model_dir,
        vcf=tmp_path / "missing.vcf",
        fasta=tmp_path / "missing.fa",
        output_dir=tmp_path / "demo",
    )

    report = build_runtime_preflight_report(request, dependency_probe=_missing_dependency)

    codes = _codes(report)
    assert report.ok is False
    assert "input.vcf.missing" in codes
    assert "input.fasta.missing" in codes
    assert "runtime.dependency_unavailable" in codes


def test_runtime_preflight_rejects_fixture_manifest_by_default(tmp_path: Path) -> None:
    model_dir, vcf, fasta = _write_inputs(tmp_path, release_id="geno-lewm-fixture-r1")

    report = build_runtime_preflight_report(
        RuntimePreflightRequest(model_dir=model_dir, vcf=vcf, fasta=fasta, output_dir=tmp_path),
        dependency_probe=_available_dependency,
    )

    assert report.ok is False
    assert "model.fixture_manifest" in _codes(report)


def test_runtime_preflight_can_write_report_and_main_json(tmp_path: Path) -> None:
    model_dir, vcf, fasta = _write_inputs(tmp_path)
    output = tmp_path / "demo" / "runtime_preflight_report.json"
    report = write_runtime_preflight_report(
        RuntimePreflightRequest(
            model_dir=model_dir,
            vcf=vcf,
            fasta=fasta,
            output_dir=tmp_path / "demo",
            require_native_runtime=False,
        ),
        output,
    )

    assert report.ok is True
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["requirements"]["native_runtime"] is False


def test_runtime_preflight_main_returns_two_for_failed_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir, _vcf, _fasta = _write_inputs(tmp_path)

    rc = main(
        [
            "--model-dir",
            str(model_dir),
            "--vcf",
            str(tmp_path / "missing.vcf"),
            "--fasta",
            str(tmp_path / "missing.fa"),
            "--output-dir",
            str(tmp_path / "demo"),
            "--no-require-native-runtime",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert "wrote" in captured.out
    assert (tmp_path / "demo" / "runtime_preflight_report.json").is_file()


def _write_inputs(
    root: Path,
    *,
    release_id: str = "geno-lewm-v0.1.0-r1",
) -> tuple[Path, Path, Path]:
    model_dir = root / "model"
    model_dir.mkdir()
    metadata = _write_model_inputs(model_dir, release_id=release_id)
    build_model_package(
        model_dir,
        metadata,
        allow_fixture_manifest="fixture" in release_id,
    )
    vcf = root / "input.vcf"
    fasta = root / "ref.fa"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t10\t.\tA\tT\t.\tPASS\t.\n",
        encoding="utf-8",
    )
    fasta.write_text(">1\nAAAAAAAAAAAAAAAAAAAA\n", encoding="utf-8")
    return model_dir, vcf, fasta


def _available_dependency(import_name: str, required: bool) -> DependencyProbe:
    return DependencyProbe(
        import_name=import_name,
        package=import_name.split(".", 1)[0],
        required=required,
        available=True,
        version="1.0.0",
        reason="available in test",
    )


def _missing_dependency(import_name: str, required: bool) -> DependencyProbe:
    return DependencyProbe(
        import_name=import_name,
        package=import_name.split(".", 1)[0],
        required=required,
        available=False,
        version=None,
        reason="missing in test",
    )


def _codes(report: RuntimePreflightReport) -> set[str]:
    return {issue.code for issue in report.issues}
