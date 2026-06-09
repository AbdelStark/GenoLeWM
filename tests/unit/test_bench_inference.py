"""Tests for the release inference benchmark wrapper."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from bench.inference import (
    ReleaseEfficiencyRequest,
    _count_vcf_alternates,
    write_release_efficiency_report,
)
from geno_lewm.errors import InputError
from geno_lewm.provenance import (
    SCHEMA_VERSION,
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    sha256_bytes,
    sha256_file,
    write_manifest,
)
from tools.release.efficiency_report import GENERATED_BY, load_efficiency_report


def test_release_efficiency_report_benchmarks_score_command(tmp_path: Path) -> None:
    model_dir = _write_model_dir(tmp_path / "model")
    vcf, fasta = _write_inputs(tmp_path)
    output = tmp_path / "efficiency_report.json"
    calls: list[tuple[str, ...]] = []
    rss = 100_000_000

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(command))
        if "--output" in command:
            output_path = Path(command[command.index("--output") + 1])
            output_path.write_text('{"sigma_calibrated":0.2}\n', encoding="utf-8")
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout='{"ok":true}\n',
            stderr="",
        )

    def memory_probe() -> int:
        nonlocal rss
        rss += 4096
        return rss

    path = write_release_efficiency_report(
        ReleaseEfficiencyRequest(
            model_dir=model_dir,
            vcf=vcf,
            fasta=fasta,
            output_json=output,
            variant="1:10:A:T",
            window="A" * 4096,
            window_start_bp=4,
            backend="cpu",
            samples=2,
            warmup_batches=1,
            commit_sha="abcdef1",
        ),
        runner=runner,
        memory_probe=memory_probe,
    )

    assert path == output
    report = load_efficiency_report(output)
    assert report.samples == 2
    assert report.warmup_batches == 1
    assert report.measurements.single_variant_latency_ms > 0
    assert report.measurements.batched_throughput_variants_per_s > 0
    assert report.measurements.peak_memory_bytes > 100_000_000
    inputs = dict(report.inputs)
    assert inputs["model_manifest"].path == "model/manifest.json"
    assert inputs["checkpoint"].path == "model/predictor.safetensors"
    assert inputs["vcf"].path == "benchmark_inputs/input.vcf"
    assert inputs["fasta"].path == "benchmark_inputs/ref.fa"
    assert inputs["single_window"].path == "inline:single_window"
    assert len(calls) == 6
    assert "--variant" in calls[0]
    assert "--window-start-bp" in calls[0]
    assert calls[0][calls[0].index("--window-start-bp") + 1] == "4"
    assert "--vcf" in calls[-1]
    assert "<redacted-inline-window>" in report.command
    assert "--window-start-bp" in report.command

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["generated_by"] == GENERATED_BY
    assert payload["dataset_snapshot"] == "geno-lewm-data-v0.1.0-r1"


def test_release_efficiency_report_rejects_failed_score_command(tmp_path: Path) -> None:
    model_dir = _write_model_dir(tmp_path / "model")
    vcf, fasta = _write_inputs(tmp_path)

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=2,
            stdout="",
            stderr="runtime missing",
        )

    with pytest.raises(InputError, match="score benchmark command failed"):
        write_release_efficiency_report(
            ReleaseEfficiencyRequest(
                model_dir=model_dir,
                vcf=vcf,
                fasta=fasta,
                output_json=tmp_path / "efficiency_report.json",
                variant="1:10:A:T",
                window="A" * 4096,
                samples=1,
                warmup_batches=0,
                commit_sha="abcdef1",
                peak_memory_bytes=100_000_000,
            ),
            runner=runner,
            memory_probe=lambda: 100_000_000,
        )


def test_count_vcf_alternates_counts_multiallelic_rows(tmp_path: Path) -> None:
    vcf = tmp_path / "input.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t10\t.\tA\tT,C\t.\tPASS\t.\n"
        "1\t20\t.\tG\t.\t.\tPASS\t.\n"
        "1\t30\t.\tC\tG\t.\tPASS\t.\n",
        encoding="utf-8",
    )

    assert _count_vcf_alternates(vcf) == 3


def _write_model_dir(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "predictor.safetensors").write_bytes(b"predictor")
    (root / "action_encoder.safetensors").write_bytes(b"action")
    (root / "calibration.parquet").write_bytes(b"calibration")
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


def _write_inputs(root: Path) -> tuple[Path, Path]:
    vcf = root / "input.vcf"
    fasta = root / "ref.fa"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t10\t.\tA\tT\t.\tPASS\t.\n",
        encoding="utf-8",
    )
    fasta.write_text(">1\nAAAAAAAAAAAAAAAAAAAA\n", encoding="utf-8")
    return vcf, fasta
