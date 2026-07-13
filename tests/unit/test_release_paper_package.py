"""Tests for the paper/demo release package verifier."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.release.dataset_package as dataset_package_module
from geno_lewm._artifact_sources import (
    CARBON_ZERO_SHOT_GENERATED_BY,
    CARBON_ZERO_SHOT_SCHEMA_VERSION,
    SCORE_JSONL_GENERATED_BY,
    SCORE_JSONL_SCHEMA_VERSION,
)
from geno_lewm.provenance import (
    SCHEMA_VERSION,
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    Receipt,
    ReceiptOutput,
    ReceiptProvenance,
    ReceiptRuntime,
    compute_output_commitment,
    sha256_bytes,
    sha256_file,
    write_manifest,
)
from geno_lewm.training.preflight import REPORT_NAME as TRAINING_PREFLIGHT_REPORT_NAME
from tests.unit.test_release_dataset_package import (
    _write_dataset_inputs,
    _write_dataset_snapshot_report,
)
from tests.unit.test_release_training_run import _write_training_run_inputs
from tools.demo.terminal_inference import (
    DEMO_MANIFEST_NAME,
    DemoArtifact,
    DemoRequest,
    write_demo_manifest,
)
from tools.release.batch_receipt_report import write_batch_receipt_report
from tools.release.dataset_integrity import DEFAULT_REPORT_NAME
from tools.release.dataset_package import build_dataset_package
from tools.release.dataset_snapshot import (
    GENERATED_BY as DATASET_SNAPSHOT_GENERATED_BY,
    INPUT_CHECK_GENERATED_BY as DATASET_INPUT_CHECK_GENERATED_BY,
    INPUT_CHECK_REPORT_NAME as DATASET_INPUT_CHECK_REPORT_NAME,
    REPORT_NAME as DATASET_SNAPSHOT_REPORT_NAME,
)
from tools.release.efficiency_report import (
    GENERATED_BY as EFFICIENCY_REPORT_GENERATED_BY,
    REPORT_NAME as EFFICIENCY_REPORT_NAME,
    parse_efficiency_report,
)
from tools.release.eval_report import parse_report_input, render_report
from tools.release.model_package import (
    GENERATED_BY as MODEL_PACKAGE_GENERATED_BY,
    MODEL_PACKAGE_NAME,
    build_model_package,
)
from tools.release.paper_draft import build_paper_draft
from tools.release.paper_package import (
    EVAL_METRICS_NAME,
    PackageIssue,
    PackagePaths,
    PackageReport,
    _verify_dataset_dir,
    _verify_eval_report,
    main,
    verify_package,
)
from tools.release.runtime_preflight import (
    DependencyProbe,
    RuntimePreflightRequest,
    write_runtime_preflight_report,
)
from tools.release.training_run import build_training_run_package


def test_verify_package_accepts_complete_artifact_set(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)

    report = verify_package(paths)

    assert report.ok is True
    assert report.model_id is not None
    assert report.issues == ()


def test_dataset_verifier_accepts_schema_1_1_role_bound_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_role_bound_dataset_verification_inputs(tmp_path, monkeypatch)
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert issues == []


def test_dataset_verifier_requires_schema_1_1_snapshot_membership_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_role_bound_dataset_verification_inputs(tmp_path, monkeypatch)
    report_path = tmp_path / DATASET_SNAPSHOT_REPORT_NAME
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    del report_payload["package"]["membership_and_split_evidence"]
    report_path.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_sha256sums(tmp_path, tmp_path / "SHA256SUMS")
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert "dataset.snapshot_report.package.membership_and_split_evidence" in {
        issue.code for issue in issues
    }


def test_dataset_verifier_requires_membership_evidence_data_card_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_role_bound_dataset_verification_inputs(tmp_path, monkeypatch)
    card_path = tmp_path / "data_card.md"
    card_path.write_text(
        card_path.read_text(encoding="utf-8").replace(
            "## Membership and Split Evidence",
            "## Bound Evidence",
        ),
        encoding="utf-8",
    )
    _refresh_sha256sums(tmp_path, tmp_path / "SHA256SUMS")
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert "dataset.card.section_missing" in {issue.code for issue in issues}


def test_dataset_verifier_keeps_schema_1_0_input_entry_shape_strict(tmp_path: Path) -> None:
    _write_dataset_dir(tmp_path)
    input_check_path = tmp_path / DATASET_INPUT_CHECK_REPORT_NAME
    input_check = json.loads(input_check_path.read_text(encoding="utf-8"))
    del input_check["inputs"][0]["description"]
    input_check_path.write_text(
        json.dumps(input_check, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    snapshot_path = tmp_path / DATASET_SNAPSHOT_REPORT_NAME
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["input_check"] = _relative_file_identity(
        tmp_path,
        DATASET_INPUT_CHECK_REPORT_NAME,
    )
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_sha256sums(tmp_path, tmp_path / "SHA256SUMS")
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert "dataset.snapshot_report.input_check.inputs" in {issue.code for issue in issues}


def test_dataset_verifier_rejects_schema_1_0_snapshot_membership_binding(
    tmp_path: Path,
) -> None:
    _write_dataset_dir(tmp_path)
    snapshot_path = tmp_path / DATASET_SNAPSHOT_REPORT_NAME
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["package"]["membership_and_split_evidence"] = {"unexpected": True}
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_sha256sums(tmp_path, tmp_path / "SHA256SUMS")
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert "dataset.snapshot_report.package.membership_and_split_evidence" in {
        issue.code for issue in issues
    }


def test_dataset_verifier_rejects_schema_1_1_snapshot_companion_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_role_bound_dataset_verification_inputs(tmp_path, monkeypatch)
    snapshot_path = tmp_path / DATASET_SNAPSHOT_REPORT_NAME
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    companion = next(
        item for item in snapshot["files"] if item["artifact_role"] == "split_companion"
    )
    companion["companion_of"] = "carbon/windows.jsonl"
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_sha256sums(tmp_path, tmp_path / "SHA256SUMS")
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert "dataset.snapshot_report.file.companion_of" in {issue.code for issue in issues}


def test_dataset_verifier_revalidates_schema_1_1_membership_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_role_bound_dataset_verification_inputs(tmp_path, monkeypatch)
    metadata_path = tmp_path / "dataset_package.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["membership_and_split_evidence"]["membership_store"]["content_identity"] = (
        "sha256:" + "f" * 64
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_sha256sums(tmp_path, tmp_path / "SHA256SUMS")
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    metadata_issues = [issue for issue in issues if issue.code == "dataset.metadata_invalid"]
    assert len(metadata_issues) == 1
    assert "membership store binding identity mismatch" in metadata_issues[0].message


def test_dataset_verifier_rejects_schema_1_1_input_check_role_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_role_bound_dataset_verification_inputs(tmp_path, monkeypatch)
    input_check_path = tmp_path / DATASET_INPUT_CHECK_REPORT_NAME
    input_check = json.loads(input_check_path.read_text(encoding="utf-8"))
    companion = next(
        item for item in input_check["inputs"] if item["artifact_role"] == "split_companion"
    )
    companion["artifact_role"] = "split_data"
    del companion["companion_of"]
    input_check_path.write_text(
        json.dumps(input_check, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    snapshot_path = tmp_path / DATASET_SNAPSHOT_REPORT_NAME
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["input_check"] = _relative_file_identity(
        tmp_path,
        DATASET_INPUT_CHECK_REPORT_NAME,
    )
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_sha256sums(tmp_path, tmp_path / "SHA256SUMS")
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert "dataset.snapshot_report.input_check.stale" in {issue.code for issue in issues}


def test_paper_package_main_outputs_json_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_complete_package(tmp_path)

    rc = main(
        [
            "--model-dir",
            str(paths.model_dir),
            "--dataset-dir",
            str(paths.dataset_dir),
            "--demo-dir",
            str(paths.demo_dir),
            "--paper-path",
            str(paths.paper_path),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out)["ok"] is True


def test_verify_package_rejects_stale_generated_paper_draft(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    assert paths.paper_path is not None
    paths.paper_path.write_text(
        paths.paper_path.read_text(encoding="utf-8").replace("eval_metrics.json", "metrics.json"),
        encoding="utf-8",
    )

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "paper.eval_metrics" in codes
    assert "paper.stale" in codes


def test_verify_package_requires_eval_config_in_generated_paper_draft(
    tmp_path: Path,
) -> None:
    paths = _write_complete_package(tmp_path)
    assert paths.paper_path is not None
    paths.paper_path.write_text(
        paths.paper_path.read_text(encoding="utf-8").replace(
            "eval_config.effective.yaml",
            "eval_config.yaml",
        ),
        encoding="utf-8",
    )

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "paper.eval_config" in codes
    assert "paper.stale" in codes


def test_verify_package_requires_dataset_snapshot_report_in_generated_paper_draft(
    tmp_path: Path,
) -> None:
    paths = _write_complete_package(tmp_path)
    assert paths.paper_path is not None
    paths.paper_path.write_text(
        paths.paper_path.read_text(encoding="utf-8").replace(
            "dataset_snapshot_report.json",
            "dataset_snapshot.json",
        ),
        encoding="utf-8",
    )

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "paper.dataset_snapshot_report" in codes
    assert "paper.stale" in codes


def test_verify_package_requires_dataset_input_check_report_in_generated_paper_draft(
    tmp_path: Path,
) -> None:
    paths = _write_complete_package(tmp_path)
    assert paths.paper_path is not None
    paths.paper_path.write_text(
        paths.paper_path.read_text(encoding="utf-8").replace(
            "dataset_input_check_report.json",
            "dataset_input_check.json",
        ),
        encoding="utf-8",
    )

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "paper.dataset_input_check_report" in codes
    assert "paper.stale" in codes


def test_verify_package_requires_model_package_in_generated_paper_draft(
    tmp_path: Path,
) -> None:
    paths = _write_complete_package(tmp_path)
    assert paths.paper_path is not None
    paths.paper_path.write_text(
        paths.paper_path.read_text(encoding="utf-8").replace(
            "model_package.json",
            "model_metadata.json",
        ),
        encoding="utf-8",
    )

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "paper.model_package" in codes
    assert "paper.stale" in codes


def test_verify_package_requires_citation_metadata_in_generated_paper_draft(
    tmp_path: Path,
) -> None:
    paths = _write_complete_package(tmp_path)
    assert paths.paper_path is not None
    paths.paper_path.write_text(
        paths.paper_path.read_text(encoding="utf-8").replace(
            "## Citation Metadata",
            "## Release Metadata",
        ),
        encoding="utf-8",
    )

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "paper.section_missing" in codes
    assert "paper.stale" in codes


def test_verify_package_requires_negative_findings_in_generated_paper_draft(
    tmp_path: Path,
) -> None:
    paths = _write_complete_package(tmp_path)
    assert paths.paper_path is not None
    paths.paper_path.write_text(
        paths.paper_path.read_text(encoding="utf-8").replace(
            "## Negative Findings",
            "## Non Results",
        ),
        encoding="utf-8",
    )

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "paper.section_missing" in codes
    assert "paper.stale" in codes


def test_verify_package_requires_utc_generated_paper_timestamp(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    assert paths.paper_path is not None
    paths.paper_path.write_text(
        paths.paper_path.read_text(encoding="utf-8").replace(
            "Generated: 2026-06-01T12:00:00Z",
            "Generated: June 1, 2026",
            1,
        ),
        encoding="utf-8",
    )

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "paper.generated_at" in codes
    assert "paper.render_failed" in codes


def test_verify_package_rejects_fixture_manifest_by_default(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path, release_id="geno-lewm-fixture-r1")

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "model.fixture_manifest" in codes
    assert "demo.runtime_preflight.fixture_manifest_allowed" in codes


def test_verify_package_can_allow_fixture_manifest_for_local_tests(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path, release_id="geno-lewm-fixture-r1")

    report = verify_package(paths, allow_fixture_manifest=True)

    assert report.ok is True


def test_verify_package_reports_missing_cards_and_bad_demo_transcript(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    (paths.model_dir / "model_card.md").unlink()
    (paths.dataset_dir / "data_card.md").unlink()
    (paths.demo_dir / "terminal-demo-transcript.md").write_text(
        "# GenoLeWM Terminal Inference Transcript\n\n- Status: failed\n",
        encoding="utf-8",
    )

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "model.card.missing" in codes
    assert "dataset.card.missing" in codes
    assert "demo.transcript.status" in codes
    assert "demo.transcript.model_id" in codes


def test_verify_package_rejects_transcript_missing_claim_boundary(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    transcript_path = paths.demo_dir / "terminal-demo-transcript.md"
    transcript_path.write_text(
        transcript_path.read_text(encoding="utf-8").replace(
            "This transcript records command behavior only. Model-quality claims require the "
            "published evaluation report linked from the release.",
            "This transcript records command behavior and model results.",
        ),
        encoding="utf-8",
    )
    _refresh_demo_manifest_artifacts(
        paths.demo_dir / DEMO_MANIFEST_NAME,
        artifacts={"terminal transcript": transcript_path},
    )

    report = verify_package(paths)

    assert report.ok is False
    assert "demo.transcript.claim_boundary" in _codes(report)


def test_verify_package_requires_model_package_metadata(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    (paths.model_dir / MODEL_PACKAGE_NAME).unlink()

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "model.metadata_missing" in codes
    assert "model.checksums.file_missing" in codes


def test_verify_package_requires_training_preflight_report(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    (paths.model_dir / TRAINING_PREFLIGHT_REPORT_NAME).unlink()

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "model.training_run.manifest_invalid" in codes
    assert "model.training_run.checksums.file_missing" in codes
    assert "model.checksums.file_missing" in codes


def test_verify_package_rejects_stale_model_card_metadata(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    card_path = paths.model_dir / "model_card.md"
    card_path.write_text(
        card_path.read_text(encoding="utf-8").replace(
            "Research-only GenoLeWM SNV scoring",
            "Research-only updated GenoLeWM SNV scoring",
        ),
        encoding="utf-8",
    )
    _write_sha256sums(paths.model_dir, _model_checksum_files(paths.model_dir))

    report = verify_package(paths)

    assert report.ok is False
    assert "model.card.stale" in _codes(report)


def test_verify_package_rejects_stale_training_run_card(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    card_path = paths.model_dir / "training_run_card.md"
    card_path.write_text(
        card_path.read_text(encoding="utf-8").replace(
            "Completed run archive for the first SNV predictor release path.",
            "Edited run archive summary that no longer matches the manifest.",
        ),
        encoding="utf-8",
    )
    _refresh_sha256sums(paths.model_dir, paths.model_dir / "training_run_SHA256SUMS")
    _write_sha256sums(paths.model_dir, _model_checksum_files(paths.model_dir))

    report = verify_package(paths)

    assert report.ok is False
    assert "model.training_run.card.stale" in _codes(report)


def test_verify_package_rejects_training_run_dataset_snapshot_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_complete_package(tmp_path)
    metadata_path = paths.model_dir / "training_run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset_snapshot_id"] = "other-dataset-snapshot"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    preflight_path = paths.model_dir / TRAINING_PREFLIGHT_REPORT_NAME
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["dataset_snapshot_id"] = "other-dataset-snapshot"
    preflight["dataset"]["snapshot_id"] = "other-dataset-snapshot"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_training_run_package(paths.model_dir, metadata_path)
    _write_sha256sums(paths.model_dir, _model_checksum_files(paths.model_dir))

    report = verify_package(paths)

    assert report.ok is False
    assert "model.training_run.dataset_snapshot_mismatch" in _codes(report)


def test_verify_package_rejects_training_run_config_path_mismatch(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    alternate_config = paths.model_dir / "alternate_train_config.yaml"
    alternate_config.write_text(
        (paths.model_dir / "train_config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    metadata_path = paths.model_dir / "training_run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_config"] = alternate_config.name
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    preflight_path = paths.model_dir / TRAINING_PREFLIGHT_REPORT_NAME
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["training_config"]["path"] = alternate_config.name
    preflight["training_config"]["sha256"] = sha256_file(alternate_config)
    preflight["training_config"]["size_bytes"] = alternate_config.stat().st_size
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_training_run_package(paths.model_dir, metadata_path)
    _write_sha256sums(paths.model_dir, _model_checksum_files(paths.model_dir))

    report = verify_package(paths)

    assert report.ok is False
    assert "model.training_run.training_config_path_mismatch" in _codes(report)


def test_verify_package_rejects_training_run_commit_mismatch(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    metadata_path = paths.model_dir / "training_run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["commit_sha"] = "bbbbbb1234567890"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_training_run_package(paths.model_dir, metadata_path)
    _write_sha256sums(paths.model_dir, _model_checksum_files(paths.model_dir))

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "model.training_run.eval_commit_mismatch" in codes
    assert "model.training_run.efficiency_commit_mismatch" in codes


def test_verify_package_requires_terminal_demo_manifest(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    (paths.demo_dir / DEMO_MANIFEST_NAME).unlink()

    report = verify_package(paths)

    assert report.ok is False
    assert "demo.manifest.missing" in _codes(report)


def test_verify_package_validates_terminal_demo_manifest_input_hashes(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    manifest_path = paths.demo_dir / DEMO_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["inputs"]["vcf"]["sha256"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = verify_package(paths)

    assert report.ok is False
    assert "demo.manifest.input.vcf.hash_mismatch" in _codes(report)


def test_verify_package_rejects_stale_terminal_demo_input_summary(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    manifest_path = paths.demo_dir / DEMO_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["inputs"]["vcf_summary"]["variant_records"] = 2
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = verify_package(paths)

    assert report.ok is False
    assert "demo.manifest.input.vcf_summary.stale" in _codes(report)


def test_verify_package_rejects_terminal_demo_manifest_score_receipt_summary_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_complete_package(tmp_path)
    manifest_path = paths.demo_dir / DEMO_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["score_receipt_batch"]["records"] = 2
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = verify_package(paths)

    assert report.ok is False
    assert "demo.manifest.score_receipt_batch.stale" in _codes(report)


def test_verify_package_rejects_terminal_demo_manifest_jsonl_field_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_complete_package(tmp_path)
    manifest_path = paths.demo_dir / DEMO_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in payload["artifacts"]:
        if artifact["label"] == "scores":
            artifact["jsonl_fields"] = ["chrom", "pos", "ref", "alt"]
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = verify_package(paths)

    assert report.ok is False
    assert "demo.manifest.artifact.field_mismatch" in _codes(report)


def test_verify_package_rejects_terminal_demo_manifest_private_inputs(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    private_vcf = tmp_path / "private-input.vcf"
    private_vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t10\t.\tA\tT\t.\tPASS\t.\n",
        encoding="utf-8",
    )
    manifest_path = paths.demo_dir / DEMO_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["inputs"]["vcf"] = _file_identity(private_vcf)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = verify_package(paths)

    assert report.ok is False
    assert "demo.manifest.input.vcf.outside_package" in _codes(report)


def test_verify_package_rejects_terminal_demo_manifest_noncanonical_output(
    tmp_path: Path,
) -> None:
    paths = _write_complete_package(tmp_path)
    alternate_scores = paths.demo_dir / "alternate_scores.jsonl"
    alternate_scores.write_text((paths.demo_dir / "scores.jsonl").read_text(encoding="utf-8"))
    manifest_path = paths.demo_dir / DEMO_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in payload["artifacts"]:
        if artifact["label"] == "scores":
            artifact.update(_file_identity(alternate_scores))
    payload["command"]["argv"][payload["command"]["argv"].index("--output") + 1] = str(
        alternate_scores
    )
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "demo.manifest.artifact.path_mismatch" in codes
    assert "demo.manifest.command.output_path" in codes


def test_verify_package_rejects_runtime_preflight_private_inputs(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    private_vcf = tmp_path / "private-runtime-input.vcf"
    private_vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t10\t.\tA\tT\t.\tPASS\t.\n",
        encoding="utf-8",
    )
    report_path = paths.demo_dir / "runtime_preflight_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["inputs"]["vcf"] = _file_identity(private_vcf)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = verify_package(paths)

    assert report.ok is False
    assert "demo.runtime_preflight.input.vcf.outside_package" in _codes(report)


def test_verify_package_rejects_runtime_preflight_fixture_allowance(
    tmp_path: Path,
) -> None:
    paths = _write_complete_package(tmp_path)
    report_path = paths.demo_dir / "runtime_preflight_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["requirements"]["fixture_manifest_allowed"] = True
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = verify_package(paths)

    assert report.ok is False
    assert "demo.runtime_preflight.fixture_manifest_allowed" in _codes(report)


def test_verify_package_rejects_runtime_preflight_private_command_inputs(
    tmp_path: Path,
) -> None:
    paths = _write_complete_package(tmp_path)
    private_vcf = tmp_path / "private-runtime-command-input.vcf"
    private_vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t10\t.\tA\tT\t.\tPASS\t.\n",
        encoding="utf-8",
    )
    report_path = paths.demo_dir / "runtime_preflight_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["command"]["argv"][payload["command"]["argv"].index("--vcf") + 1] = str(private_vcf)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = verify_package(paths)

    assert report.ok is False
    assert "demo.runtime_preflight.command.vcf_outside_package" in _codes(report)


def test_verify_package_rejects_runtime_preflight_command_drift(
    tmp_path: Path,
) -> None:
    paths = _write_complete_package(tmp_path)
    report_path = paths.demo_dir / "runtime_preflight_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    argv = payload["command"]["argv"]
    argv[argv.index("--backend") + 1] = "cuda"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = verify_package(paths)

    assert report.ok is False
    assert "demo.manifest.command.runtime_preflight_mismatch" in _codes(report)


def test_verify_package_requires_terminal_demo_manifest_runtime_preflight_summary(
    tmp_path: Path,
) -> None:
    paths = _write_complete_package(tmp_path)
    manifest_path = paths.demo_dir / DEMO_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["runtime_preflight"]
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = verify_package(paths)

    assert report.ok is False
    assert "demo.manifest.runtime_preflight" in _codes(report)


def test_verify_package_rejects_stale_terminal_demo_manifest_runtime_preflight_summary(
    tmp_path: Path,
) -> None:
    paths = _write_complete_package(tmp_path)
    manifest_path = paths.demo_dir / DEMO_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["runtime_preflight"]["command"]["argv"][-1] = "999"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = verify_package(paths)

    assert report.ok is False
    assert "demo.manifest.runtime_preflight.stale" in _codes(report)


def test_verify_package_rejects_batch_receipt_report_model_mismatch(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    manifest = _load_manifest(paths.model_dir)

    _rewrite_demo_receipts(
        paths,
        model_id=sha256_bytes(b"other-model"),
        calibration_hash=manifest.calibration.hash,
    )

    report = verify_package(paths)

    assert report.ok is False
    assert "demo.batch_receipt_report.model_id" in _codes(report)


def test_verify_package_rejects_batch_receipt_report_calibration_mismatch(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    manifest = _load_manifest(paths.model_dir)

    _rewrite_demo_receipts(
        paths,
        model_id=manifest.model_id(),
        calibration_hash=sha256_bytes(b"other-calibration"),
    )

    report = verify_package(paths)

    assert report.ok is False
    assert "demo.batch_receipt_report.calibration_hash" in _codes(report)


def test_verify_package_reports_manifest_and_dataset_hash_mismatches(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    (paths.model_dir / "predictor.safetensors").write_bytes(b"tampered predictor")
    (paths.dataset_dir / "clinvar" / "eval.vcf").write_text("tampered\n", encoding="utf-8")

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "model.artifact_hash_mismatch" in codes
    assert "model.checksums.hash_mismatch" in codes
    assert "dataset.file.hash_mismatch" in codes
    assert "dataset.checksums.hash_mismatch" in codes


def test_verify_package_rejects_placeholder_eval_report(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    (paths.model_dir / "eval_report.md").write_text(
        "# Evaluation Report\n\n"
        "Generated by: geno-lewm-eval-all\n\n"
        "## Results\nTBD.\n\n"
        "## Limitations\nKnown limitations.\n\n"
        "## Negative Findings\nNegative findings are explicitly reported.\n\n"
        "## Conclusions\nMeasured conclusions.\n",
        encoding="utf-8",
    )
    _write_sha256sums(
        paths.model_dir,
        (
            "manifest.json",
            "predictor.safetensors",
            "action_encoder.safetensors",
            "calibration.parquet",
            "train_config.yaml",
            "eval_report.md",
            EVAL_METRICS_NAME,
            EFFICIENCY_REPORT_NAME,
            "model_card.md",
        ),
    )

    report = verify_package(paths)

    assert report.ok is False
    assert "model.artifact_hash_mismatch" in _codes(report)
    assert "model.eval_report.placeholder" in _codes(report)


def test_verify_eval_report_requires_generated_artifact_table(tmp_path: Path) -> None:
    report_path = tmp_path / "eval_report.md"
    metrics_path = tmp_path / EVAL_METRICS_NAME
    metrics_path.write_text(json.dumps(_eval_metrics_payload()), encoding="utf-8")
    report_path.write_text(
        "# Evaluation Report\n\n"
        "Generated by: geno-lewm-eval-all\n\n"
        "## Summary\n"
        f"- Model id: {sha256_bytes(b'model')}\n"
        "- Dataset snapshot: data-v1\n"
        "- Result status: measured metrics from the input JSON payload.\n"
        "- Claim boundary: planned targets are not reported as results.\n\n"
        "## Results\nMeasured AUROC table.\n\n"
        "## Artifacts\nNo table.\n\n"
        "## Limitations\nKnown limitations.\n\n"
        "## Negative Findings\nNegative findings are explicitly reported.\n\n"
        "## Conclusions\nMeasured conclusions.\n",
        encoding="utf-8",
    )
    issues: list[PackageIssue] = []

    _verify_eval_report(report_path, metrics_path, issues)

    codes = {issue.code for issue in issues}
    assert "model.eval_report.artifact.checkpoint" in codes
    assert "model.eval_report.artifact.config" in codes
    assert "model.eval_report.artifact.dataset_manifest" in codes
    assert "model.eval_report.artifact.eval_config" in codes
    assert "model.eval_report.artifact.efficiency_report" in codes


def test_verify_eval_report_requires_baseline_score_artifact_row(tmp_path: Path) -> None:
    report_path = tmp_path / "eval_report.md"
    metrics_path = tmp_path / EVAL_METRICS_NAME
    metrics_path.write_text(json.dumps(_eval_metrics_payload(baseline=True)), encoding="utf-8")
    report_path.write_text(
        _generated_eval_report_text(
            baseline_row=(
                "| auroc | clinvar_coding | 0.73 | area | carbon_zero_shot | "
                "0.70 | 0.03 | higher | 0.70 to 0.76 | 1200 | measured |"
            ),
            baseline_artifact=False,
        ),
        encoding="utf-8",
    )
    issues: list[PackageIssue] = []

    _verify_eval_report(report_path, metrics_path, issues)

    assert "model.eval_report.baseline_artifact_missing" in {issue.code for issue in issues}


def test_verify_package_requires_eval_metrics_json(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    (paths.model_dir / EVAL_METRICS_NAME).unlink()

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "model.eval_metrics.missing" in codes
    assert "model.checksums.file_missing" in codes


def test_verify_package_requires_efficiency_report(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    (paths.model_dir / EFFICIENCY_REPORT_NAME).unlink()

    report = verify_package(paths)

    codes = _codes(report)
    assert report.ok is False
    assert "model.efficiency_report.missing" in codes
    assert "model.checksums.file_missing" in codes


def test_verify_package_rejects_stale_eval_report(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    report_path = paths.model_dir / "eval_report.md"
    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace("0.73", "0.74", 1),
        encoding="utf-8",
    )
    _write_sha256sums(
        paths.model_dir,
        (
            "manifest.json",
            "predictor.safetensors",
            "action_encoder.safetensors",
            "calibration.parquet",
            "train_config.yaml",
            "eval_report.md",
            EVAL_METRICS_NAME,
            EFFICIENCY_REPORT_NAME,
            "model_card.md",
        ),
    )

    report = verify_package(paths)

    assert report.ok is False
    assert "model.artifact_hash_mismatch" in _codes(report)
    assert "model.eval_report.stale" in _codes(report)


def test_verify_package_rejects_eval_metrics_release_mismatch(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    metrics_path = paths.model_dir / EVAL_METRICS_NAME
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["model_release"] = "other-release"
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (paths.model_dir / "eval_report.md").write_text(
        render_report(parse_report_input(payload)),
        encoding="utf-8",
    )

    report = verify_package(paths)

    assert report.ok is False
    assert "model.eval_metrics.model_release_mismatch" in _codes(report)


def test_verify_package_rejects_efficiency_dataset_mismatch(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    report_path = paths.model_dir / EFFICIENCY_REPORT_NAME
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["dataset_snapshot"] = "other-dataset"
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_package(paths)

    assert report.ok is False
    assert "model.efficiency_report.dataset_snapshot_mismatch" in _codes(report)


def test_verify_package_rejects_missing_eval_score_artifact(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    (paths.model_dir / "eval" / "scores.jsonl").unlink()

    report = verify_package(paths)

    assert report.ok is False
    codes = _codes(report)
    assert "model.eval_artifact.scores.missing" in codes
    assert "model.checksums.file_missing" in codes


def test_verify_package_requires_eval_score_checksum_entry(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    checksum_path = paths.model_dir / "SHA256SUMS"
    checksum_path.write_text(
        "\n".join(
            line
            for line in checksum_path.read_text(encoding="utf-8").splitlines()
            if not line.endswith("  eval/scores.jsonl")
        )
        + "\n",
        encoding="utf-8",
    )

    report = verify_package(paths)

    assert report.ok is False
    assert "model.checksums.entry_missing" in _codes(report)


def test_verify_package_rejects_duplicate_checksum_entries(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    checksum_path = paths.model_dir / "SHA256SUMS"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    checksum_path.write_text("\n".join((*lines, lines[0])) + "\n", encoding="utf-8")

    report = verify_package(paths)

    assert report.ok is False
    assert "model.checksums.invalid" in _codes(report)


def test_verify_package_rejects_absolute_eval_artifact_path(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    metrics_path = paths.model_dir / EVAL_METRICS_NAME
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["artifacts"]["eval_config"] = str(tmp_path / "outside" / "eval_config.yaml")
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (paths.model_dir / "eval_report.md").write_text(
        render_report(parse_report_input(payload)),
        encoding="utf-8",
    )

    report = verify_package(paths)

    assert report.ok is False
    assert "model.eval_artifact.eval_config.path" in _codes(report)


def test_verify_package_requires_eval_metrics_eval_config_artifact(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    metrics_path = paths.model_dir / EVAL_METRICS_NAME
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    del payload["artifacts"]["eval_config"]
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_package(paths)

    assert report.ok is False
    codes = _codes(report)
    assert "model.eval_metrics.invalid" in codes
    assert "paper.render_failed" in codes


def test_verify_package_requires_eval_report_eval_config_artifact(tmp_path: Path) -> None:
    paths = _write_complete_package(tmp_path)
    report_path = paths.model_dir / "eval_report.md"
    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace(
            "| eval_config | eval_config.effective.yaml |\n",
            "",
        ),
        encoding="utf-8",
    )

    report = verify_package(paths)

    assert report.ok is False
    codes = _codes(report)
    assert "model.eval_report.artifact.eval_config" in codes
    assert "model.eval_report.stale" in codes


def _write_complete_package(
    root: Path,
    *,
    release_id: str = "geno-lewm-v0.1.0-r1",
) -> PackagePaths:
    model_dir = root / "model"
    dataset_dir = root / "dataset"
    demo_dir = root / "demo"
    paper_path = root / "paper.md"
    model_dir.mkdir()
    dataset_dir.mkdir()
    demo_dir.mkdir()

    _write_model_dir(model_dir, release_id=release_id)
    _write_dataset_dir(dataset_dir)
    manifest = _load_manifest(model_dir)
    vcf = demo_dir / "input.vcf"
    fasta = demo_dir / "ref.fa"
    _write_demo_inputs(vcf, fasta)
    runtime_preflight = write_runtime_preflight_report(
        RuntimePreflightRequest(
            model_dir=model_dir,
            vcf=vcf,
            fasta=fasta,
            output_dir=demo_dir,
            backend="cpu",
            allow_fixture_manifest="fixture" in release_id,
        ),
        demo_dir / "runtime_preflight_report.json",
        generated_at="2026-06-01T12:00:00Z",
        dependency_probe=_available_dependency,
    )
    assert runtime_preflight.ok is True
    scores = demo_dir / "scores.jsonl"
    receipts = demo_dir / "receipts.jsonl"
    receipt_output = ReceiptOutput(
        sigma_raw=0.1,
        sigma_calibrated=0.2,
        bucket_id="coding_missense|mid|none",
        confidence=0.9,
        low_confidence=False,
    )
    scores.write_text(
        json.dumps(
            {
                "schema_version": SCORE_JSONL_SCHEMA_VERSION,
                "generated_by": SCORE_JSONL_GENERATED_BY,
                "chrom": "1",
                "pos": 10,
                "ref": "A",
                "alt": "T",
                "sigma_raw": receipt_output.sigma_raw,
                "sigma_calibrated": receipt_output.sigma_calibrated,
                "bucket_id": receipt_output.bucket_id,
                "confidence": receipt_output.confidence,
                "low_confidence": receipt_output.low_confidence,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    receipts.write_text(
        _receipt_json(
            model_id=manifest.model_id(),
            calibration_hash=manifest.calibration.hash,
            output=receipt_output,
            row_index=1,
        )
        + "\n",
        encoding="utf-8",
    )
    batch_report = write_batch_receipt_report(
        scores,
        receipts,
        demo_dir / "batch_receipt_report.json",
        generated_at="2026-06-01T12:00:00Z",
    )
    (demo_dir / "terminal-demo-transcript.md").write_text(
        "\n".join(
            [
                "# GenoLeWM Terminal Inference Transcript",
                "",
                "- Generated: 2026-06-01T12:00:00Z",
                "- Status: passed",
                "- Exit code: 0",
                f"- Model release: {manifest.release_id}",
                f"- Model version: {manifest.model_version}",
                f"- Model id: {manifest.model_id()}",
                "- Input VCF records: 1",
                "- Input alternate alleles: 1",
                "- Input contigs: 1",
                "- First input variants: 1:10:A>T",
                "- Scores: demo/scores.jsonl",
                "- Receipts: demo/receipts.jsonl",
                "- Runtime preflight report: demo/runtime_preflight_report.json",
                "- Batch receipt report: demo/batch_receipt_report.json",
                "- Demo manifest: demo/terminal_demo_manifest.json",
                "",
                "## Command",
                "",
                "```console",
                "$ geno-lewm-score --model-dir model --vcf input.vcf --fasta ref.fa",
                "```",
                "",
                "## Output Artifacts",
                "",
                "| Artifact | Path | SHA-256 | Bytes | JSONL rows |",
                "| --- | --- | --- | ---: | ---: |",
                f"| scores | demo/scores.jsonl | {sha256_file(scores)} | {scores.stat().st_size} | 1 |",
                f"| receipts | demo/receipts.jsonl | {sha256_file(receipts)} | {receipts.stat().st_size} | 1 |",
                (
                    "| runtime preflight report | demo/runtime_preflight_report.json | "
                    f"{sha256_file(demo_dir / 'runtime_preflight_report.json')} | "
                    f"{(demo_dir / 'runtime_preflight_report.json').stat().st_size} | - |"
                ),
                (
                    "| batch receipt report | demo/batch_receipt_report.json | "
                    f"{sha256_file(batch_report)} | {batch_report.stat().st_size} | - |"
                ),
                f"- Scores SHA-256: {sha256_file(scores)}",
                "- Scores JSONL rows: 1",
                f"- Scores JSONL fields: {_jsonl_field_list(scores)}",
                f"- Receipts SHA-256: {sha256_file(receipts)}",
                "- Receipts JSONL rows: 1",
                f"- Receipts JSONL fields: {_jsonl_field_list(receipts)}",
                f"- Runtime Preflight Report SHA-256: {sha256_file(demo_dir / 'runtime_preflight_report.json')}",
                f"- Batch Receipt Report SHA-256: {sha256_file(batch_report)}",
                "",
                "## Score And Receipt Summary",
                "",
                "- Records: 1",
                f"- Score fields: {_jsonl_field_list(scores)}",
                f"- Receipt fields: {_jsonl_field_list(receipts)}",
                "- Checked score fields: sigma_raw, sigma_calibrated, bucket_id, confidence, low_confidence",
                "- Receipt stream: jsonl_per_scored_alternate_v1",
                "- Receipt schema: 1.0.0",
                f"- Receipt model id: {manifest.model_id()}",
                f"- Calibration hash: {manifest.calibration.hash}",
                "- Runtime backend: cpu",
                "- Runtime device: CPU",
                "",
                "## Artifact Inputs",
                "",
                "- Model directory: model",
                "- Manifest: model/manifest.json",
                "- VCF: demo/input.vcf",
                "- FASTA: demo/ref.fa",
                "",
                "This transcript records command behavior only. Model-quality claims require the "
                "published evaluation report linked from the release.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_terminal_demo_manifest(
        model_dir=model_dir,
        demo_dir=demo_dir,
        vcf=vcf,
        fasta=fasta,
        scores=scores,
        receipts=receipts,
        batch_report=batch_report,
        manifest=manifest,
    )
    build_paper_draft(
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        demo_dir=demo_dir,
        output=paper_path,
        generated_at="2026-06-01T12:00:00Z",
    )
    return PackagePaths(
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        demo_dir=demo_dir,
        paper_path=paper_path,
    )


def _write_model_dir(root: Path, *, release_id: str) -> None:
    training_metadata = _write_training_run_inputs(root)
    _write_eval_report_artifacts(root, release_id=release_id)
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        model_name="geno-lewm",
        model_version="0.1.0",
        release_id=release_id,
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
    build_training_run_package(root, training_metadata)
    metadata_path = root / "model_release_metadata.json"
    metadata_path.write_text(
        json.dumps(_model_metadata(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_model_package(
        root,
        metadata_path,
        allow_fixture_manifest="fixture" in release_id,
    )


def _model_metadata() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "generated_by": MODEL_PACKAGE_GENERATED_BY,
        "generated_at": "2026-06-01T12:00:00Z",
        "summary": "SNV predictor checkpoint for the first GenoLeWM paper/demo release.",
        "data": [
            "Training data snapshot: geno-lewm-data-v0.1.0-r1.",
            "Evaluation data is documented in eval_report.md.",
        ],
        "hardware": ["Local CPU fixture package; final release records training accelerator."],
        "license": "Apache-2.0 for GenoLeWM metadata; upstream encoder and data terms apply.",
        "intended_use": "Research-only GenoLeWM SNV scoring and reproducibility experiments.",
        "limitations": [
            "Not a clinical diagnostic model.",
            "Performance claims are limited to the generated evaluation report.",
        ],
        "training": [
            "Configuration is recorded in train_config.yaml.",
            "Manifest records all checkpoint artifact hashes.",
        ],
        "evaluation": [
            "eval_report.md is generated from measured metrics JSON.",
            "The model card does not restate benchmark claims.",
        ],
        "runtime": [
            "Load through GenoLeWM runtime with manifest verification.",
            "Requires compatible Carbon encoder revision from the manifest.",
        ],
        "release_notes": [
            "Publish with dataset snapshot and terminal demo transcript links.",
            "Run tools.release.paper_package before uploading artifacts.",
        ],
        "extra_files": [
            TRAINING_PREFLIGHT_REPORT_NAME,
            "training_run_manifest.json",
            "training_run_card.md",
            "training_run_SHA256SUMS",
        ],
    }


def _model_checksum_files(model_dir: Path) -> tuple[str, ...]:
    manifest = _load_manifest(model_dir)
    return (
        "manifest.json",
        MODEL_PACKAGE_NAME,
        "model_card.md",
        manifest.predictor.file,
        manifest.action_encoder.file,
        manifest.calibration.file,
        manifest.training.config_file,
        manifest.eval.file,
        EVAL_METRICS_NAME,
        EFFICIENCY_REPORT_NAME,
        "eval/scores.jsonl",
        TRAINING_PREFLIGHT_REPORT_NAME,
        "training_run_manifest.json",
        "training_run_card.md",
        "training_run_SHA256SUMS",
    )


def _write_dataset_dir(root: Path) -> None:
    metadata_path = _write_dataset_inputs(root)
    build_dataset_package(root, metadata_path)
    _write_dataset_snapshot_report(root)


def _write_schema_1_1_dataset_snapshot_report(root: Path) -> None:
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    snapshot_files: list[dict[str, object]] = []
    for manifest_file in manifest["files"]:
        path = root / manifest_file["path"]
        snapshot_file = {
            "path": manifest_file["path"],
            "source_path": f"sources/{manifest_file['path']}",
            "source_sha256": sha256_file(path),
            "source_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "already_exists": False,
        }
        snapshot_file.update(
            {
                key: manifest_file[key]
                for key in (
                    "split",
                    "records",
                    "artifact_role",
                    "companion_of",
                    "description",
                )
                if key in manifest_file
            }
        )
        snapshot_files.append(snapshot_file)

    inputs = []
    for item in snapshot_files:
        input_item = {
            "kind": "dataset_artifact",
            "source_path": item["source_path"],
            "staged_path": item["path"],
            "sha256": item["source_sha256"],
            "size_bytes": item["source_size_bytes"],
        }
        input_item.update(
            {
                key: item[key]
                for key in ("split", "artifact_role", "companion_of", "description")
                if key in item
            }
        )
        inputs.append(input_item)
    input_check_payload = {
        "ok": True,
        "schema_version": "1.0.0",
        "generated_by": DATASET_INPUT_CHECK_GENERATED_BY,
        "snapshot_id": manifest["snapshot_id"],
        "snapshot_spec": {
            "path": "dataset_snapshot.json",
            "sha256": "sha256:" + "1" * 64,
            "size_bytes": 1024,
        },
        "source_count": len(inputs),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in inputs),
        "inputs": inputs,
    }
    (root / DATASET_INPUT_CHECK_REPORT_NAME).write_text(
        json.dumps(input_check_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    package_files = [
        {
            key: item[key]
            for key in (
                "path",
                "sha256",
                "size_bytes",
                "split",
                "records",
                "artifact_role",
                "companion_of",
                "description",
            )
            if key in item
        }
        for item in snapshot_files
    ]
    snapshot_payload = {
        "schema_version": "1.0.0",
        "generated_by": DATASET_SNAPSHOT_GENERATED_BY,
        "generated_at": manifest["generated_at"],
        "snapshot_id": manifest["snapshot_id"],
        "report_path": DATASET_SNAPSHOT_REPORT_NAME,
        "snapshot_spec": input_check_payload["snapshot_spec"],
        "input_check_path": DATASET_INPUT_CHECK_REPORT_NAME,
        "input_check": _relative_file_identity(root, DATASET_INPUT_CHECK_REPORT_NAME),
        "metadata_path": "dataset_package.json",
        "package": {
            "snapshot_id": manifest["snapshot_id"],
            "metadata": _relative_file_identity(root, "dataset_package.json"),
            "manifest_path": "dataset_manifest.json",
            "manifest": _relative_file_identity(root, "dataset_manifest.json"),
            "data_card_path": "data_card.md",
            "data_card": _relative_file_identity(root, "data_card.md"),
            "integrity_path": DEFAULT_REPORT_NAME,
            "integrity": _relative_file_identity(root, DEFAULT_REPORT_NAME),
            "checksums_path": "SHA256SUMS",
            "files": package_files,
            "membership_and_split_evidence": manifest["membership_and_split_evidence"],
        },
        "files": snapshot_files,
    }
    (root / DATASET_SNAPSHOT_REPORT_NAME).write_text(
        json.dumps(snapshot_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_sha256sums(
        root,
        (
            "data_card.md",
            "dataset_package.json",
            "dataset_manifest.json",
            DEFAULT_REPORT_NAME,
            DATASET_INPUT_CHECK_REPORT_NAME,
            DATASET_SNAPSHOT_REPORT_NAME,
            *(file["path"] for file in manifest["files"]),
        ),
    )


def _write_role_bound_dataset_verification_inputs(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_path = _write_dataset_inputs(root)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    companion_path = root / "clinvar" / "eval-copy.vcf"
    companion_path.write_bytes((root / "clinvar" / "eval.vcf").read_bytes())
    store_root = root / "membership" / "store"
    store_root.mkdir(parents=True)
    for name in (
        "manifest.json",
        "memberships.parquet",
        "lookup.sqlite",
        "snapshot-lineage.json",
        "build-receipt.json",
    ):
        (store_root / name).write_text(name + "\n", encoding="utf-8")
    schema_path = root / "contract" / "membership-split-evidence.schema.json"
    schema_path.parent.mkdir()
    schema_path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": [
                    "artifact_id",
                    "schema_version",
                    "membership_store",
                    "training_windows",
                    "streams",
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    membership_manifest = SimpleNamespace(
        artifact_id="fixture-membership",
        content_identity="sha256:" + "1" * 64,
        physical_identity="sha256:" + "2" * 64,
        rowset_sha256="sha256:" + "3" * 64,
    )
    report_path = root / "evidence" / "membership-split-evidence.json"
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "artifact_id": "fixture-split-evidence",
                "schema_version": "fixture.membership-split-evidence.v1",
                "membership_store": {
                    "artifact_id": membership_manifest.artifact_id,
                    "content_identity": membership_manifest.content_identity,
                    "physical_identity": membership_manifest.physical_identity,
                    "rowset_sha256": membership_manifest.rowset_sha256,
                },
                "training_windows": {
                    "source": {"artifact_path": "carbon/windows.jsonl"},
                    "sha256": sha256_file(root / "carbon" / "windows.jsonl"),
                    "size_bytes": (root / "carbon" / "windows.jsonl").stat().st_size,
                    "record_count": 1,
                    "split": "train",
                },
                "streams": {
                    "evaluation": {
                        "role": "evaluation",
                        "record_count": 1,
                        "labels_jsonl": _relative_file_identity(root, "clinvar/eval.vcf"),
                        "vcf": _relative_file_identity(root, "clinvar/eval-copy.vcf"),
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    payload["schema_version"] = "1.1.0"
    payload["splits"]["evaluation"] = payload["splits"].pop("eval_clinvar_coding")
    files = payload["files"]
    files[0]["artifact_role"] = "split_data"
    files[1]["artifact_role"] = "split_data"
    files[1]["split"] = "evaluation"
    files.append(
        {
            "path": "clinvar/eval-copy.vcf",
            "artifact_role": "split_companion",
            "split": "evaluation",
            "records": 1,
            "companion_of": "clinvar/eval.vcf",
            "description": "alternate evaluation encoding",
        }
    )
    evidence_paths = [
        *(
            f"membership/store/{name}"
            for name in (
                "manifest.json",
                "memberships.parquet",
                "lookup.sqlite",
                "snapshot-lineage.json",
                "build-receipt.json",
            )
        ),
        "evidence/membership-split-evidence.json",
        "contract/membership-split-evidence.schema.json",
    ]
    files.extend({"path": path, "artifact_role": "evidence"} for path in evidence_paths)
    payload["membership_and_split_evidence"] = {
        "membership_store": {
            "path": "membership/store",
            "artifact_id": membership_manifest.artifact_id,
            "content_identity": membership_manifest.content_identity,
            "physical_identity": membership_manifest.physical_identity,
            "rowset_sha256": membership_manifest.rowset_sha256,
        },
        "report": {
            "path": "evidence/membership-split-evidence.json",
            "schema_path": "contract/membership-split-evidence.schema.json",
            "artifact_id": "fixture-split-evidence",
            "schema_version": "fixture.membership-split-evidence.v1",
        },
    }

    class _Store:
        def __init__(self) -> None:
            self.manifest = membership_manifest

        def __enter__(self) -> _Store:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        dataset_package_module,
        "MembershipStore",
        SimpleNamespace(open=lambda *_args, **_kwargs: _Store()),
    )
    monkeypatch.setattr(
        dataset_package_module,
        "MEMBERSHIP_SPLIT_EVIDENCE_SCHEMA_SHA256",
        sha256_file(schema_path),
    )
    monkeypatch.setattr(
        dataset_package_module,
        "MEMBERSHIP_SPLIT_EVIDENCE_SCHEMA_VERSION",
        "fixture.membership-split-evidence.v1",
    )
    monkeypatch.setattr(
        dataset_package_module,
        "_require_publication_eligible_split_evidence",
        lambda _report: None,
    )
    metadata_path = root / "dataset_package.json"
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_dataset_package(root, metadata_path)
    _write_schema_1_1_dataset_snapshot_report(root)


def _relative_file_identity(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_demo_inputs(vcf: Path, fasta: Path) -> None:
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t10\t.\tA\tT\t.\tPASS\t.\n",
        encoding="utf-8",
    )
    fasta.write_text(">1\nAAAAAAAAAAAAAAAAAAAA\n", encoding="utf-8")


def _generated_eval_report_text(
    *,
    baseline_row: str | None = None,
    baseline_artifact: bool = True,
) -> str:
    metric_row = baseline_row or (
        "| auroc | clinvar_coding | 0.73 | area | not reported | - | - | higher | "
        "0.70 to 0.76 | 1200 | measured |"
    )
    artifacts = [
        "| checkpoint | model/predictor.safetensors |",
        "| config | model/train_config.yaml |",
        "| dataset_manifest | dataset/dataset_manifest.json |",
        "| eval_config | eval_config.effective.yaml |",
        "| efficiency_report | model/efficiency_report.json |",
    ]
    if baseline_artifact:
        artifacts.append("| baseline_scores | eval/carbon_zero_shot_scores.jsonl |")
    return (
        "# Evaluation Report\n\n"
        "Generated by: geno-lewm-eval-all\n"
        "Generated: 2026-06-01T12:00:00Z\n\n"
        "## Summary\n\n"
        "- Model release: geno-lewm-v0.1.0-r1\n"
        f"- Model id: {sha256_bytes(b'eval-report-model')}\n"
        "- Dataset snapshot: geno-lewm-data-v0.1.0-r1\n"
        "- Commit: abcdef1234567890\n"
        "- Hardware: local CPU fixture\n"
        "- Result status: measured metrics from the input JSON payload.\n"
        "- Claim boundary: planned targets are not reported as results.\n\n"
        "## Results\n\n"
        "| Metric | Split | Value | Unit | Baseline | Baseline Value | Delta vs Baseline | Direction | CI | N | Notes |\n"
        "| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- | ---: | --- |\n"
        f"{metric_row}\n\n"
        "## Artifacts\n\n"
        "| Artifact | Path or identifier |\n"
        "| --- | --- |\n" + "\n".join(artifacts) + "\n\n"
        "## Limitations\n\n"
        "- Known limitations.\n\n"
        "## Negative Findings\n\n"
        "- Negative findings are explicitly reported.\n\n"
        "## Conclusions\n\n"
        "- Measured conclusions.\n"
    )


def _write_eval_report_artifacts(
    root: Path,
    *,
    baseline: bool = False,
    release_id: str = "geno-lewm-v0.1.0-r1",
) -> None:
    _write_efficiency_report(root, release_id=release_id)
    _write_eval_score_artifacts(root, baseline=baseline)
    (root / "eval_config.effective.yaml").write_text(
        "metrics:\n  aggregate: true\n",
        encoding="utf-8",
    )
    payload = _eval_metrics_payload(baseline=baseline, release_id=release_id)
    (root / EVAL_METRICS_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = render_report(parse_report_input(payload))
    (root / "eval_report.md").write_text(report, encoding="utf-8")


def _write_eval_score_artifacts(root: Path, *, baseline: bool = False) -> None:
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        eval_dir / "scores.jsonl",
        [
            {
                "schema_version": SCORE_JSONL_SCHEMA_VERSION,
                "generated_by": SCORE_JSONL_GENERATED_BY,
                "chrom": "1",
                "pos": 10,
                "ref": "A",
                "alt": "T",
                "sigma_calibrated": 0.73,
            }
        ],
    )
    if baseline:
        _write_jsonl(
            eval_dir / "carbon_zero_shot_scores.jsonl",
            [
                {
                    "schema_version": CARBON_ZERO_SHOT_SCHEMA_VERSION,
                    "generated_by": CARBON_ZERO_SHOT_GENERATED_BY,
                    "chrom": "1",
                    "pos": 10,
                    "ref": "A",
                    "alt": "T",
                    "carbon_zero_shot_score": 0.70,
                }
            ],
        )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _eval_metrics_payload(
    *,
    baseline: bool = False,
    release_id: str = "geno-lewm-v0.1.0-r1",
) -> dict[str, object]:
    variant_keys_hash = sha256_bytes(b"clinvar-coding-variant-keys")
    metric: dict[str, object] = {
        "name": "auroc",
        "value": 0.73,
        "split": "clinvar_coding",
        "unit": "area",
        "higher_is_better": True,
        "ci_low": 0.70,
        "ci_high": 0.76,
        "n": 1200,
        "notes": "measured",
    }
    artifacts = {
        "checkpoint": "model/predictor.safetensors",
        "config": "model/train_config.yaml",
        "dataset_manifest": "dataset/dataset_manifest.json",
        "eval_config": "eval_config.effective.yaml",
        "efficiency_report": f"model/{EFFICIENCY_REPORT_NAME}",
        "scores": "eval/scores.jsonl",
    }
    if baseline:
        metric.update(
            {
                "baseline": "carbon_zero_shot",
                "baseline_value": 0.70,
                "delta_vs_baseline": 0.03,
                "evaluated_variant_keys_sha256": variant_keys_hash,
                "baseline_evaluated_variant_keys_sha256": variant_keys_hash,
            }
        )
        artifacts["baseline_scores"] = "eval/carbon_zero_shot_scores.jsonl"
    conclusion = "The auroc metric value 0.73 on clinvar_coding was measured."
    if baseline:
        conclusion = (
            "The auroc metric value 0.73 on clinvar_coding has delta 0.03 versus carbon_zero_shot."
        )
    return {
        "schema_version": "1.0.0",
        "generated_by": "geno-lewm-eval-all",
        "generated_at": "2026-06-01T12:00:00Z",
        "model_id": sha256_bytes(b"eval-report-model"),
        "model_release": release_id,
        "dataset_snapshot": "geno-lewm-data-v0.1.0-r1",
        "commit": "abcdef1234567890",
        "hardware": "local CPU fixture",
        "metrics": [metric],
        "artifacts": artifacts,
        "limitations": ["Known limitations."],
        "negative_findings": ["No clinical utility claim is measured by this fixture report."],
        "conclusions": [conclusion],
    }


def _write_efficiency_report(
    root: Path,
    *,
    release_id: str = "geno-lewm-v0.1.0-r1",
) -> None:
    report = parse_efficiency_report(_efficiency_payload(root, release_id=release_id))
    (root / EFFICIENCY_REPORT_NAME).write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _efficiency_payload(
    root: Path,
    *,
    release_id: str = "geno-lewm-v0.1.0-r1",
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "generated_by": EFFICIENCY_REPORT_GENERATED_BY,
        "generated_at": "2026-06-01T12:00:00Z",
        "model_id": sha256_bytes(b"eval-report-model"),
        "model_release": release_id,
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
        "hardware": "local CPU fixture",
        "runtime": "Python fixture runtime",
        "warmup_batches": 1,
        "samples": 8,
        "measurements": {
            "single_variant_latency_ms": 12.5,
            "batched_throughput_variants_per_s": 512.0,
            "peak_memory_bytes": 123456789,
        },
        "inputs": {
            "checkpoint": {
                "path": "model/predictor.safetensors",
                "sha256": sha256_file(root / "predictor.safetensors"),
                "size_bytes": (root / "predictor.safetensors").stat().st_size,
            },
            "dataset_manifest": {
                "path": "dataset/dataset_manifest.json",
                "sha256": sha256_bytes(b"dataset-manifest"),
                "size_bytes": 1024,
            },
        },
        "limitations": ["Local fixture run; release numbers require the published checkpoint."],
    }


def _available_dependency(import_name: str, required: bool) -> DependencyProbe:
    return DependencyProbe(
        import_name=import_name,
        package=import_name.split(".", 1)[0],
        required=required,
        available=True,
        version="1.0.0",
        reason="available in test",
    )


def _rewrite_demo_receipts(
    paths: PackagePaths,
    *,
    model_id: str,
    calibration_hash: str,
) -> None:
    scores = paths.demo_dir / "scores.jsonl"
    receipts = paths.demo_dir / "receipts.jsonl"
    output = ReceiptOutput(
        sigma_raw=0.1,
        sigma_calibrated=0.2,
        bucket_id="coding_missense|mid|none",
        confidence=0.9,
        low_confidence=False,
    )
    receipts.write_text(
        _receipt_json(
            model_id=model_id,
            calibration_hash=calibration_hash,
            output=output,
            row_index=1,
        )
        + "\n",
        encoding="utf-8",
    )
    batch_report = write_batch_receipt_report(
        scores,
        receipts,
        paths.demo_dir / "batch_receipt_report.json",
        generated_at="2026-06-01T12:00:00Z",
    )
    transcript_path = paths.demo_dir / "terminal-demo-transcript.md"
    transcript = transcript_path.read_text(encoding="utf-8")
    transcript = _replace_line(
        transcript,
        "- Receipts SHA-256:",
        f"- Receipts SHA-256: {sha256_file(receipts)}",
    )
    transcript = _replace_line(
        transcript,
        "- Batch Receipt Report SHA-256:",
        f"- Batch Receipt Report SHA-256: {sha256_file(batch_report)}",
    )
    transcript_path.write_text(transcript, encoding="utf-8")
    _refresh_demo_manifest_artifacts(
        paths.demo_dir / DEMO_MANIFEST_NAME,
        artifacts={
            "receipts": receipts,
            "batch receipt report": batch_report,
            "terminal transcript": transcript_path,
        },
    )


def _refresh_demo_manifest_artifacts(
    manifest_path: Path,
    *,
    artifacts: dict[str, Path],
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in payload["artifacts"]:
        path = artifacts.get(artifact["label"])
        if path is None:
            continue
        artifact["sha256"] = sha256_file(path)
        artifact["size_bytes"] = path.stat().st_size
        if artifact["label"] in {"scores", "receipts"}:
            artifact["jsonl_records"] = 1
            artifact["jsonl_fields"] = _jsonl_fields(path)
        if artifact["label"] == "batch receipt report":
            payload["score_receipt_batch"] = _batch_receipt_summary(path)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl_fields(path: Path) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            assert isinstance(payload, dict)
            for key in payload:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    return fields


def _jsonl_field_list(path: Path) -> str:
    return ", ".join(_jsonl_fields(path))


def _batch_receipt_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    required = (
        "records",
        "model_id",
        "calibration_hash",
        "receipt_schema_version",
        "receipt_stream",
        "checked_score_fields",
        "runtime",
    )
    return {field: payload[field] for field in required}


def _replace_line(text: str, prefix: str, replacement: str) -> str:
    return (
        "\n".join(replacement if line.startswith(prefix) else line for line in text.splitlines())
        + "\n"
    )


def _write_terminal_demo_manifest(
    *,
    model_dir: Path,
    demo_dir: Path,
    vcf: Path,
    fasta: Path,
    scores: Path,
    receipts: Path,
    batch_report: Path,
    manifest: Manifest,
) -> None:
    command = (
        "geno-lewm-score",
        "--quiet",
        "--no-banner",
        "--model-dir",
        str(model_dir),
        "--backend",
        "cpu",
        "--vcf",
        str(vcf),
        "--fasta",
        str(fasta),
        "--output",
        str(scores),
        "--receipt",
        str(receipts),
        "--batch-size",
        "64",
        "--no-progress",
    )
    write_demo_manifest(
        request=DemoRequest(
            model_dir=model_dir,
            vcf=vcf,
            fasta=fasta,
            output_dir=demo_dir,
            backend="cpu",
        ),
        model_manifest=manifest,
        command=command,
        completed=subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout='{"output_path":"scores.jsonl"}\n',
            stderr="",
        ),
        artifacts=(
            DemoArtifact(
                label="scores",
                path=scores,
                sha256=sha256_file(scores),
                size_bytes=scores.stat().st_size,
                jsonl_records=1,
                jsonl_fields=tuple(_jsonl_fields(scores)),
            ),
            DemoArtifact(
                label="receipts",
                path=receipts,
                sha256=sha256_file(receipts),
                size_bytes=receipts.stat().st_size,
                jsonl_records=1,
                jsonl_fields=tuple(_jsonl_fields(receipts)),
            ),
            DemoArtifact(
                label="runtime preflight report",
                path=demo_dir / "runtime_preflight_report.json",
                sha256=sha256_file(demo_dir / "runtime_preflight_report.json"),
                size_bytes=(demo_dir / "runtime_preflight_report.json").stat().st_size,
            ),
            DemoArtifact(
                label="batch receipt report",
                path=batch_report,
                sha256=sha256_file(batch_report),
                size_bytes=batch_report.stat().st_size,
            ),
        ),
        runtime_preflight_summary=_runtime_preflight_summary(
            demo_dir / "runtime_preflight_report.json"
        ),
        score_receipt_summary=_batch_receipt_summary(batch_report),
        generated_at=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
    )


def _runtime_preflight_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    command = payload["command"]
    assert isinstance(command, dict)
    argv = command["argv"]
    assert isinstance(argv, list)
    requirements = payload["requirements"]
    assert isinstance(requirements, dict)
    return {
        "schema_version": payload["schema_version"],
        "generated_by": payload["generated_by"],
        "ok": payload["ok"],
        "model_id": payload["model_id"],
        "release_id": payload["release_id"],
        "requested_backend": payload["requested_backend"],
        "selected_backend": payload["selected_backend"],
        "requirements": requirements,
        "command": {
            "argv": argv,
            "shell": command["shell"],
        },
    }


def _write_sha256sums(root: Path, files: tuple[str, ...]) -> None:
    lines = []
    for relative in files:
        digest = sha256_file(root / relative).removeprefix("sha256:")
        lines.append(f"{digest}  {relative}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _refresh_sha256sums(root: Path, checksums_path: Path) -> None:
    files = tuple(
        line.split(maxsplit=1)[1]
        for line in checksums_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    lines = []
    for relative in files:
        digest = sha256_file(root / relative).removeprefix("sha256:")
        lines.append(f"{digest}  {relative}")
    checksums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _file_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _load_manifest(model_dir: Path) -> Manifest:
    from geno_lewm.provenance import load_manifest

    return load_manifest(model_dir / "manifest.json")


def _receipt_json(
    *,
    model_id: str,
    calibration_hash: str,
    output: ReceiptOutput,
    row_index: int,
) -> str:
    receipt = Receipt(
        schema_version="1.0.0",
        model_id=model_id,
        input_commitment="sha256:" + f"{row_index}".zfill(64),
        output=output,
        output_commitment=compute_output_commitment(output),
        calibration_hash=calibration_hash,
        runtime=ReceiptRuntime(
            backend="cpu",
            device="CPU",
            geno_lewm_version="0.1.0",
            carbon_revision="main",
        ),
        timestamp=f"2026-06-01T12:00:0{row_index}Z",
        provenance=ReceiptProvenance(
            kind="checksum_only",
            details={
                "scope": "vcf_row",
                "receipt_stream": "jsonl_per_scored_alternate_v1",
                "row_index": row_index,
            },
        ),
    )
    return receipt.to_canonical_json().decode("utf-8")


def _codes(report: PackageReport) -> set[str]:
    return {issue.code for issue in report.issues}
