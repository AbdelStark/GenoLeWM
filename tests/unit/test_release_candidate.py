"""Tests for the aggregate release-candidate report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.training.preflight import REPORT_NAME as TRAINING_PREFLIGHT_REPORT_NAME
from tests.unit.test_release_hub_release import _write_release_candidate
from tools.demo.terminal_inference import DEMO_MANIFEST_NAME
from tools.release import release_candidate
from tools.release.efficiency_report import REPORT_NAME as EFFICIENCY_REPORT_NAME
from tools.release.hub_release import UploadFile
from tools.release.issue_refs import issue_ref_payload
from tools.release.model_package import EVAL_CONFIG_NAME, EVAL_METRICS_NAME, MODEL_PACKAGE_NAME
from tools.release.release_candidate import (
    PublicArtifactCheck,
    PublicLinkCheck,
    build_release_candidate_report,
)


def test_release_candidate_report_binds_package_hub_links_and_artifacts(tmp_path: Path) -> None:
    paths = _write_release_candidate(tmp_path)

    report = build_release_candidate_report(
        model_dir=paths["model_dir"],
        dataset_dir=paths["dataset_dir"],
        demo_dir=paths["demo_dir"],
        paper_path=paths["paper_path"],
        repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
        demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
        paper_url="https://arxiv.org/abs/2606.00001",
        commit_sha="abcdef1234567890",
        generated_at="2026-06-01T12:00:00Z",
        public_link_probe=_ok_public_link,
        public_artifact_probe=_ok_public_artifacts,
    )

    assert report.ready is True
    assert report.package_ok is True
    assert report.hub_plan is not None
    assert report.hub_plan.generated_at == report.generated_at
    assert report.model_id is not None and report.model_id.startswith("sha256:")
    assert report.dataset_snapshot_id == "geno-lewm-data-v0.1.0-r1"
    assert (
        report.urls["model"] == "https://huggingface.co/AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1"
    )
    assert report.artifacts["eval_metrics"] is not None
    assert report.artifacts["eval_config"] is not None
    assert report.artifacts["model_package"] is not None
    assert report.artifacts["eval_report"] is not None
    assert report.artifacts["efficiency_report"] is not None
    assert report.artifacts["predictor"] is not None
    assert report.artifacts["action_encoder"] is not None
    assert report.artifacts["calibration"] is not None
    assert report.artifacts["training_config"] is not None
    assert report.artifacts["training_run_manifest"] is not None
    assert report.artifacts["training_run_card"] is not None
    assert report.artifacts["training_run_checksums"] is not None
    assert report.artifacts["training_preflight_report"] is not None
    assert report.artifacts["runtime_preflight"] is not None
    assert report.artifacts["batch_receipt_report"] is not None
    assert report.artifacts["terminal_demo_manifest"] is not None
    assert report.artifacts["dataset_package"] is not None
    assert report.artifacts["dataset_snapshot_report"] is not None
    assert report.artifacts["dataset_input_check_report"] is not None
    assert report.artifacts["paper"] is not None
    assert EVAL_METRICS_NAME in {file.destination for file in report.hub_plan.files}
    assert EVAL_CONFIG_NAME in {file.destination for file in report.hub_plan.files}
    assert {file.destination for file in report.hub_plan.dataset_files} >= {
        "dataset_manifest.json",
        "dataset_package.json",
        "dataset_input_check_report.json",
        "carbon/windows.jsonl",
        "clinvar/eval.vcf",
    }
    assert {file.destination for file in report.hub_plan.demo_files} >= {
        DEMO_MANIFEST_NAME,
        "terminal-demo-transcript.md",
        "scores.jsonl",
        "receipts.jsonl",
    }
    assert report.public_links_required is True
    assert {check.name for check in report.public_link_checks} == {
        "model",
        "dataset",
        "demo",
        "paper",
    }
    assert {check.name for check in report.public_artifact_checks} == {
        "model",
        "dataset",
        "demo",
        "paper",
    }
    readiness = {item.code: item for item in report.readiness}
    assert set(readiness) == {
        "package_verifier",
        "model_package",
        "dataset_package",
        "terminal_demo",
        "paper_artifact",
        "public_links",
        "public_artifacts",
        "hub_publication_plan",
    }
    assert all(item.ok for item in readiness.values())
    assert "model_files=" in " ".join(readiness["hub_publication_plan"].evidence)
    assert "dataset_files=" in " ".join(readiness["hub_publication_plan"].evidence)
    assert "demo_files=" in " ".join(readiness["hub_publication_plan"].evidence)
    assert readiness["model_package"].issue_refs == (164, 165, 101)
    assert readiness["dataset_package"].issue_refs == (163,)
    assert readiness["terminal_demo"].issue_refs == (166,)
    assert readiness["paper_artifact"].issue_refs == (167,)
    assert readiness["public_links"].issue_refs == (163, 166, 167, 101)
    assert readiness["public_artifacts"].issue_refs == (163, 166, 167, 101)
    assert readiness["hub_publication_plan"].issue_refs == (167, 101)
    assert report.blockers == ()


def test_release_candidate_report_blocks_unreachable_public_links(tmp_path: Path) -> None:
    paths = _write_release_candidate(tmp_path)

    report = build_release_candidate_report(
        model_dir=paths["model_dir"],
        dataset_dir=paths["dataset_dir"],
        demo_dir=paths["demo_dir"],
        paper_path=paths["paper_path"],
        repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        dataset_url="https://huggingface.co/datasets/AbdelStark/missing-dataset",
        demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
        paper_url="https://arxiv.org/abs/2606.00001",
        commit_sha="abcdef1234567890",
        public_link_probe=_dataset_missing_public_link,
        public_artifact_probe=_ok_public_artifacts,
    )

    assert report.ready is False
    assert report.package_ok is True
    assert report.hub_plan is not None
    assert {blocker.code for blocker in report.blockers} == {"public_link.dataset.unreachable"}
    assert report.blockers[0].issue_refs == (163,)
    public_links = next(item for item in report.readiness if item.code == "public_links")
    assert public_links.ok is False
    assert "public_link.dataset.unreachable" in public_links.blockers


def test_release_candidate_report_requires_public_paper_url_when_paper_path_is_set(
    tmp_path: Path,
) -> None:
    paths = _write_release_candidate(tmp_path)

    report = build_release_candidate_report(
        model_dir=paths["model_dir"],
        dataset_dir=paths["dataset_dir"],
        demo_dir=paths["demo_dir"],
        paper_path=paths["paper_path"],
        repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
        demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
        commit_sha="abcdef1234567890",
        public_link_probe=_ok_public_link,
        public_artifact_probe=_ok_public_artifacts,
    )

    assert report.ready is False
    assert "paper.url_missing" in {blocker.code for blocker in report.blockers}
    assert {
        blocker.issue_refs for blocker in report.blockers if blocker.code == "paper.url_missing"
    } == {(167,)}


def test_release_candidate_report_returns_blockers_for_invalid_package(tmp_path: Path) -> None:
    paths = _write_release_candidate(tmp_path)
    (paths["demo_dir"] / "runtime_preflight_report.json").unlink()

    report = build_release_candidate_report(
        model_dir=paths["model_dir"],
        dataset_dir=paths["dataset_dir"],
        demo_dir=paths["demo_dir"],
        paper_path=paths["paper_path"],
        repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
        demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
        commit_sha="abcdef1234567890",
        require_public_links=False,
    )

    assert report.ready is False
    assert report.package_ok is False
    assert report.hub_plan is None
    codes = {blocker.code for blocker in report.blockers}
    assert "package.demo.runtime_preflight.missing" in codes
    assert "package.failed" in codes
    readiness = {item.code: item for item in report.readiness}
    assert readiness["package_verifier"].ok is False
    assert "demo.runtime_preflight.missing" in readiness["package_verifier"].blockers
    assert readiness["terminal_demo"].ok is False
    assert readiness["hub_publication_plan"].blockers == ("hub_plan.missing",)


def test_release_candidate_report_blocks_missing_public_artifacts(tmp_path: Path) -> None:
    paths = _write_release_candidate(tmp_path)

    report = build_release_candidate_report(
        model_dir=paths["model_dir"],
        dataset_dir=paths["dataset_dir"],
        demo_dir=paths["demo_dir"],
        paper_path=paths["paper_path"],
        repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
        demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
        paper_url="https://arxiv.org/abs/2606.00001",
        commit_sha="abcdef1234567890",
        public_link_probe=_ok_public_link,
        public_artifact_probe=_missing_demo_public_artifacts,
    )

    assert report.ready is False
    assert {blocker.code for blocker in report.blockers} == {"public_artifact.demo.missing"}
    assert report.blockers[0].issue_refs == (166,)
    public_artifacts = next(item for item in report.readiness if item.code == "public_artifacts")
    assert public_artifacts.ok is False
    assert "public_artifact.demo.missing_files" in public_artifacts.blockers


def test_release_candidate_report_blocks_public_artifact_size_mismatches(
    tmp_path: Path,
) -> None:
    paths = _write_release_candidate(tmp_path)

    report = build_release_candidate_report(
        model_dir=paths["model_dir"],
        dataset_dir=paths["dataset_dir"],
        demo_dir=paths["demo_dir"],
        paper_path=paths["paper_path"],
        repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
        demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
        paper_url="https://arxiv.org/abs/2606.00001",
        commit_sha="abcdef1234567890",
        public_link_probe=_ok_public_link,
        public_artifact_probe=_size_mismatch_model_public_artifacts,
    )

    assert report.ready is False
    assert {blocker.code for blocker in report.blockers} == {"public_artifact.model.size_mismatch"}
    public_artifacts = next(item for item in report.readiness if item.code == "public_artifacts")
    assert public_artifacts.ok is False
    assert "public_artifact.model.size_mismatch" in public_artifacts.blockers


def test_release_candidate_report_blocks_public_paper_hash_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_release_candidate(tmp_path)

    report = build_release_candidate_report(
        model_dir=paths["model_dir"],
        dataset_dir=paths["dataset_dir"],
        demo_dir=paths["demo_dir"],
        paper_path=paths["paper_path"],
        repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
        demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
        paper_url="https://arxiv.org/abs/2606.00001",
        commit_sha="abcdef1234567890",
        public_link_probe=_ok_public_link,
        public_artifact_probe=_hash_mismatch_paper_public_artifacts,
    )

    assert report.ready is False
    assert {blocker.code for blocker in report.blockers} == {"public_artifact.paper.hash_mismatch"}
    assert report.blockers[0].issue_refs == (167,)
    public_artifacts = next(item for item in report.readiness if item.code == "public_artifacts")
    assert public_artifacts.ok is False
    assert "public_artifact.paper.hash_mismatch" in public_artifacts.blockers


def test_public_artifact_probe_blocks_hash_mismatches(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = (
        UploadFile(
            source="/tmp/model_card.md",
            destination="model_card.md",
            sha256="sha256:" + "a" * 64,
            size_bytes=12,
        ),
    )

    monkeypatch.setattr(
        release_candidate,
        "_open_json_for_status",
        lambda _url, _timeout_seconds: (
            200,
            {"siblings": [{"rfilename": "model_card.md"}]},
        ),
    )
    monkeypatch.setattr(
        release_candidate,
        "_hash_and_size_url",
        lambda _url, _timeout_seconds: ("sha256:" + "b" * 64, 12),
    )

    check = release_candidate._probe_public_artifacts(
        "model",
        "https://huggingface.co/AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        expected,
        1.0,
    )

    assert check.ok is False
    assert check.missing == ()
    assert check.hash_mismatches == ("model_card.md",)
    assert check.size_mismatches == ()
    assert check.verified_count == 0


def test_public_artifact_probe_blocks_size_mismatches(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = (
        UploadFile(
            source="/tmp/model_card.md",
            destination="model_card.md",
            sha256="sha256:" + "a" * 64,
            size_bytes=12,
        ),
    )

    monkeypatch.setattr(
        release_candidate,
        "_open_json_for_status",
        lambda _url, _timeout_seconds: (
            200,
            {"siblings": [{"rfilename": "model_card.md"}]},
        ),
    )
    monkeypatch.setattr(
        release_candidate,
        "_hash_and_size_url",
        lambda _url, _timeout_seconds: ("sha256:" + "a" * 64, 13),
    )

    check = release_candidate._probe_public_artifacts(
        "model",
        "https://huggingface.co/AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        expected,
        1.0,
    )

    assert check.ok is False
    assert check.missing == ()
    assert check.hash_mismatches == ()
    assert check.size_mismatches == ("model_card.md",)
    assert check.verified_count == 0


def test_public_artifact_probe_blocks_unexpected_public_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = (
        UploadFile(
            source="/tmp/model_card.md",
            destination="model_card.md",
            sha256="sha256:" + "a" * 64,
            size_bytes=12,
        ),
    )

    monkeypatch.setattr(
        release_candidate,
        "_open_json_for_status",
        lambda _url, _timeout_seconds: (
            200,
            {"siblings": [{"rfilename": "model_card.md"}, {"rfilename": "private.bin"}]},
        ),
    )
    monkeypatch.setattr(
        release_candidate,
        "_hash_and_size_url",
        lambda _url, _timeout_seconds: ("sha256:" + "a" * 64, 12),
    )

    check = release_candidate._probe_public_artifacts(
        "model",
        "https://huggingface.co/AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        expected,
        1.0,
    )

    assert check.ok is False
    assert check.missing == ()
    assert check.hash_mismatches == ()
    assert check.size_mismatches == ()
    assert check.unexpected == ("private.bin",)
    assert check.verified_count == 1


def test_release_candidate_main_writes_report_and_uses_ready_exit_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_release_candidate(tmp_path)
    output = tmp_path / "release_candidate_report.json"
    monkeypatch.setattr(release_candidate, "_probe_public_url", _ok_public_link)
    monkeypatch.setattr(release_candidate, "_probe_public_artifacts", _ok_public_artifacts)

    rc = release_candidate.main(
        [
            "--model-dir",
            str(paths["model_dir"]),
            "--dataset-dir",
            str(paths["dataset_dir"]),
            "--demo-dir",
            str(paths["demo_dir"]),
            "--paper-path",
            str(paths["paper_path"]),
            "--repo-id",
            "AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
            "--dataset-url",
            "https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
            "--demo-url",
            "https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            "--paper-url",
            "https://arxiv.org/abs/2606.00001",
            "--commit-sha",
            "abcdef1234567890",
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "wrote" in captured.out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is True
    assert str(tmp_path) not in json.dumps(payload)
    assert payload["package"]["ok"] is True
    assert payload["public_links"]["required"] is True
    assert len(payload["public_links"]["checks"]) == 4
    assert payload["public_artifacts"]["required"] is True
    assert len(payload["public_artifacts"]["checks"]) == 4
    assert payload["artifacts"]["eval_metrics"]["path"].endswith(EVAL_METRICS_NAME)
    assert payload["artifacts"]["eval_metrics"]["path"].startswith("model/")
    assert payload["artifacts"]["eval_config"]["path"].endswith(EVAL_CONFIG_NAME)
    assert payload["artifacts"]["eval_config"]["path"].startswith("model/")
    assert payload["artifacts"]["model_package"]["path"].endswith(MODEL_PACKAGE_NAME)
    assert payload["artifacts"]["efficiency_report"]["path"].endswith(EFFICIENCY_REPORT_NAME)
    assert payload["artifacts"]["training_run_checksums"]["path"].endswith(
        "training_run_SHA256SUMS"
    )
    assert payload["artifacts"]["training_preflight_report"]["path"].endswith(
        TRAINING_PREFLIGHT_REPORT_NAME
    )
    assert {file["destination"] for file in payload["hub_plan"]["dataset_files"]} >= {
        "dataset_manifest.json",
        "carbon/windows.jsonl",
    }
    assert {file["destination"] for file in payload["hub_plan"]["demo_files"]} >= {
        DEMO_MANIFEST_NAME,
        "terminal-demo-transcript.md",
    }
    assert {item["code"] for item in payload["readiness"]} >= {
        "package_verifier",
        "hub_publication_plan",
        "public_links",
        "public_artifacts",
    }
    readiness = {item["code"]: item for item in payload["readiness"]}
    assert readiness["dataset_package"]["issue_refs"] == issue_ref_payload((163,))
    assert readiness["model_package"]["issue_refs"] == issue_ref_payload((164, 165, 101))
    assert readiness["terminal_demo"]["issue_refs"] == issue_ref_payload((166,))
    assert readiness["public_artifacts"]["issue_refs"] == issue_ref_payload((163, 166, 167, 101))
    assert all(item["ok"] for item in payload["readiness"])


def test_release_candidate_main_returns_two_for_failed_candidate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_release_candidate(tmp_path)
    (paths["demo_dir"] / "terminal-demo-transcript.md").unlink()
    output = tmp_path / "release_candidate_report.json"
    monkeypatch.setattr(release_candidate, "_probe_public_url", _ok_public_link)
    monkeypatch.setattr(release_candidate, "_probe_public_artifacts", _ok_public_artifacts)

    rc = release_candidate.main(
        [
            "--model-dir",
            str(paths["model_dir"]),
            "--dataset-dir",
            str(paths["dataset_dir"]),
            "--demo-dir",
            str(paths["demo_dir"]),
            "--paper-path",
            str(paths["paper_path"]),
            "--repo-id",
            "AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
            "--dataset-url",
            "https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
            "--demo-url",
            "https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            "--paper-url",
            "https://arxiv.org/abs/2606.00001",
            "--commit-sha",
            "abcdef1234567890",
            "--output",
            str(output),
        ]
    )
    capsys.readouterr()

    assert rc == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is False
    assert payload["blockers"]
    blocker = next(item for item in payload["blockers"] if item["code"] == "package.failed")
    assert blocker["issue_refs"] == issue_ref_payload((163, 164, 165, 166, 167, 101))


def test_release_candidate_main_can_skip_public_link_check_for_offline_fixtures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_release_candidate(tmp_path, release_id="geno-lewm-fixture-r1")
    output = tmp_path / "release_candidate_report.json"

    rc = release_candidate.main(
        [
            "--model-dir",
            str(paths["model_dir"]),
            "--dataset-dir",
            str(paths["dataset_dir"]),
            "--demo-dir",
            str(paths["demo_dir"]),
            "--paper-path",
            str(paths["paper_path"]),
            "--repo-id",
            "AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
            "--dataset-url",
            "https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
            "--demo-url",
            "https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            "--paper-url",
            "https://arxiv.org/abs/2606.00001",
            "--commit-sha",
            "abcdef1234567890",
            "--output",
            str(output),
            "--allow-fixture-manifest",
            "--skip-public-link-check",
        ]
    )
    capsys.readouterr()

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is True
    assert str(tmp_path) not in json.dumps(payload)
    assert payload["public_links"] == {"required": False, "checks": []}
    assert payload["public_artifacts"] == {"required": False, "checks": []}
    public_links = next(item for item in payload["readiness"] if item["code"] == "public_links")
    assert public_links["ok"] is True
    assert public_links["evidence"] == ["required=false", "fixture_rehearsal=true"]
    public_artifacts = next(
        item for item in payload["readiness"] if item["code"] == "public_artifacts"
    )
    assert public_artifacts["ok"] is True
    assert public_artifacts["evidence"] == ["required=false", "fixture_rehearsal=true"]


def test_release_candidate_main_rejects_skipped_public_checks_for_release_candidates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_release_candidate(tmp_path)
    output = tmp_path / "release_candidate_report.json"

    rc = release_candidate.main(
        [
            "--model-dir",
            str(paths["model_dir"]),
            "--dataset-dir",
            str(paths["dataset_dir"]),
            "--demo-dir",
            str(paths["demo_dir"]),
            "--paper-path",
            str(paths["paper_path"]),
            "--repo-id",
            "AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
            "--dataset-url",
            "https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
            "--demo-url",
            "https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            "--paper-url",
            "https://arxiv.org/abs/2606.00001",
            "--commit-sha",
            "abcdef1234567890",
            "--output",
            str(output),
            "--skip-public-link-check",
        ]
    )
    capsys.readouterr()

    assert rc == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is False
    assert "public_links.skipped_for_release" in {
        blocker["code"] for blocker in payload["blockers"]
    }
    readiness = {item["code"]: item for item in payload["readiness"]}
    assert readiness["public_links"]["ok"] is False
    assert readiness["public_artifacts"]["ok"] is False
    assert readiness["public_links"]["blockers"] == ["public_links.skipped_for_release"]
    assert readiness["public_artifacts"]["blockers"] == ["public_artifacts.skipped_for_release"]


def _ok_public_link(name: str, url: str, timeout_seconds: float) -> PublicLinkCheck:
    assert timeout_seconds > 0
    return PublicLinkCheck(name=name, url=url, ok=True, status_code=200)


def _ok_public_artifacts(
    name: str,
    url: str,
    expected: tuple[UploadFile, ...],
    timeout_seconds: float,
) -> PublicArtifactCheck:
    assert timeout_seconds > 0
    return PublicArtifactCheck(
        name=name,
        url=url,
        ok=True,
        expected_count=len(expected),
        observed_count=len(expected),
        verified_count=len(expected),
        status_code=200,
    )


def _missing_demo_public_artifacts(
    name: str,
    url: str,
    expected: tuple[UploadFile, ...],
    timeout_seconds: float,
) -> PublicArtifactCheck:
    if name != "demo":
        return _ok_public_artifacts(name, url, expected, timeout_seconds)
    assert timeout_seconds > 0
    return PublicArtifactCheck(
        name=name,
        url=url,
        ok=False,
        expected_count=len(expected),
        observed_count=max(0, len(expected) - 1),
        verified_count=max(0, len(expected) - 1),
        missing=("terminal-demo-transcript.md",),
        status_code=200,
    )


def _size_mismatch_model_public_artifacts(
    name: str,
    url: str,
    expected: tuple[UploadFile, ...],
    timeout_seconds: float,
) -> PublicArtifactCheck:
    if name != "model":
        return _ok_public_artifacts(name, url, expected, timeout_seconds)
    assert timeout_seconds > 0
    return PublicArtifactCheck(
        name=name,
        url=url,
        ok=False,
        expected_count=len(expected),
        observed_count=len(expected),
        verified_count=max(0, len(expected) - 1),
        size_mismatches=("model_card.md",),
        status_code=200,
    )


def _hash_mismatch_paper_public_artifacts(
    name: str,
    url: str,
    expected: tuple[UploadFile, ...],
    timeout_seconds: float,
) -> PublicArtifactCheck:
    if name != "paper":
        return _ok_public_artifacts(name, url, expected, timeout_seconds)
    assert timeout_seconds > 0
    return PublicArtifactCheck(
        name=name,
        url=url,
        ok=False,
        expected_count=len(expected),
        observed_count=len(expected),
        verified_count=0,
        hash_mismatches=("paper.md",),
        status_code=200,
    )


def _dataset_missing_public_link(name: str, url: str, timeout_seconds: float) -> PublicLinkCheck:
    assert timeout_seconds > 0
    if name == "dataset":
        return PublicLinkCheck(name=name, url=url, ok=False, status_code=404, error="HTTP 404")
    return _ok_public_link(name, url, timeout_seconds)
