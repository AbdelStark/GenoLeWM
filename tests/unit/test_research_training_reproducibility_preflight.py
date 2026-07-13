# SPDX-License-Identifier: Apache-2.0
"""Tests for the immutable issue #47 H200 job preflight."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from tools.research.training_reproducibility_preflight import (
    EXPECTED_BASELINE_CONFIG_PATH,
    EXPECTED_CARBON_RUNTIME_HASH,
    EXPECTED_CONTAINER_IMAGE,
    EXPECTED_DATASET_PATH,
    EXPECTED_DATASET_REFERENCE_PATH,
    EXPECTED_DATASET_REPO,
    EXPECTED_DATASET_REVISION,
    EXPECTED_DETERMINISTIC_CONFIG_PATH,
    EXPECTED_UPLOAD_REPO,
    TrainingReproducibilityPreflightRequest,
    build_training_reproducibility_preflight_report,
)


def test_preflight_accepts_clean_exact_sha_contract(tmp_path: Path) -> None:
    repo, commit_sha = _write_clean_repo(tmp_path)
    request = _request(repo, commit_sha)

    report = build_training_reproducibility_preflight_report(
        request,
        generated_at="2026-07-13T00:00:00Z",
    )

    assert report.ok is True
    assert report.issues == ()
    payload = report.to_dict()
    assert payload["repository"]["observed_commit_sha"] == commit_sha
    assert payload["job"]["execution_order"] == [
        "baseline_a",
        "deterministic_a",
        "deterministic_b",
        "baseline_b",
    ]
    assert payload["job"]["expected_sample_count"] == 4_000
    assert payload["job"]["expected_carbon_runtime_hash"] == EXPECTED_CARBON_RUNTIME_HASH
    assert payload["job"]["upload_repo"] == EXPECTED_UPLOAD_REPO
    assert payload["dataset"]["reference"]["path"] == EXPECTED_DATASET_REFERENCE_PATH.as_posix()
    assert payload["dataset"]["reference"]["sha256"].startswith("sha256:")
    assert payload["dataset"]["source"] == {
        "repo_id": EXPECTED_DATASET_REPO,
        "repo_type": "model",
        "revision": EXPECTED_DATASET_REVISION,
        "path": EXPECTED_DATASET_PATH,
    }


def test_preflight_rejects_any_baseline_config_drift(tmp_path: Path) -> None:
    repo, commit_sha = _write_clean_repo(tmp_path)
    baseline = repo / EXPECTED_BASELINE_CONFIG_PATH
    baseline.write_text(
        baseline.read_text(encoding="utf-8").replace("lr: 3.0e-5", "lr: 3.1e-5"),
        encoding="utf-8",
    )

    report = build_training_reproducibility_preflight_report(_request(repo, commit_sha))

    assert report.ok is False
    codes = {issue.code for issue in report.issues}
    assert "config.baseline_diff" in codes
    assert "repository.worktree_dirty" in codes


def test_preflight_rejects_unpinned_dataset_revision(tmp_path: Path) -> None:
    repo, commit_sha = _write_clean_repo(tmp_path)
    request = replace(_request(repo, commit_sha), dataset_revision="main")

    report = build_training_reproducibility_preflight_report(request)

    assert report.ok is False
    assert "request.dataset_revision_mismatch" in {issue.code for issue in report.issues}


def test_preflight_rejects_unpinned_carbon_runtime(tmp_path: Path) -> None:
    repo, commit_sha = _write_clean_repo(tmp_path)
    request = replace(_request(repo, commit_sha), expected_carbon_runtime_hash="sha256:" + "0" * 64)

    report = build_training_reproducibility_preflight_report(request)

    assert report.ok is False
    assert "request.expected_carbon_runtime_hash_mismatch" in {
        issue.code for issue in report.issues
    }


def test_preflight_rejects_unpinned_upload_repo(tmp_path: Path) -> None:
    repo, commit_sha = _write_clean_repo(tmp_path)
    request = replace(_request(repo, commit_sha), upload_repo="other/runs")

    report = build_training_reproducibility_preflight_report(request)

    assert report.ok is False
    assert "request.upload_repo_mismatch" in {issue.code for issue in report.issues}


def _write_clean_repo(tmp_path: Path) -> tuple[Path, str]:
    source_root = Path.cwd()
    repo = tmp_path / "repo"
    for relative in (
        EXPECTED_DETERMINISTIC_CONFIG_PATH,
        EXPECTED_BASELINE_CONFIG_PATH,
        EXPECTED_DATASET_REFERENCE_PATH,
    ):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    commit_sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, commit_sha


def _request(repo: Path, commit_sha: str) -> TrainingReproducibilityPreflightRequest:
    return TrainingReproducibilityPreflightRequest(
        repo_root=repo,
        deterministic_config=EXPECTED_DETERMINISTIC_CONFIG_PATH,
        baseline_config=EXPECTED_BASELINE_CONFIG_PATH,
        dataset_reference=EXPECTED_DATASET_REFERENCE_PATH,
        expected_commit_sha=commit_sha,
        run_name=f"geno-lewm-repro-h200-{commit_sha[:12]}-500-r1",
        run_attempt=1,
        steps=500,
        expected_sample_count=4_000,
        dataset_repo=EXPECTED_DATASET_REPO,
        dataset_revision=EXPECTED_DATASET_REVISION,
        dataset_path=EXPECTED_DATASET_PATH,
        carbon_model_dir="/carbon",
        expected_carbon_runtime_hash=EXPECTED_CARBON_RUNTIME_HASH,
        upload_repo=EXPECTED_UPLOAD_REPO,
        container_image=EXPECTED_CONTAINER_IMAGE,
        min_cuda_vram_gb=120.0,
        max_throughput_drop=0.15,
        max_repeat_spread=0.05,
    )
