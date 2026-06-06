# SPDX-License-Identifier: Apache-2.0
"""Build a machine-readable first paper/demo release candidate report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final
from urllib import error as urllib_error, request as urllib_request
from urllib.parse import quote, urlparse

from geno_lewm.errors import GenoLeWMError, exit_code_for
from geno_lewm.provenance import Manifest, load_manifest, sha256_file
from geno_lewm.training.preflight import REPORT_NAME as TRAINING_PREFLIGHT_REPORT_NAME
from tools.demo.terminal_inference import DEMO_MANIFEST_NAME
from tools.release.batch_receipt_report import REPORT_NAME as BATCH_RECEIPT_REPORT_NAME
from tools.release.dataset_integrity import DEFAULT_REPORT_NAME as DATASET_INTEGRITY_NAME
from tools.release.dataset_snapshot import (
    INPUT_CHECK_REPORT_NAME as DATASET_INPUT_CHECK_REPORT_NAME,
    REPORT_NAME as DATASET_SNAPSHOT_REPORT_NAME,
)
from tools.release.efficiency_report import REPORT_NAME as EFFICIENCY_REPORT_NAME
from tools.release.hub_release import HubReleasePlan, UploadFile, build_hub_release_plan
from tools.release.issue_refs import (
    ALL_RELEASE_ISSUES,
    DATASET_ISSUE,
    DEMO_ISSUE,
    EVAL_ISSUE,
    MODEL_RELEASE_ISSUE,
    PAPER_ISSUE,
    TRAINING_ISSUE,
    issue_ref_payload,
)
from tools.release.model_package import EVAL_CONFIG_NAME, EVAL_METRICS_NAME, MODEL_PACKAGE_NAME
from tools.release.paper_package import PackageIssue, PackagePaths, verify_package
from tools.release.runtime_preflight import REPORT_NAME as RUNTIME_PREFLIGHT_REPORT_NAME

REPORT_NAME: Final = "release_candidate_report.json"
SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.release_candidate"
DEFAULT_PUBLIC_LINK_TIMEOUT_SECONDS: Final = 10.0


@dataclass(frozen=True, slots=True)
class CandidateBlocker:
    """One reason a release candidate is not publishable yet."""

    code: str
    path: str
    message: str

    @property
    def issue_refs(self) -> tuple[int, ...]:
        """Live GitHub release blockers that own this failure mode."""
        return _issue_refs_for_blocker(self.code)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "path": _public_report_path(self.path),
            "message": self.message,
            "issue_refs": issue_ref_payload(self.issue_refs),
        }


@dataclass(frozen=True, slots=True)
class PublicLinkCheck:
    """Reachability result for one public release link."""

    name: str
    url: str
    ok: bool
    status_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "url": self.url,
            "ok": self.ok,
            "status_code": self.status_code,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class PublicArtifactCheck:
    """Remote artifact-listing and checksum result for one release target."""

    name: str
    url: str
    ok: bool
    expected_count: int
    observed_count: int | None = None
    verified_count: int = 0
    missing: tuple[str, ...] = ()
    hash_mismatches: tuple[str, ...] = ()
    size_mismatches: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    status_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "url": self.url,
            "ok": self.ok,
            "expected_count": self.expected_count,
            "observed_count": self.observed_count,
            "verified_count": self.verified_count,
            "missing": list(self.missing),
            "hash_mismatches": list(self.hash_mismatches),
            "size_mismatches": list(self.size_mismatches),
            "unexpected": list(self.unexpected),
            "status_code": self.status_code,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class ReadinessItem:
    """One high-level release-readiness requirement and its evidence."""

    code: str
    ok: bool
    message: str
    evidence: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    @property
    def issue_refs(self) -> tuple[int, ...]:
        """Live GitHub release blockers that own this readiness row."""
        return _issue_refs_for_readiness(self.code)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "ok": self.ok,
            "message": self.message,
            "evidence": list(self.evidence),
            "blockers": list(self.blockers),
            "issue_refs": issue_ref_payload(self.issue_refs),
        }


@dataclass(frozen=True, slots=True)
class ReleaseCandidateReport:
    """Aggregate publication decision for the first paper/demo release."""

    schema_version: str
    generated_by: str
    generated_at: str
    ready: bool
    release_id: str | None
    model_id: str | None
    dataset_snapshot_id: str | None
    commit_sha: str
    repo_id: str
    urls: dict[str, str | None]
    public_links_required: bool
    public_link_checks: tuple[PublicLinkCheck, ...]
    public_artifact_checks: tuple[PublicArtifactCheck, ...]
    artifacts: dict[str, dict[str, object] | None]
    package_ok: bool
    package_issues: tuple[PackageIssue, ...]
    hub_plan: HubReleasePlan | None
    readiness: tuple[ReadinessItem, ...]
    blockers: tuple[CandidateBlocker, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "ready": self.ready,
            "release_id": self.release_id,
            "model_id": self.model_id,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "commit_sha": self.commit_sha,
            "repo_id": self.repo_id,
            "urls": self.urls,
            "public_links": {
                "required": self.public_links_required,
                "checks": [check.to_dict() for check in self.public_link_checks],
            },
            "public_artifacts": {
                "required": self.public_links_required,
                "checks": [check.to_dict() for check in self.public_artifact_checks],
            },
            "artifacts": self.artifacts,
            "package": {
                "ok": self.package_ok,
                "issues": [issue.to_dict() for issue in self.package_issues],
            },
            "hub_plan": None if self.hub_plan is None else self.hub_plan.to_dict(),
            "readiness": [item.to_dict() for item in self.readiness],
            "blockers": [blocker.to_dict() for blocker in self.blockers],
        }


def build_release_candidate_report(
    *,
    model_dir: Path,
    dataset_dir: Path,
    demo_dir: Path,
    repo_id: str,
    dataset_url: str,
    demo_url: str,
    commit_sha: str,
    paper_path: Path | None = None,
    paper_url: str | None = None,
    generated_at: str | None = None,
    allow_fixture_manifest: bool = False,
    require_public_links: bool = True,
    public_link_timeout_seconds: float = DEFAULT_PUBLIC_LINK_TIMEOUT_SECONDS,
    public_link_probe: Callable[[str, str, float], PublicLinkCheck] | None = None,
    public_artifact_probe: (
        Callable[[str, str, tuple[UploadFile, ...], float], PublicArtifactCheck] | None
    ) = None,
) -> ReleaseCandidateReport:
    """Verify and summarize a paper/demo release candidate."""
    generated_at = generated_at or _utc_now()
    blockers: list[CandidateBlocker] = []
    package_report = verify_package(
        PackagePaths(
            model_dir=model_dir,
            dataset_dir=dataset_dir,
            demo_dir=demo_dir,
            paper_path=paper_path,
        ),
        allow_fixture_manifest=allow_fixture_manifest,
    )
    blockers.extend(_blockers_from_package(package_report.issues))

    manifest = _try_load_manifest(model_dir / "manifest.json", blockers)
    dataset_snapshot = _dataset_snapshot_id(dataset_dir / "dataset_manifest.json", blockers)
    artifacts = _artifact_identities(
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        demo_dir=demo_dir,
        paper_path=paper_path,
        manifest=manifest,
    )
    urls: dict[str, str | None] = {
        "model": _model_url(repo_id),
        "dataset": dataset_url,
        "demo": demo_url,
        "paper": paper_url,
    }
    hub_plan: HubReleasePlan | None = None
    if package_report.ok:
        try:
            hub_plan = build_hub_release_plan(
                model_dir=model_dir,
                dataset_dir=dataset_dir,
                demo_dir=demo_dir,
                repo_id=repo_id,
                dataset_url=dataset_url,
                demo_url=demo_url,
                commit_sha=commit_sha,
                paper_path=paper_path,
                paper_url=paper_url,
                generated_at=generated_at,
                allow_fixture_manifest=allow_fixture_manifest,
            )
        except GenoLeWMError as exc:
            blockers.append(
                CandidateBlocker(
                    code="hub_plan.invalid",
                    path="model",
                    message=exc.message or str(exc),
                )
            )
    else:
        blockers.append(
            CandidateBlocker(
                code="package.failed",
                path="model",
                message="paper/demo package verifier did not pass",
            )
        )

    public_link_checks: tuple[PublicLinkCheck, ...] = ()
    public_artifact_checks: tuple[PublicArtifactCheck, ...] = ()
    if not require_public_links and not allow_fixture_manifest:
        blockers.append(
            CandidateBlocker(
                code="public_links.skipped_for_release",
                path="model",
                message=(
                    "public link and artifact hash checks can only be skipped "
                    "when fixture manifests are explicitly allowed"
                ),
            )
        )
    if require_public_links:
        if paper_path is not None and paper_url is None:
            blockers.append(
                CandidateBlocker(
                    code="paper.url_missing",
                    path=_public_report_path(paper_path),
                    message="paper_url is required when public link checks are enabled",
                )
            )
        link_probe = public_link_probe or _probe_public_url
        public_link_checks = _check_public_links(
            urls,
            timeout_seconds=public_link_timeout_seconds,
            probe=link_probe,
        )
        blockers.extend(_blockers_from_public_links(public_link_checks))
        if hub_plan is not None:
            artifact_probe = public_artifact_probe or _probe_public_artifacts
            public_artifact_checks = _check_public_artifacts(
                urls=urls,
                hub_plan=hub_plan,
                timeout_seconds=public_link_timeout_seconds,
                probe=artifact_probe,
            )
            blockers.extend(_blockers_from_public_artifacts(public_artifact_checks))

    readiness = _readiness_items(
        package_ok=package_report.ok,
        package_issues=package_report.issues,
        artifacts=artifacts,
        hub_plan=hub_plan,
        public_links_required=require_public_links,
        public_checks_skip_allowed=allow_fixture_manifest,
        public_link_checks=public_link_checks,
        public_artifact_checks=public_artifact_checks,
        paper_path=paper_path,
        paper_url=paper_url,
        release_id=None if manifest is None else manifest.release_id,
        model_id=None if manifest is None else manifest.model_id(),
        dataset_snapshot_id=dataset_snapshot,
    )
    ready = (
        package_report.ok
        and hub_plan is not None
        and not blockers
        and all(item.ok for item in readiness)
    )
    return ReleaseCandidateReport(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        generated_at=generated_at,
        ready=ready,
        release_id=None if manifest is None else manifest.release_id,
        model_id=None if manifest is None else manifest.model_id(),
        dataset_snapshot_id=dataset_snapshot,
        commit_sha=commit_sha,
        repo_id=repo_id,
        urls=urls,
        public_links_required=require_public_links,
        public_link_checks=public_link_checks,
        public_artifact_checks=public_artifact_checks,
        artifacts=artifacts,
        package_ok=package_report.ok,
        package_issues=package_report.issues,
        hub_plan=hub_plan,
        readiness=readiness,
        blockers=tuple(blockers),
    )


def write_release_candidate_report(
    *,
    model_dir: Path,
    dataset_dir: Path,
    demo_dir: Path,
    repo_id: str,
    dataset_url: str,
    demo_url: str,
    commit_sha: str,
    output: Path,
    paper_path: Path | None = None,
    paper_url: str | None = None,
    generated_at: str | None = None,
    allow_fixture_manifest: bool = False,
    require_public_links: bool = True,
    public_link_timeout_seconds: float = DEFAULT_PUBLIC_LINK_TIMEOUT_SECONDS,
) -> ReleaseCandidateReport:
    """Build and write ``release_candidate_report.json``."""
    report = build_release_candidate_report(
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        demo_dir=demo_dir,
        repo_id=repo_id,
        dataset_url=dataset_url,
        demo_url=demo_url,
        commit_sha=commit_sha,
        paper_path=paper_path,
        paper_url=paper_url,
        generated_at=generated_at,
        allow_fixture_manifest=allow_fixture_manifest,
        require_public_links=require_public_links,
        public_link_timeout_seconds=public_link_timeout_seconds,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = write_release_candidate_report(
            model_dir=args.model_dir,
            dataset_dir=args.dataset_dir,
            demo_dir=args.demo_dir,
            paper_path=args.paper_path,
            repo_id=args.repo_id,
            dataset_url=args.dataset_url,
            demo_url=args.demo_url,
            paper_url=args.paper_url,
            commit_sha=args.commit_sha,
            output=args.output,
            allow_fixture_manifest=args.allow_fixture_manifest,
            require_public_links=not args.skip_public_link_check,
            public_link_timeout_seconds=args.public_link_timeout,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(f"wrote {args.output}\n")
    return 0 if report.ready else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build release_candidate_report.json for a paper/demo release.",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--paper-path", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dataset-url", required=True)
    parser.add_argument("--demo-url", required=True)
    parser.add_argument("--paper-url")
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-fixture-manifest",
        action="store_true",
        help="Allow fixture/test manifests for local verifier tests only.",
    )
    parser.add_argument(
        "--skip-public-link-check",
        action="store_true",
        help="Skip public URL reachability checks for offline fixture tests only.",
    )
    parser.add_argument(
        "--public-link-timeout",
        type=float,
        default=DEFAULT_PUBLIC_LINK_TIMEOUT_SECONDS,
        help="Timeout in seconds for each public URL reachability check.",
    )
    return parser


def _blockers_from_public_links(
    checks: tuple[PublicLinkCheck, ...],
) -> tuple[CandidateBlocker, ...]:
    return tuple(
        CandidateBlocker(
            code=f"public_link.{check.name}.unreachable",
            path=check.url,
            message=check.error
            or (
                "public release link did not return a successful HTTP status"
                if check.status_code is not None
                else "public release link could not be reached"
            ),
        )
        for check in checks
        if not check.ok
    )


def _blockers_from_public_artifacts(
    checks: tuple[PublicArtifactCheck, ...],
) -> tuple[CandidateBlocker, ...]:
    blockers: list[CandidateBlocker] = []
    for check in checks:
        if check.ok:
            continue
        if check.missing:
            blockers.append(
                CandidateBlocker(
                    code=f"public_artifact.{check.name}.missing",
                    path=check.url,
                    message="missing public artifacts: " + ", ".join(check.missing),
                )
            )
        if check.hash_mismatches:
            blockers.append(
                CandidateBlocker(
                    code=f"public_artifact.{check.name}.hash_mismatch",
                    path=check.url,
                    message="hash-mismatched public artifacts: " + ", ".join(check.hash_mismatches),
                )
            )
        if check.size_mismatches:
            blockers.append(
                CandidateBlocker(
                    code=f"public_artifact.{check.name}.size_mismatch",
                    path=check.url,
                    message="size-mismatched public artifacts: " + ", ".join(check.size_mismatches),
                )
            )
        if check.unexpected:
            blockers.append(
                CandidateBlocker(
                    code=f"public_artifact.{check.name}.unexpected",
                    path=check.url,
                    message="unexpected public artifacts: " + ", ".join(check.unexpected),
                )
            )
        if not (
            check.missing or check.hash_mismatches or check.size_mismatches or check.unexpected
        ):
            blockers.append(
                CandidateBlocker(
                    code=f"public_artifact.{check.name}.check_failed",
                    path=check.url,
                    message=check.error or "public artifact check failed",
                )
            )
    return tuple(blockers)


def _blockers_from_package(issues: tuple[PackageIssue, ...]) -> tuple[CandidateBlocker, ...]:
    return tuple(
        CandidateBlocker(
            code=f"package.{issue.code}",
            path=issue.path,
            message=issue.message,
        )
        for issue in issues
        if issue.severity == "error"
    )


def _readiness_items(
    *,
    package_ok: bool,
    package_issues: tuple[PackageIssue, ...],
    artifacts: dict[str, dict[str, object] | None],
    hub_plan: HubReleasePlan | None,
    public_links_required: bool,
    public_checks_skip_allowed: bool,
    public_link_checks: tuple[PublicLinkCheck, ...],
    public_artifact_checks: tuple[PublicArtifactCheck, ...],
    paper_path: Path | None,
    paper_url: str | None,
    release_id: str | None,
    model_id: str | None,
    dataset_snapshot_id: str | None,
) -> tuple[ReadinessItem, ...]:
    return (
        _package_readiness(package_ok, package_issues),
        _artifact_group_readiness(
            code="model_package",
            label="Model package",
            artifacts=artifacts,
            required=(
                "model_manifest",
                "model_package",
                "model_card",
                "model_checksums",
                "eval_metrics",
                "eval_config",
                "eval_report",
                "efficiency_report",
                "predictor",
                "action_encoder",
                "calibration",
                "training_config",
                "training_run_manifest",
                "training_run_card",
                "training_run_checksums",
                "training_preflight_report",
            ),
            evidence=tuple(
                item
                for item in (
                    None if release_id is None else f"release_id={release_id}",
                    None if model_id is None else f"model_id={model_id}",
                    _inventory_evidence(
                        "model_files", None if hub_plan is None else hub_plan.files
                    ),
                )
                if item is not None
            ),
        ),
        _artifact_group_readiness(
            code="dataset_package",
            label="Dataset package",
            artifacts=artifacts,
            required=(
                "dataset_manifest",
                "dataset_package",
                "dataset_snapshot_report",
                "dataset_input_check_report",
                "data_card",
                "dataset_integrity",
                "dataset_checksums",
            ),
            evidence=tuple(
                item
                for item in (
                    None
                    if dataset_snapshot_id is None
                    else f"dataset_snapshot_id={dataset_snapshot_id}",
                    _inventory_evidence(
                        "dataset_files",
                        None if hub_plan is None else hub_plan.dataset_files,
                    ),
                )
                if item is not None
            ),
        ),
        _artifact_group_readiness(
            code="terminal_demo",
            label="Terminal demo",
            artifacts=artifacts,
            required=(
                "terminal_transcript",
                "terminal_demo_manifest",
                "runtime_preflight",
                "batch_receipt_report",
                "scores_jsonl",
                "receipts_jsonl",
            ),
            evidence=(
                _inventory_evidence(
                    "demo_files", None if hub_plan is None else hub_plan.demo_files
                ),
            ),
        ),
        _paper_readiness(artifacts=artifacts, paper_path=paper_path, paper_url=paper_url),
        _public_link_readiness(
            required=public_links_required,
            skip_allowed=public_checks_skip_allowed,
            checks=public_link_checks,
            paper_path=paper_path,
            paper_url=paper_url,
        ),
        _public_artifact_readiness(
            required=public_links_required,
            skip_allowed=public_checks_skip_allowed,
            checks=public_artifact_checks,
            hub_plan=hub_plan,
        ),
        _hub_plan_readiness(hub_plan),
    )


def _package_readiness(
    package_ok: bool,
    package_issues: tuple[PackageIssue, ...],
) -> ReadinessItem:
    error_codes = tuple(issue.code for issue in package_issues if issue.severity == "error")
    return ReadinessItem(
        code="package_verifier",
        ok=package_ok,
        message=(
            "paper/demo package verifier passed"
            if package_ok
            else "paper/demo package verifier reported errors"
        ),
        evidence=("tools.release.paper_package",),
        blockers=error_codes,
    )


def _artifact_group_readiness(
    *,
    code: str,
    label: str,
    artifacts: dict[str, dict[str, object] | None],
    required: tuple[str, ...],
    evidence: tuple[str, ...] = (),
) -> ReadinessItem:
    missing = tuple(name for name in required if artifacts.get(name) is None)
    return ReadinessItem(
        code=code,
        ok=not missing and all(not item.endswith("=0") for item in evidence),
        message=(
            f"{label} artifacts are present" if not missing else f"{label} artifacts are incomplete"
        ),
        evidence=evidence,
        blockers=tuple(f"artifact.{name}.missing" for name in missing),
    )


def _paper_readiness(
    *,
    artifacts: dict[str, dict[str, object] | None],
    paper_path: Path | None,
    paper_url: str | None,
) -> ReadinessItem:
    if paper_path is None:
        return ReadinessItem(
            code="paper_artifact",
            ok=True,
            message="paper artifact was not requested for this candidate",
            evidence=("paper_path=not_requested",),
        )
    blockers: list[str] = []
    if artifacts.get("paper") is None:
        blockers.append("artifact.paper.missing")
    if paper_url is None:
        blockers.append("paper_url.missing")
    return ReadinessItem(
        code="paper_artifact",
        ok=not blockers,
        message="paper artifact and public URL are present"
        if not blockers
        else "paper is incomplete",
        evidence=(
            f"paper_path={_public_report_path(paper_path)}",
            *(() if paper_url is None else (f"paper_url={paper_url}",)),
        ),
        blockers=tuple(blockers),
    )


def _public_link_readiness(
    *,
    required: bool,
    skip_allowed: bool,
    checks: tuple[PublicLinkCheck, ...],
    paper_path: Path | None,
    paper_url: str | None,
) -> ReadinessItem:
    if not required:
        if not skip_allowed:
            return ReadinessItem(
                code="public_links",
                ok=False,
                message="public link checks were skipped for a non-fixture candidate",
                blockers=("public_links.skipped_for_release",),
            )
        return ReadinessItem(
            code="public_links",
            ok=True,
            message="public link checks were skipped for this candidate",
            evidence=("required=false", "fixture_rehearsal=true"),
        )
    required_names = {"model", "dataset", "demo"}
    if paper_path is not None or paper_url is not None:
        required_names.add("paper")
    observed = {check.name for check in checks}
    missing = tuple(sorted(required_names - observed))
    failed = tuple(check.name for check in checks if not check.ok)
    return ReadinessItem(
        code="public_links",
        ok=not missing and not failed,
        message=(
            "required public artifact links are reachable"
            if not missing and not failed
            else "one or more required public artifact links are missing or unreachable"
        ),
        evidence=tuple(
            f"{check.name}=HTTP {check.status_code}"
            if check.status_code is not None
            else check.name
            for check in checks
            if check.ok
        ),
        blockers=(
            *(f"public_link.{name}.missing" for name in missing),
            *(f"public_link.{name}.unreachable" for name in failed),
        ),
    )


def _public_artifact_readiness(
    *,
    required: bool,
    skip_allowed: bool,
    checks: tuple[PublicArtifactCheck, ...],
    hub_plan: HubReleasePlan | None,
) -> ReadinessItem:
    if not required:
        if not skip_allowed:
            return ReadinessItem(
                code="public_artifacts",
                ok=False,
                message="public artifact hash checks were skipped for a non-fixture candidate",
                blockers=("public_artifacts.skipped_for_release",),
            )
        return ReadinessItem(
            code="public_artifacts",
            ok=True,
            message="public artifact hash checks were skipped for this candidate",
            evidence=("required=false", "fixture_rehearsal=true"),
        )
    if hub_plan is None:
        return ReadinessItem(
            code="public_artifacts",
            ok=False,
            message="public artifact hash checks could not run without a Hub plan",
            blockers=("hub_plan.missing",),
        )
    required_names = set(_expected_public_artifacts(hub_plan))
    observed = {check.name for check in checks}
    missing_checks = tuple(sorted(required_names - observed))
    failed = tuple(check for check in checks if not check.ok)
    return ReadinessItem(
        code="public_artifacts",
        ok=not missing_checks and not failed,
        message=(
            "public artifacts are listed and match expected hashes and sizes"
            if not missing_checks and not failed
            else (
                "one or more public artifacts are missing, unexpected, "
                "hash mismatched, or size mismatched"
            )
        ),
        evidence=tuple(
            f"{check.name}={check.verified_count}/{check.expected_count}"
            for check in checks
            if check.ok
        ),
        blockers=(
            *(f"public_artifact.{name}.missing_check" for name in missing_checks),
            *(f"public_artifact.{check.name}.missing_files" for check in failed if check.missing),
            *(
                f"public_artifact.{check.name}.hash_mismatch"
                for check in failed
                if check.hash_mismatches
            ),
            *(
                f"public_artifact.{check.name}.size_mismatch"
                for check in failed
                if check.size_mismatches
            ),
            *(
                f"public_artifact.{check.name}.unexpected_files"
                for check in failed
                if check.unexpected
            ),
            *(
                f"public_artifact.{check.name}.check_failed"
                for check in failed
                if check.error
                and not check.missing
                and not check.hash_mismatches
                and not check.size_mismatches
                and not check.unexpected
            ),
        ),
    )


def _hub_plan_readiness(hub_plan: HubReleasePlan | None) -> ReadinessItem:
    if hub_plan is None:
        return ReadinessItem(
            code="hub_publication_plan",
            ok=False,
            message="Hub publication dry-run plan is missing",
            blockers=("hub_plan.missing",),
        )
    command_text = "\n".join(hub_plan.commands)
    blockers: list[str] = []
    if not hub_plan.files:
        blockers.append("hub_plan.model_files.missing")
    if not hub_plan.dataset_files:
        blockers.append("hub_plan.dataset_files.missing")
    if not hub_plan.demo_files:
        blockers.append("hub_plan.demo_files.missing")
    if "huggingface-cli upload" not in command_text or "--repo-type model" not in command_text:
        blockers.append("hub_plan.model_upload_command.missing")
    if "--repo-type dataset" not in command_text:
        blockers.append("hub_plan.dataset_upload_command.missing")
    if "gh release upload" not in command_text:
        blockers.append("hub_plan.demo_upload_command.missing")
    return ReadinessItem(
        code="hub_publication_plan",
        ok=not blockers,
        message=(
            "Hub dry-run includes model, dataset, and demo upload plans"
            if not blockers
            else "Hub dry-run publication plan is incomplete"
        ),
        evidence=(
            _inventory_evidence("model_files", hub_plan.files),
            _inventory_evidence("dataset_files", hub_plan.dataset_files),
            _inventory_evidence("demo_files", hub_plan.demo_files),
            f"commands={len(hub_plan.commands)}",
        ),
        blockers=tuple(blockers),
    )


def _inventory_evidence(name: str, files: tuple[Any, ...] | None) -> str:
    return f"{name}={0 if files is None else len(files)}"


def _issue_refs_for_readiness(code: str) -> tuple[int, ...]:
    return {
        "package_verifier": ALL_RELEASE_ISSUES,
        "model_package": (TRAINING_ISSUE, EVAL_ISSUE, MODEL_RELEASE_ISSUE),
        "dataset_package": (DATASET_ISSUE,),
        "terminal_demo": (DEMO_ISSUE,),
        "paper_artifact": (PAPER_ISSUE,),
        "public_links": (
            DATASET_ISSUE,
            DEMO_ISSUE,
            PAPER_ISSUE,
            MODEL_RELEASE_ISSUE,
        ),
        "public_artifacts": (
            DATASET_ISSUE,
            DEMO_ISSUE,
            PAPER_ISSUE,
            MODEL_RELEASE_ISSUE,
        ),
        "hub_publication_plan": (PAPER_ISSUE, MODEL_RELEASE_ISSUE),
    }.get(code, ())


def _issue_refs_for_blocker(code: str) -> tuple[int, ...]:
    if code == "package.failed":
        return ALL_RELEASE_ISSUES
    if code.startswith("package.dataset."):
        return (DATASET_ISSUE,)
    if code.startswith("package.demo."):
        return (DEMO_ISSUE,)
    if code.startswith("package.paper."):
        return (PAPER_ISSUE,)
    if code.startswith("package.model.training"):
        return (TRAINING_ISSUE, MODEL_RELEASE_ISSUE)
    if code.startswith(
        (
            "package.model.eval",
            "package.model.efficiency",
            "package.model.baseline",
        )
    ):
        return (EVAL_ISSUE, MODEL_RELEASE_ISSUE)
    if code.startswith("package.model."):
        return (MODEL_RELEASE_ISSUE,)
    if code.startswith(("public_link.dataset.", "public_artifact.dataset.")):
        return (DATASET_ISSUE,)
    if code.startswith(("public_link.demo.", "public_artifact.demo.")):
        return (DEMO_ISSUE,)
    if code.startswith(("public_link.paper.", "public_artifact.paper.")):
        return (PAPER_ISSUE,)
    if code.startswith(("public_link.model.", "public_artifact.model.")):
        return (MODEL_RELEASE_ISSUE,)
    if code.startswith("paper."):
        return (PAPER_ISSUE,)
    if code.startswith("hub_plan.dataset"):
        return (DATASET_ISSUE, MODEL_RELEASE_ISSUE)
    if code.startswith("hub_plan.demo"):
        return (DEMO_ISSUE, MODEL_RELEASE_ISSUE)
    if code.startswith("hub_plan.model"):
        return (MODEL_RELEASE_ISSUE,)
    if code.startswith("hub_plan."):
        return (PAPER_ISSUE, MODEL_RELEASE_ISSUE)
    if code.startswith(("public_links.", "public_artifacts.")):
        return (
            DATASET_ISSUE,
            DEMO_ISSUE,
            PAPER_ISSUE,
            MODEL_RELEASE_ISSUE,
        )
    return ()


def _artifact_identities(
    *,
    model_dir: Path,
    dataset_dir: Path,
    demo_dir: Path,
    paper_path: Path | None,
    manifest: Manifest | None,
) -> dict[str, dict[str, object] | None]:
    model_artifacts: dict[str, dict[str, object] | None] = {
        "predictor": None,
        "action_encoder": None,
        "calibration": None,
        "training_config": None,
        "manifest_eval_report": None,
    }
    if manifest is not None:
        model_artifacts = {
            "predictor": _artifact_identity(
                model_dir / manifest.predictor.file,
                root=model_dir,
                prefix="model",
            ),
            "action_encoder": _artifact_identity(
                model_dir / manifest.action_encoder.file,
                root=model_dir,
                prefix="model",
            ),
            "calibration": _artifact_identity(
                model_dir / manifest.calibration.file,
                root=model_dir,
                prefix="model",
            ),
            "training_config": _artifact_identity(
                model_dir / manifest.training.config_file,
                root=model_dir,
                prefix="model",
            ),
            "manifest_eval_report": _artifact_identity(
                model_dir / manifest.eval.file,
                root=model_dir,
                prefix="model",
            ),
        }
    return {
        "model_manifest": _artifact_identity(
            model_dir / "manifest.json", root=model_dir, prefix="model"
        ),
        "model_package": _artifact_identity(
            model_dir / MODEL_PACKAGE_NAME, root=model_dir, prefix="model"
        ),
        "model_card": _artifact_identity(
            model_dir / "model_card.md", root=model_dir, prefix="model"
        ),
        "model_checksums": _artifact_identity(
            model_dir / "SHA256SUMS", root=model_dir, prefix="model"
        ),
        **model_artifacts,
        "eval_metrics": _artifact_identity(
            model_dir / EVAL_METRICS_NAME, root=model_dir, prefix="model"
        ),
        "eval_config": _artifact_identity(
            model_dir / EVAL_CONFIG_NAME, root=model_dir, prefix="model"
        ),
        "eval_report": _artifact_identity(
            model_dir / "eval_report.md", root=model_dir, prefix="model"
        ),
        "efficiency_report": _artifact_identity(
            model_dir / EFFICIENCY_REPORT_NAME,
            root=model_dir,
            prefix="model",
        ),
        "training_run_manifest": _artifact_identity(
            model_dir / "training_run_manifest.json",
            root=model_dir,
            prefix="model",
        ),
        "training_run_card": _artifact_identity(
            model_dir / "training_run_card.md",
            root=model_dir,
            prefix="model",
        ),
        "training_run_checksums": _artifact_identity(
            model_dir / "training_run_SHA256SUMS",
            root=model_dir,
            prefix="model",
        ),
        "training_preflight_report": _artifact_identity(
            model_dir / TRAINING_PREFLIGHT_REPORT_NAME,
            root=model_dir,
            prefix="model",
        ),
        "dataset_manifest": _artifact_identity(
            dataset_dir / "dataset_manifest.json",
            root=dataset_dir,
            prefix="dataset",
        ),
        "dataset_package": _artifact_identity(
            dataset_dir / "dataset_package.json",
            root=dataset_dir,
            prefix="dataset",
        ),
        "dataset_snapshot_report": _artifact_identity(
            dataset_dir / DATASET_SNAPSHOT_REPORT_NAME,
            root=dataset_dir,
            prefix="dataset",
        ),
        "dataset_input_check_report": _artifact_identity(
            dataset_dir / DATASET_INPUT_CHECK_REPORT_NAME,
            root=dataset_dir,
            prefix="dataset",
        ),
        "data_card": _artifact_identity(
            dataset_dir / "data_card.md", root=dataset_dir, prefix="dataset"
        ),
        "dataset_integrity": _artifact_identity(
            dataset_dir / DATASET_INTEGRITY_NAME,
            root=dataset_dir,
            prefix="dataset",
        ),
        "dataset_checksums": _artifact_identity(
            dataset_dir / "SHA256SUMS",
            root=dataset_dir,
            prefix="dataset",
        ),
        "terminal_transcript": _artifact_identity(
            demo_dir / "terminal-demo-transcript.md",
            root=demo_dir,
            prefix="demo",
        ),
        "terminal_demo_manifest": _artifact_identity(
            demo_dir / DEMO_MANIFEST_NAME,
            root=demo_dir,
            prefix="demo",
        ),
        "runtime_preflight": _artifact_identity(
            demo_dir / RUNTIME_PREFLIGHT_REPORT_NAME,
            root=demo_dir,
            prefix="demo",
        ),
        "batch_receipt_report": _artifact_identity(
            demo_dir / BATCH_RECEIPT_REPORT_NAME,
            root=demo_dir,
            prefix="demo",
        ),
        "scores_jsonl": _artifact_identity(demo_dir / "scores.jsonl", root=demo_dir, prefix="demo"),
        "receipts_jsonl": _artifact_identity(
            demo_dir / "receipts.jsonl", root=demo_dir, prefix="demo"
        ),
        "paper": None if paper_path is None else _artifact_identity(paper_path),
    }


def _artifact_identity(
    path: Path,
    *,
    root: Path | None = None,
    prefix: str | None = None,
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return {
        "path": _public_artifact_path(path, root=root, prefix=prefix),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _public_artifact_path(path: Path, *, root: Path | None, prefix: str | None) -> str:
    if root is not None:
        try:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
            return f"{prefix}/{relative}" if prefix else relative
        except (OSError, RuntimeError, ValueError):
            pass
    return _public_report_path(path)


def _public_report_path(value: str | Path) -> str:
    text = str(value)
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return text
    path = Path(text)
    if not path.is_absolute() and ".." not in path.parts and path.parts:
        return path.as_posix()
    return path.name


def _try_load_manifest(path: Path, blockers: list[CandidateBlocker]) -> Any | None:
    if not path.is_file():
        return None
    try:
        return load_manifest(path)
    except GenoLeWMError as exc:
        blockers.append(
            CandidateBlocker(
                code="model.manifest_invalid",
                path=str(path),
                message=exc.message or str(exc),
            )
        )
        return None


def _dataset_snapshot_id(path: Path, blockers: list[CandidateBlocker]) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(
            CandidateBlocker(
                code="dataset.manifest_invalid",
                path=str(path),
                message=str(exc),
            )
        )
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("snapshot_id")
    return value if isinstance(value, str) and value else None


def _model_url(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id.strip()}"


def _check_public_links(
    urls: dict[str, str | None],
    *,
    timeout_seconds: float,
    probe: Callable[[str, str, float], PublicLinkCheck],
) -> tuple[PublicLinkCheck, ...]:
    checks: list[PublicLinkCheck] = []
    for name in ("model", "dataset", "demo", "paper"):
        url = urls.get(name)
        if url is None:
            continue
        checks.append(probe(name, url, timeout_seconds))
    return tuple(checks)


def _check_public_artifacts(
    *,
    urls: dict[str, str | None],
    hub_plan: HubReleasePlan,
    timeout_seconds: float,
    probe: Callable[[str, str, tuple[UploadFile, ...], float], PublicArtifactCheck],
) -> tuple[PublicArtifactCheck, ...]:
    expected = _expected_public_artifacts(hub_plan, urls=urls)
    checks: list[PublicArtifactCheck] = []
    for name, expected_files in expected.items():
        url = urls.get(name)
        if url is None:
            continue
        checks.append(probe(name, url, expected_files, timeout_seconds))
    return tuple(checks)


def _expected_public_artifacts(
    hub_plan: HubReleasePlan,
    *,
    urls: dict[str, str | None] | None = None,
) -> dict[str, tuple[UploadFile, ...]]:
    expected = {
        "model": hub_plan.files,
        "dataset": hub_plan.dataset_files,
        "demo": _unique_demo_assets(hub_plan.demo_files),
    }
    if hub_plan.paper_file is not None:
        expected["paper"] = (hub_plan.paper_file,)
        if urls is not None and _paper_asset_is_on_demo_release(
            demo_url=urls.get("demo"),
            paper_url=urls.get("paper"),
        ):
            expected["demo"] = _unique_demo_assets((*expected["demo"], hub_plan.paper_file))
    return expected


def _paper_asset_is_on_demo_release(
    *,
    demo_url: str | None,
    paper_url: str | None,
) -> bool:
    if not demo_url or not paper_url:
        return False
    demo = urlparse(demo_url)
    paper = urlparse(paper_url)
    demo_parts = tuple(part for part in demo.path.strip("/").split("/") if part)
    paper_parts = tuple(part for part in paper.path.strip("/").split("/") if part)
    return (
        demo.netloc == "github.com"
        and paper.netloc == "github.com"
        and len(demo_parts) >= 5
        and len(paper_parts) >= 6
        and demo_parts[2:4] == ("releases", "tag")
        and paper_parts[2:4] == ("releases", "download")
        and demo_parts[:2] == paper_parts[:2]
        and demo_parts[4] == paper_parts[4]
    )


def _probe_public_url(name: str, url: str, timeout_seconds: float) -> PublicLinkCheck:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return PublicLinkCheck(name=name, url=url, ok=False, error="URL must use http(s)")
    try:
        status = _open_for_status(url, timeout_seconds, method="HEAD")
    except urllib_error.HTTPError as exc:
        if exc.code not in {403, 405}:
            return PublicLinkCheck(
                name=name,
                url=url,
                ok=False,
                status_code=exc.code,
                error=f"HTTP {exc.code}",
            )
        try:
            status = _open_for_status(url, timeout_seconds, method="GET")
        except urllib_error.HTTPError as fallback_exc:
            return PublicLinkCheck(
                name=name,
                url=url,
                ok=False,
                status_code=fallback_exc.code,
                error=f"HTTP {fallback_exc.code}",
            )
        except (urllib_error.URLError, OSError) as fallback_exc:
            return PublicLinkCheck(
                name=name,
                url=url,
                ok=False,
                error=str(fallback_exc),
            )
    except (urllib_error.URLError, OSError) as exc:
        return PublicLinkCheck(name=name, url=url, ok=False, error=str(exc))
    return PublicLinkCheck(
        name=name,
        url=url,
        ok=200 <= status < 400,
        status_code=status,
        error=None if 200 <= status < 400 else f"HTTP {status}",
    )


def _probe_public_artifacts(
    name: str,
    url: str,
    expected: tuple[UploadFile, ...],
    timeout_seconds: float,
) -> PublicArtifactCheck:
    if name == "paper":
        return _probe_public_paper_artifact(url, expected, timeout_seconds)
    api_url = _artifact_listing_api_url(name, url)
    if api_url is None:
        return PublicArtifactCheck(
            name=name,
            url=url,
            ok=False,
            expected_count=len(expected),
            error="public artifact listing is only supported for Hugging Face repos and GitHub release tags",
        )
    try:
        status, payload = _open_json_for_status(api_url, timeout_seconds)
    except urllib_error.HTTPError as exc:
        return PublicArtifactCheck(
            name=name,
            url=url,
            ok=False,
            expected_count=len(expected),
            status_code=exc.code,
            error=f"HTTP {exc.code}",
        )
    except (json.JSONDecodeError, urllib_error.URLError, OSError) as exc:
        return PublicArtifactCheck(
            name=name,
            url=url,
            ok=False,
            expected_count=len(expected),
            error=str(exc),
        )
    expected_files = _expected_public_file_map(expected)
    observed = _filter_observed_public_artifacts(
        name,
        _observed_public_artifacts(name, payload),
    )
    missing = tuple(sorted(set(expected_files) - set(observed)))
    unexpected = tuple(sorted(set(observed) - set(expected_files)))
    hash_mismatches: list[str] = []
    size_mismatches: list[str] = []
    verified_count = 0
    for destination, expected_file in expected_files.items():
        if destination in missing:
            continue
        download_url = _artifact_download_url(
            name=name,
            listing_url=url,
            destination=destination,
            observed=observed,
        )
        if download_url is None:
            hash_mismatches.append(destination)
            continue
        try:
            observed_hash, observed_size = _hash_and_size_url(download_url, timeout_seconds)
        except (urllib_error.HTTPError, urllib_error.URLError, OSError) as exc:
            hash_mismatches.append(f"{destination} ({exc})")
            continue
        if observed_hash != expected_file.sha256:
            hash_mismatches.append(destination)
            continue
        if observed_size != expected_file.size_bytes:
            size_mismatches.append(destination)
            continue
        verified_count += 1
    return PublicArtifactCheck(
        name=name,
        url=url,
        ok=not missing and not hash_mismatches and not size_mismatches and not unexpected,
        expected_count=len(expected_files),
        observed_count=len(observed),
        verified_count=verified_count,
        missing=missing,
        hash_mismatches=tuple(hash_mismatches),
        size_mismatches=tuple(size_mismatches),
        unexpected=unexpected,
        status_code=status,
        error=None
        if not missing and not hash_mismatches and not size_mismatches and not unexpected
        else "public artifacts are missing, hash mismatched, size mismatched, or unexpected",
    )


def _probe_public_paper_artifact(
    url: str,
    expected: tuple[UploadFile, ...],
    timeout_seconds: float,
) -> PublicArtifactCheck:
    if len(expected) != 1:
        return PublicArtifactCheck(
            name="paper",
            url=url,
            ok=False,
            expected_count=len(expected),
            error="paper artifact check requires exactly one expected paper file",
        )
    expected_file = expected[0]
    try:
        observed_hash, observed_size = _hash_and_size_url(url, timeout_seconds)
    except (urllib_error.HTTPError, urllib_error.URLError, OSError) as exc:
        return PublicArtifactCheck(
            name="paper",
            url=url,
            ok=False,
            expected_count=1,
            observed_count=0,
            verified_count=0,
            hash_mismatches=(f"{expected_file.destination} ({exc})",),
            error=str(exc),
        )
    hash_mismatches = () if observed_hash == expected_file.sha256 else (expected_file.destination,)
    size_mismatches = (
        () if observed_size == expected_file.size_bytes else (expected_file.destination,)
    )
    ok = not hash_mismatches and not size_mismatches
    return PublicArtifactCheck(
        name="paper",
        url=url,
        ok=ok,
        expected_count=1,
        observed_count=1,
        verified_count=1 if ok else 0,
        hash_mismatches=hash_mismatches,
        size_mismatches=size_mismatches,
        status_code=200,
        error=None if ok else "public paper artifact hash or size mismatch",
    )


def _unique_demo_assets(files: tuple[UploadFile, ...]) -> tuple[UploadFile, ...]:
    by_name: dict[str, UploadFile] = {}
    for file in files:
        by_name.setdefault(
            Path(file.destination).name,
            UploadFile(
                source=file.source,
                destination=Path(file.destination).name,
                sha256=file.sha256,
                size_bytes=file.size_bytes,
            ),
        )
    return tuple(by_name[name] for name in sorted(by_name))


def _expected_public_file_map(expected: tuple[UploadFile, ...]) -> dict[str, UploadFile]:
    files: dict[str, UploadFile] = {}
    for file in expected:
        files[file.destination] = file
    return files


def _filter_observed_public_artifacts(
    name: str,
    observed: dict[str, str | None],
) -> dict[str, str | None]:
    ignored = {".gitattributes"} if name in {"model", "dataset"} else set()
    if not ignored:
        return observed
    return {path: url for path, url in observed.items() if path not in ignored}


def _artifact_listing_api_url(name: str, url: str) -> str | None:
    parsed = urlparse(url)
    parts = tuple(part for part in parsed.path.strip("/").split("/") if part)
    if name == "model" and parsed.netloc == "huggingface.co" and len(parts) >= 2:
        return f"https://huggingface.co/api/models/{parts[0]}/{parts[1]}"
    if (
        name == "dataset"
        and parsed.netloc == "huggingface.co"
        and len(parts) >= 3
        and parts[0] == "datasets"
    ):
        return f"https://huggingface.co/api/datasets/{parts[1]}/{parts[2]}"
    if (
        name == "demo"
        and parsed.netloc == "github.com"
        and len(parts) >= 5
        and parts[2:4] == ("releases", "tag")
    ):
        owner, repo, _releases, _tag, tag = parts[:5]
        return f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    return None


def _observed_public_artifacts(name: str, payload: object) -> dict[str, str | None]:
    if not isinstance(payload, dict):
        return {}
    if name in {"model", "dataset"}:
        siblings = payload.get("siblings")
        if not isinstance(siblings, list):
            return {}
        return {
            value: None
            for item in siblings
            if isinstance(item, dict)
            if isinstance(value := item.get("rfilename"), str)
        }
    if name == "demo":
        assets = payload.get("assets")
        if not isinstance(assets, list):
            return {}
        return {
            value: browser_download_url if isinstance(browser_download_url, str) else None
            for item in assets
            if isinstance(item, dict)
            if isinstance(value := item.get("name"), str)
            for browser_download_url in (item.get("browser_download_url"),)
        }
    return {}


def _artifact_download_url(
    *,
    name: str,
    listing_url: str,
    destination: str,
    observed: dict[str, str | None],
) -> str | None:
    if name == "demo":
        return observed.get(destination)
    parsed = urlparse(listing_url)
    parts = tuple(part for part in parsed.path.strip("/").split("/") if part)
    quoted_destination = quote(destination, safe="/")
    if name == "model" and parsed.netloc == "huggingface.co" and len(parts) >= 2:
        return f"https://huggingface.co/{parts[0]}/{parts[1]}/resolve/main/{quoted_destination}"
    if (
        name == "dataset"
        and parsed.netloc == "huggingface.co"
        and len(parts) >= 3
        and parts[0] == "datasets"
    ):
        return (
            f"https://huggingface.co/datasets/{parts[1]}/{parts[2]}"
            f"/resolve/main/{quoted_destination}"
        )
    return None


def _open_for_status(url: str, timeout_seconds: float, *, method: str) -> int:
    request = urllib_request.Request(
        url,
        method=method,
        headers={"User-Agent": "GenoLeWM-release-candidate/1.0"},
    )
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        return int(response.status)


def _open_json_for_status(url: str, timeout_seconds: float) -> tuple[int, object]:
    request = urllib_request.Request(
        url,
        method="GET",
        headers={"User-Agent": "GenoLeWM-release-candidate/1.0"},
    )
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        data = response.read().decode("utf-8")
        return int(response.status), json.loads(data)


def _hash_and_size_url(url: str, timeout_seconds: float) -> tuple[str, int]:
    request = urllib_request.Request(
        url,
        method="GET",
        headers={"User-Agent": "GenoLeWM-release-candidate/1.0"},
    )
    digest = hashlib.sha256()
    size_bytes = 0
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
    return "sha256:" + digest.hexdigest(), size_bytes


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
