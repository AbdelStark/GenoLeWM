# SPDX-License-Identifier: Apache-2.0
"""Bind final Hub publication and clean-machine replay evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import quote, urlparse

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file
from tools.release.clean_machine_demo import (
    DEFAULT_REPORT_NAME as CLEAN_MACHINE_REPORT_NAME,
    GENERATED_BY as CLEAN_MACHINE_GENERATED_BY,
    SCHEMA_VERSION as CLEAN_MACHINE_SCHEMA_VERSION,
)
from tools.release.hub_publish import (
    DEFAULT_PLAN_OUTPUT,
    DEFAULT_PUBLISH_OUTPUT,
    GENERATED_BY as HUB_PUBLISH_GENERATED_BY,
    SCHEMA_VERSION as HUB_PUBLISH_SCHEMA_VERSION,
)
from tools.release.hub_release import (
    GENERATED_BY as HUB_RELEASE_GENERATED_BY,
    SCHEMA_VERSION as HUB_RELEASE_SCHEMA_VERSION,
)
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
from tools.release.release_candidate import (
    GENERATED_BY as RELEASE_CANDIDATE_GENERATED_BY,
    REPORT_NAME as RELEASE_CANDIDATE_REPORT_NAME,
    SCHEMA_VERSION as RELEASE_CANDIDATE_SCHEMA_VERSION,
)

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.publication_report"
DEFAULT_OUTPUT: Final = "publication_evidence_report.json"

Severity = Literal["error", "warning"]
PUBLIC_ARTIFACT_ISSUES: Final = (
    DATASET_ISSUE,
    DEMO_ISSUE,
    PAPER_ISSUE,
    MODEL_RELEASE_ISSUE,
)
PUBLIC_REPLAY_ISSUES: Final = (DATASET_ISSUE, DEMO_ISSUE, MODEL_RELEASE_ISSUE)
DEMO_REPLAY_ISSUES: Final = (DEMO_ISSUE, MODEL_RELEASE_ISSUE)
PUBLISH_ISSUES: Final = (PAPER_ISSUE, MODEL_RELEASE_ISSUE)
REQUIRED_CANDIDATE_ARTIFACTS: Final = (
    "model_manifest",
    "model_package",
    "model_card",
    "model_checksums",
    "predictor",
    "action_encoder",
    "calibration",
    "training_config",
    "training_run_manifest",
    "training_run_card",
    "training_run_checksums",
    "training_preflight_report",
    "eval_metrics",
    "eval_config",
    "eval_report",
    "efficiency_report",
    "dataset_manifest",
    "dataset_package",
    "dataset_snapshot_report",
    "dataset_input_check_report",
    "data_card",
    "dataset_integrity",
    "dataset_checksums",
    "terminal_transcript",
    "terminal_demo_manifest",
    "runtime_preflight",
    "batch_receipt_report",
    "scores_jsonl",
    "receipts_jsonl",
)
CANDIDATE_ARTIFACT_UPLOADS: Final[Mapping[str, tuple[str, str | None]]] = {
    "model_manifest": ("model", "manifest.json"),
    "model_package": ("model", "model_package.json"),
    "model_card": ("model", "model_card.md"),
    "model_checksums": ("model", "SHA256SUMS"),
    "predictor": ("model", None),
    "action_encoder": ("model", None),
    "calibration": ("model", None),
    "training_config": ("model", None),
    "training_run_manifest": ("model", "training_run_manifest.json"),
    "training_run_card": ("model", "training_run_card.md"),
    "training_run_checksums": ("model", "training_run_SHA256SUMS"),
    "training_preflight_report": ("model", "training_preflight_report.json"),
    "eval_metrics": ("model", "eval_metrics.json"),
    "eval_config": ("model", "eval_config.effective.yaml"),
    "eval_report": ("model", "eval_report.md"),
    "efficiency_report": ("model", "efficiency_report.json"),
    "dataset_manifest": ("dataset", "dataset_manifest.json"),
    "dataset_package": ("dataset", "dataset_package.json"),
    "dataset_snapshot_report": ("dataset", "dataset_snapshot_report.json"),
    "dataset_input_check_report": ("dataset", "dataset_input_check_report.json"),
    "data_card": ("dataset", "data_card.md"),
    "dataset_integrity": ("dataset", "split_integrity.json"),
    "dataset_checksums": ("dataset", "SHA256SUMS"),
    "terminal_transcript": ("demo", "terminal-demo-transcript.md"),
    "terminal_demo_manifest": ("demo", "terminal_demo_manifest.json"),
    "runtime_preflight": ("demo", "runtime_preflight_report.json"),
    "batch_receipt_report": ("demo", "batch_receipt_report.json"),
    "scores_jsonl": ("demo", "scores.jsonl"),
    "receipts_jsonl": ("demo", "receipts.jsonl"),
}
CANDIDATE_READINESS_ISSUES: Final = {
    "package_verifier": ALL_RELEASE_ISSUES,
    "model_package": (TRAINING_ISSUE, EVAL_ISSUE, MODEL_RELEASE_ISSUE),
    "dataset_package": (DATASET_ISSUE,),
    "terminal_demo": (DEMO_ISSUE,),
    "paper_artifact": (PAPER_ISSUE,),
    "public_links": (DATASET_ISSUE, DEMO_ISSUE, PAPER_ISSUE, MODEL_RELEASE_ISSUE),
    "public_artifacts": (DATASET_ISSUE, DEMO_ISSUE, PAPER_ISSUE, MODEL_RELEASE_ISSUE),
    "hub_publication_plan": (PAPER_ISSUE, MODEL_RELEASE_ISSUE),
}
CANDIDATE_PUBLIC_LINK_ISSUES: Final = {
    "model": (MODEL_RELEASE_ISSUE,),
    "dataset": (DATASET_ISSUE,),
    "demo": (DEMO_ISSUE,),
    "paper": (PAPER_ISSUE,),
}
CANDIDATE_PUBLIC_ARTIFACT_ISSUES: Final = {
    "model": (MODEL_RELEASE_ISSUE,),
    "dataset": (DATASET_ISSUE,),
    "demo": (DEMO_ISSUE,),
    "paper": (PAPER_ISSUE,),
}
CANDIDATE_ARTIFACT_ISSUES: Final[Mapping[str, tuple[int, ...]]] = {
    "model_manifest": (MODEL_RELEASE_ISSUE,),
    "model_package": (TRAINING_ISSUE, EVAL_ISSUE, MODEL_RELEASE_ISSUE),
    "model_card": (PAPER_ISSUE, MODEL_RELEASE_ISSUE),
    "model_checksums": (MODEL_RELEASE_ISSUE,),
    "predictor": (TRAINING_ISSUE, MODEL_RELEASE_ISSUE),
    "action_encoder": (TRAINING_ISSUE, MODEL_RELEASE_ISSUE),
    "calibration": (TRAINING_ISSUE, MODEL_RELEASE_ISSUE),
    "training_config": (TRAINING_ISSUE, MODEL_RELEASE_ISSUE),
    "training_run_manifest": (TRAINING_ISSUE, MODEL_RELEASE_ISSUE),
    "training_run_card": (TRAINING_ISSUE, PAPER_ISSUE, MODEL_RELEASE_ISSUE),
    "training_run_checksums": (TRAINING_ISSUE, MODEL_RELEASE_ISSUE),
    "training_preflight_report": (TRAINING_ISSUE, MODEL_RELEASE_ISSUE),
    "eval_metrics": (EVAL_ISSUE, PAPER_ISSUE, MODEL_RELEASE_ISSUE),
    "eval_config": (EVAL_ISSUE, PAPER_ISSUE, MODEL_RELEASE_ISSUE),
    "eval_report": (EVAL_ISSUE, PAPER_ISSUE, MODEL_RELEASE_ISSUE),
    "efficiency_report": (EVAL_ISSUE, PAPER_ISSUE, MODEL_RELEASE_ISSUE),
    "dataset_manifest": (DATASET_ISSUE,),
    "dataset_package": (DATASET_ISSUE,),
    "dataset_snapshot_report": (DATASET_ISSUE, PAPER_ISSUE),
    "dataset_input_check_report": (DATASET_ISSUE, PAPER_ISSUE),
    "data_card": (DATASET_ISSUE, PAPER_ISSUE),
    "dataset_integrity": (DATASET_ISSUE,),
    "dataset_checksums": (DATASET_ISSUE,),
    "terminal_transcript": (DEMO_ISSUE, PAPER_ISSUE),
    "terminal_demo_manifest": DEMO_REPLAY_ISSUES,
    "runtime_preflight": DEMO_REPLAY_ISSUES,
    "batch_receipt_report": DEMO_REPLAY_ISSUES,
    "scores_jsonl": DEMO_REPLAY_ISSUES,
    "receipts_jsonl": DEMO_REPLAY_ISSUES,
    "paper": PUBLISH_ISSUES,
}


@dataclass(frozen=True, slots=True)
class ReportIdentity:
    """File identity for a top-level publication evidence report."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PublicationIssue:
    """One publication evidence consistency issue."""

    severity: Severity
    code: str
    path: str
    message: str

    @property
    def issue_refs(self) -> tuple[int, ...]:
        """Live GitHub release blockers that own this failure mode."""
        return _issue_refs_for_publication_issue(self.code)

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "issue_refs": issue_ref_payload(self.issue_refs),
        }


@dataclass(frozen=True, slots=True)
class PublicationEvidenceReport:
    """Machine-readable final publication evidence binder."""

    schema_version: str
    generated_by: str
    generated_at: str
    ok: bool
    source_reports: dict[str, ReportIdentity]
    model_id: str | None
    release_id: str | None
    urls: dict[str, str]
    paper_artifact: dict[str, Any] | None
    release_candidate_artifacts: dict[str, dict[str, Any]]
    release_candidate_readiness: tuple[dict[str, Any], ...]
    release_candidate_public_links: dict[str, Any]
    release_candidate_public_artifacts: dict[str, Any]
    downloaded_artifact_count: int
    downloaded_artifacts: tuple[dict[str, Any], ...]
    replay_artifact_count: int
    replay_artifacts: tuple[dict[str, Any], ...]
    issues: tuple[PublicationIssue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "ok": self.ok,
            "source_reports": {
                name: identity.to_dict() for name, identity in self.source_reports.items()
            },
            "model_id": self.model_id,
            "release_id": self.release_id,
            "urls": self.urls,
            "paper_artifact": self.paper_artifact,
            "release_candidate_artifacts": self.release_candidate_artifacts,
            "release_candidate_readiness": list(self.release_candidate_readiness),
            "release_candidate_public_links": self.release_candidate_public_links,
            "release_candidate_public_artifacts": self.release_candidate_public_artifacts,
            "downloaded_artifact_count": self.downloaded_artifact_count,
            "downloaded_artifacts": list(self.downloaded_artifacts),
            "replay_artifact_count": self.replay_artifact_count,
            "replay_artifacts": list(self.replay_artifacts),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def build_publication_evidence_report(
    *,
    plan_path: Path,
    release_candidate_path: Path,
    publish_report_path: Path,
    clean_machine_report_path: Path,
    generated_at: str | None = None,
) -> PublicationEvidenceReport:
    """Build the final evidence report after public replay succeeds."""
    issues: list[PublicationIssue] = []
    plan = _load_json_object(plan_path, "hub release plan")
    candidate = _load_json_object(release_candidate_path, "release candidate report")
    publish = _load_json_object(publish_report_path, "Hub publish report")
    clean_machine = _load_json_object(clean_machine_report_path, "clean-machine demo report")

    _verify_source_report_headers(
        plan=plan,
        plan_path=plan_path,
        candidate=candidate,
        release_candidate_path=release_candidate_path,
        publish=publish,
        publish_report_path=publish_report_path,
        clean_machine=clean_machine,
        clean_machine_report_path=clean_machine_report_path,
        issues=issues,
    )
    _verify_plan_candidate_identity(
        plan=plan,
        plan_path=plan_path,
        candidate=candidate,
        release_candidate_path=release_candidate_path,
        issues=issues,
    )
    _verify_plan_upload_inventories(plan, plan_path, issues)
    paper_artifact = _verify_plan_paper_file(plan, plan_path, issues)
    release_candidate_artifacts = _release_candidate_artifacts(
        candidate,
        release_candidate_path,
        issues,
    )
    if candidate.get("hub_plan") != plan:
        _issue(
            issues,
            "error",
            "candidate.hub_plan_mismatch",
            release_candidate_path,
            "release_candidate_report.json hub_plan does not match hub_release_plan.json",
        )
    if publish.get("plan") != plan:
        _issue(
            issues,
            "error",
            "publish.plan_mismatch",
            publish_report_path,
            "hub_publish_report.json plan does not match hub_release_plan.json",
        )
    if publish.get("final_candidate_report") != candidate:
        _issue(
            issues,
            "error",
            "publish.candidate_mismatch",
            publish_report_path,
            "hub_publish_report.json final candidate does not match release_candidate_report.json",
        )
    if publish.get("final_candidate_ready") is not True:
        _issue(
            issues,
            "error",
            "publish.final_candidate_not_ready",
            publish_report_path,
            "hub_publish_report.json did not record final_candidate_ready=true",
        )
    if candidate.get("ready") is not True:
        _issue(
            issues,
            "error",
            "candidate.not_ready",
            release_candidate_path,
            "release_candidate_report.json must have ready=true",
        )
    _verify_candidate_readiness(
        candidate,
        release_candidate_path,
        issues,
    )
    _verify_candidate_public_checks(
        candidate,
        plan,
        release_candidate_path,
        issues,
    )
    package = clean_machine.get("package")
    if not isinstance(package, dict) or package.get("ok") is not True:
        _issue(
            issues,
            "error",
            "clean_machine.package_not_ok",
            clean_machine_report_path,
            "clean-machine package verification must pass",
        )
    _verify_clean_machine_candidate_identity(
        clean_machine=clean_machine,
        clean_machine_report_path=clean_machine_report_path,
        release_candidate_path=release_candidate_path,
        issues=issues,
    )
    replay_artifacts = _replay_artifacts(clean_machine, clean_machine_report_path, issues)
    downloaded_artifacts = _downloaded_artifacts(clean_machine, clean_machine_report_path, issues)
    _verify_replay_demo_manifest_identity(
        candidate=candidate,
        clean_machine_report_path=clean_machine_report_path,
        replay_artifacts=replay_artifacts,
        downloaded_artifacts=downloaded_artifacts,
        issues=issues,
    )
    _verify_downloaded_artifact_coverage(
        plan=plan,
        clean_machine=clean_machine,
        clean_machine_report_path=clean_machine_report_path,
        downloaded_artifacts=downloaded_artifacts,
        issues=issues,
    )
    _verify_release_candidate_artifact_publication(
        plan=plan,
        release_candidate_artifacts=release_candidate_artifacts,
        downloaded_artifacts=downloaded_artifacts,
        clean_machine_report_path=clean_machine_report_path,
        issues=issues,
    )
    public_downloaded_artifacts = _public_artifact_paths(
        downloaded_artifacts,
        report_path=clean_machine_report_path,
    )
    public_replay_artifacts = _public_artifact_paths(
        replay_artifacts,
        report_path=clean_machine_report_path,
    )

    return PublicationEvidenceReport(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        generated_at=generated_at or _utc_now(),
        ok=not any(issue.severity == "error" for issue in issues),
        source_reports={
            "hub_release_plan": _identity(plan_path, reported_path=plan_path.name),
            "release_candidate": _identity(
                release_candidate_path,
                reported_path=release_candidate_path.name,
            ),
            "hub_publish": _identity(publish_report_path, reported_path=publish_report_path.name),
            "clean_machine_demo": _identity(
                clean_machine_report_path,
                reported_path=clean_machine_report_path.name,
            ),
        },
        model_id=_optional_str(candidate.get("model_id")),
        release_id=_optional_str(candidate.get("release_id")),
        urls=_string_dict(candidate.get("urls")),
        paper_artifact=paper_artifact,
        release_candidate_artifacts=release_candidate_artifacts,
        release_candidate_readiness=_release_candidate_readiness_summary(candidate),
        release_candidate_public_links=_release_candidate_public_links_summary(candidate),
        release_candidate_public_artifacts=_release_candidate_public_artifacts_summary(candidate),
        downloaded_artifact_count=len(public_downloaded_artifacts),
        downloaded_artifacts=public_downloaded_artifacts,
        replay_artifact_count=len(public_replay_artifacts),
        replay_artifacts=public_replay_artifacts,
        issues=tuple(issues),
    )


def write_publication_evidence_report(
    *,
    plan_path: Path,
    release_candidate_path: Path,
    publish_report_path: Path,
    clean_machine_report_path: Path,
    output: Path,
    generated_at: str | None = None,
) -> PublicationEvidenceReport:
    """Build and write ``publication_evidence_report.json``."""
    report = build_publication_evidence_report(
        plan_path=plan_path,
        release_candidate_path=release_candidate_path,
        publish_report_path=publish_report_path,
        clean_machine_report_path=clean_machine_report_path,
        generated_at=generated_at,
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
        report = write_publication_evidence_report(
            plan_path=args.plan,
            release_candidate_path=args.release_candidate,
            publish_report_path=args.publish_report,
            clean_machine_report_path=args.clean_machine_demo_report,
            output=args.output,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(f"wrote {args.output}\n")
    return 0 if report.ok else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind Hub publication and clean-machine replay reports.",
    )
    parser.add_argument("--plan", type=Path, default=Path(DEFAULT_PLAN_OUTPUT))
    parser.add_argument(
        "--release-candidate",
        type=Path,
        default=Path(RELEASE_CANDIDATE_REPORT_NAME),
    )
    parser.add_argument("--publish-report", type=Path, default=Path(DEFAULT_PUBLISH_OUTPUT))
    parser.add_argument(
        "--clean-machine-demo-report",
        type=Path,
        default=Path("clean-machine-public-replay") / CLEAN_MACHINE_REPORT_NAME,
    )
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    return parser


def _verify_source_report_headers(
    *,
    plan: dict[str, Any],
    plan_path: Path,
    candidate: dict[str, Any],
    release_candidate_path: Path,
    publish: dict[str, Any],
    publish_report_path: Path,
    clean_machine: dict[str, Any],
    clean_machine_report_path: Path,
    issues: list[PublicationIssue],
) -> None:
    _require_report_header(
        plan,
        plan_path,
        code_prefix="plan",
        schema_version=HUB_RELEASE_SCHEMA_VERSION,
        generated_by=HUB_RELEASE_GENERATED_BY,
        issues=issues,
    )
    _require_report_header(
        candidate,
        release_candidate_path,
        code_prefix="candidate",
        schema_version=RELEASE_CANDIDATE_SCHEMA_VERSION,
        generated_by=RELEASE_CANDIDATE_GENERATED_BY,
        issues=issues,
    )
    _require_report_header(
        publish,
        publish_report_path,
        code_prefix="publish",
        schema_version=HUB_PUBLISH_SCHEMA_VERSION,
        generated_by=HUB_PUBLISH_GENERATED_BY,
        issues=issues,
    )
    _require_report_header(
        clean_machine,
        clean_machine_report_path,
        code_prefix="clean_machine",
        schema_version=CLEAN_MACHINE_SCHEMA_VERSION,
        generated_by=CLEAN_MACHINE_GENERATED_BY,
        issues=issues,
    )


def _require_report_header(
    payload: dict[str, Any],
    path: Path,
    *,
    code_prefix: str,
    schema_version: str,
    generated_by: str,
    issues: list[PublicationIssue],
) -> None:
    if payload.get("schema_version") != schema_version:
        _issue(
            issues,
            "error",
            f"{code_prefix}.schema_version",
            path,
            f"schema_version must be {schema_version}",
        )
    if payload.get("generated_by") != generated_by:
        _issue(
            issues,
            "error",
            f"{code_prefix}.generated_by",
            path,
            f"generated_by must be {generated_by}",
        )


def _verify_plan_candidate_identity(
    *,
    plan: dict[str, Any],
    plan_path: Path,
    candidate: dict[str, Any],
    release_candidate_path: Path,
    issues: list[PublicationIssue],
) -> None:
    if plan.get("ready") is not True:
        _issue(
            issues,
            "error",
            "plan.not_ready",
            plan_path,
            "hub_release_plan.json must have ready=true",
        )
    for key in ("release_id", "model_id", "commit_sha", "repo_id"):
        if plan.get(key) != candidate.get(key):
            _issue(
                issues,
                "error",
                f"plan.{key}_mismatch",
                plan_path,
                f"hub_release_plan.json {key} must match release_candidate_report.json",
            )
    urls = candidate.get("urls")
    if not isinstance(urls, dict):
        _issue(
            issues,
            "error",
            "candidate.urls",
            release_candidate_path,
            "release_candidate_report.json urls must be an object",
        )
        return
    expected_model_url = None
    repo_id = plan.get("repo_id")
    if isinstance(repo_id, str) and repo_id:
        expected_model_url = f"https://huggingface.co/{repo_id}"
    _compare_url_identity(
        observed=urls.get("model"),
        expected=expected_model_url,
        plan_path=plan_path,
        issues=issues,
        code="plan.model_url_mismatch",
        message="candidate model URL must match the Hub repo id in hub_release_plan.json",
    )
    for key, code in (
        ("dataset_url", "plan.dataset_url_mismatch"),
        ("demo_url", "plan.demo_url_mismatch"),
        ("paper_url", "plan.paper_url_mismatch"),
    ):
        url_key = key.removesuffix("_url")
        _compare_url_identity(
            observed=urls.get(url_key),
            expected=plan.get(key),
            plan_path=plan_path,
            issues=issues,
            code=code,
            message=f"candidate {url_key} URL must match hub_release_plan.json {key}",
        )


def _compare_url_identity(
    *,
    observed: object,
    expected: object,
    plan_path: Path,
    issues: list[PublicationIssue],
    code: str,
    message: str,
) -> None:
    if expected is None:
        if observed not in {None, ""}:
            _issue(issues, "error", code, plan_path, message)
        return
    if observed != expected:
        _issue(issues, "error", code, plan_path, message)


def _release_candidate_artifacts(
    candidate: dict[str, Any],
    release_candidate_path: Path,
    issues: list[PublicationIssue],
) -> dict[str, dict[str, Any]]:
    raw = candidate.get("artifacts")
    if not isinstance(raw, dict):
        _issue(
            issues,
            "error",
            "candidate.artifacts",
            release_candidate_path,
            "release_candidate_report.json artifacts must be an object",
        )
        return {}
    artifacts: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            _issue(
                issues,
                "error",
                "candidate.artifacts.key",
                release_candidate_path,
                "release_candidate_report.json artifact names must be non-empty strings",
            )
            continue
        if value is None:
            continue
        if not isinstance(value, dict):
            _issue(
                issues,
                "error",
                f"candidate.artifacts.{key}",
                release_candidate_path,
                f"release_candidate_report.json artifact {key} must be an object",
            )
            continue
        identity = _release_candidate_artifact_identity(
            value,
            release_candidate_path,
            issues,
            key=key,
        )
        if identity is not None:
            artifacts[key] = identity
    required = list(REQUIRED_CANDIDATE_ARTIFACTS)
    urls = candidate.get("urls")
    if isinstance(urls, dict) and urls.get("paper") not in {None, ""}:
        required.append("paper")
    for key in required:
        if key not in artifacts:
            _issue(
                issues,
                "error",
                f"candidate.artifacts.{key}.missing",
                release_candidate_path,
                f"release_candidate_report.json must include artifact identity for {key}",
            )
    return artifacts


def _verify_candidate_readiness(
    candidate: dict[str, Any],
    release_candidate_path: Path,
    issues: list[PublicationIssue],
) -> None:
    raw_blockers = candidate.get("blockers")
    if not isinstance(raw_blockers, list):
        _issue(
            issues,
            "error",
            "candidate.blockers",
            release_candidate_path,
            "release_candidate_report.json must include a blockers list",
        )
    elif raw_blockers:
        _issue(
            issues,
            "error",
            "candidate.blockers_not_empty",
            release_candidate_path,
            "release_candidate_report.json must not carry blockers when ready=true",
        )

    raw_readiness = candidate.get("readiness")
    if not isinstance(raw_readiness, list) or not raw_readiness:
        _issue(
            issues,
            "error",
            "candidate.readiness_missing",
            release_candidate_path,
            "release_candidate_report.json must include the readiness checklist",
        )
        return

    observed: dict[str, int] = {}
    for index, item in enumerate(raw_readiness):
        if not isinstance(item, dict):
            _issue(
                issues,
                "error",
                "candidate.readiness.item",
                release_candidate_path,
                f"readiness[{index}] must be an object",
            )
            continue
        code = item.get("code")
        if not isinstance(code, str) or not code:
            _issue(
                issues,
                "error",
                "candidate.readiness.code",
                release_candidate_path,
                f"readiness[{index}] code must be a non-empty string",
            )
            continue
        if code not in CANDIDATE_READINESS_ISSUES:
            _issue(
                issues,
                "error",
                "candidate.readiness.unknown",
                release_candidate_path,
                f"release_candidate_report.json contains unknown readiness row: {code}",
            )
            continue
        previous_index = observed.get(code)
        if previous_index is not None:
            _issue(
                issues,
                "error",
                f"candidate.readiness.{code}.duplicate",
                release_candidate_path,
                (
                    f"readiness[{index}] duplicates readiness row {code} "
                    f"from readiness[{previous_index}]"
                ),
            )
            continue
        observed[code] = index
        if item.get("ok") is not True:
            _issue(
                issues,
                "error",
                f"candidate.readiness.{code}.not_ok",
                release_candidate_path,
                f"release_candidate_report.json readiness row {code} must have ok=true",
            )
        raw_item_blockers = item.get("blockers")
        if not isinstance(raw_item_blockers, list):
            _issue(
                issues,
                "error",
                f"candidate.readiness.{code}.blockers",
                release_candidate_path,
                f"readiness row {code} must include a blockers list",
            )
        elif raw_item_blockers:
            _issue(
                issues,
                "error",
                f"candidate.readiness.{code}.blockers_present",
                release_candidate_path,
                f"readiness row {code} must not carry blockers when ready=true",
            )
        expected_issue_refs = issue_ref_payload(CANDIDATE_READINESS_ISSUES[code])
        if item.get("issue_refs") != expected_issue_refs:
            _issue(
                issues,
                "error",
                f"candidate.readiness.{code}.issue_refs",
                release_candidate_path,
                f"readiness row {code} issue_refs must match the live release tracker map",
            )

    for code in CANDIDATE_READINESS_ISSUES:
        if code not in observed:
            _issue(
                issues,
                "error",
                f"candidate.readiness.{code}.missing",
                release_candidate_path,
                f"release_candidate_report.json is missing readiness row {code}",
            )


def _verify_candidate_public_checks(
    candidate: dict[str, Any],
    plan: dict[str, Any],
    release_candidate_path: Path,
    issues: list[PublicationIssue],
) -> None:
    paper_required = plan.get("paper_url") not in {None, ""}
    urls = candidate.get("urls")
    if isinstance(urls, dict) and urls.get("paper") not in {None, ""}:
        paper_required = True
    link_names = ("model", "dataset", "demo", *(("paper",) if paper_required else ()))
    _verify_candidate_public_links(
        candidate,
        release_candidate_path,
        issues,
        expected_names=link_names,
    )
    artifact_names = ("model", "dataset", "demo", *(("paper",) if paper_required else ()))
    _verify_candidate_public_artifacts(
        candidate,
        release_candidate_path,
        issues,
        expected_names=artifact_names,
    )


def _verify_candidate_public_links(
    candidate: dict[str, Any],
    release_candidate_path: Path,
    issues: list[PublicationIssue],
    *,
    expected_names: tuple[str, ...],
) -> None:
    raw = candidate.get("public_links")
    if not isinstance(raw, dict):
        _issue(
            issues,
            "error",
            "candidate.public_links",
            release_candidate_path,
            "release_candidate_report.json must include public_links",
        )
        return
    if raw.get("required") is not True:
        _issue(
            issues,
            "error",
            "candidate.public_links.required",
            release_candidate_path,
            "release_candidate_report.json must require public link checks",
        )
    checks = raw.get("checks")
    if not isinstance(checks, list) or not checks:
        _issue(
            issues,
            "error",
            "candidate.public_links.checks",
            release_candidate_path,
            "release_candidate_report.json public_links.checks must be a non-empty list",
        )
        return
    urls = candidate.get("urls")
    observed: dict[str, int] = {}
    expected = set(expected_names)
    for index, item in enumerate(checks):
        if not isinstance(item, dict):
            _issue(
                issues,
                "error",
                "candidate.public_links.item",
                release_candidate_path,
                f"public_links.checks[{index}] must be an object",
            )
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            _issue(
                issues,
                "error",
                "candidate.public_links.name",
                release_candidate_path,
                f"public_links.checks[{index}] name must be a non-empty string",
            )
            continue
        if name not in expected:
            _issue(
                issues,
                "error",
                "candidate.public_links.unexpected",
                release_candidate_path,
                f"release_candidate_report.json has unexpected public link check: {name}",
            )
            continue
        previous_index = observed.get(name)
        if previous_index is not None:
            _issue(
                issues,
                "error",
                f"candidate.public_links.{name}.duplicate",
                release_candidate_path,
                (
                    f"public_links.checks[{index}] duplicates {name} "
                    f"from public_links.checks[{previous_index}]"
                ),
            )
            continue
        observed[name] = index
        if item.get("ok") is not True:
            _issue(
                issues,
                "error",
                f"candidate.public_links.{name}.not_ok",
                release_candidate_path,
                f"public link check {name} must have ok=true",
            )
        status_code = item.get("status_code")
        if (
            item.get("ok") is True
            and status_code is not None
            and (
                not isinstance(status_code, int)
                or isinstance(status_code, bool)
                or not 200 <= status_code < 400
            )
        ):
            _issue(
                issues,
                "error",
                f"candidate.public_links.{name}.status_code",
                release_candidate_path,
                f"public link check {name} status_code must be a successful HTTP status",
            )
        expected_url = urls.get(name) if isinstance(urls, dict) else None
        if isinstance(expected_url, str) and item.get("url") != expected_url:
            _issue(
                issues,
                "error",
                f"candidate.public_links.{name}.url_mismatch",
                release_candidate_path,
                f"public link check {name} URL must match release_candidate_report.json urls",
            )
    for name in expected_names:
        if name not in observed:
            _issue(
                issues,
                "error",
                f"candidate.public_links.{name}.missing",
                release_candidate_path,
                f"release_candidate_report.json is missing public link check {name}",
            )


def _verify_candidate_public_artifacts(
    candidate: dict[str, Any],
    release_candidate_path: Path,
    issues: list[PublicationIssue],
    *,
    expected_names: tuple[str, ...],
) -> None:
    raw = candidate.get("public_artifacts")
    if not isinstance(raw, dict):
        _issue(
            issues,
            "error",
            "candidate.public_artifacts",
            release_candidate_path,
            "release_candidate_report.json must include public_artifacts",
        )
        return
    if raw.get("required") is not True:
        _issue(
            issues,
            "error",
            "candidate.public_artifacts.required",
            release_candidate_path,
            "release_candidate_report.json must require public artifact checks",
        )
    checks = raw.get("checks")
    if not isinstance(checks, list) or not checks:
        _issue(
            issues,
            "error",
            "candidate.public_artifacts.checks",
            release_candidate_path,
            "release_candidate_report.json public_artifacts.checks must be a non-empty list",
        )
        return
    observed: dict[str, int] = {}
    expected = set(expected_names)
    for index, item in enumerate(checks):
        if not isinstance(item, dict):
            _issue(
                issues,
                "error",
                "candidate.public_artifacts.item",
                release_candidate_path,
                f"public_artifacts.checks[{index}] must be an object",
            )
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            _issue(
                issues,
                "error",
                "candidate.public_artifacts.name",
                release_candidate_path,
                f"public_artifacts.checks[{index}] name must be a non-empty string",
            )
            continue
        if name not in expected:
            _issue(
                issues,
                "error",
                "candidate.public_artifacts.unexpected",
                release_candidate_path,
                f"release_candidate_report.json has unexpected public artifact check: {name}",
            )
            continue
        previous_index = observed.get(name)
        if previous_index is not None:
            _issue(
                issues,
                "error",
                f"candidate.public_artifacts.{name}.duplicate",
                release_candidate_path,
                (
                    f"public_artifacts.checks[{index}] duplicates {name} "
                    f"from public_artifacts.checks[{previous_index}]"
                ),
            )
            continue
        observed[name] = index
        if item.get("ok") is not True:
            _issue(
                issues,
                "error",
                f"candidate.public_artifacts.{name}.not_ok",
                release_candidate_path,
                f"public artifact check {name} must have ok=true",
            )
        expected_count = item.get("expected_count")
        observed_count = item.get("observed_count")
        verified_count = item.get("verified_count")
        if (
            not isinstance(expected_count, int)
            or isinstance(expected_count, bool)
            or expected_count <= 0
            or observed_count != expected_count
            or verified_count != expected_count
        ):
            _issue(
                issues,
                "error",
                f"candidate.public_artifacts.{name}.count_mismatch",
                release_candidate_path,
                (f"public artifact check {name} must verify every expected artifact exactly once"),
            )
        for field, code_suffix in (
            ("missing", "missing_files"),
            ("hash_mismatches", "hash_mismatch"),
            ("size_mismatches", "size_mismatch"),
            ("unexpected", "unexpected_files"),
        ):
            value = item.get(field)
            if not isinstance(value, list):
                _issue(
                    issues,
                    "error",
                    f"candidate.public_artifacts.{name}.{field}",
                    release_candidate_path,
                    f"public artifact check {name} {field} must be a list",
                )
            elif value:
                _issue(
                    issues,
                    "error",
                    f"candidate.public_artifacts.{name}.{code_suffix}",
                    release_candidate_path,
                    f"public artifact check {name} must not report {field}",
                )
    for name in expected_names:
        if name not in observed:
            _issue(
                issues,
                "error",
                f"candidate.public_artifacts.{name}.missing",
                release_candidate_path,
                f"release_candidate_report.json is missing public artifact check {name}",
            )


def _release_candidate_readiness_summary(candidate: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = candidate.get("readiness")
    if not isinstance(raw, list):
        return ()
    items: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        code = item.get("code")
        if not isinstance(code, str) or not code:
            continue
        items.append(
            {
                "code": code,
                "ok": item.get("ok") is True,
                "message": _optional_str(item.get("message")),
                "evidence": _string_list(item.get("evidence")),
                "blockers": _string_list(item.get("blockers")),
                "issue_refs": _issue_ref_list(item.get("issue_refs")),
            }
        )
    return tuple(items)


def _release_candidate_public_links_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    raw = candidate.get("public_links")
    if not isinstance(raw, dict):
        return {"required": False, "checks": []}
    checks = raw.get("checks")
    return {
        "required": raw.get("required") is True,
        "checks": [_public_link_check_summary(item) for item in checks if isinstance(item, dict)]
        if isinstance(checks, list)
        else [],
    }


def _public_link_check_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _optional_str(item.get("name")),
        "url": _optional_str(item.get("url")),
        "ok": item.get("ok") is True,
        "status_code": _optional_positive_int(item.get("status_code")),
        "error": _optional_str(item.get("error")),
    }


def _release_candidate_public_artifacts_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    raw = candidate.get("public_artifacts")
    if not isinstance(raw, dict):
        return {"required": False, "checks": []}
    checks = raw.get("checks")
    return {
        "required": raw.get("required") is True,
        "checks": [
            _public_artifact_check_summary(item) for item in checks if isinstance(item, dict)
        ]
        if isinstance(checks, list)
        else [],
    }


def _public_artifact_check_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _optional_str(item.get("name")),
        "url": _optional_str(item.get("url")),
        "ok": item.get("ok") is True,
        "expected_count": _optional_nonnegative_int(item.get("expected_count")),
        "observed_count": _optional_nonnegative_int(item.get("observed_count")),
        "verified_count": _optional_nonnegative_int(item.get("verified_count")),
        "missing": _string_list(item.get("missing")),
        "hash_mismatches": _string_list(item.get("hash_mismatches")),
        "size_mismatches": _string_list(item.get("size_mismatches")),
        "unexpected": _string_list(item.get("unexpected")),
        "status_code": _optional_positive_int(item.get("status_code")),
        "error": _optional_str(item.get("error")),
    }


def _release_candidate_artifact_identity(
    raw: dict[str, Any],
    release_candidate_path: Path,
    issues: list[PublicationIssue],
    *,
    key: str,
) -> dict[str, Any] | None:
    path = raw.get("path")
    sha256 = raw.get("sha256")
    size_bytes = raw.get("size_bytes")
    if not isinstance(path, str) or not path:
        _issue(
            issues,
            "error",
            f"candidate.artifacts.{key}.path",
            release_candidate_path,
            f"release_candidate_report.json artifact {key} path is required",
        )
        return None
    path_obj = Path(path)
    if path_obj.is_absolute() or ".." in path_obj.parts or not path_obj.parts:
        _issue(
            issues,
            "error",
            f"candidate.artifacts.{key}.path",
            release_candidate_path,
            f"release_candidate_report.json artifact {key} path must be public-safe",
        )
        return None
    if not isinstance(sha256, str) or not _looks_like_sha256(sha256):
        _issue(
            issues,
            "error",
            f"candidate.artifacts.{key}.sha256",
            release_candidate_path,
            f"release_candidate_report.json artifact {key} sha256 must be sha256:<64hex>",
        )
        return None
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
        _issue(
            issues,
            "error",
            f"candidate.artifacts.{key}.size_bytes",
            release_candidate_path,
            f"release_candidate_report.json artifact {key} size_bytes must be positive",
        )
        return None
    return {
        "path": path_obj.as_posix(),
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def _verify_clean_machine_candidate_identity(
    *,
    clean_machine: dict[str, Any],
    clean_machine_report_path: Path,
    release_candidate_path: Path,
    issues: list[PublicationIssue],
) -> None:
    candidate_identity = clean_machine.get("release_candidate_report_identity")
    if not isinstance(candidate_identity, dict):
        _issue(
            issues,
            "error",
            "clean_machine.candidate_identity_missing",
            clean_machine_report_path,
            "clean-machine report must include release_candidate_report_identity",
        )
        return
    if candidate_identity.get("label") != "release candidate report":
        _issue(
            issues,
            "error",
            "clean_machine.candidate_identity_label",
            clean_machine_report_path,
            "release_candidate_report_identity label must be release candidate report",
        )
    raw_report_path = clean_machine.get("release_candidate_report")
    if not isinstance(raw_report_path, str) or not raw_report_path:
        _issue(
            issues,
            "error",
            "clean_machine.candidate_path_missing",
            clean_machine_report_path,
            "clean-machine report must include release_candidate_report",
        )
    elif candidate_identity.get("path") != raw_report_path:
        _issue(
            issues,
            "error",
            "clean_machine.candidate_identity_path",
            clean_machine_report_path,
            "release_candidate_report_identity path must match release_candidate_report",
        )
    if candidate_identity.get("sha256") != sha256_file(release_candidate_path):
        _issue(
            issues,
            "error",
            "clean_machine.candidate_hash_mismatch",
            clean_machine_report_path,
            "clean-machine report was not replayed from this release_candidate_report.json",
        )
    size_bytes = candidate_identity.get("size_bytes")
    expected_size = release_candidate_path.stat().st_size
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
        _issue(
            issues,
            "error",
            "clean_machine.candidate_size_bytes",
            clean_machine_report_path,
            "release_candidate_report_identity size_bytes must be positive",
        )
    elif size_bytes != expected_size:
        _issue(
            issues,
            "error",
            "clean_machine.candidate_size_mismatch",
            clean_machine_report_path,
            "clean-machine report release_candidate_report_identity size does not match release_candidate_report.json",
        )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(f"{label} is missing", details={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            f"{label} JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError(f"{label} must be a JSON object", details={"path": str(path)})
    return payload


def _replay_artifacts(
    clean_machine: dict[str, Any],
    path: Path,
    issues: list[PublicationIssue],
) -> tuple[dict[str, Any], ...]:
    raw = clean_machine.get("replay_artifacts")
    if not isinstance(raw, list) or not raw:
        _issue(
            issues,
            "error",
            "clean_machine.replay_artifacts_missing",
            path,
            "clean-machine report must include replay artifacts",
        )
        return ()
    labels: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            _issue(
                issues,
                "error",
                "clean_machine.replay_artifact_invalid",
                path,
                f"replay_artifacts[{index}] must be an object",
            )
            continue
        label = item.get("label")
        if isinstance(label, str):
            labels.add(label)
        if not isinstance(label, str) or not label:
            _issue(issues, "error", "clean_machine.replay_artifact.label", path, "label required")
        _verify_artifact_identity(
            item,
            path,
            issues,
            code_prefix="clean_machine.replay_artifact",
            index=index,
            label="replay_artifacts",
        )
        artifacts.append(dict(item))
    required = {
        "terminal transcript",
        "terminal demo manifest",
        "scores",
        "receipts",
        "runtime preflight report",
        "batch receipt report",
    }
    for label in sorted(required - labels):
        _issue(
            issues,
            "error",
            "clean_machine.replay_artifact.missing",
            path,
            f"missing replay artifact: {label}",
        )
    return tuple(artifacts)


def _verify_replay_demo_manifest_identity(
    *,
    candidate: dict[str, Any],
    clean_machine_report_path: Path,
    replay_artifacts: tuple[dict[str, Any], ...],
    downloaded_artifacts: tuple[dict[str, Any], ...],
    issues: list[PublicationIssue],
) -> None:
    artifact = _artifact_by_label(replay_artifacts, "terminal demo manifest")
    if artifact is None:
        return
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return
    manifest_path = _reported_file_path(raw_path, clean_machine_report_path)
    payload = _load_json_artifact(
        manifest_path,
        code_prefix="clean_machine.replay_manifest",
        issues=issues,
    )
    if payload is None:
        return
    if payload.get("status") != "passed":
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.status",
            manifest_path,
            "terminal demo manifest status must be passed",
        )
    model = payload.get("model")
    if not isinstance(model, dict):
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.model",
            manifest_path,
            "terminal demo manifest must include a model object",
        )
        return
    expected_model_id = candidate.get("model_id")
    if model.get("model_id") != expected_model_id:
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.model_id",
            manifest_path,
            "terminal demo manifest model_id must match release_candidate_report.json",
        )
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.inputs",
            manifest_path,
            "terminal demo manifest must include an inputs object",
        )
        return
    expected_manifest = _downloaded_artifact_by_group_path(
        downloaded_artifacts,
        group="model",
        suffix="manifest.json",
    )
    model_manifest = inputs.get("model_manifest")
    if not isinstance(model_manifest, dict):
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.model_manifest",
            manifest_path,
            "terminal demo manifest inputs must include model_manifest identity",
        )
        return
    if expected_manifest is None:
        return
    if model_manifest.get("sha256") != expected_manifest.get("sha256"):
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.model_manifest_hash",
            manifest_path,
            "terminal demo manifest model_manifest hash must match the downloaded public manifest.json",
        )
    if model_manifest.get("size_bytes") != expected_manifest.get("size_bytes"):
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.model_manifest_size",
            manifest_path,
            "terminal demo manifest model_manifest size must match the downloaded public manifest.json",
        )
    for input_key in ("vcf", "fasta"):
        _verify_replay_manifest_demo_input(
            inputs=inputs,
            input_key=input_key,
            manifest_path=manifest_path,
            clean_machine_report_path=clean_machine_report_path,
            downloaded_artifacts=downloaded_artifacts,
            issues=issues,
        )
    _verify_replay_manifest_artifacts(
        payload=payload,
        manifest_path=manifest_path,
        clean_machine_report_path=clean_machine_report_path,
        replay_artifacts=replay_artifacts,
        issues=issues,
    )
    _verify_replay_manifest_score_receipt_batch(
        payload=payload,
        manifest_path=manifest_path,
        clean_machine_report_path=clean_machine_report_path,
        replay_artifacts=replay_artifacts,
        issues=issues,
    )
    _verify_replay_manifest_runtime_preflight(
        payload=payload,
        manifest_path=manifest_path,
        clean_machine_report_path=clean_machine_report_path,
        replay_artifacts=replay_artifacts,
        downloaded_artifacts=downloaded_artifacts,
        issues=issues,
    )


def _verify_replay_manifest_demo_input(
    *,
    inputs: dict[str, Any],
    input_key: str,
    manifest_path: Path,
    clean_machine_report_path: Path,
    downloaded_artifacts: tuple[dict[str, Any], ...],
    issues: list[PublicationIssue],
) -> None:
    identity = inputs.get(input_key)
    code_prefix = f"clean_machine.replay_manifest.{input_key}"
    if not isinstance(identity, dict):
        _issue(
            issues,
            "error",
            code_prefix,
            manifest_path,
            f"terminal demo manifest inputs must include {input_key} identity",
        )
        return
    raw_path = identity.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        _issue(
            issues,
            "error",
            f"{code_prefix}_path",
            manifest_path,
            f"terminal demo manifest {input_key} input path is required",
        )
        return
    expected = _downloaded_demo_artifact_by_path(downloaded_artifacts, raw_path)
    if expected is None:
        _issue(
            issues,
            "error",
            f"{code_prefix}_download_missing",
            manifest_path,
            f"downloaded demo artifacts must include replayed {input_key} input",
        )
        return
    _compare_replay_manifest_identity(
        identity,
        expected,
        manifest_path=manifest_path,
        clean_machine_report_path=clean_machine_report_path,
        code_prefix=code_prefix,
        label=f"terminal demo manifest {input_key} input",
        issues=issues,
    )


def _downloaded_demo_artifact_by_path(
    downloaded_artifacts: tuple[dict[str, Any], ...],
    path: str,
) -> dict[str, Any] | None:
    expected_name = Path(path).name
    for artifact in downloaded_artifacts:
        if artifact.get("group") != "demo":
            continue
        raw_path = artifact.get("path")
        if raw_path == path:
            return artifact
        if isinstance(raw_path, str) and Path(raw_path).name == expected_name:
            return artifact
    return None


def _compare_replay_manifest_identity(
    observed: dict[str, Any],
    expected: dict[str, Any],
    *,
    manifest_path: Path,
    clean_machine_report_path: Path,
    code_prefix: str,
    label: str,
    issues: list[PublicationIssue],
) -> None:
    for field in ("sha256", "size_bytes"):
        if observed.get(field) != expected.get(field):
            _issue(
                issues,
                "error",
                f"{code_prefix}_{field}",
                manifest_path,
                f"{label} {field} must match the downloaded public demo artifact",
            )
    observed_path = observed.get("path")
    expected_path = expected.get("path")
    if not isinstance(observed_path, str) or not isinstance(expected_path, str):
        return
    if observed_path == expected_path:
        return
    try:
        paths_match = (
            _reported_file_path(observed_path, clean_machine_report_path).resolve()
            == _reported_file_path(expected_path, clean_machine_report_path).resolve()
        )
    except (OSError, RuntimeError):
        paths_match = False
    if not paths_match:
        _issue(
            issues,
            "error",
            f"{code_prefix}_path",
            manifest_path,
            f"{label} path must match the downloaded public demo artifact",
        )


def _verify_replay_manifest_artifacts(
    *,
    payload: dict[str, Any],
    manifest_path: Path,
    clean_machine_report_path: Path,
    replay_artifacts: tuple[dict[str, Any], ...],
    issues: list[PublicationIssue],
) -> None:
    raw = payload.get("artifacts")
    if not isinstance(raw, list) or not raw:
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.artifacts",
            manifest_path,
            "terminal demo manifest must include replay artifact identities",
        )
        return
    manifest_artifacts: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            _issue(
                issues,
                "error",
                "clean_machine.replay_manifest.artifact",
                manifest_path,
                f"artifacts[{index}] must be an object",
            )
            continue
        label = item.get("label")
        if isinstance(label, str) and label:
            manifest_artifacts[label] = item
    for report_artifact in replay_artifacts:
        label = report_artifact.get("label")
        if label == "terminal demo manifest":
            continue
        if not isinstance(label, str) or not label:
            continue
        manifest_artifact = manifest_artifacts.get(label)
        if manifest_artifact is None:
            _issue(
                issues,
                "error",
                "clean_machine.replay_manifest.artifact_missing",
                manifest_path,
                f"terminal demo manifest is missing replay artifact identity: {label}",
            )
            continue
        _verify_replay_manifest_artifact_path(
            manifest_artifact=manifest_artifact,
            report_artifact=report_artifact,
            manifest_path=manifest_path,
            clean_machine_report_path=clean_machine_report_path,
            issues=issues,
            label=label,
        )
        for field in ("sha256", "size_bytes"):
            if manifest_artifact.get(field) != report_artifact.get(field):
                _issue(
                    issues,
                    "error",
                    f"clean_machine.replay_manifest.artifact_{field}",
                    manifest_path,
                    f"terminal demo manifest artifact {label} {field} must match clean-machine replay report",
                )


def _verify_replay_manifest_artifact_path(
    *,
    manifest_artifact: dict[str, Any],
    report_artifact: dict[str, Any],
    manifest_path: Path,
    clean_machine_report_path: Path,
    issues: list[PublicationIssue],
    label: str,
) -> None:
    manifest_raw_path = manifest_artifact.get("path")
    report_raw_path = report_artifact.get("path")
    if not isinstance(manifest_raw_path, str) or not manifest_raw_path:
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.artifact_path",
            manifest_path,
            f"terminal demo manifest artifact {label} path must be a non-empty string",
        )
        return
    if not isinstance(report_raw_path, str) or not report_raw_path:
        return
    manifest_artifact_path = _reported_file_path(manifest_raw_path, clean_machine_report_path)
    report_artifact_path = _reported_file_path(report_raw_path, clean_machine_report_path)
    if manifest_artifact_path.resolve() != report_artifact_path.resolve():
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.artifact_path",
            manifest_path,
            f"terminal demo manifest artifact {label} path must match clean-machine replay report",
        )


def _verify_replay_manifest_score_receipt_batch(
    *,
    payload: dict[str, Any],
    manifest_path: Path,
    clean_machine_report_path: Path,
    replay_artifacts: tuple[dict[str, Any], ...],
    issues: list[PublicationIssue],
) -> None:
    summary = payload.get("score_receipt_batch")
    if not isinstance(summary, dict):
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.score_receipt_batch",
            manifest_path,
            "terminal demo manifest must include score_receipt_batch",
        )
        return
    artifact = _artifact_by_label(replay_artifacts, "batch receipt report")
    if artifact is None:
        return
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return
    batch_report_path = _reported_file_path(raw_path, clean_machine_report_path)
    batch_report = _load_json_artifact(
        batch_report_path,
        code_prefix="clean_machine.replay_manifest.batch_receipt_report",
        issues=issues,
    )
    if batch_report is None:
        return
    required = (
        "records",
        "model_id",
        "calibration_hash",
        "receipt_schema_version",
        "receipt_stream",
        "checked_score_fields",
        "runtime",
    )
    missing_summary = [field for field in required if field not in summary]
    if missing_summary:
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.score_receipt_batch.missing",
            manifest_path,
            f"score_receipt_batch is missing: {', '.join(missing_summary)}",
        )
        return
    missing_report = [field for field in required if field not in batch_report]
    if missing_report:
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.batch_receipt_report.missing",
            batch_report_path,
            f"batch receipt report is missing: {', '.join(missing_report)}",
        )
        return
    expected = {field: batch_report[field] for field in required}
    if summary != expected:
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.score_receipt_batch.stale",
            manifest_path,
            "terminal demo manifest score_receipt_batch must match replayed batch_receipt_report.json",
        )


def _verify_replay_manifest_runtime_preflight(
    *,
    payload: dict[str, Any],
    manifest_path: Path,
    clean_machine_report_path: Path,
    replay_artifacts: tuple[dict[str, Any], ...],
    downloaded_artifacts: tuple[dict[str, Any], ...],
    issues: list[PublicationIssue],
) -> None:
    summary = payload.get("runtime_preflight")
    if not isinstance(summary, dict):
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.runtime_preflight",
            manifest_path,
            "terminal demo manifest must include runtime_preflight",
        )
        return
    artifact = _artifact_by_label(replay_artifacts, "runtime preflight report")
    if artifact is None:
        return
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return
    preflight_path = _reported_file_path(raw_path, clean_machine_report_path)
    preflight = _load_json_artifact(
        preflight_path,
        code_prefix="clean_machine.replay_manifest.runtime_preflight_report",
        issues=issues,
    )
    if preflight is None:
        return
    expected = _runtime_preflight_manifest_summary(preflight)
    if expected is None:
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.runtime_preflight_report.invalid",
            preflight_path,
            "runtime preflight report is missing command.argv or requirements",
        )
        return
    if summary != expected:
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.runtime_preflight.stale",
            manifest_path,
            "terminal demo manifest runtime_preflight must match replayed runtime_preflight_report.json",
        )
    _verify_runtime_preflight_public_identities(
        preflight,
        preflight_path=preflight_path,
        clean_machine_report_path=clean_machine_report_path,
        downloaded_artifacts=downloaded_artifacts,
        issues=issues,
    )


def _verify_runtime_preflight_public_identities(
    payload: dict[str, Any],
    *,
    preflight_path: Path,
    clean_machine_report_path: Path,
    downloaded_artifacts: tuple[dict[str, Any], ...],
    issues: list[PublicationIssue],
) -> None:
    manifest_identity = payload.get("manifest")
    if not isinstance(manifest_identity, dict):
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.runtime_preflight_report.manifest",
            preflight_path,
            "runtime preflight report must include downloaded model manifest identity",
        )
    else:
        expected_manifest = _downloaded_artifact_by_group_path(
            downloaded_artifacts,
            group="model",
            suffix="manifest.json",
        )
        if expected_manifest is not None:
            _compare_replay_manifest_identity(
                manifest_identity,
                expected_manifest,
                manifest_path=preflight_path,
                clean_machine_report_path=clean_machine_report_path,
                code_prefix="clean_machine.replay_manifest.runtime_preflight_report.manifest",
                label="runtime preflight model manifest",
                issues=issues,
            )
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        _issue(
            issues,
            "error",
            "clean_machine.replay_manifest.runtime_preflight_report.inputs",
            preflight_path,
            "runtime preflight report must include input identities",
        )
        return
    for input_key in ("vcf", "fasta"):
        identity = inputs.get(input_key)
        code_prefix = f"clean_machine.replay_manifest.runtime_preflight_report.{input_key}"
        if not isinstance(identity, dict):
            _issue(
                issues,
                "error",
                code_prefix,
                preflight_path,
                f"runtime preflight report must include {input_key} input identity",
            )
            continue
        if identity.get("ok") is not True:
            _issue(
                issues,
                "error",
                f"{code_prefix}_not_ok",
                preflight_path,
                f"runtime preflight report {input_key} input identity must have ok=true",
            )
        raw_path = identity.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            _issue(
                issues,
                "error",
                f"{code_prefix}_path",
                preflight_path,
                f"runtime preflight report {input_key} input path is required",
            )
            continue
        expected = _downloaded_demo_artifact_by_path(downloaded_artifacts, raw_path)
        if expected is None:
            _issue(
                issues,
                "error",
                f"{code_prefix}_download_missing",
                preflight_path,
                f"downloaded demo artifacts must include runtime preflight {input_key} input",
            )
            continue
        _compare_replay_manifest_identity(
            identity,
            expected,
            manifest_path=preflight_path,
            clean_machine_report_path=clean_machine_report_path,
            code_prefix=code_prefix,
            label=f"runtime preflight {input_key} input",
            issues=issues,
        )


def _runtime_preflight_manifest_summary(payload: dict[str, Any]) -> dict[str, Any] | None:
    command = payload.get("command")
    requirements = payload.get("requirements")
    if not isinstance(command, dict) or not isinstance(requirements, dict):
        return None
    argv = command.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(part, str) or not part for part in argv)
    ):
        return None
    return {
        "schema_version": payload.get("schema_version"),
        "generated_by": payload.get("generated_by"),
        "ok": payload.get("ok"),
        "model_id": payload.get("model_id"),
        "release_id": payload.get("release_id"),
        "requested_backend": payload.get("requested_backend"),
        "selected_backend": payload.get("selected_backend"),
        "requirements": dict(requirements),
        "command": {
            "argv": list(argv),
            "shell": command.get("shell"),
        },
    }


def _artifact_by_label(
    artifacts: tuple[dict[str, Any], ...],
    label: str,
) -> dict[str, Any] | None:
    for artifact in artifacts:
        if artifact.get("label") == label:
            return artifact
    return None


def _downloaded_artifact_by_group_path(
    artifacts: tuple[dict[str, Any], ...],
    *,
    group: str,
    suffix: str,
) -> dict[str, Any] | None:
    for artifact in artifacts:
        raw_path = artifact.get("path")
        if (
            artifact.get("group") == group
            and isinstance(raw_path, str)
            and Path(raw_path).as_posix().endswith(suffix)
        ):
            return artifact
    return None


def _load_json_artifact(
    path: Path,
    *,
    code_prefix: str,
    issues: list[PublicationIssue],
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        _issue(issues, "error", f"{code_prefix}.missing", path, "JSON artifact is missing")
        return None
    except json.JSONDecodeError as exc:
        _issue(
            issues,
            "error",
            f"{code_prefix}.json",
            path,
            f"JSON artifact is invalid at line {exc.lineno}, column {exc.colno}",
        )
        return None
    if not isinstance(payload, dict):
        _issue(issues, "error", f"{code_prefix}.object", path, "JSON artifact must be an object")
        return None
    return payload


def _downloaded_artifacts(
    clean_machine: dict[str, Any],
    path: Path,
    issues: list[PublicationIssue],
) -> tuple[dict[str, Any], ...]:
    raw = clean_machine.get("downloaded_artifacts")
    if not isinstance(raw, list) or not raw:
        _issue(
            issues,
            "error",
            "clean_machine.downloads_missing",
            path,
            "clean-machine report must include downloaded public artifacts",
        )
        return ()
    groups: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            _issue(
                issues,
                "error",
                "clean_machine.downloaded_artifact_invalid",
                path,
                f"downloaded_artifacts[{index}] must be an object",
            )
            continue
        group = item.get("group")
        if isinstance(group, str):
            groups.add(group)
        if group not in {"model", "dataset", "demo"}:
            _issue(
                issues,
                "error",
                "clean_machine.downloaded_artifact.group",
                path,
                f"downloaded_artifacts[{index}] group must be model, dataset, or demo",
            )
        source_url = item.get("source_url")
        if not isinstance(source_url, str) or not _is_http_url(source_url):
            _issue(
                issues,
                "error",
                "clean_machine.downloaded_artifact.source_url",
                path,
                f"downloaded_artifacts[{index}] source_url must be an http(s) URL",
            )
        _verify_artifact_identity(
            item,
            path,
            issues,
            code_prefix="clean_machine.downloaded_artifact",
            index=index,
            label="downloaded_artifacts",
        )
        artifacts.append(dict(item))
    for group in sorted({"model", "dataset", "demo"} - groups):
        _issue(
            issues,
            "error",
            "clean_machine.downloaded_artifact.missing_group",
            path,
            f"missing downloaded artifact group: {group}",
        )
    return tuple(artifacts)


def _verify_downloaded_artifact_coverage(
    *,
    plan: dict[str, Any],
    clean_machine: dict[str, Any],
    clean_machine_report_path: Path,
    downloaded_artifacts: tuple[dict[str, Any], ...],
    issues: list[PublicationIssue],
) -> None:
    expected = {
        "model": _expected_upload_files(plan, "files", by_basename=False),
        "dataset": _expected_upload_files(plan, "dataset_files", by_basename=False),
        "demo": _expected_upload_files(plan, "demo_files", by_basename=True),
    }
    expected_source_urls = _expected_download_source_urls(plan)
    observed = _observed_downloaded_files(
        clean_machine=clean_machine,
        clean_machine_report_path=clean_machine_report_path,
        downloaded_artifacts=downloaded_artifacts,
        issues=issues,
    )
    for group, expected_files in expected.items():
        observed_files = observed.get(group, {})
        for destination, expected_hash in expected_files.items():
            observed_identity = observed_files.get(destination)
            if observed_identity is None:
                _issue(
                    issues,
                    "error",
                    "clean_machine.downloaded_artifact.missing_expected",
                    clean_machine_report_path,
                    f"missing downloaded {group} artifact: {destination}",
                )
                continue
            if observed_identity.get("sha256") != expected_hash:
                _issue(
                    issues,
                    "error",
                    "clean_machine.downloaded_artifact.unexpected_hash",
                    clean_machine_report_path,
                    f"downloaded {group} artifact {destination} does not match Hub plan hash",
                )
            expected_source_url = expected_source_urls.get(group, {}).get(destination)
            if (
                expected_source_url is not None
                and observed_identity.get("source_url") != expected_source_url
            ):
                _issue(
                    issues,
                    "error",
                    "clean_machine.downloaded_artifact.source_url_mismatch",
                    clean_machine_report_path,
                    f"downloaded {group} artifact {destination} did not come from the planned public URL",
                )
        for destination in sorted(set(observed_files) - set(expected_files)):
            _issue(
                issues,
                "error",
                "clean_machine.downloaded_artifact.unexpected",
                clean_machine_report_path,
                f"unexpected downloaded {group} artifact: {destination}",
            )


def _verify_release_candidate_artifact_publication(
    *,
    plan: dict[str, Any],
    release_candidate_artifacts: dict[str, dict[str, Any]],
    downloaded_artifacts: tuple[dict[str, Any], ...],
    clean_machine_report_path: Path,
    issues: list[PublicationIssue],
) -> None:
    for key, (group, expected_destination) in CANDIDATE_ARTIFACT_UPLOADS.items():
        candidate_identity = release_candidate_artifacts.get(key)
        if candidate_identity is None:
            continue
        destination = _candidate_upload_destination(
            candidate_identity,
            key=key,
            group=group,
            expected_destination=expected_destination,
            path=clean_machine_report_path,
            issues=issues,
        )
        if destination is None:
            continue
        plan_identity = _plan_upload_identity(plan, group=group, destination=destination)
        if plan_identity is None:
            _issue(
                issues,
                "error",
                f"candidate.artifacts.{key}.plan_missing",
                clean_machine_report_path,
                f"Hub plan must include release candidate artifact {key}",
            )
        else:
            _compare_candidate_artifact_identity(
                candidate_identity,
                plan_identity,
                path=clean_machine_report_path,
                code_prefix=f"candidate.artifacts.{key}.plan",
                label=f"release candidate artifact {key}",
                issues=issues,
            )
        download_identity = _downloaded_artifact_identity(
            downloaded_artifacts,
            group=group,
            destination=destination,
        )
        if download_identity is None:
            _issue(
                issues,
                "error",
                f"candidate.artifacts.{key}.download_missing",
                clean_machine_report_path,
                f"downloaded public artifacts must include release candidate artifact {key}",
            )
            continue
        _compare_candidate_artifact_identity(
            candidate_identity,
            download_identity,
            path=clean_machine_report_path,
            code_prefix=f"candidate.artifacts.{key}.download",
            label=f"release candidate artifact {key}",
            issues=issues,
        )


def _candidate_upload_destination(
    candidate_identity: dict[str, Any],
    *,
    key: str,
    group: str,
    expected_destination: str | None,
    path: Path,
    issues: list[PublicationIssue],
) -> str | None:
    raw_path = candidate_identity.get("path")
    if not isinstance(raw_path, str):
        return None
    expected_prefix = f"{group}/"
    if expected_destination is not None:
        expected_path = f"{expected_prefix}{expected_destination}"
        if raw_path != expected_path:
            _issue(
                issues,
                "error",
                f"candidate.artifacts.{key}.path_mismatch",
                path,
                f"release candidate artifact {key} must point to {expected_path}",
            )
        return expected_destination
    if raw_path.startswith(expected_prefix) and raw_path != expected_prefix:
        return raw_path.removeprefix(expected_prefix)
    _issue(
        issues,
        "error",
        f"candidate.artifacts.{key}.path_mismatch",
        path,
        f"release candidate artifact {key} must point under {expected_prefix}",
    )
    return None


def _plan_upload_identity(
    plan: dict[str, Any],
    *,
    group: str,
    destination: str,
) -> dict[str, Any] | None:
    key = {"model": "files", "dataset": "dataset_files", "demo": "demo_files"}.get(group)
    if key is None:
        return None
    raw = plan.get(key)
    if not isinstance(raw, list):
        return None
    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get("destination") == destination:
            return {
                "sha256": item.get("sha256"),
                "size_bytes": item.get("size_bytes"),
            }
    return None


def _downloaded_artifact_identity(
    artifacts: tuple[dict[str, Any], ...],
    *,
    group: str,
    destination: str,
) -> dict[str, Any] | None:
    suffix = f"{group}/{destination}"
    for item in artifacts:
        raw_path = item.get("path")
        if (
            item.get("group") == group
            and isinstance(raw_path, str)
            and Path(raw_path).as_posix().endswith(suffix)
        ):
            return {
                "sha256": item.get("sha256"),
                "size_bytes": item.get("size_bytes"),
            }
    return None


def _compare_candidate_artifact_identity(
    candidate_identity: dict[str, Any],
    expected_identity: dict[str, Any],
    *,
    path: Path,
    code_prefix: str,
    label: str,
    issues: list[PublicationIssue],
) -> None:
    if candidate_identity.get("sha256") != expected_identity.get("sha256"):
        _issue(
            issues,
            "error",
            f"{code_prefix}_hash",
            path,
            f"{label} SHA-256 must match published artifact identity",
        )
    if candidate_identity.get("size_bytes") != expected_identity.get("size_bytes"):
        _issue(
            issues,
            "error",
            f"{code_prefix}_size",
            path,
            f"{label} size must match published artifact identity",
        )


def _expected_upload_files(
    plan: dict[str, Any],
    key: str,
    *,
    by_basename: bool,
) -> dict[str, str]:
    raw = plan.get(key)
    if not isinstance(raw, list):
        return {}
    expected: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        destination = item.get("destination")
        sha256 = item.get("sha256")
        if not isinstance(destination, str) or not isinstance(sha256, str):
            continue
        expected[Path(destination).name if by_basename else destination] = sha256
    return expected


def _expected_download_source_urls(plan: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        "model": _expected_model_download_source_urls(plan),
        "dataset": _expected_dataset_download_source_urls(plan),
        "demo": _expected_demo_download_source_urls(plan),
    }


def _expected_model_download_source_urls(plan: dict[str, Any]) -> dict[str, str]:
    repo_id = plan.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        return {}
    result: dict[str, str] = {}
    for destination in _expected_upload_files(plan, "files", by_basename=False):
        result[destination] = (
            f"https://huggingface.co/{repo_id}/resolve/main/{quote(destination, safe='/')}"
        )
    return result


def _expected_dataset_download_source_urls(plan: dict[str, Any]) -> dict[str, str]:
    dataset_url = plan.get("dataset_url")
    if not isinstance(dataset_url, str):
        return {}
    parsed = urlparse(dataset_url)
    parts = tuple(part for part in parsed.path.strip("/").split("/") if part)
    if parsed.netloc != "huggingface.co" or len(parts) < 3 or parts[0] != "datasets":
        return {}
    repo_id = f"{parts[1]}/{parts[2]}"
    result: dict[str, str] = {}
    for destination in _expected_upload_files(plan, "dataset_files", by_basename=False):
        result[destination] = (
            f"https://huggingface.co/datasets/{repo_id}/resolve/main/{quote(destination, safe='/')}"
        )
    return result


def _expected_demo_download_source_urls(plan: dict[str, Any]) -> dict[str, str]:
    demo_url = plan.get("demo_url")
    if not isinstance(demo_url, str):
        return {}
    parsed = urlparse(demo_url)
    parts = tuple(part for part in parsed.path.strip("/").split("/") if part)
    if parsed.netloc != "github.com" or len(parts) < 5 or parts[2:4] != ("releases", "tag"):
        return {}
    owner, repo, _releases, _tag, tag = parts[:5]
    result: dict[str, str] = {}
    for asset_name in _expected_upload_files(plan, "demo_files", by_basename=True):
        result[asset_name] = (
            f"https://github.com/{owner}/{repo}/releases/download/"
            f"{quote(tag, safe='')}/{quote(asset_name, safe='')}"
        )
    return result


def _verify_plan_upload_inventories(
    plan: dict[str, Any],
    plan_path: Path,
    issues: list[PublicationIssue],
) -> None:
    for key, by_basename in (
        ("files", False),
        ("dataset_files", False),
        ("demo_files", True),
    ):
        raw = plan.get(key)
        if not isinstance(raw, list):
            continue
        seen_destinations: dict[str, int] = {}
        seen_public_names: dict[str, int] = {}
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                _issue(
                    issues,
                    "error",
                    f"plan.{key}.entry",
                    plan_path,
                    f"{key}[{index}] must be an object",
                )
                continue
            destination = item.get("destination")
            if not isinstance(destination, str) or not destination.strip():
                _issue(
                    issues,
                    "error",
                    f"plan.{key}.destination",
                    plan_path,
                    f"{key}[{index}] destination is required",
                )
                continue
            if not _is_package_relative(destination):
                _issue(
                    issues,
                    "error",
                    f"plan.{key}.destination",
                    plan_path,
                    f"{key}[{index}] destination must be package-relative",
                )
                continue
            previous_index = seen_destinations.get(destination)
            if previous_index is not None:
                _issue(
                    issues,
                    "error",
                    f"plan.{key}.duplicate_destination",
                    plan_path,
                    (
                        f"{key}[{index}] duplicates destination {destination} "
                        f"from {key}[{previous_index}]"
                    ),
                )
            seen_destinations[destination] = index
            if by_basename:
                public_name = Path(destination).name
                previous_public_index = seen_public_names.get(public_name)
                if previous_public_index is not None:
                    _issue(
                        issues,
                        "error",
                        f"plan.{key}.duplicate_asset_name",
                        plan_path,
                        (
                            f"{key}[{index}] duplicates public asset name {public_name} "
                            f"from {key}[{previous_public_index}]"
                        ),
                    )
                seen_public_names[public_name] = index
            sha256 = item.get("sha256")
            if not isinstance(sha256, str) or not _looks_like_sha256(sha256):
                _issue(
                    issues,
                    "error",
                    f"plan.{key}.sha256",
                    plan_path,
                    f"{key}[{index}] sha256 must be sha256:<64hex>",
                )


def _verify_plan_paper_file(
    plan: dict[str, Any],
    plan_path: Path,
    issues: list[PublicationIssue],
) -> dict[str, Any] | None:
    if plan.get("paper_url") in {None, ""}:
        return None
    raw = plan.get("paper_file")
    if not isinstance(raw, dict):
        _issue(
            issues,
            "error",
            "plan.paper_file",
            plan_path,
            "hub_release_plan.json must bind the verified paper file when paper_url is set",
        )
        return None
    source = raw.get("source")
    source_path: Path | None = None
    if not isinstance(source, str) or not source.strip():
        _issue(
            issues,
            "error",
            "plan.paper_file.source",
            plan_path,
            "paper_file.source must record the verified paper file path",
        )
    else:
        source_path = _reported_file_path(source, plan_path)
        if not source_path.is_file():
            _issue(
                issues,
                "error",
                "plan.paper_file.source_missing",
                source_path,
                "paper_file.source must exist when publication evidence is generated",
            )
    destination = raw.get("destination")
    if not isinstance(destination, str) or not _is_package_relative(destination):
        _issue(
            issues,
            "error",
            "plan.paper_file.destination",
            plan_path,
            "paper_file.destination must be package-relative",
        )
    sha256 = raw.get("sha256")
    if not isinstance(sha256, str) or not _looks_like_sha256(sha256):
        _issue(
            issues,
            "error",
            "plan.paper_file.sha256",
            plan_path,
            "paper_file.sha256 must be sha256:<64hex>",
        )
    elif source_path is not None and source_path.is_file():
        observed_hash = sha256_file(source_path)
        if observed_hash != sha256:
            _issue(
                issues,
                "error",
                "plan.paper_file.hash_mismatch",
                source_path,
                "paper_file.sha256 must match paper_file.source bytes",
            )
    size_bytes = raw.get("size_bytes")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
        _issue(
            issues,
            "error",
            "plan.paper_file.size_bytes",
            plan_path,
            "paper_file.size_bytes must be a positive integer",
        )
    elif source_path is not None and source_path.is_file():
        observed_size = source_path.stat().st_size
        if observed_size != size_bytes:
            _issue(
                issues,
                "error",
                "plan.paper_file.size_mismatch",
                source_path,
                "paper_file.size_bytes must match paper_file.source bytes",
            )
    if (
        isinstance(source, str)
        and source.strip()
        and isinstance(destination, str)
        and _is_package_relative(destination)
        and isinstance(sha256, str)
        and _looks_like_sha256(sha256)
        and isinstance(size_bytes, int)
        and not isinstance(size_bytes, bool)
        and size_bytes > 0
    ):
        return {
            "source": _public_path_reference(source_path if source_path is not None else source),
            "destination": destination,
            "sha256": sha256,
            "size_bytes": size_bytes,
            "url": plan.get("paper_url"),
        }
    return None


def _is_package_relative(value: str) -> bool:
    path = Path(value)
    return (
        "://" not in value
        and not path.is_absolute()
        and ".." not in path.parts
        and bool(path.parts)
    )


def _observed_downloaded_files(
    *,
    clean_machine: dict[str, Any],
    clean_machine_report_path: Path,
    downloaded_artifacts: tuple[dict[str, Any], ...],
    issues: list[PublicationIssue],
) -> dict[str, dict[str, dict[str, str]]]:
    roots = {
        "model": _report_root(clean_machine, "model_dir", clean_machine_report_path, issues),
        "dataset": _report_root(clean_machine, "dataset_dir", clean_machine_report_path, issues),
        "demo": _report_root(clean_machine, "demo_dir", clean_machine_report_path, issues),
    }
    observed: dict[str, dict[str, dict[str, str]]] = {"model": {}, "dataset": {}, "demo": {}}
    for item in downloaded_artifacts:
        group = item.get("group")
        raw_path = item.get("path")
        sha256 = item.get("sha256")
        source_url = item.get("source_url")
        if group not in observed or not isinstance(raw_path, str) or not isinstance(sha256, str):
            continue
        root = roots[group]
        if root is None:
            continue
        path = _reported_file_path(raw_path, clean_machine_report_path)
        destination = _download_destination(path, root)
        if destination is None:
            _issue(
                issues,
                "error",
                "clean_machine.downloaded_artifact.outside_root",
                clean_machine_report_path,
                f"downloaded {group} artifact is outside {group}_dir",
            )
            continue
        observed[group][Path(destination).name if group == "demo" else destination] = {
            "sha256": sha256,
            "source_url": source_url if isinstance(source_url, str) else "",
        }
    return observed


def _download_destination(path: Path, root: Path) -> str | None:
    try:
        destination = path.relative_to(root)
    except ValueError:
        return None
    if not destination.parts or ".." in destination.parts:
        return None
    return destination.as_posix()


def _report_root(
    clean_machine: dict[str, Any],
    field: str,
    report_path: Path,
    issues: list[PublicationIssue],
) -> Path | None:
    value = clean_machine.get(field)
    if not isinstance(value, str) or not value:
        _issue(
            issues,
            "error",
            f"clean_machine.{field}",
            report_path,
            f"clean-machine report must include {field}",
        )
        return None
    return _reported_file_path(value, report_path)


def _verify_artifact_identity(
    item: dict[str, Any],
    report_path: Path,
    issues: list[PublicationIssue],
    *,
    code_prefix: str,
    index: int,
    label: str,
) -> None:
    raw_path = item.get("path")
    sha256 = item.get("sha256")
    size_bytes = item.get("size_bytes")
    if not isinstance(raw_path, str) or not raw_path:
        _issue(
            issues,
            "error",
            f"{code_prefix}.path",
            report_path,
            f"{label}[{index}] path is required",
        )
        return
    artifact_path = _reported_file_path(raw_path, report_path)
    if not artifact_path.is_file():
        _issue(
            issues,
            "error",
            f"{code_prefix}.missing",
            artifact_path,
            f"{label}[{index}] file is missing",
        )
        return
    if not isinstance(sha256, str) or not _looks_like_sha256(sha256):
        _issue(
            issues,
            "error",
            f"{code_prefix}.sha256",
            report_path,
            f"{label}[{index}] sha256 must be sha256:<64hex>",
        )
    else:
        observed_hash = sha256_file(artifact_path)
        if observed_hash != sha256:
            _issue(
                issues,
                "error",
                f"{code_prefix}.hash_mismatch",
                artifact_path,
                f"{label}[{index}] expected {sha256}, observed {observed_hash}",
            )
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
        _issue(
            issues,
            "error",
            f"{code_prefix}.size_bytes",
            report_path,
            f"{label}[{index}] size_bytes must be positive",
        )
    else:
        observed_size = artifact_path.stat().st_size
        if observed_size != size_bytes:
            _issue(
                issues,
                "error",
                f"{code_prefix}.size_mismatch",
                artifact_path,
                f"{label}[{index}] expected {size_bytes} bytes, observed {observed_size}",
            )


def _reported_file_path(raw_path: str, report_path: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return report_path.parent / candidate


def _public_artifact_paths(
    artifacts: tuple[dict[str, Any], ...],
    *,
    report_path: Path,
) -> tuple[dict[str, Any], ...]:
    public: list[dict[str, Any]] = []
    for artifact in artifacts:
        item = dict(artifact)
        raw_path = item.get("path")
        if isinstance(raw_path, str) and raw_path:
            item["path"] = _public_path_reference(
                _reported_file_path(raw_path, report_path),
                root=report_path.parent,
            )
        public.append(item)
    return tuple(public)


def _public_path_reference(value: str | Path, *, root: Path | None = None) -> str:
    path = Path(value)
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except (OSError, RuntimeError, ValueError):
            pass
    if not path.is_absolute() and ".." not in path.parts and path.parts:
        return path.as_posix()
    return path.name


def _identity(path: Path, *, reported_path: str | None = None) -> ReportIdentity:
    if not path.is_file():
        raise InputError("publication report input is missing", details={"path": str(path)})
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise InputError("publication report input is empty", details={"path": str(path)})
    return ReportIdentity(
        path=reported_path or _public_path_reference(path),
        sha256=sha256_file(path),
        size_bytes=size_bytes,
    )


def _issue(
    issues: list[PublicationIssue],
    severity: Severity,
    code: str,
    path: Path,
    message: str,
) -> None:
    issues.append(
        PublicationIssue(
            severity=severity,
            code=code,
            path=str(path),
            message=message,
        )
    )


def _issue_refs_for_publication_issue(code: str) -> tuple[int, ...]:
    if code in {"candidate.not_ready", "clean_machine.package_not_ok"}:
        return ALL_RELEASE_ISSUES
    if code.startswith("candidate."):
        return _candidate_issue_refs(code)
    if code.startswith("publish."):
        return PUBLISH_ISSUES
    if code.startswith("plan."):
        return _plan_issue_refs(code)
    if code.startswith("clean_machine."):
        return _clean_machine_issue_refs(code)
    return ()


def _candidate_issue_refs(code: str) -> tuple[int, ...]:
    if code == "candidate.urls":
        return PUBLIC_ARTIFACT_ISSUES
    if code.startswith("candidate.public_links."):
        return _candidate_public_issue_refs(
            code,
            prefix="candidate.public_links",
            issue_map=CANDIDATE_PUBLIC_LINK_ISSUES,
            default=PUBLIC_ARTIFACT_ISSUES,
        )
    if code.startswith("candidate.public_artifacts."):
        return _candidate_public_issue_refs(
            code,
            prefix="candidate.public_artifacts",
            issue_map=CANDIDATE_PUBLIC_ARTIFACT_ISSUES,
            default=PUBLIC_REPLAY_ISSUES,
        )
    if code == "candidate.public_links":
        return PUBLIC_ARTIFACT_ISSUES
    if code == "candidate.public_artifacts":
        return PUBLIC_REPLAY_ISSUES
    if code.startswith("candidate.artifacts."):
        artifact_name = code.removeprefix("candidate.artifacts.").split(".", maxsplit=1)[0]
        return CANDIDATE_ARTIFACT_ISSUES.get(
            artifact_name,
            (PAPER_ISSUE, MODEL_RELEASE_ISSUE),
        )
    return ALL_RELEASE_ISSUES


def _candidate_public_issue_refs(
    code: str,
    *,
    prefix: str,
    issue_map: Mapping[str, tuple[int, ...]],
    default: tuple[int, ...],
) -> tuple[int, ...]:
    for name, issue_refs in issue_map.items():
        if code.startswith(f"{prefix}.{name}."):
            return issue_refs
    return default


def _plan_issue_refs(code: str) -> tuple[int, ...]:
    if code.startswith("plan.dataset_"):
        return (DATASET_ISSUE,)
    if code.startswith("plan.demo_"):
        return (DEMO_ISSUE,)
    if code.startswith("plan.paper_"):
        return (PAPER_ISSUE,)
    if code.startswith("plan.files."):
        return (MODEL_RELEASE_ISSUE,)
    if code.startswith("plan.dataset_files."):
        return (DATASET_ISSUE,)
    if code.startswith("plan.demo_files."):
        return (DEMO_ISSUE,)
    if code.startswith("plan.paper_file"):
        return (PAPER_ISSUE,)
    if code in {"plan.model_id_mismatch", "plan.repo_id_mismatch", "plan.model_url_mismatch"}:
        return (MODEL_RELEASE_ISSUE,)
    if code in {"plan.release_id_mismatch", "plan.commit_sha_mismatch"}:
        return PUBLISH_ISSUES
    return PUBLIC_ARTIFACT_ISSUES


def _clean_machine_issue_refs(code: str) -> tuple[int, ...]:
    if code == "clean_machine.model_dir":
        return (MODEL_RELEASE_ISSUE,)
    if code == "clean_machine.dataset_dir":
        return (DATASET_ISSUE,)
    if code == "clean_machine.demo_dir":
        return (DEMO_ISSUE,)
    if code.startswith("clean_machine.replay_manifest."):
        return DEMO_REPLAY_ISSUES
    if code.startswith("clean_machine.replay_artifact"):
        return DEMO_REPLAY_ISSUES
    if code == "clean_machine.replay_artifacts_missing":
        return DEMO_REPLAY_ISSUES
    if code == "clean_machine.downloads_missing":
        return PUBLIC_REPLAY_ISSUES
    if code.startswith("clean_machine.downloaded_artifact"):
        return PUBLIC_REPLAY_ISSUES
    if code.startswith("clean_machine.candidate_"):
        return PUBLIC_REPLAY_ISSUES
    return PUBLIC_REPLAY_ISSUES


def _string_dict(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items() if isinstance(item, str)}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _issue_ref_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    refs: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        number = item.get("number")
        url = item.get("url")
        if isinstance(number, int) and not isinstance(number, bool) and isinstance(url, str):
            refs.append({"number": number, "url": url})
    return refs


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_nonnegative_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _optional_positive_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _looks_like_sha256(value: str) -> bool:
    return (
        len(value) == len("sha256:") + 64
        and value.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in value.removeprefix("sha256:"))
    )


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
