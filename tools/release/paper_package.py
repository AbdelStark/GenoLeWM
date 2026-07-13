# SPDX-License-Identifier: Apache-2.0
"""Validate the artifact set for the first paper/demo release."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from geno_lewm._artifact_sources import (
    CARBON_ZERO_SHOT_GENERATED_BY,
    SCORE_JSONL_GENERATED_BY,
)
from geno_lewm.data import MEMBERSHIP_STORE_SCHEMA_VERSION, V03_CHROMOSOME_ROLES
from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import Manifest, canonical_json_sha256, load_manifest, sha256_file
from geno_lewm.provenance.hashing import looks_like_sha256
from geno_lewm.training.preflight import REPORT_NAME as TRAINING_PREFLIGHT_REPORT_NAME
from tools.demo.terminal_inference import (
    DEMO_MANIFEST_NAME,
    DEMO_MANIFEST_SCHEMA_VERSION,
    GENERATED_BY as DEMO_MANIFEST_GENERATED_BY,
    summarize_vcf_input,
)
from tools.release.batch_receipt_report import (
    REPORT_NAME as BATCH_RECEIPT_REPORT_NAME,
    build_batch_receipt_report,
)
from tools.release.dataset_integrity import (
    DEFAULT_REPORT_NAME,
    GENERATED_BY as DATASET_INTEGRITY_GENERATED_BY,
    build_dataset_integrity_report,
)
from tools.release.dataset_package import (
    ARTIFACT_ROLES,
    DatasetPackage,
    load_dataset_package,
    render_data_card,
)
from tools.release.dataset_snapshot import (
    GENERATED_BY as DATASET_SNAPSHOT_GENERATED_BY,
    INPUT_CHECK_GENERATED_BY as DATASET_INPUT_CHECK_GENERATED_BY,
    INPUT_CHECK_REPORT_NAME as DATASET_INPUT_CHECK_REPORT_NAME,
    REPORT_NAME as DATASET_SNAPSHOT_REPORT_NAME,
)
from tools.release.efficiency_report import (
    REPORT_NAME as EFFICIENCY_REPORT_NAME,
    EfficiencyReport,
    load_efficiency_report,
)
from tools.release.eval_report import EvalReportInput, load_report_input, render_report
from tools.release.model_package import (
    MODEL_PACKAGE_NAME,
    load_model_package,
    render_model_card,
)
from tools.release.paper_draft import (
    GENERATED_BY as PAPER_DRAFT_GENERATED_BY,
    render_paper_draft,
)
from tools.release.runtime_preflight import (
    GENERATED_BY as RUNTIME_PREFLIGHT_GENERATED_BY,
    REPORT_NAME as RUNTIME_PREFLIGHT_REPORT_NAME,
    SCHEMA_VERSION as RUNTIME_PREFLIGHT_SCHEMA_VERSION,
)
from tools.release.serious_completion_paper import (
    SeriousCompletionPaperPaths,
    verify_serious_completion_paper,
)
from tools.release.training_run import (
    CARD_NAME as TRAINING_RUN_CARD_NAME,
    CHECKSUMS_NAME as TRAINING_RUN_CHECKSUMS_NAME,
    MANIFEST_NAME as TRAINING_RUN_MANIFEST_NAME,
    TrainingRunManifest,
    render_training_run_card,
    verify_training_run_manifest,
)

Severity = Literal["error", "warning"]
PLACEHOLDER_RE: Final = re.compile(
    r"\b(?:tbd|todo|placeholder|coming soon|fake|dummy|lorem ipsum)\b",
    re.IGNORECASE,
)
EVAL_METRICS_NAME: Final = "eval_metrics.json"


@dataclass(frozen=True, slots=True)
class PackagePaths:
    """Paths that make up a candidate paper/demo release package."""

    model_dir: Path
    dataset_dir: Path
    demo_dir: Path
    paper_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PackageIssue:
    """One release-package verification issue."""

    severity: Severity
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class PackageReport:
    """Verification result for a candidate paper/demo release package."""

    ok: bool
    model_id: str | None
    issues: tuple[PackageIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "model_id": self.model_id,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class RuntimePreflightEvidence:
    """Runtime preflight data needed to validate the terminal-demo manifest."""

    command: tuple[str, ...] | None
    manifest_summary: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class _DatasetVerification:
    snapshot_id: str | None
    membership_runtime_binding: dict[str, object] | None
    package_valid: bool


def verify_package(
    paths: PackagePaths,
    *,
    allow_fixture_manifest: bool = False,
) -> PackageReport:
    """Verify a candidate paper/demo release package."""
    issues: list[PackageIssue] = []
    manifest = _verify_model_dir(
        paths.model_dir,
        paths.dataset_dir,
        issues,
        allow_fixture_manifest=allow_fixture_manifest,
    )
    dataset = _verify_dataset_dir(paths.dataset_dir, issues)
    _verify_training_dataset_membership_binding(paths.model_dir, dataset, issues)
    _verify_demo_dir(
        paths.demo_dir,
        issues,
        model_id=None if manifest is None else manifest.model_id(),
        calibration_hash=None if manifest is None else manifest.calibration.hash,
        model_dir=paths.model_dir,
        allow_fixture_manifest=allow_fixture_manifest,
    )
    if paths.paper_path is not None:
        _verify_paper_path(
            paths.paper_path,
            issues,
            model_id=None if manifest is None else manifest.model_id(),
            dataset_snapshot=dataset.snapshot_id,
            model_dir=paths.model_dir,
            dataset_dir=paths.dataset_dir,
            demo_dir=paths.demo_dir,
        )
    return PackageReport(
        ok=not any(issue.severity == "error" for issue in issues),
        model_id=None if manifest is None else manifest.model_id(),
        issues=tuple(issues),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report: Any
        if args.serious_completion:
            if args.suite_dir is None:
                raise InputError("--suite-dir is required with --serious-completion")
            if args.planning_demo_dir is None:
                raise InputError("--planning-demo-dir is required with --serious-completion")
            report = verify_serious_completion_paper(
                SeriousCompletionPaperPaths(
                    suite_dir=args.suite_dir,
                    planning_demo_dir=args.planning_demo_dir,
                    paper_path=args.paper_path,
                )
            )
        else:
            if args.model_dir is None:
                raise InputError("--model-dir is required")
            if args.dataset_dir is None:
                raise InputError("--dataset-dir is required")
            if args.demo_dir is None:
                raise InputError("--demo-dir is required")
            report = verify_package(
                PackagePaths(
                    model_dir=args.model_dir,
                    dataset_dir=args.dataset_dir,
                    demo_dir=args.demo_dir,
                    paper_path=args.paper_path,
                ),
                allow_fixture_manifest=args.allow_fixture_manifest,
            )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return 0 if report.ok else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a paper/demo release artifact package.",
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--demo-dir", type=Path)
    parser.add_argument(
        "--serious-completion",
        action="store_true",
        help="Verify the v0.2 serious-completion paper against suite and planning-demo artifacts.",
    )
    parser.add_argument("--suite-dir", type=Path)
    parser.add_argument("--planning-demo-dir", type=Path)
    parser.add_argument("--paper-path", type=Path)
    parser.add_argument(
        "--allow-fixture-manifest",
        action="store_true",
        help="Allow fixture/test model manifests for local verifier tests only.",
    )
    return parser


def _verify_model_dir(
    model_dir: Path,
    dataset_dir: Path,
    issues: list[PackageIssue],
    *,
    allow_fixture_manifest: bool,
) -> Manifest | None:
    manifest_path = model_dir / "manifest.json"
    manifest: Manifest | None = None
    if not manifest_path.is_file():
        _error(issues, "model.manifest_missing", manifest_path, "model manifest is required")
    else:
        try:
            manifest = load_manifest(manifest_path)
        except GenoLeWMError as exc:
            _error(issues, "model.manifest_invalid", manifest_path, exc.message or str(exc))
        else:
            if not allow_fixture_manifest and _looks_like_fixture_manifest(manifest):
                _error(
                    issues,
                    "model.fixture_manifest",
                    manifest_path,
                    "fixture/test manifests cannot back a paper/demo release",
                )
            _verify_manifest_artifacts(model_dir, manifest, issues)

    required_checksum_files = [
        "manifest.json",
        MODEL_PACKAGE_NAME,
        "model_card.md",
        EVAL_METRICS_NAME,
        EFFICIENCY_REPORT_NAME,
        TRAINING_PREFLIGHT_REPORT_NAME,
        TRAINING_RUN_MANIFEST_NAME,
        TRAINING_RUN_CARD_NAME,
        TRAINING_RUN_CHECKSUMS_NAME,
    ]
    if manifest is None:
        required_checksum_files.extend(("eval_report.md", "train_config.yaml"))
    else:
        required_checksum_files.extend(
            (
                manifest.predictor.file,
                manifest.action_encoder.file,
                manifest.calibration.file,
                manifest.training.config_file,
                manifest.eval.file,
            )
        )
    model_card = model_dir / "model_card.md"
    _require_markdown_sections(
        model_card,
        issues,
        code_prefix="model.card",
        sections=("Data", "Hardware", "License", "Intended Use", "Limitations"),
    )
    _verify_model_package_metadata(model_dir, manifest, issues)
    if manifest is not None:
        eval_input = _verify_eval_report(
            model_dir / manifest.eval.file,
            model_dir / EVAL_METRICS_NAME,
            issues,
            model_dir=model_dir,
            dataset_dir=dataset_dir,
        )
    else:
        eval_input = _verify_eval_report(
            model_dir / "eval_report.md",
            model_dir / EVAL_METRICS_NAME,
            issues,
            model_dir=model_dir,
            dataset_dir=dataset_dir,
        )
    efficiency_report = _verify_efficiency_report(
        model_dir / EFFICIENCY_REPORT_NAME,
        issues,
    )
    if eval_input is not None:
        required_checksum_files.extend(_model_checksum_eval_artifacts(eval_input))
    if manifest is not None:
        _verify_release_evidence_identity(
            manifest,
            eval_input,
            model_dir / EVAL_METRICS_NAME,
            efficiency_report,
            model_dir / EFFICIENCY_REPORT_NAME,
            issues,
        )
    _verify_training_run_dir(
        model_dir,
        issues,
        manifest=manifest,
        eval_input=eval_input,
        efficiency_report=efficiency_report,
    )
    _verify_sha256sums(
        model_dir,
        issues,
        code_prefix="model.checksums",
        required_files=tuple(dict.fromkeys(required_checksum_files)),
    )
    return manifest


def _model_checksum_eval_artifacts(eval_input: EvalReportInput) -> tuple[str, ...]:
    files: list[str] = []
    for _, raw_path in eval_input.artifacts:
        relative = _model_relative_eval_artifact(raw_path)
        if relative is not None:
            files.append(relative)
    return tuple(dict.fromkeys(files))


def _model_relative_eval_artifact(raw_path: str) -> str | None:
    if "://" in raw_path:
        return None
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return None
    parts = relative.parts
    if parts[0] == "dataset":
        return None
    if parts[0] == "model":
        if len(parts) == 1:
            return None
        return Path(*parts[1:]).as_posix()
    return relative.as_posix()


def _verify_model_package_metadata(
    model_dir: Path,
    manifest: Manifest | None,
    issues: list[PackageIssue],
) -> None:
    metadata_path = model_dir / MODEL_PACKAGE_NAME
    card_path = model_dir / "model_card.md"
    if not metadata_path.is_file():
        _error(
            issues,
            "model.metadata_missing",
            metadata_path,
            f"{MODEL_PACKAGE_NAME} is required",
        )
        return
    try:
        package = load_model_package(metadata_path)
    except GenoLeWMError as exc:
        _error(
            issues,
            "model.metadata_invalid",
            metadata_path,
            exc.message or str(exc),
        )
        return
    if manifest is None or not card_path.is_file():
        return
    expected = render_model_card(manifest, package)
    try:
        observed = card_path.read_text(encoding="utf-8")
    except OSError as exc:
        _error(issues, "model.card.read_failed", card_path, str(exc))
        return
    if observed != expected:
        _error(
            issues,
            "model.card.stale",
            card_path,
            f"model_card.md does not match {MODEL_PACKAGE_NAME}",
        )


def _verify_training_run_dir(
    model_dir: Path,
    issues: list[PackageIssue],
    *,
    manifest: Manifest | None = None,
    eval_input: EvalReportInput | None = None,
    efficiency_report: EfficiencyReport | None = None,
) -> None:
    manifest_path = model_dir / TRAINING_RUN_MANIFEST_NAME
    card_path = model_dir / TRAINING_RUN_CARD_NAME
    artifact_files: tuple[str, ...] = ()
    run_manifest: TrainingRunManifest | None = None
    if not manifest_path.is_file():
        _error(
            issues,
            "model.training_run.manifest_missing",
            manifest_path,
            f"{TRAINING_RUN_MANIFEST_NAME} is required",
        )
    else:
        try:
            run_manifest = verify_training_run_manifest(
                model_dir,
                manifest_path,
                require_preflight=True,
            )
        except GenoLeWMError as exc:
            _error(
                issues,
                "model.training_run.manifest_invalid",
                manifest_path,
                exc.message or str(exc),
            )
        else:
            artifact_files = tuple(artifact.path for artifact in run_manifest.artifacts)
            _verify_training_run_identity(
                manifest_path,
                run_manifest,
                manifest=manifest,
                eval_input=eval_input,
                efficiency_report=efficiency_report,
                issues=issues,
            )
    _require_markdown_sections(
        card_path,
        issues,
        code_prefix="model.training_run.card",
        sections=(
            "Run Identity",
            "Command",
            "Hardware",
            "Runtime",
            "Reproducibility",
            "Monitoring",
            "Artifacts",
            "Result Summary",
            "Limitations",
        ),
    )
    _verify_training_run_card(card_path, run_manifest, issues)
    _verify_named_sha256sums(
        model_dir,
        model_dir / TRAINING_RUN_CHECKSUMS_NAME,
        issues,
        code_prefix="model.training_run.checksums",
        required_files=(
            TRAINING_RUN_MANIFEST_NAME,
            TRAINING_RUN_CARD_NAME,
            TRAINING_PREFLIGHT_REPORT_NAME,
            *artifact_files,
        ),
    )


def _verify_training_run_identity(
    manifest_path: Path,
    run_manifest: TrainingRunManifest,
    *,
    manifest: Manifest | None,
    eval_input: EvalReportInput | None,
    efficiency_report: EfficiencyReport | None,
    issues: list[PackageIssue],
) -> None:
    if manifest is not None:
        expected_snapshot = _manifest_dataset_snapshot(manifest)
        if expected_snapshot is not None and run_manifest.dataset_snapshot_id != expected_snapshot:
            _error(
                issues,
                "model.training_run.dataset_snapshot_mismatch",
                manifest_path,
                "training_run_manifest.json dataset_snapshot_id must match manifest training data snapshot",
            )
        config_artifact = next(
            (artifact for artifact in run_manifest.artifacts if artifact.kind == "training_config"),
            None,
        )
        if config_artifact is not None:
            if config_artifact.path != manifest.training.config_file:
                _error(
                    issues,
                    "model.training_run.training_config_path_mismatch",
                    manifest_path,
                    "training_run_manifest.json training_config path must match manifest training config",
                )
            if config_artifact.sha256 != manifest.training.hash:
                _error(
                    issues,
                    "model.training_run.training_config_hash_mismatch",
                    manifest_path,
                    "training_run_manifest.json training_config hash must match manifest training hash",
                )
    if eval_input is not None and run_manifest.commit_sha.lower() != eval_input.commit.lower():
        _error(
            issues,
            "model.training_run.eval_commit_mismatch",
            manifest_path,
            "training_run_manifest.json commit_sha must match eval_metrics.json commit",
        )
    if (
        efficiency_report is not None
        and run_manifest.commit_sha.lower() != efficiency_report.commit.lower()
    ):
        _error(
            issues,
            "model.training_run.efficiency_commit_mismatch",
            manifest_path,
            "training_run_manifest.json commit_sha must match efficiency_report.json commit",
        )


def _verify_training_run_card(
    card_path: Path,
    manifest: TrainingRunManifest | None,
    issues: list[PackageIssue],
) -> None:
    if manifest is None or not card_path.is_file():
        return
    expected = render_training_run_card(manifest)
    try:
        observed = card_path.read_text(encoding="utf-8")
    except OSError as exc:
        _error(issues, "model.training_run.card.read_failed", card_path, str(exc))
        return
    if observed != expected:
        _error(
            issues,
            "model.training_run.card.stale",
            card_path,
            f"{TRAINING_RUN_CARD_NAME} does not match {TRAINING_RUN_MANIFEST_NAME}",
        )


def _verify_manifest_artifacts(
    model_dir: Path,
    manifest: Manifest,
    issues: list[PackageIssue],
) -> None:
    expected = {
        manifest.predictor.file: manifest.predictor.hash,
        manifest.action_encoder.file: manifest.action_encoder.hash,
        manifest.calibration.file: manifest.calibration.hash,
        manifest.training.config_file: manifest.training.hash,
        manifest.eval.file: manifest.eval.hash,
    }
    for relative, expected_hash in expected.items():
        path = model_dir / relative
        if not path.is_file():
            _error(issues, "model.artifact_missing", path, "manifest artifact is missing")
            continue
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            _error(
                issues,
                "model.artifact_hash_mismatch",
                path,
                f"expected {expected_hash}, observed {observed_hash}",
            )


def _verify_eval_report(
    path: Path,
    metrics_path: Path,
    issues: list[PackageIssue],
    *,
    model_dir: Path | None = None,
    dataset_dir: Path | None = None,
) -> EvalReportInput | None:
    _require_markdown_sections(
        path,
        issues,
        code_prefix="model.eval_report",
        sections=(
            "Summary",
            "Results",
            "Artifacts",
            "Limitations",
            "Negative Findings",
            "Conclusions",
        ),
    )
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if not re.search(r"(?im)^generated(?: by|:)", text):
        _error(
            issues,
            "model.eval_report.not_generated",
            path,
            "eval report must identify the generator or generation command",
        )
    if PLACEHOLDER_RE.search(text):
        _error(
            issues,
            "model.eval_report.placeholder",
            path,
            "eval report cannot contain placeholder wording",
        )
    required_patterns = {
        "model.eval_report.result_status": r"(?m)^- Result status: measured metrics",
        "model.eval_report.claim_boundary": r"(?m)^- Claim boundary: planned targets",
        "model.eval_report.model_id": r"(?m)^- Model id: sha256:[0-9a-f]{64}$",
        "model.eval_report.dataset_snapshot": r"(?m)^- Dataset snapshot: .+",
        "model.eval_report.artifact.checkpoint": r"(?m)^\|\s*checkpoint\s*\|",
        "model.eval_report.artifact.config": r"(?m)^\|\s*config\s*\|",
        "model.eval_report.artifact.dataset_manifest": (r"(?m)^\|\s*dataset_manifest\s*\|"),
        "model.eval_report.artifact.efficiency_report": (r"(?m)^\|\s*efficiency_report\s*\|"),
        "model.eval_report.artifact.eval_config": r"(?m)^\|\s*eval_config\s*\|",
    }
    for code, pattern in required_patterns.items():
        if re.search(pattern, text) is None:
            _error(issues, code, path, f"missing generated eval-report marker: {code}")
    if (
        _eval_report_has_baseline_rows(text)
        and re.search(
            r"(?m)^\|\s*(?:baseline_scores|input_\d+\.baseline_scores)\s*\|",
            text,
        )
        is None
    ):
        _error(
            issues,
            "model.eval_report.baseline_artifact_missing",
            path,
            "baseline result rows require a baseline score artifact row",
        )
    return _verify_eval_report_matches_metrics(
        path,
        metrics_path,
        text,
        issues,
        model_dir=model_dir,
        dataset_dir=dataset_dir,
    )


def _verify_eval_report_matches_metrics(
    report_path: Path,
    metrics_path: Path,
    report_text: str,
    issues: list[PackageIssue],
    *,
    model_dir: Path | None = None,
    dataset_dir: Path | None = None,
) -> EvalReportInput | None:
    if not metrics_path.is_file():
        _error(
            issues,
            "model.eval_metrics.missing",
            metrics_path,
            f"{EVAL_METRICS_NAME} is required",
        )
        return None
    try:
        report_input = load_report_input(metrics_path)
    except GenoLeWMError as exc:
        _error(
            issues,
            "model.eval_metrics.invalid",
            metrics_path,
            exc.message or str(exc),
        )
        return None
    expected = render_report(report_input)
    if report_text != expected:
        _error(
            issues,
            "model.eval_report.stale",
            report_path,
            f"eval_report.md does not match render of {EVAL_METRICS_NAME}",
        )
    if model_dir is not None and dataset_dir is not None:
        _verify_eval_artifact_files(
            report_input,
            metrics_path,
            issues,
            model_dir=model_dir,
            dataset_dir=dataset_dir,
        )
    return report_input


def _verify_eval_artifact_files(
    report_input: EvalReportInput,
    metrics_path: Path,
    issues: list[PackageIssue],
    *,
    model_dir: Path,
    dataset_dir: Path,
) -> None:
    for key, raw_path in report_input.artifacts:
        target = _resolve_eval_artifact_path(
            raw_path,
            metrics_path=metrics_path,
            model_dir=model_dir,
            dataset_dir=dataset_dir,
            issues=issues,
            key=key,
        )
        if target is None:
            continue
        if not target.is_file():
            _error(
                issues,
                f"model.eval_artifact.{key}.missing",
                target,
                "eval metrics artifact path does not exist in the release package",
            )
            continue
        if key == "scores" or key.endswith(".scores"):
            _verify_jsonl_generated_by(
                target,
                expected=SCORE_JSONL_GENERATED_BY,
                issues=issues,
                code=f"model.eval_artifact.{key}.generated_by",
            )
        elif key == "baseline_scores" or key.endswith(".baseline_scores"):
            _verify_jsonl_generated_by(
                target,
                expected=CARBON_ZERO_SHOT_GENERATED_BY,
                issues=issues,
                code=f"model.eval_artifact.{key}.generated_by",
            )


def _resolve_eval_artifact_path(
    raw_path: str,
    *,
    metrics_path: Path,
    model_dir: Path,
    dataset_dir: Path,
    issues: list[PackageIssue],
    key: str,
) -> Path | None:
    if "://" in raw_path:
        _error(
            issues,
            f"model.eval_artifact.{key}.path",
            metrics_path,
            "eval artifact paths must be package-relative, not URLs",
        )
        return None
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        _error(
            issues,
            f"model.eval_artifact.{key}.path",
            metrics_path,
            "eval artifact paths must be relative and stay inside the release package",
        )
        return None
    parts = relative.parts
    if not parts:
        _error(
            issues,
            f"model.eval_artifact.{key}.path",
            metrics_path,
            "eval artifact path must be non-empty",
        )
        return None
    if parts[0] == "model":
        return model_dir.joinpath(*parts[1:])
    if parts[0] == "dataset":
        return dataset_dir.joinpath(*parts[1:])
    if parts[0] == "eval":
        return model_dir / relative
    return metrics_path.parent / relative


def _verify_jsonl_generated_by(
    path: Path,
    *,
    expected: str,
    issues: list[PackageIssue],
    code: str,
) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            rows = 0
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                rows += 1
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    _error(issues, code, path, f"line {line_no}: invalid JSONL row: {exc.msg}")
                    return
                if not isinstance(payload, dict):
                    _error(issues, code, path, f"line {line_no}: JSONL row must be an object")
                    return
                if payload.get("generated_by") != expected:
                    _error(
                        issues,
                        code,
                        path,
                        f"line {line_no}: generated_by must be {expected}",
                    )
                    return
    except OSError as exc:
        _error(issues, code, path, str(exc))
        return
    if rows == 0:
        _error(issues, code, path, "JSONL artifact must contain at least one row")


def _verify_efficiency_report(
    path: Path,
    issues: list[PackageIssue],
) -> EfficiencyReport | None:
    if not path.is_file():
        _error(
            issues,
            "model.efficiency_report.missing",
            path,
            f"{EFFICIENCY_REPORT_NAME} is required",
        )
        return None
    try:
        return load_efficiency_report(path)
    except GenoLeWMError as exc:
        _error(
            issues,
            "model.efficiency_report.invalid",
            path,
            exc.message or str(exc),
        )
        return None


def _verify_release_evidence_identity(
    manifest: Manifest,
    eval_input: EvalReportInput | None,
    metrics_path: Path,
    efficiency_report: EfficiencyReport | None,
    efficiency_path: Path,
    issues: list[PackageIssue],
) -> None:
    if eval_input is None or efficiency_report is None:
        return
    if eval_input.model_release != manifest.release_id:
        _error(
            issues,
            "model.eval_metrics.model_release_mismatch",
            metrics_path,
            "eval_metrics.json model_release must match manifest release_id",
        )
    if efficiency_report.model_release != manifest.release_id:
        _error(
            issues,
            "model.efficiency_report.model_release_mismatch",
            efficiency_path,
            "efficiency_report.json model_release must match manifest release_id",
        )
    expected_snapshot = _manifest_dataset_snapshot(manifest)
    if expected_snapshot is not None and eval_input.dataset_snapshot != expected_snapshot:
        _error(
            issues,
            "model.eval_metrics.dataset_snapshot_mismatch",
            metrics_path,
            "eval_metrics.json dataset_snapshot must match manifest training data snapshot",
        )
    if expected_snapshot is not None and efficiency_report.dataset_snapshot != expected_snapshot:
        _error(
            issues,
            "model.efficiency_report.dataset_snapshot_mismatch",
            efficiency_path,
            "efficiency_report.json dataset_snapshot must match manifest training data snapshot",
        )
    if eval_input.dataset_snapshot != efficiency_report.dataset_snapshot:
        _error(
            issues,
            "model.eval_efficiency.dataset_snapshot_mismatch",
            metrics_path,
            "eval_metrics.json and efficiency_report.json dataset snapshots must match",
        )
    if eval_input.model_id != efficiency_report.model_id:
        _error(
            issues,
            "model.eval_efficiency.model_id_mismatch",
            metrics_path,
            "eval_metrics.json and efficiency_report.json model ids must match",
        )
    if eval_input.commit.lower() != efficiency_report.commit:
        _error(
            issues,
            "model.eval_efficiency.commit_mismatch",
            metrics_path,
            "eval_metrics.json and efficiency_report.json commits must match",
        )


def _manifest_dataset_snapshot(manifest: Manifest) -> str | None:
    for key in ("snapshot", "dataset_snapshot", "snapshot_id"):
        value = manifest.training.data_snapshot.get(key)
        if value:
            return value
    if len(manifest.training.data_snapshot) == 1:
        return next(iter(manifest.training.data_snapshot.values()))
    return None


def _eval_report_has_baseline_rows(text: str) -> bool:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 11 or cells[0] in {"Metric", "Artifact"}:
            continue
        baseline = cells[4]
        if baseline and baseline not in {"not reported", "-"}:
            return True
    return False


def _verify_dataset_dir(
    dataset_dir: Path,
    issues: list[PackageIssue],
) -> _DatasetVerification:
    _require_markdown_sections(
        dataset_dir / "data_card.md",
        issues,
        code_prefix="dataset.card",
        sections=("Sources", "License", "Preprocessing", "Splits", "Limitations"),
    )
    package = _verify_dataset_package_metadata(dataset_dir, issues)
    has_membership_and_split_evidence = (
        package is not None and package.membership_and_split_evidence is not None
    )
    if has_membership_and_split_evidence and (dataset_dir / "data_card.md").is_file():
        _require_markdown_sections(
            dataset_dir / "data_card.md",
            issues,
            code_prefix="dataset.card",
            sections=("Membership and Split Evidence",),
        )
    manifest_path = dataset_dir / "dataset_manifest.json"
    dataset_files: tuple[str, ...] = ()
    snapshot_id: str | None = None
    if not manifest_path.is_file():
        _error(
            issues,
            "dataset.manifest_missing",
            manifest_path,
            "dataset_manifest.json is required",
        )
    else:
        dataset_files, snapshot_id = _verify_dataset_manifest(dataset_dir, manifest_path, issues)
        _verify_dataset_integrity(dataset_dir, manifest_path, issues)
        _verify_dataset_snapshot_report(
            dataset_dir,
            manifest_path,
            snapshot_id=snapshot_id,
            issues=issues,
        )
    _verify_sha256sums(
        dataset_dir,
        issues,
        code_prefix="dataset.checksums",
        required_files=(
            "data_card.md",
            "dataset_package.json",
            "dataset_manifest.json",
            DEFAULT_REPORT_NAME,
            DATASET_INPUT_CHECK_REPORT_NAME,
            DATASET_SNAPSHOT_REPORT_NAME,
            *dataset_files,
        ),
    )
    return _DatasetVerification(
        snapshot_id=snapshot_id,
        membership_runtime_binding=(
            None if package is None else _dataset_membership_runtime_binding(package)
        ),
        package_valid=package is not None,
    )


def _verify_dataset_package_metadata(
    dataset_dir: Path,
    issues: list[PackageIssue],
) -> DatasetPackage | None:
    metadata_path = dataset_dir / "dataset_package.json"
    manifest_path = dataset_dir / "dataset_manifest.json"
    data_card_path = dataset_dir / "data_card.md"
    if not metadata_path.is_file():
        _error(
            issues,
            "dataset.metadata_missing",
            metadata_path,
            "dataset_package.json is required",
        )
        return None
    try:
        package = load_dataset_package(dataset_dir, metadata_path)
    except GenoLeWMError as exc:
        _error(
            issues,
            "dataset.metadata_invalid",
            metadata_path,
            exc.message or str(exc),
        )
        return None
    if manifest_path.is_file():
        try:
            observed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _error(issues, "dataset.manifest_invalid", manifest_path, str(exc))
        else:
            expected_manifest = package.manifest()
            if observed_manifest != expected_manifest:
                _error(
                    issues,
                    "dataset.manifest_stale",
                    manifest_path,
                    "dataset_manifest.json does not match dataset_package.json",
                )
    if data_card_path.is_file():
        try:
            observed_card = data_card_path.read_text(encoding="utf-8")
        except OSError as exc:
            _error(issues, "dataset.card.read_failed", data_card_path, str(exc))
        else:
            expected_integrity = _expected_dataset_integrity_for_card(
                dataset_dir,
                manifest_path,
            )
            expected_card = render_data_card(package, integrity_report=expected_integrity)
            if observed_card != expected_card:
                _error(
                    issues,
                    "dataset.card.stale",
                    data_card_path,
                    "data_card.md does not match dataset_package.json",
                )
    return package


def _dataset_membership_runtime_binding(
    package: DatasetPackage,
) -> dict[str, object] | None:
    evidence = package.membership_and_split_evidence
    if evidence is None:
        return None
    policy = {
        "schema_version": MEMBERSHIP_STORE_SCHEMA_VERSION,
        "membership_content_identity": evidence.membership_store.content_identity,
        "excluded_chromosomes": [
            *V03_CHROMOSOME_ROLES.validation,
            *V03_CHROMOSOME_ROLES.evaluation,
        ],
        "selection": "chromosome_roles",
        "lookup": "lookup.sqlite",
    }
    return {
        **evidence.to_dict(),
        "holdout_policy": policy,
        "holdout_policy_identity": canonical_json_sha256(policy),
    }


def _verify_training_dataset_membership_binding(
    model_dir: Path,
    dataset: _DatasetVerification,
    issues: list[PackageIssue],
) -> None:
    if not dataset.package_valid:
        return
    manifest_path = model_dir / TRAINING_RUN_MANIFEST_NAME
    try:
        run_manifest = verify_training_run_manifest(
            model_dir,
            manifest_path,
            require_preflight=True,
        )
    except GenoLeWMError:
        return
    if run_manifest.membership_and_split_evidence != dataset.membership_runtime_binding:
        _error(
            issues,
            "model.training_run.dataset_membership_mismatch",
            manifest_path,
            (
                "training-run membership store, split report, and holdout policy must "
                "match the verified dataset package"
            ),
        )


def _expected_dataset_integrity_for_card(
    dataset_dir: Path,
    manifest_path: Path,
) -> dict[str, object] | None:
    if not manifest_path.is_file():
        return None
    generated_at = _existing_dataset_integrity_generated_at(dataset_dir / DEFAULT_REPORT_NAME)
    try:
        return build_dataset_integrity_report(
            dataset_dir,
            manifest_path,
            generated_at=generated_at,
        ).to_dict()
    except GenoLeWMError:
        return None


def _existing_dataset_integrity_generated_at(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    generated_at = payload.get("generated_at")
    return generated_at if isinstance(generated_at, str) and generated_at else None


def _verify_dataset_integrity(
    dataset_dir: Path,
    manifest_path: Path,
    issues: list[PackageIssue],
) -> None:
    report_path = dataset_dir / DEFAULT_REPORT_NAME
    if not report_path.is_file():
        _error(
            issues,
            "dataset.integrity_missing",
            report_path,
            f"{DEFAULT_REPORT_NAME} is required",
        )
        return
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _error(issues, "dataset.integrity_invalid", report_path, str(exc))
        return
    if not isinstance(payload, dict):
        _error(issues, "dataset.integrity_invalid", report_path, "report must be an object")
        return
    if payload.get("generated_by") != DATASET_INTEGRITY_GENERATED_BY:
        _error(
            issues,
            "dataset.integrity.generated_by",
            report_path,
            f"generated_by must be {DATASET_INTEGRITY_GENERATED_BY}",
        )
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at:
        _error(
            issues,
            "dataset.integrity.generated_at",
            report_path,
            "generated_at must be a non-empty string",
        )
        return
    try:
        expected = build_dataset_integrity_report(
            dataset_dir,
            manifest_path,
            generated_at=generated_at,
        ).to_dict()
    except GenoLeWMError as exc:
        _error(
            issues,
            "dataset.integrity_failed",
            report_path,
            exc.message or str(exc),
        )
        return
    if payload != expected:
        _error(
            issues,
            "dataset.integrity_stale",
            report_path,
            "split integrity report is stale or does not match dataset_manifest.json",
        )


def _verify_dataset_snapshot_report(
    dataset_dir: Path,
    manifest_path: Path,
    *,
    snapshot_id: str | None,
    issues: list[PackageIssue],
) -> None:
    report_path = dataset_dir / DATASET_SNAPSHOT_REPORT_NAME
    if not report_path.is_file():
        _error(
            issues,
            "dataset.snapshot_report.missing",
            report_path,
            f"{DATASET_SNAPSHOT_REPORT_NAME} is required",
        )
        return
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _error(issues, "dataset.snapshot_report.invalid", report_path, str(exc))
        return
    if not isinstance(payload, dict):
        _error(issues, "dataset.snapshot_report.invalid", report_path, "report must be an object")
        return
    if payload.get("schema_version") != "1.0.0":
        _error(
            issues,
            "dataset.snapshot_report.schema_version",
            report_path,
            "schema_version must be 1.0.0",
        )
    if payload.get("generated_by") != DATASET_SNAPSHOT_GENERATED_BY:
        _error(
            issues,
            "dataset.snapshot_report.generated_by",
            report_path,
            f"generated_by must be {DATASET_SNAPSHOT_GENERATED_BY}",
        )
    if not isinstance(payload.get("generated_at"), str) or not payload.get("generated_at"):
        _error(
            issues,
            "dataset.snapshot_report.generated_at",
            report_path,
            "generated_at must be a non-empty string",
        )
    if snapshot_id is not None and payload.get("snapshot_id") != snapshot_id:
        _error(
            issues,
            "dataset.snapshot_report.snapshot_id",
            report_path,
            "snapshot report snapshot_id does not match dataset_manifest.json",
        )
    if payload.get("report_path") != DATASET_SNAPSHOT_REPORT_NAME:
        _error(
            issues,
            "dataset.snapshot_report.report_path",
            report_path,
            f"report_path must be {DATASET_SNAPSHOT_REPORT_NAME}",
        )
    if payload.get("metadata_path") != "dataset_package.json":
        _error(
            issues,
            "dataset.snapshot_report.metadata_path",
            report_path,
            "metadata_path must be dataset_package.json",
        )
    if payload.get("input_check_path") != DATASET_INPUT_CHECK_REPORT_NAME:
        _error(
            issues,
            "dataset.snapshot_report.input_check_path",
            report_path,
            f"input_check_path must be {DATASET_INPUT_CHECK_REPORT_NAME}",
        )
    _verify_snapshot_source_spec(payload.get("snapshot_spec"), report_path, issues)
    _verify_snapshot_input_check(
        payload.get("input_check"),
        snapshot_files=payload.get("files"),
        dataset_dir=dataset_dir,
        report_path=report_path,
        snapshot_id=snapshot_id,
        issues=issues,
    )
    _verify_snapshot_package_block(
        payload.get("package"),
        snapshot_files=payload.get("files"),
        membership_and_split_evidence=_dataset_manifest_membership_and_split_evidence(
            manifest_path
        ),
        dataset_dir=dataset_dir,
        report_path=report_path,
        snapshot_id=snapshot_id,
        issues=issues,
    )
    manifest_files = _dataset_manifest_file_index(dataset_dir, manifest_path, issues)
    _verify_snapshot_report_files(
        payload.get("files"),
        dataset_dir=dataset_dir,
        report_path=report_path,
        manifest_files=manifest_files,
        issues=issues,
    )


def _verify_snapshot_source_spec(
    raw: object,
    report_path: Path,
    issues: list[PackageIssue],
) -> None:
    if not isinstance(raw, dict):
        _error(
            issues,
            "dataset.snapshot_report.snapshot_spec",
            report_path,
            "snapshot_spec must be an object",
        )
        return
    path = raw.get("path")
    if not isinstance(path, str) or not _is_public_relative_reference(path):
        _error(
            issues,
            "dataset.snapshot_report.snapshot_spec.path",
            report_path,
            "snapshot spec path must be a public-safe relative reference",
        )
    digest = raw.get("sha256")
    if not isinstance(digest, str) or not looks_like_sha256(digest):
        _error(
            issues,
            "dataset.snapshot_report.snapshot_spec.sha256",
            report_path,
            "snapshot spec sha256 must be a sha256:<64hex> string",
        )
    if not _is_positive_int(raw.get("size_bytes")):
        _error(
            issues,
            "dataset.snapshot_report.snapshot_spec.size_bytes",
            report_path,
            "snapshot spec size_bytes must be a positive integer",
        )


def _verify_snapshot_input_check(
    raw: object,
    *,
    snapshot_files: object,
    dataset_dir: Path,
    report_path: Path,
    snapshot_id: str | None,
    issues: list[PackageIssue],
) -> None:
    code_prefix = "dataset.snapshot_report.input_check"
    _verify_snapshot_package_artifact_identity(
        raw,
        key="input_check",
        expected_relative=DATASET_INPUT_CHECK_REPORT_NAME,
        dataset_dir=dataset_dir,
        report_path=report_path,
        issues=issues,
        code_prefix=code_prefix,
    )
    path = dataset_dir / DATASET_INPUT_CHECK_REPORT_NAME
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _error(issues, f"{code_prefix}.invalid", path, str(exc))
        return
    if not isinstance(payload, dict):
        _error(issues, f"{code_prefix}.invalid", path, "input check report must be an object")
        return
    if payload.get("schema_version") != "1.0.0":
        _error(issues, f"{code_prefix}.schema_version", path, "schema_version must be 1.0.0")
    if payload.get("generated_by") != DATASET_INPUT_CHECK_GENERATED_BY:
        _error(
            issues,
            f"{code_prefix}.generated_by",
            path,
            f"generated_by must be {DATASET_INPUT_CHECK_GENERATED_BY}",
        )
    if snapshot_id is not None and payload.get("snapshot_id") != snapshot_id:
        _error(
            issues,
            f"{code_prefix}.snapshot_id",
            path,
            "input check snapshot_id does not match dataset_manifest.json",
        )
    raw_inputs = payload.get("inputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        _error(issues, f"{code_prefix}.inputs", path, "inputs must be a non-empty list")
        return
    expected = _expected_input_check_entries(snapshot_files)
    observed = [
        _normalized_input_check_entry(item) for item in raw_inputs if isinstance(item, dict)
    ]
    if None in observed or len(observed) != len(raw_inputs):
        _error(issues, f"{code_prefix}.inputs", path, "input entries must be valid objects")
        return
    if observed != expected:
        _error(
            issues,
            f"{code_prefix}.stale",
            path,
            "input check report does not match dataset_snapshot_report.json source identities",
        )


def _expected_input_check_entries(snapshot_files: object) -> list[dict[str, object]]:
    if not isinstance(snapshot_files, list):
        return []
    expected: list[dict[str, object]] = []
    for item in snapshot_files:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        source_path = item.get("source_path")
        source_sha256 = item.get("source_sha256")
        source_size_bytes = item.get("source_size_bytes")
        if not all(isinstance(value, str) for value in (path, source_path, source_sha256)):
            continue
        if not _is_positive_int(source_size_bytes):
            continue
        semantic_fields = _normalized_snapshot_input_semantic_fields(item)
        if semantic_fields is None:
            continue
        entry: dict[str, object] = {
            "source_path": source_path,
            "staged_path": path,
            **semantic_fields,
            "sha256": source_sha256,
            "size_bytes": source_size_bytes,
        }
        expected.append(entry)
    return expected


def _normalized_input_check_entry(item: dict[str, object]) -> dict[str, object] | None:
    source_path = item.get("source_path")
    staged_path = item.get("staged_path")
    sha256 = item.get("sha256")
    size_bytes = item.get("size_bytes")
    if not isinstance(source_path, str):
        return None
    if not isinstance(staged_path, str):
        return None
    if not isinstance(sha256, str) or not looks_like_sha256(sha256):
        return None
    if not _is_positive_int(size_bytes):
        return None
    if not _is_public_relative_reference(source_path) or not _is_public_relative_reference(
        staged_path
    ):
        return None
    semantic_fields = _normalized_snapshot_input_semantic_fields(item)
    if semantic_fields is None:
        return None
    normalized: dict[str, object] = {
        "source_path": source_path,
        "staged_path": staged_path,
        **semantic_fields,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }
    return normalized


def _normalized_snapshot_input_semantic_fields(
    item: dict[str, object],
) -> dict[str, object] | None:
    if "artifact_role" not in item:
        split = item.get("split")
        description = item.get("description")
        if not isinstance(split, str) or not isinstance(description, str):
            return None
        return {"split": split, "description": description}

    artifact_role = item.get("artifact_role")
    if not isinstance(artifact_role, str) or artifact_role not in ARTIFACT_ROLES:
        return None
    semantic_fields: dict[str, object] = {"artifact_role": artifact_role}
    if artifact_role == "evidence":
        if "split" in item or "companion_of" in item:
            return None
    else:
        split = item.get("split")
        if not isinstance(split, str) or not split:
            return None
        semantic_fields["split"] = split
    if artifact_role == "split_companion":
        companion_of = item.get("companion_of")
        if not isinstance(companion_of, str) or not companion_of:
            return None
        semantic_fields["companion_of"] = companion_of
    elif "companion_of" in item:
        return None
    if "description" in item:
        description = item["description"]
        if not isinstance(description, str) or not description:
            return None
        semantic_fields["description"] = description
    return semantic_fields


def _verify_snapshot_package_block(
    raw: object,
    *,
    snapshot_files: object,
    membership_and_split_evidence: object | None,
    dataset_dir: Path,
    report_path: Path,
    snapshot_id: str | None,
    issues: list[PackageIssue],
) -> None:
    if not isinstance(raw, dict):
        _error(
            issues,
            "dataset.snapshot_report.package",
            report_path,
            "package must be an object",
        )
        return
    expected = {
        "manifest_path": "dataset_manifest.json",
        "data_card_path": "data_card.md",
        "integrity_path": DEFAULT_REPORT_NAME,
        "checksums_path": "SHA256SUMS",
    }
    if snapshot_id is not None and raw.get("snapshot_id") != snapshot_id:
        _error(
            issues,
            "dataset.snapshot_report.package.snapshot_id",
            report_path,
            "package snapshot_id does not match dataset_manifest.json",
        )
    for key, expected_value in expected.items():
        if raw.get(key) != expected_value:
            _error(
                issues,
                f"dataset.snapshot_report.package.{key}",
                report_path,
                f"package.{key} must be {expected_value}",
            )
    identities = {
        "metadata": "dataset_package.json",
        "manifest": "dataset_manifest.json",
        "data_card": "data_card.md",
        "integrity": DEFAULT_REPORT_NAME,
    }
    for key, expected_value in identities.items():
        _verify_snapshot_package_artifact_identity(
            raw.get(key),
            key=key,
            expected_relative=expected_value,
            dataset_dir=dataset_dir,
            report_path=report_path,
            issues=issues,
        )
    if raw.get("membership_and_split_evidence") != membership_and_split_evidence:
        _error(
            issues,
            "dataset.snapshot_report.package.membership_and_split_evidence",
            report_path,
            (
                "package.membership_and_split_evidence must match the verified "
                "dataset manifest binding"
            ),
        )
    _verify_snapshot_package_files(
        raw.get("files"),
        snapshot_files=snapshot_files,
        report_path=report_path,
        issues=issues,
    )


def _dataset_manifest_membership_and_split_evidence(manifest_path: Path) -> object | None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload.get("membership_and_split_evidence")


def _verify_snapshot_package_files(
    raw: object,
    *,
    snapshot_files: object,
    report_path: Path,
    issues: list[PackageIssue],
) -> None:
    if not isinstance(raw, list) or not raw:
        _error(
            issues,
            "dataset.snapshot_report.package.files",
            report_path,
            "package.files must be a non-empty list",
        )
        return
    if not isinstance(snapshot_files, list) or not snapshot_files:
        return
    expected: list[dict[str, object]] = []
    for item in snapshot_files:
        if not isinstance(item, dict):
            return
        expected.append(
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
        )
    if raw != expected:
        _error(
            issues,
            "dataset.snapshot_report.package.files_stale",
            report_path,
            "package.files does not match dataset snapshot file identities",
        )


def _verify_snapshot_package_artifact_identity(
    raw: object,
    *,
    key: str,
    expected_relative: str,
    dataset_dir: Path,
    report_path: Path,
    issues: list[PackageIssue],
    code_prefix: str | None = None,
) -> None:
    if code_prefix is None:
        code_prefix = f"dataset.snapshot_report.package.{key}"
    if not isinstance(raw, dict):
        _error(
            issues,
            code_prefix,
            report_path,
            f"package.{key} must be an artifact identity object",
        )
        return
    if raw.get("path") != expected_relative:
        _error(
            issues,
            f"{code_prefix}.path",
            report_path,
            f"package.{key}.path must be {expected_relative}",
        )
    path = _safe_relative(dataset_dir, expected_relative, issues, code=f"{code_prefix}.path")
    if path is None:
        return
    if not path.is_file():
        _error(
            issues,
            f"{code_prefix}.missing",
            path,
            f"package.{key}.path target is missing",
        )
        return
    expected_sha = raw.get("sha256")
    if not isinstance(expected_sha, str) or not looks_like_sha256(expected_sha):
        _error(
            issues,
            f"{code_prefix}.sha256",
            report_path,
            f"package.{key}.sha256 must be a sha256:<64hex> string",
        )
    elif sha256_file(path) != expected_sha:
        _error(
            issues,
            f"{code_prefix}.hash_mismatch",
            path,
            f"package.{key}.sha256 does not match current file content",
        )
    if raw.get("size_bytes") != path.stat().st_size:
        _error(
            issues,
            f"{code_prefix}.size_bytes",
            report_path,
            f"package.{key}.size_bytes does not match current file size",
        )


def _dataset_manifest_file_index(
    dataset_dir: Path,
    manifest_path: Path,
    issues: list[PackageIssue],
) -> dict[str, dict[str, object]]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _error(issues, "dataset.snapshot_report.manifest_invalid", manifest_path, str(exc))
        return {}
    raw_files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(raw_files, list):
        return {}
    files: dict[str, dict[str, object]] = {}
    for item in raw_files:
        if not isinstance(item, dict):
            continue
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            continue
        if _safe_relative(dataset_dir, relative, issues, code="dataset.snapshot_report.path"):
            files[relative] = item
    return files


def _verify_snapshot_report_files(
    raw: object,
    *,
    dataset_dir: Path,
    report_path: Path,
    manifest_files: dict[str, dict[str, object]],
    issues: list[PackageIssue],
) -> None:
    if not isinstance(raw, list) or not raw:
        _error(
            issues,
            "dataset.snapshot_report.files",
            report_path,
            "files must be a non-empty list",
        )
        return
    observed_paths: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            _error(
                issues,
                "dataset.snapshot_report.file.invalid",
                report_path,
                f"files[{index}] must be an object",
            )
            continue
        _verify_snapshot_report_file(
            item,
            index=index,
            dataset_dir=dataset_dir,
            report_path=report_path,
            manifest_files=manifest_files,
            observed_paths=observed_paths,
            issues=issues,
        )
    missing = sorted(set(manifest_files) - observed_paths)
    for relative in missing:
        _error(
            issues,
            "dataset.snapshot_report.file.missing",
            report_path,
            f"snapshot report is missing manifest file {relative}",
        )


def _verify_snapshot_report_file(
    item: dict[str, object],
    *,
    index: int,
    dataset_dir: Path,
    report_path: Path,
    manifest_files: dict[str, dict[str, object]],
    observed_paths: set[str],
    issues: list[PackageIssue],
) -> None:
    relative = item.get("path")
    if not isinstance(relative, str) or not relative:
        _error(
            issues,
            "dataset.snapshot_report.file.path",
            report_path,
            f"files[{index}].path must be a non-empty string",
        )
        return
    if relative in observed_paths:
        _error(
            issues,
            "dataset.snapshot_report.file.duplicate",
            report_path,
            f"files[{index}].path appears more than once",
        )
        return
    observed_paths.add(relative)
    manifest_file = manifest_files.get(relative)
    if manifest_file is None:
        _error(
            issues,
            "dataset.snapshot_report.file.manifest",
            report_path,
            f"files[{index}].path is not in dataset_manifest.json",
        )
        return
    source_path = item.get("source_path")
    if not isinstance(source_path, str) or not _is_public_relative_reference(source_path):
        _error(
            issues,
            "dataset.snapshot_report.file.source_path",
            report_path,
            f"files[{index}].source_path must be a public-safe relative reference",
        )
    source_sha256 = item.get("source_sha256")
    if not isinstance(source_sha256, str) or not looks_like_sha256(source_sha256):
        _error(
            issues,
            "dataset.snapshot_report.file.source_sha256",
            report_path,
            f"files[{index}].source_sha256 must be a sha256:<64hex> string",
        )
    if not _is_positive_int(item.get("source_size_bytes")):
        _error(
            issues,
            "dataset.snapshot_report.file.source_size_bytes",
            report_path,
            f"files[{index}].source_size_bytes must be a positive integer",
        )
    _compare_snapshot_file_identity(
        item,
        manifest_file,
        dataset_dir=dataset_dir,
        report_path=report_path,
        relative=relative,
        index=index,
        issues=issues,
    )


def _compare_snapshot_file_identity(
    item: dict[str, object],
    manifest_file: dict[str, object],
    *,
    dataset_dir: Path,
    report_path: Path,
    relative: str,
    index: int,
    issues: list[PackageIssue],
) -> None:
    for key in ("split", "records", "artifact_role", "companion_of", "description"):
        if item.get(key) != manifest_file.get(key):
            _error(
                issues,
                f"dataset.snapshot_report.file.{key}",
                report_path,
                f"files[{index}].{key} does not match dataset_manifest.json",
            )
    path = _safe_relative(dataset_dir, relative, issues, code="dataset.snapshot_report.file.path")
    if path is None:
        return
    if not path.is_file():
        _error(
            issues,
            "dataset.snapshot_report.file.file_missing",
            path,
            "snapshot report file target is missing",
        )
        return
    expected_sha = item.get("sha256")
    if not isinstance(expected_sha, str) or not looks_like_sha256(expected_sha):
        _error(
            issues,
            "dataset.snapshot_report.file.sha256",
            report_path,
            f"files[{index}].sha256 must be a sha256:<64hex> string",
        )
    elif sha256_file(path) != expected_sha:
        _error(
            issues,
            "dataset.snapshot_report.file.hash_mismatch",
            path,
            f"files[{index}].sha256 does not match current file content",
        )
    if item.get("size_bytes") != path.stat().st_size:
        _error(
            issues,
            "dataset.snapshot_report.file.size_bytes",
            report_path,
            f"files[{index}].size_bytes does not match current file size",
        )


def _is_public_relative_reference(value: str) -> bool:
    candidate = Path(value)
    return bool(value.strip()) and not candidate.is_absolute() and ".." not in candidate.parts


def _is_positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _verify_dataset_manifest(
    dataset_dir: Path,
    manifest_path: Path,
    issues: list[PackageIssue],
) -> tuple[tuple[str, ...], str | None]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _error(issues, "dataset.manifest_invalid", manifest_path, str(exc))
        return (), None
    if not isinstance(payload, dict):
        _error(issues, "dataset.manifest_invalid", manifest_path, "manifest must be an object")
        return (), None
    required = {"schema_version", "snapshot_id", "sources", "splits", "files"}
    missing = sorted(required - set(payload))
    if missing:
        _error(
            issues,
            "dataset.manifest_missing_keys",
            manifest_path,
            f"missing keys: {', '.join(missing)}",
        )
    snapshot_id: str | None = None
    for key in ("schema_version", "snapshot_id"):
        if not isinstance(payload.get(key), str) or not payload.get(key):
            _error(issues, f"dataset.manifest.{key}", manifest_path, f"{key} must be non-empty")
        elif key == "snapshot_id":
            snapshot_id = payload[key]
    if not isinstance(payload.get("sources"), list) or not payload.get("sources"):
        _error(
            issues, "dataset.manifest.sources", manifest_path, "sources must be a non-empty list"
        )
    if not isinstance(payload.get("splits"), dict) or not payload.get("splits"):
        _error(
            issues, "dataset.manifest.splits", manifest_path, "splits must be a non-empty object"
        )
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        _error(issues, "dataset.manifest.files", manifest_path, "files must be a non-empty list")
        return (), snapshot_id
    manifest_files: list[str] = []
    for index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, dict):
            _error(
                issues,
                "dataset.file.invalid",
                manifest_path,
                f"files[{index}] must be an object",
            )
            continue
        relative = raw_file.get("path")
        expected_hash = raw_file.get("sha256")
        if not isinstance(relative, str) or not relative:
            _error(issues, "dataset.file.path", manifest_path, f"files[{index}].path is required")
            continue
        manifest_files.append(relative)
        path = _safe_relative(dataset_dir, relative, issues, code="dataset.file.path")
        if path is None:
            continue
        if not isinstance(expected_hash, str) or not looks_like_sha256(expected_hash):
            _error(
                issues,
                "dataset.file.sha256",
                manifest_path,
                f"files[{index}].sha256 must be a sha256:<64hex> string",
            )
            continue
        if not path.is_file():
            _error(issues, "dataset.file.missing", path, "dataset file is missing")
            continue
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            _error(
                issues,
                "dataset.file.hash_mismatch",
                path,
                f"expected {expected_hash}, observed {observed_hash}",
            )
    return tuple(manifest_files), snapshot_id


def _verify_demo_dir(
    demo_dir: Path,
    issues: list[PackageIssue],
    *,
    model_id: str | None,
    calibration_hash: str | None,
    model_dir: Path,
    allow_fixture_manifest: bool,
) -> None:
    transcript = demo_dir / "terminal-demo-transcript.md"
    if not transcript.is_file():
        _error(
            issues,
            "demo.transcript_missing",
            transcript,
            "terminal-demo-transcript.md is required",
        )
        return
    text = transcript.read_text(encoding="utf-8")
    required_patterns = {
        "demo.transcript.title": r"(?m)^# GenoLeWM Terminal Inference Transcript$",
        "demo.transcript.generated": r"(?m)^- Generated: .+Z$",
        "demo.transcript.status": r"(?m)^- Status: passed$",
        "demo.transcript.exit_code": r"(?m)^- Exit code: 0$",
        "demo.transcript.command": r"geno-lewm-score",
        "demo.transcript.command_section": r"(?m)^## Command$",
        "demo.transcript.model_release": r"(?m)^- Model release: .+",
        "demo.transcript.model_version": r"(?m)^- Model version: .+",
        "demo.transcript.model_id": r"(?m)^- Model id: sha256:[0-9a-f]{64}$",
        "demo.transcript.input_vcf_records": r"(?m)^- Input VCF records: [1-9][0-9]*$",
        "demo.transcript.input_alternate_alleles": (
            r"(?m)^- Input alternate alleles: [1-9][0-9]*$"
        ),
        "demo.transcript.input_contigs": r"(?m)^- Input contigs: .+",
        "demo.transcript.first_input_variants": r"(?m)^- First input variants: .+",
        "demo.transcript.scores": r"(?m)^- Scores: .+",
        "demo.transcript.receipts": r"(?m)^- Receipts: .+",
        "demo.transcript.runtime_preflight": r"(?m)^- Runtime preflight report: .+",
        "demo.transcript.batch_report": r"(?m)^- Batch receipt report: .+",
        "demo.transcript.demo_manifest": r"(?m)^- Demo manifest: .+terminal_demo_manifest\.json$",
        "demo.transcript.scores_hash": r"(?m)^- Scores SHA-256: sha256:[0-9a-f]{64}$",
        "demo.transcript.receipts_hash": r"(?m)^- Receipts SHA-256: sha256:[0-9a-f]{64}$",
        "demo.transcript.runtime_preflight_hash": (
            r"(?m)^- Runtime Preflight Report SHA-256: sha256:[0-9a-f]{64}$"
        ),
        "demo.transcript.batch_report_hash": (
            r"(?m)^- Batch Receipt Report SHA-256: sha256:[0-9a-f]{64}$"
        ),
        "demo.transcript.scores_rows": r"(?m)^- Scores JSONL rows: [1-9][0-9]*$",
        "demo.transcript.receipts_rows": r"(?m)^- Receipts JSONL rows: [1-9][0-9]*$",
        "demo.transcript.output_artifacts": r"(?m)^## Output Artifacts$",
        "demo.transcript.score_receipt_summary": r"(?m)^## Score And Receipt Summary$",
        "demo.transcript.artifact_inputs": r"(?m)^## Artifact Inputs$",
        "demo.transcript.manifest_input": r"(?m)^- Manifest: .+manifest\.json$",
        "demo.transcript.vcf_input": r"(?m)^- VCF: .+",
        "demo.transcript.fasta_input": r"(?m)^- FASTA: .+",
        "demo.transcript.claim_boundary": (
            r"(?m)^This transcript records command behavior only\. Model-quality claims require "
            r"the published evaluation report linked from the release\.$"
        ),
    }
    for code, pattern in required_patterns.items():
        if re.search(pattern, text) is None:
            _error(issues, code, transcript, f"missing transcript marker: {code}")
    if not allow_fixture_manifest and re.search(r"(?i)fixture|dummy|test manifest", text):
        _error(
            issues,
            "demo.transcript.fixture_marker",
            transcript,
            "transcript contains fixture/test wording",
        )
    _verify_demo_jsonl_artifact(
        demo_dir / "scores.jsonl",
        text,
        issues,
        label="Scores",
        code_prefix="demo.scores",
    )
    _verify_demo_jsonl_artifact(
        demo_dir / "receipts.jsonl",
        text,
        issues,
        label="Receipts",
        code_prefix="demo.receipts",
    )
    runtime_preflight = _verify_demo_runtime_preflight_report(
        demo_dir / RUNTIME_PREFLIGHT_REPORT_NAME,
        text,
        issues,
        model_id=model_id,
        model_dir=model_dir,
        demo_dir=demo_dir,
        allow_fixture_manifest=allow_fixture_manifest,
    )
    batch_receipt_report = _verify_demo_batch_receipt_report(
        demo_dir / BATCH_RECEIPT_REPORT_NAME,
        text,
        issues,
        model_id=model_id,
        calibration_hash=calibration_hash,
    )
    _verify_demo_manifest(
        demo_dir / DEMO_MANIFEST_NAME,
        text,
        issues,
        model_id=model_id,
        model_dir=model_dir,
        demo_dir=demo_dir,
        batch_receipt_report=batch_receipt_report,
        runtime_preflight_command=None if runtime_preflight is None else runtime_preflight.command,
        runtime_preflight_summary=(
            None if runtime_preflight is None else runtime_preflight.manifest_summary
        ),
    )


def _verify_demo_manifest(
    path: Path,
    transcript_text: str,
    issues: list[PackageIssue],
    *,
    model_id: str | None,
    model_dir: Path,
    demo_dir: Path,
    batch_receipt_report: dict[str, Any] | None,
    runtime_preflight_command: tuple[str, ...] | None,
    runtime_preflight_summary: dict[str, Any] | None,
) -> None:
    if not path.is_file():
        _error(issues, "demo.manifest.missing", path, f"{DEMO_MANIFEST_NAME} is required")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _error(issues, "demo.manifest.invalid", path, str(exc))
        return
    if not isinstance(payload, dict):
        _error(issues, "demo.manifest.invalid", path, "manifest must be an object")
        return
    if payload.get("schema_version") != DEMO_MANIFEST_SCHEMA_VERSION:
        _error(issues, "demo.manifest.schema_version", path, "demo manifest schema is unsupported")
    if payload.get("generated_by") != DEMO_MANIFEST_GENERATED_BY:
        _error(issues, "demo.manifest.generated_by", path, "demo manifest generator is invalid")
    if payload.get("status") != "passed":
        _error(issues, "demo.manifest.status", path, "demo manifest status must be passed")
    if DEMO_MANIFEST_NAME not in transcript_text:
        _error(issues, "demo.manifest.transcript_marker", path, "transcript does not name manifest")
    _verify_demo_manifest_command(
        payload.get("command"),
        path,
        issues,
        model_dir=model_dir,
        demo_dir=demo_dir,
        runtime_preflight_command=runtime_preflight_command,
    )
    _verify_demo_manifest_model(payload.get("model"), path, issues, model_id=model_id)
    _verify_demo_manifest_inputs(
        payload.get("inputs"),
        path,
        issues,
        model_dir=model_dir,
        demo_dir=demo_dir,
    )
    _verify_demo_manifest_artifacts(payload.get("artifacts"), path, issues, demo_dir=demo_dir)
    _verify_demo_manifest_runtime_preflight(
        payload.get("runtime_preflight"),
        path,
        issues,
        expected_summary=runtime_preflight_summary,
    )
    _verify_demo_manifest_score_receipt_batch(
        payload.get("score_receipt_batch"),
        path,
        issues,
        batch_receipt_report=batch_receipt_report,
    )


def _verify_demo_manifest_command(
    raw: object,
    path: Path,
    issues: list[PackageIssue],
    *,
    model_dir: Path,
    demo_dir: Path,
    runtime_preflight_command: tuple[str, ...] | None,
) -> None:
    if not isinstance(raw, dict):
        _error(issues, "demo.manifest.command", path, "command must be an object")
        return
    if raw.get("returncode") != 0:
        _error(issues, "demo.manifest.command.returncode", path, "demo command must exit zero")
    argv = raw.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(part, str) or not part for part in argv)
    ):
        _error(issues, "demo.manifest.command.argv", path, "argv must be a non-empty string list")
        return
    required = {"--model-dir", "--vcf", "--fasta", "--output", "--receipt", "--no-progress"}
    if argv[0] != "geno-lewm-score" or sorted(required - set(argv)):
        _error(
            issues,
            "demo.manifest.command.argv",
            path,
            "manifest command must cover the geno-lewm-score terminal demo invocation",
        )
        return
    _verify_score_command_paths(
        argv,
        path,
        issues,
        code_prefix="demo.manifest.command",
        model_dir=model_dir,
        demo_dir=demo_dir,
    )
    if runtime_preflight_command is not None and tuple(argv) != runtime_preflight_command:
        _error(
            issues,
            "demo.manifest.command.runtime_preflight_mismatch",
            path,
            "demo manifest command must match the command covered by runtime_preflight_report.json",
        )


def _verify_demo_manifest_model(
    raw: object,
    path: Path,
    issues: list[PackageIssue],
    *,
    model_id: str | None,
) -> None:
    if not isinstance(raw, dict):
        _error(issues, "demo.manifest.model", path, "model must be an object")
        return
    if model_id is not None and raw.get("model_id") != model_id:
        _error(issues, "demo.manifest.model_id", path, "demo manifest model id is stale")
    for key in ("release_id", "model_version", "model_id", "encoder_id", "encoder_revision"):
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            _error(issues, f"demo.manifest.model.{key}", path, f"{key} must be a non-empty string")


def _verify_demo_manifest_inputs(
    raw: object,
    path: Path,
    issues: list[PackageIssue],
    *,
    model_dir: Path,
    demo_dir: Path,
) -> None:
    if not isinstance(raw, dict):
        _error(issues, "demo.manifest.inputs", path, "inputs must be an object")
        return
    expected = {
        "model_manifest": (model_dir,),
        "vcf": (demo_dir,),
        "fasta": (demo_dir,),
    }
    vcf_path: Path | None = None
    for key, allowed_roots in expected.items():
        item = raw.get(key)
        code = f"demo.manifest.input.{key}"
        if not isinstance(item, dict):
            _error(issues, code, path, f"{key} input identity is required")
            continue
        target = _verify_preflight_file_identity(
            item,
            path,
            issues,
            code=code,
            allowed_roots=allowed_roots,
            base_dirs=(path.parent, model_dir.parent, demo_dir.parent),
        )
        if key == "model_manifest" and target is not None:
            expected_manifest = model_dir / "manifest.json"
            if target.resolve() != expected_manifest.resolve():
                _error(
                    issues,
                    f"{code}.path_mismatch",
                    target,
                    "demo manifest must identify the verified model manifest",
                )
        elif key == "vcf":
            vcf_path = target
    _verify_demo_manifest_vcf_summary(raw.get("vcf_summary"), path, issues, vcf_path=vcf_path)


def _verify_demo_manifest_vcf_summary(
    raw: object,
    path: Path,
    issues: list[PackageIssue],
    *,
    vcf_path: Path | None,
) -> None:
    if not isinstance(raw, dict):
        _error(
            issues,
            "demo.manifest.input.vcf_summary",
            path,
            "vcf_summary input block is required",
        )
        return
    if vcf_path is None or not vcf_path.is_file():
        return
    try:
        expected = summarize_vcf_input(vcf_path)
    except GenoLeWMError as exc:
        _error(
            issues,
            "demo.manifest.input.vcf_summary.unreadable",
            vcf_path,
            exc.message or str(exc),
        )
        return
    if raw != expected:
        _error(
            issues,
            "demo.manifest.input.vcf_summary.stale",
            path,
            "vcf_summary does not match the packaged demo VCF",
        )


def _verify_demo_manifest_artifacts(
    raw: object,
    path: Path,
    issues: list[PackageIssue],
    *,
    demo_dir: Path,
) -> None:
    if not isinstance(raw, list) or not raw:
        _error(issues, "demo.manifest.artifacts", path, "artifacts must be a non-empty list")
        return
    expected_labels = {
        "scores": demo_dir / "scores.jsonl",
        "receipts": demo_dir / "receipts.jsonl",
        "runtime preflight report": demo_dir / RUNTIME_PREFLIGHT_REPORT_NAME,
        "batch receipt report": demo_dir / BATCH_RECEIPT_REPORT_NAME,
        "terminal transcript": demo_dir / "terminal-demo-transcript.md",
    }
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            _error(issues, "demo.manifest.artifact", path, f"artifacts[{index}] must be an object")
            continue
        label = item.get("label")
        if not isinstance(label, str) or not label:
            _error(issues, "demo.manifest.artifact.label", path, "artifact label is required")
            continue
        seen.add(label)
        _verify_preflight_file_identity(
            item,
            path,
            issues,
            code="demo.manifest.artifact",
            allowed_roots=(demo_dir,),
            base_dirs=(path.parent, demo_dir.parent),
        )
        expected_path = expected_labels.get(label)
        if expected_path is not None:
            raw_path = item.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                continue
            observed_path = _identity_target(
                raw_path, path, base_dirs=(path.parent, demo_dir.parent)
            )
            if not _same_path(observed_path, expected_path):
                _error(
                    issues,
                    "demo.manifest.artifact.path_mismatch",
                    observed_path,
                    f"{label} artifact must point at {expected_path.name}",
                )
        if label in {"scores", "receipts"}:
            _verify_demo_manifest_jsonl_artifact(
                item,
                path,
                issues,
                label=label,
                demo_dir=demo_dir,
            )
    for label in sorted(set(expected_labels) - seen):
        _error(issues, "demo.manifest.artifact.missing", path, f"missing artifact: {label}")


def _verify_demo_manifest_jsonl_artifact(
    item: dict[str, object],
    path: Path,
    issues: list[PackageIssue],
    *,
    label: str,
    demo_dir: Path,
) -> None:
    rows = item.get("jsonl_records")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
        _error(
            issues,
            "demo.manifest.artifact.jsonl_records",
            path,
            f"{label} artifact must record a positive JSONL row count",
        )
        return
    raw_path = item.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return
    artifact_path = _identity_target(raw_path, path, base_dirs=(path.parent, demo_dir.parent))
    if not artifact_path.is_file():
        return
    try:
        observed, observed_fields = _inspect_jsonl_artifact(artifact_path)
    except (OSError, json.JSONDecodeError) as exc:
        _error(issues, "demo.manifest.artifact.invalid_jsonl", artifact_path, str(exc))
        return
    if observed != rows:
        _error(
            issues,
            "demo.manifest.artifact.row_count_mismatch",
            artifact_path,
            f"expected {rows}, observed {observed}",
        )
    fields = item.get("jsonl_fields")
    if (
        not isinstance(fields, list)
        or not fields
        or any(not isinstance(field, str) or not field for field in fields)
    ):
        _error(
            issues,
            "demo.manifest.artifact.jsonl_fields",
            path,
            f"{label} artifact must record JSONL field names",
        )
        return
    if tuple(fields) != observed_fields:
        _error(
            issues,
            "demo.manifest.artifact.field_mismatch",
            artifact_path,
            f"expected {_field_list(observed_fields)}, observed {_field_list(tuple(fields))}",
        )


def _verify_demo_manifest_score_receipt_batch(
    raw: object,
    path: Path,
    issues: list[PackageIssue],
    *,
    batch_receipt_report: dict[str, Any] | None,
) -> None:
    if not isinstance(raw, dict):
        _error(
            issues,
            "demo.manifest.score_receipt_batch",
            path,
            "score_receipt_batch must be an object",
        )
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
    missing = [field for field in required if field not in raw]
    if missing:
        _error(
            issues,
            "demo.manifest.score_receipt_batch.missing",
            path,
            f"score_receipt_batch is missing: {', '.join(missing)}",
        )
        return
    records = raw.get("records")
    if not isinstance(records, int) or isinstance(records, bool) or records <= 0:
        _error(
            issues,
            "demo.manifest.score_receipt_batch.records",
            path,
            "records must be a positive integer",
        )
    for field in ("model_id", "calibration_hash"):
        value = raw.get(field)
        if not isinstance(value, str) or not looks_like_sha256(value):
            _error(
                issues,
                f"demo.manifest.score_receipt_batch.{field}",
                path,
                f"{field} must be sha256:<64hex>",
            )
    if not _is_non_empty_str_list(raw.get("checked_score_fields")):
        _error(
            issues,
            "demo.manifest.score_receipt_batch.checked_score_fields",
            path,
            "checked_score_fields must be a non-empty string list",
        )
    if not isinstance(raw.get("runtime"), dict):
        _error(
            issues,
            "demo.manifest.score_receipt_batch.runtime",
            path,
            "runtime must be an object",
        )
    if batch_receipt_report is None:
        return
    expected = {field: batch_receipt_report[field] for field in required}
    if raw != expected:
        _error(
            issues,
            "demo.manifest.score_receipt_batch.stale",
            path,
            "score_receipt_batch does not match batch_receipt_report.json",
        )


def _verify_demo_manifest_runtime_preflight(
    raw: object,
    path: Path,
    issues: list[PackageIssue],
    *,
    expected_summary: dict[str, Any] | None,
) -> None:
    if not isinstance(raw, dict):
        _error(
            issues,
            "demo.manifest.runtime_preflight",
            path,
            "runtime_preflight must be an object",
        )
        return
    required = (
        "schema_version",
        "generated_by",
        "ok",
        "model_id",
        "release_id",
        "requested_backend",
        "selected_backend",
        "requirements",
        "command",
    )
    missing = [field for field in required if field not in raw]
    if missing:
        _error(
            issues,
            "demo.manifest.runtime_preflight.missing",
            path,
            f"runtime_preflight is missing: {', '.join(missing)}",
        )
        return
    if not isinstance(raw.get("requirements"), dict):
        _error(
            issues,
            "demo.manifest.runtime_preflight.requirements",
            path,
            "runtime_preflight.requirements must be an object",
        )
    command = raw.get("command")
    if not isinstance(command, dict) or not _is_non_empty_str_list(command.get("argv")):
        _error(
            issues,
            "demo.manifest.runtime_preflight.command",
            path,
            "runtime_preflight.command.argv must be a non-empty string list",
        )
    if expected_summary is None:
        return
    if raw != expected_summary:
        _error(
            issues,
            "demo.manifest.runtime_preflight.stale",
            path,
            "runtime_preflight does not match runtime_preflight_report.json",
        )


def _verify_demo_batch_receipt_report(
    path: Path,
    transcript_text: str,
    issues: list[PackageIssue],
    *,
    model_id: str | None,
    calibration_hash: str | None,
) -> dict[str, Any] | None:
    if not path.is_file():
        _error(issues, "demo.batch_receipt_report.missing", path, f"{path.name} is required")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _error(issues, "demo.batch_receipt_report.invalid", path, str(exc))
        return None
    if not isinstance(payload, dict):
        _error(issues, "demo.batch_receipt_report.invalid", path, "report must be an object")
        return None
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at:
        _error(
            issues,
            "demo.batch_receipt_report.generated_at",
            path,
            "generated_at must be a non-empty string",
        )
        return None
    try:
        expected = build_batch_receipt_report(
            path.parent / "scores.jsonl",
            path.parent / "receipts.jsonl",
            generated_at=generated_at,
        ).to_dict()
    except GenoLeWMError as exc:
        _error(
            issues,
            "demo.batch_receipt_report.failed",
            path,
            exc.message or str(exc),
        )
        return None
    if payload != expected:
        _error(
            issues,
            "demo.batch_receipt_report.stale",
            path,
            "batch receipt report is stale or does not match score/receipt JSONL artifacts",
        )
        return None
    if model_id is not None and payload.get("model_id") != model_id:
        _error(
            issues,
            "demo.batch_receipt_report.model_id",
            path,
            "batch receipt report model id does not match model manifest",
        )
    if calibration_hash is not None and payload.get("calibration_hash") != calibration_hash:
        _error(
            issues,
            "demo.batch_receipt_report.calibration_hash",
            path,
            "batch receipt report calibration hash does not match model manifest",
        )
    digest = sha256_file(path)
    if f"- Batch Receipt Report SHA-256: {digest}" not in transcript_text:
        _error(
            issues,
            "demo.batch_receipt_report.hash_mismatch",
            path,
            "transcript does not record this batch receipt report hash",
        )
    return payload


def _verify_demo_runtime_preflight_report(
    path: Path,
    transcript_text: str,
    issues: list[PackageIssue],
    *,
    model_id: str | None,
    model_dir: Path,
    demo_dir: Path,
    allow_fixture_manifest: bool,
) -> RuntimePreflightEvidence | None:
    if not path.is_file():
        _error(issues, "demo.runtime_preflight.missing", path, f"{path.name} is required")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _error(issues, "demo.runtime_preflight.invalid", path, str(exc))
        return None
    if not isinstance(payload, dict):
        _error(issues, "demo.runtime_preflight.invalid", path, "report must be an object")
        return None
    if payload.get("schema_version") != RUNTIME_PREFLIGHT_SCHEMA_VERSION:
        _error(
            issues,
            "demo.runtime_preflight.schema_version",
            path,
            "runtime preflight schema version is unsupported",
        )
    if payload.get("generated_by") != RUNTIME_PREFLIGHT_GENERATED_BY:
        _error(
            issues,
            "demo.runtime_preflight.generated_by",
            path,
            "runtime preflight report must identify its generator",
        )
    if payload.get("ok") is not True:
        _error(issues, "demo.runtime_preflight.failed", path, "runtime preflight did not pass")
    if model_id is not None and payload.get("model_id") != model_id:
        _error(
            issues,
            "demo.runtime_preflight.model_id",
            path,
            "runtime preflight model id does not match model manifest",
        )
    _verify_preflight_requirements(
        payload.get("requirements"),
        path,
        issues,
        allow_fixture_manifest=allow_fixture_manifest,
    )
    runtime_preflight_command = _verify_preflight_command(
        payload.get("command"),
        path,
        issues,
        model_dir=model_dir,
        demo_dir=demo_dir,
    )
    _verify_preflight_dependencies(payload.get("dependencies"), path, issues)
    _verify_preflight_network_guard(payload.get("network_guard"), path, issues)
    _verify_preflight_file_group(
        payload.get("inputs"),
        path,
        issues,
        code_prefix="demo.runtime_preflight.input",
        expected_keys=("vcf", "fasta"),
        allowed_roots=(demo_dir,),
        base_dirs=(path.parent, demo_dir.parent),
    )
    _verify_preflight_artifacts(
        payload.get("artifacts"),
        path,
        issues,
        base_dirs=(path.parent, model_dir.parent, demo_dir.parent),
    )
    digest = sha256_file(path)
    if f"- Runtime Preflight Report SHA-256: {digest}" not in transcript_text:
        _error(
            issues,
            "demo.runtime_preflight.hash_mismatch",
            path,
            "transcript does not record this runtime preflight report hash",
        )
    return RuntimePreflightEvidence(
        command=runtime_preflight_command,
        manifest_summary=_runtime_preflight_manifest_summary(
            payload,
            runtime_preflight_command,
        ),
    )


def _runtime_preflight_manifest_summary(
    payload: dict[str, Any],
    command: tuple[str, ...] | None,
) -> dict[str, Any] | None:
    requirements = payload.get("requirements")
    if command is None or not isinstance(requirements, dict):
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
            "argv": list(command),
            "shell": shlex.join(command),
        },
    }


def _verify_demo_jsonl_artifact(
    path: Path,
    transcript_text: str,
    issues: list[PackageIssue],
    *,
    label: str,
    code_prefix: str,
) -> None:
    if not path.is_file():
        _error(issues, f"{code_prefix}.missing", path, f"{path.name} is required")
        return
    if path.stat().st_size <= 0:
        _error(issues, f"{code_prefix}.empty", path, f"{path.name} must be non-empty")
        return
    try:
        records, fields = _inspect_jsonl_artifact(path)
    except json.JSONDecodeError as exc:
        _error(
            issues,
            f"{code_prefix}.invalid_jsonl",
            path,
            f"line {exc.lineno}: {exc.msg}",
        )
        return
    except OSError as exc:
        _error(issues, f"{code_prefix}.read_failed", path, str(exc))
        return
    if records <= 0:
        _error(issues, f"{code_prefix}.no_rows", path, f"{path.name} has no JSONL rows")
        return
    digest = sha256_file(path)
    if f"- {label} SHA-256: {digest}" not in transcript_text:
        _error(
            issues,
            f"{code_prefix}.hash_mismatch",
            path,
            "transcript does not record this artifact hash",
        )
    if f"- {label} JSONL rows: {records}" not in transcript_text:
        _error(
            issues,
            f"{code_prefix}.row_count_mismatch",
            path,
            "transcript does not record this artifact row count",
        )
    if f"- {label} JSONL fields: {_field_list(fields)}" not in transcript_text:
        _error(
            issues,
            f"{code_prefix}.field_mismatch",
            path,
            "transcript does not record this artifact field list",
        )


def _count_jsonl_records(path: Path) -> int:
    records, _fields = _inspect_jsonl_artifact(path)
    return records


def _inspect_jsonl_artifact(path: Path) -> tuple[int, tuple[str, ...]]:
    records = 0
    fields: list[str] = []
    seen_fields: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise json.JSONDecodeError("JSONL rows must be objects", line, line_no)
            for key in payload:
                if key not in seen_fields:
                    fields.append(key)
                    seen_fields.add(key)
            records += 1
    return records, tuple(fields)


def _field_list(fields: tuple[str, ...]) -> str:
    return ", ".join(fields) if fields else "-"


def _is_non_empty_str_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item) for item in value)
    )


def _verify_preflight_requirements(
    raw: object,
    path: Path,
    issues: list[PackageIssue],
    *,
    allow_fixture_manifest: bool,
) -> None:
    if not isinstance(raw, dict):
        _error(
            issues,
            "demo.runtime_preflight.requirements",
            path,
            "requirements must be an object",
        )
        return
    if raw.get("native_runtime") is not True:
        _error(
            issues,
            "demo.runtime_preflight.native_runtime_not_required",
            path,
            "release preflight must require native runtime dependencies",
        )
    if not allow_fixture_manifest and raw.get("fixture_manifest_allowed") is not False:
        _error(
            issues,
            "demo.runtime_preflight.fixture_manifest_allowed",
            path,
            "release preflight must not allow fixture/test manifests",
        )


def _verify_preflight_command(
    raw: object,
    path: Path,
    issues: list[PackageIssue],
    *,
    model_dir: Path,
    demo_dir: Path,
) -> tuple[str, ...] | None:
    if not isinstance(raw, dict):
        _error(issues, "demo.runtime_preflight.command", path, "command must be an object")
        return None
    argv = raw.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(part, str) or not part for part in argv)
    ):
        _error(
            issues,
            "demo.runtime_preflight.command.argv",
            path,
            "command.argv must be a non-empty string list",
        )
        return None
    required = {"--model-dir", "--vcf", "--fasta", "--output", "--receipt", "--no-progress"}
    missing = sorted(required - set(argv))
    if argv[0] != "geno-lewm-score" or missing:
        _error(
            issues,
            "demo.runtime_preflight.command.argv",
            path,
            "command must cover the geno-lewm-score terminal demo invocation",
        )
        return None
    _verify_score_command_paths(
        argv,
        path,
        issues,
        code_prefix="demo.runtime_preflight.command",
        model_dir=model_dir,
        demo_dir=demo_dir,
    )
    return tuple(argv)


def _verify_score_command_paths(
    argv: list[str],
    path: Path,
    issues: list[PackageIssue],
    *,
    code_prefix: str,
    model_dir: Path,
    demo_dir: Path,
) -> None:
    exact_paths = {
        "--model-dir": model_dir,
        "--output": demo_dir / "scores.jsonl",
        "--receipt": demo_dir / "receipts.jsonl",
    }
    base_dirs = (path.parent, model_dir.parent, demo_dir.parent)
    for flag, expected in exact_paths.items():
        raw = _flag_value(argv, flag)
        if raw is None:
            _error(issues, f"{code_prefix}.{flag.removeprefix('--')}", path, f"{flag} is required")
            continue
        observed = _identity_target(raw, path, base_dirs=base_dirs)
        if not _same_path(observed, expected):
            _error(
                issues,
                f"{code_prefix}.{flag.removeprefix('--')}_path",
                observed,
                f"{flag} must point at {expected}",
            )
    for flag in ("--vcf", "--fasta"):
        raw = _flag_value(argv, flag)
        if raw is None:
            _error(issues, f"{code_prefix}.{flag.removeprefix('--')}", path, f"{flag} is required")
            continue
        observed = _identity_target(raw, path, base_dirs=(path.parent, demo_dir.parent))
        if not _path_is_within(observed, demo_dir):
            _error(
                issues,
                f"{code_prefix}.{flag.removeprefix('--')}_outside_package",
                observed,
                f"{flag} must point inside the demo package",
            )


def _flag_value(argv: list[str], flag: str) -> str | None:
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    value_index = index + 1
    if value_index >= len(argv):
        return None
    value = argv[value_index]
    if value.startswith("--"):
        return None
    return value


def _verify_preflight_dependencies(
    raw: object,
    path: Path,
    issues: list[PackageIssue],
) -> None:
    if not isinstance(raw, list) or not raw:
        _error(
            issues,
            "demo.runtime_preflight.dependencies",
            path,
            "dependencies must be a non-empty list",
        )
        return
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            _error(
                issues,
                "demo.runtime_preflight.dependency",
                path,
                f"dependencies[{index}] must be an object",
            )
            continue
        if item.get("required") is True and item.get("available") is not True:
            _error(
                issues,
                "demo.runtime_preflight.dependency_unavailable",
                path,
                f"required dependency is unavailable: {item.get('import_name')}",
            )


def _verify_preflight_network_guard(
    raw: object,
    path: Path,
    issues: list[PackageIssue],
) -> None:
    if not isinstance(raw, dict) or raw.get("ok") is not True:
        _error(
            issues,
            "demo.runtime_preflight.network_guard",
            path,
            "runtime preflight must demonstrate fail-closed network guard behavior",
        )


def _verify_preflight_file_group(
    raw: object,
    path: Path,
    issues: list[PackageIssue],
    *,
    code_prefix: str,
    expected_keys: tuple[str, ...],
    allowed_roots: tuple[Path, ...] = (),
    base_dirs: tuple[Path, ...] = (),
) -> None:
    if not isinstance(raw, dict):
        _error(issues, f"{code_prefix}.invalid", path, "file group must be an object")
        return
    for key in expected_keys:
        item = raw.get(key)
        if not isinstance(item, dict):
            _error(issues, f"{code_prefix}.{key}", path, f"{key} identity is required")
            continue
        _verify_preflight_file_identity(
            item,
            path,
            issues,
            code=f"{code_prefix}.{key}",
            allowed_roots=allowed_roots,
            base_dirs=base_dirs,
        )


def _verify_preflight_artifacts(
    raw: object,
    path: Path,
    issues: list[PackageIssue],
    *,
    base_dirs: tuple[Path, ...] = (),
) -> None:
    if not isinstance(raw, list) or not raw:
        _error(
            issues,
            "demo.runtime_preflight.artifacts",
            path,
            "artifacts must be a non-empty list",
        )
        return
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            _error(
                issues,
                "demo.runtime_preflight.artifact",
                path,
                f"artifacts[{index}] must be an object",
            )
            continue
        _verify_preflight_file_identity(
            item,
            path,
            issues,
            code="demo.runtime_preflight.artifact",
            base_dirs=base_dirs,
        )
        if item.get("ok") is not True:
            _error(
                issues,
                "demo.runtime_preflight.artifact_not_ok",
                path,
                f"artifact preflight did not pass: {item.get('path')}",
            )


def _verify_preflight_file_identity(
    item: dict[str, object],
    path: Path,
    issues: list[PackageIssue],
    *,
    code: str,
    allowed_roots: tuple[Path, ...] = (),
    base_dirs: tuple[Path, ...] = (),
) -> Path | None:
    raw_path = item.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        _error(issues, f"{code}.path", path, "file identity path must be non-empty")
        return None
    artifact_path = _identity_target(raw_path, path, base_dirs=base_dirs)
    if allowed_roots and not any(_path_is_within(artifact_path, root) for root in allowed_roots):
        _error(
            issues,
            f"{code}.outside_package",
            artifact_path,
            "file identity must stay inside the release package",
        )
    digest = item.get("sha256")
    if not isinstance(digest, str) or not looks_like_sha256(digest):
        _error(issues, f"{code}.sha256", path, "file identity must include sha256:<64hex>")
        return artifact_path
    if not artifact_path.is_file():
        _error(issues, f"{code}.missing", artifact_path, "preflight file target is missing")
        return artifact_path
    observed = sha256_file(artifact_path)
    if observed != digest:
        _error(
            issues,
            f"{code}.hash_mismatch",
            artifact_path,
            f"expected {digest}, observed {observed}",
        )
    size_bytes = item.get("size_bytes")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
        _error(issues, f"{code}.size_bytes", path, "file identity size_bytes must be positive")
    elif artifact_path.stat().st_size != size_bytes:
        _error(
            issues,
            f"{code}.size_mismatch",
            artifact_path,
            f"expected {size_bytes}, observed {artifact_path.stat().st_size}",
        )
    return artifact_path


def _identity_target(
    raw_path: str,
    owner_path: Path,
    *,
    base_dirs: tuple[Path, ...] = (),
) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    roots = base_dirs or (owner_path.parent,)
    candidates = tuple(root / candidate for root in roots)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except (OSError, RuntimeError):
        return False


def _verify_paper_path(
    path: Path,
    issues: list[PackageIssue],
    *,
    model_id: str | None,
    dataset_snapshot: str | None,
    model_dir: Path,
    dataset_dir: Path,
    demo_dir: Path,
) -> None:
    _require_markdown_sections(
        path,
        issues,
        code_prefix="paper",
        sections=(
            "Citation Metadata",
            "Results",
            "Conclusions",
            "Negative Findings",
            "Limitations",
            "Artifact Availability",
        ),
    )
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    required_patterns = {
        "paper.generated_by": rf"(?m)^Generated by: {re.escape(PAPER_DRAFT_GENERATED_BY)}$",
        "paper.generated_at": r"(?m)^Generated: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
        "paper.model_package": r"model_package\.json",
        "paper.dataset_package": r"dataset_package\.json",
        "paper.dataset_input_check_report": r"dataset_input_check_report\.json",
        "paper.dataset_snapshot_report": r"dataset_snapshot_report\.json",
        "paper.eval_metrics": r"eval_metrics\.json",
        "paper.eval_config": r"eval_config\.effective\.yaml",
        "paper.eval_report": r"eval_report\.md",
        "paper.efficiency_report": r"efficiency_report\.json",
        "paper.transcript": r"terminal-demo-transcript\.md",
        "paper.demo_manifest": r"terminal_demo_manifest\.json",
        "paper.runtime_preflight_report": r"runtime_preflight_report\.json",
        "paper.batch_receipt_report": r"batch_receipt_report\.json",
    }
    for code, pattern in required_patterns.items():
        if re.search(pattern, text) is None:
            _error(issues, code, path, f"missing paper marker: {code}")
    if model_id is not None and model_id not in text:
        _error(issues, "paper.model_id", path, "paper does not name the verified model id")
    if dataset_snapshot is not None and dataset_snapshot not in text:
        _error(
            issues,
            "paper.dataset_snapshot",
            path,
            "paper does not name the verified dataset snapshot id",
        )
    if PLACEHOLDER_RE.search(text):
        _error(issues, "paper.placeholder", path, "paper cannot contain placeholder wording")
    _verify_paper_matches_artifacts(
        path,
        text,
        issues,
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        demo_dir=demo_dir,
    )


def _verify_paper_matches_artifacts(
    path: Path,
    text: str,
    issues: list[PackageIssue],
    *,
    model_dir: Path,
    dataset_dir: Path,
    demo_dir: Path,
) -> None:
    title = _paper_title(text)
    generated_at = _paper_generated_at(text)
    if title is None:
        _error(issues, "paper.title", path, "generated paper must start with an H1 title")
        return
    if generated_at is None:
        return
    try:
        expected = render_paper_draft(
            model_dir=model_dir,
            dataset_dir=dataset_dir,
            demo_dir=demo_dir,
            title=title,
            generated_at=generated_at,
        )
    except GenoLeWMError as exc:
        _error(issues, "paper.render_failed", path, exc.message or str(exc))
        return
    if text != expected:
        _error(
            issues,
            "paper.stale",
            path,
            "paper draft does not match render of current release artifacts",
        )


def _paper_title(text: str) -> str | None:
    match = re.search(r"(?m)^# (?P<title>.+)$", text)
    if match is None:
        return None
    title = match.group("title").strip()
    return title or None


def _paper_generated_at(text: str) -> str | None:
    match = re.search(r"(?m)^Generated: (?P<generated_at>.+)$", text)
    if match is None:
        return None
    generated_at = match.group("generated_at").strip()
    return generated_at or None


def _verify_sha256sums(
    directory: Path,
    issues: list[PackageIssue],
    *,
    code_prefix: str,
    required_files: tuple[str, ...],
) -> None:
    path = directory / "SHA256SUMS"
    if not path.is_file():
        _error(issues, f"{code_prefix}.missing", path, "SHA256SUMS is required")
        return
    _verify_named_sha256sums(
        directory,
        path,
        issues,
        code_prefix=code_prefix,
        required_files=required_files,
    )


def _verify_named_sha256sums(
    directory: Path,
    path: Path,
    issues: list[PackageIssue],
    *,
    code_prefix: str,
    required_files: tuple[str, ...],
) -> None:
    if not path.is_file():
        _error(issues, f"{code_prefix}.missing", path, f"{path.name} is required")
        return
    try:
        entries = _parse_sha256sums(path.read_text(encoding="utf-8"))
    except InputError as exc:
        _error(issues, f"{code_prefix}.invalid", path, exc.message or str(exc))
        return
    for relative in required_files:
        if relative not in entries:
            _error(
                issues,
                f"{code_prefix}.entry_missing",
                path,
                f"SHA256SUMS is missing entry for {relative}",
            )
    for relative, expected_hash in entries.items():
        artifact_path = _safe_relative(directory, relative, issues, code=f"{code_prefix}.path")
        if artifact_path is None:
            continue
        if not artifact_path.is_file():
            _error(issues, f"{code_prefix}.file_missing", artifact_path, "checksum target missing")
            continue
        observed_hash = sha256_file(artifact_path)
        if observed_hash != expected_hash:
            _error(
                issues,
                f"{code_prefix}.hash_mismatch",
                artifact_path,
                f"expected {expected_hash}, observed {observed_hash}",
            )


def _parse_sha256sums(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise InputError("SHA256SUMS lines must be '<hex>  <path>'", details={"line": line_no})
        hex_digest, relative = parts
        expected_hash = "sha256:" + hex_digest.lower()
        if not looks_like_sha256(expected_hash):
            raise InputError(
                "SHA256SUMS contains an invalid SHA-256 digest",
                details={"line": line_no, "digest": hex_digest},
            )
        if relative in entries:
            raise InputError(
                "SHA256SUMS contains duplicate paths",
                details={"line": line_no, "path": relative},
            )
        entries[relative] = expected_hash
    if not entries:
        raise InputError("SHA256SUMS must contain at least one entry")
    return entries


def _require_markdown_sections(
    path: Path,
    issues: list[PackageIssue],
    *,
    code_prefix: str,
    sections: tuple[str, ...],
) -> None:
    if not path.is_file():
        _error(issues, f"{code_prefix}.missing", path, f"{path.name} is required")
        return
    text = path.read_text(encoding="utf-8")
    for section in sections:
        pattern = r"(?im)^#{1,3}\s+" + re.escape(section).replace(r"\ ", r"[- ]") + r"\b"
        if re.search(pattern, text) is None:
            _error(
                issues,
                f"{code_prefix}.section_missing",
                path,
                f"missing markdown section: {section}",
            )


def _safe_relative(
    root: Path,
    relative: str,
    issues: list[PackageIssue],
    *,
    code: str,
) -> Path | None:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        _error(
            issues, code, root / relative, "artifact paths must be relative and stay inside root"
        )
        return None
    return root / candidate


def _looks_like_fixture_manifest(manifest: Manifest) -> bool:
    parts = [
        manifest.release_id,
        manifest.model_version,
        *manifest.training.data_snapshot.keys(),
        *manifest.training.data_snapshot.values(),
    ]
    text = " ".join(parts).lower()
    return any(token in text for token in ("fixture", "dummy", "test"))


def _error(
    issues: list[PackageIssue],
    code: str,
    path: Path,
    message: str,
) -> None:
    issues.append(PackageIssue("error", code, str(path), message))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
