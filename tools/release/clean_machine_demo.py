# SPDX-License-Identifier: Apache-2.0
"""Replay the terminal demo from public release artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final
from urllib import error as urllib_error, request as urllib_request
from urllib.parse import quote, urlparse

from geno_lewm.errors import GenoLeWMError, InputError, ResourceError, exit_code_for
from geno_lewm.provenance import load_manifest, sha256_file
from geno_lewm.provenance.hashing import looks_like_sha256
from tools.demo.terminal_inference import (
    DEMO_MANIFEST_NAME,
    DEMO_MANIFEST_SCHEMA_VERSION,
    GENERATED_BY as DEMO_MANIFEST_GENERATED_BY,
    DemoRequest,
    run_demo_transcript,
)
from tools.release.hub_release import (
    GENERATED_BY as HUB_RELEASE_GENERATED_BY,
    SCHEMA_VERSION as HUB_RELEASE_SCHEMA_VERSION,
)
from tools.release.paper_package import PackageIssue, PackagePaths, verify_package
from tools.release.release_candidate import (
    GENERATED_BY as RELEASE_CANDIDATE_GENERATED_BY,
    SCHEMA_VERSION as RELEASE_CANDIDATE_SCHEMA_VERSION,
)

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.clean_machine_demo"
DEFAULT_TIMEOUT_SECONDS: Final = 30.0
DEFAULT_REPORT_NAME: Final = "clean_machine_demo_report.json"
REQUIRED_READINESS_CODES: Final = (
    "package_verifier",
    "model_package",
    "dataset_package",
    "terminal_demo",
    "paper_artifact",
    "public_links",
    "public_artifacts",
    "hub_publication_plan",
)


@dataclass(frozen=True, slots=True)
class DownloadedArtifact:
    """One public artifact downloaded for clean-machine demo replay."""

    group: str
    source_url: str
    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "group": self.group,
            "source_url": self.source_url,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class ReplayArtifact:
    """One local artifact generated during clean-machine replay."""

    label: str
    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class CleanMachineDemoReport:
    """Machine-readable clean-machine demo replay evidence."""

    schema_version: str
    generated_by: str
    generated_at: str
    release_candidate_report: str
    release_candidate_report_identity: ReplayArtifact
    model_dir: str
    dataset_dir: str
    demo_dir: str
    replay_dir: str
    package_ok: bool
    package_issues: tuple[PackageIssue, ...]
    transcript_path: str
    demo_manifest_path: str
    downloaded_artifacts: tuple[DownloadedArtifact, ...]
    replay_artifacts: tuple[ReplayArtifact, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "release_candidate_report": self.release_candidate_report,
            "release_candidate_report_identity": (self.release_candidate_report_identity.to_dict()),
            "model_dir": self.model_dir,
            "dataset_dir": self.dataset_dir,
            "demo_dir": self.demo_dir,
            "replay_dir": self.replay_dir,
            "package": {
                "ok": self.package_ok,
                "issues": [issue.to_dict() for issue in self.package_issues],
            },
            "transcript_path": self.transcript_path,
            "demo_manifest_path": self.demo_manifest_path,
            "downloaded_artifacts": [artifact.to_dict() for artifact in self.downloaded_artifacts],
            "replay_artifacts": [artifact.to_dict() for artifact in self.replay_artifacts],
        }


def replay_public_terminal_demo(
    *,
    release_candidate_report: Path,
    output_dir: Path,
    backend: str = "auto",
    batch_size: int = 64,
    command: str = "geno-lewm-score",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    require_native_runtime: bool = True,
    hf_token: str | None = None,
    github_token: str | None = None,
    generated_at: str | None = None,
) -> CleanMachineDemoReport:
    """Download public release artifacts and rerun the terminal demo."""
    payload = _load_ready_release_candidate(release_candidate_report)
    urls = _require_dict(payload.get("urls"), "urls")
    hub_plan = _require_dict(payload.get("hub_plan"), "hub_plan")
    _verify_hub_plan_header(hub_plan)
    _verify_candidate_plan_identity(payload, urls, hub_plan)
    model_files = _upload_files(hub_plan, "files")
    dataset_files = _upload_files(hub_plan, "dataset_files")
    demo_files = _upload_files(hub_plan, "demo_files")
    model_url = _require_str(urls.get("model"), "urls.model")
    dataset_url = _require_str(urls.get("dataset"), "urls.dataset")
    demo_url = _require_str(urls.get("demo"), "urls.demo")

    model_dir = output_dir / "model"
    dataset_dir = output_dir / "dataset"
    demo_dir = output_dir / "demo"
    replay_dir = output_dir / "replay"
    downloads: list[DownloadedArtifact] = []
    for file in model_files:
        public_url = _model_download_url(model_url, file["destination"])
        downloads.append(
            _download_checked(
                group="model",
                source_url=public_url,
                destination=model_dir / file["destination"],
                expected_sha256=file["sha256"],
                timeout_seconds=timeout_seconds,
                token=hf_token,
                report_root=output_dir,
            )
        )
    for file in dataset_files:
        public_url = _dataset_download_url(dataset_url, file["destination"])
        downloads.append(
            _download_checked(
                group="dataset",
                source_url=public_url,
                destination=dataset_dir / file["destination"],
                expected_sha256=file["sha256"],
                timeout_seconds=timeout_seconds,
                token=hf_token,
                report_root=output_dir,
            )
        )
    demo_asset_urls = _github_release_asset_urls(demo_url, timeout_seconds, token=github_token)
    for file in _unique_demo_files(demo_files):
        asset_name = Path(file["destination"]).name
        demo_public_url = demo_asset_urls.get(asset_name)
        if demo_public_url is None:
            raise InputError(
                "public demo release is missing an expected asset",
                details={"asset": asset_name, "demo_url": demo_url},
            )
        downloads.append(
            _download_checked(
                group="demo",
                source_url=demo_public_url,
                destination=demo_dir / asset_name,
                expected_sha256=file["sha256"],
                timeout_seconds=timeout_seconds,
                token=None,
                report_root=output_dir,
            )
        )

    package_report = verify_package(
        PackagePaths(
            model_dir=model_dir,
            dataset_dir=dataset_dir,
            demo_dir=demo_dir,
        )
    )
    if not package_report.ok:
        raise InputError(
            "downloaded public release package is invalid",
            details={"issues": [issue.to_dict() for issue in package_report.issues]},
        )

    vcf, fasta = _demo_inputs_from_manifest(demo_dir / DEMO_MANIFEST_NAME, demo_dir)
    transcript = run_demo_transcript(
        DemoRequest(
            model_dir=model_dir,
            vcf=vcf,
            fasta=fasta,
            output_dir=replay_dir,
            backend=backend,
            batch_size=batch_size,
            command=command,
            require_native_runtime=require_native_runtime,
        )
    )
    replay_artifacts = _collect_replay_artifacts(replay_dir, report_root=output_dir)
    _verify_replayed_demo_manifest(
        replay_dir / DEMO_MANIFEST_NAME,
        model_dir=model_dir,
        vcf=vcf,
        fasta=fasta,
        report_root=output_dir,
    )
    release_candidate_reference = release_candidate_report.name
    report = CleanMachineDemoReport(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        generated_at=generated_at or _utc_now(),
        release_candidate_report=release_candidate_reference,
        release_candidate_report_identity=_inspect_replay_artifact(
            "release candidate report",
            release_candidate_report,
            reported_path=release_candidate_reference,
        ),
        model_dir=_report_relative_path(model_dir, output_dir),
        dataset_dir=_report_relative_path(dataset_dir, output_dir),
        demo_dir=_report_relative_path(demo_dir, output_dir),
        replay_dir=_report_relative_path(replay_dir, output_dir),
        package_ok=package_report.ok,
        package_issues=package_report.issues,
        transcript_path=_report_relative_path(transcript, output_dir),
        demo_manifest_path=_report_relative_path(replay_dir / DEMO_MANIFEST_NAME, output_dir),
        downloaded_artifacts=tuple(downloads),
        replay_artifacts=replay_artifacts,
    )
    _write_json(output_dir / DEFAULT_REPORT_NAME, report.to_dict())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = replay_public_terminal_demo(
            release_candidate_report=args.release_candidate_report,
            output_dir=args.output_dir,
            backend=args.backend,
            batch_size=args.batch_size,
            command=args.command,
            timeout_seconds=args.timeout,
            require_native_runtime=not args.no_require_native_runtime,
            hf_token=_env_token("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"),
            github_token=_env_token("GH_TOKEN", "GITHUB_TOKEN"),
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(f"{report.transcript_path}\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download public release artifacts and replay the terminal demo.",
    )
    parser.add_argument("--release-candidate-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--command", default="geno-lewm-score")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--no-require-native-runtime",
        action="store_true",
        help="Skip native-runtime dependency checks for local tests only.",
    )
    return parser


def _load_ready_release_candidate(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(
            "release candidate report is missing", details={"path": str(path)}
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "release candidate report JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("release candidate report must be a JSON object")
    if payload.get("ready") is not True:
        raise InputError("clean-machine demo replay requires a ready release candidate")
    _require_report_header(
        payload,
        label="release candidate report",
        schema_version=RELEASE_CANDIDATE_SCHEMA_VERSION,
        generated_by=RELEASE_CANDIDATE_GENERATED_BY,
    )
    _verify_candidate_blockers(payload)
    _verify_candidate_readiness(payload)
    _verify_candidate_public_checks(payload)
    return payload


def _verify_candidate_blockers(payload: dict[str, Any]) -> None:
    blockers = payload.get("blockers")
    if not isinstance(blockers, list):
        raise InputError("release candidate report blockers must be a list")
    if blockers:
        raise InputError(
            "clean-machine demo replay requires an unblocked release candidate",
            details={"blockers": blockers},
        )


def _verify_candidate_readiness(payload: dict[str, Any]) -> None:
    raw_readiness = payload.get("readiness")
    if not isinstance(raw_readiness, list):
        raise InputError("release candidate report readiness must be a list")
    readiness_by_code: dict[str, dict[str, Any]] = {}
    for item in raw_readiness:
        if not isinstance(item, dict):
            raise InputError("release candidate readiness entries must be objects")
        code = item.get("code")
        if isinstance(code, str):
            readiness_by_code[code] = item
    missing = tuple(code for code in REQUIRED_READINESS_CODES if code not in readiness_by_code)
    if missing:
        raise InputError(
            "release candidate report is missing required readiness rows",
            details={"missing": list(missing)},
        )
    failed = tuple(
        code for code in REQUIRED_READINESS_CODES if readiness_by_code[code].get("ok") is not True
    )
    if failed:
        raise InputError(
            "clean-machine demo replay requires passing release-candidate readiness",
            details={"failed": list(failed)},
        )


def _verify_candidate_public_checks(payload: dict[str, Any]) -> None:
    urls = _require_dict(payload.get("urls"), "urls")
    hub_plan = _require_dict(payload.get("hub_plan"), "hub_plan")
    paper_required = hub_plan.get("paper_file") is not None or urls.get("paper") not in {None, ""}
    expected_links = {"model", "dataset", "demo"}
    expected_artifacts = {"model", "dataset", "demo"}
    if paper_required:
        expected_links.add("paper")
        expected_artifacts.add("paper")
    _verify_public_link_checks(payload.get("public_links"), expected_names=expected_links)
    _verify_public_artifact_checks(
        payload.get("public_artifacts"),
        expected_names=expected_artifacts,
    )


def _verify_public_link_checks(value: object, *, expected_names: set[str]) -> None:
    section = _require_dict(value, "public_links")
    if section.get("required") is not True:
        raise InputError("clean-machine demo replay requires public link checks")
    checks = _named_check_objects(section.get("checks"), "public_links.checks")
    missing = tuple(sorted(expected_names - set(checks)))
    failed = tuple(sorted(name for name, check in checks.items() if check.get("ok") is not True))
    if missing or failed:
        raise InputError(
            "clean-machine demo replay requires passing public link checks",
            details={"missing": list(missing), "failed": list(failed)},
        )


def _verify_public_artifact_checks(value: object, *, expected_names: set[str]) -> None:
    section = _require_dict(value, "public_artifacts")
    if section.get("required") is not True:
        raise InputError("clean-machine demo replay requires public artifact checks")
    checks = _named_check_objects(section.get("checks"), "public_artifacts.checks")
    missing = tuple(sorted(expected_names - set(checks)))
    failed: list[str] = []
    incomplete: list[str] = []
    for name, check in checks.items():
        if check.get("ok") is not True:
            failed.append(name)
            continue
        expected_count = check.get("expected_count")
        verified_count = check.get("verified_count")
        if (
            not isinstance(expected_count, int)
            or isinstance(expected_count, bool)
            or expected_count <= 0
            or verified_count != expected_count
        ):
            incomplete.append(name)
            continue
        for key in ("missing", "hash_mismatches", "size_mismatches", "unexpected"):
            if check.get(key) not in ([], ()):
                failed.append(name)
                break
    if missing or failed or incomplete:
        raise InputError(
            "clean-machine demo replay requires passing public artifact checks",
            details={
                "missing": list(missing),
                "failed": sorted(set(failed)),
                "incomplete": sorted(set(incomplete)),
            },
        )


def _named_check_objects(value: object, field: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise InputError("release candidate field must be a list", details={"field": field})
    checks: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, dict):
            raise InputError(
                "release candidate check entries must be objects", details={"field": field}
            )
        name = item.get("name")
        if isinstance(name, str) and name:
            checks[name] = item
    return checks


def _require_report_header(
    payload: dict[str, Any],
    *,
    label: str,
    schema_version: str,
    generated_by: str,
) -> None:
    if payload.get("schema_version") != schema_version:
        raise InputError(
            f"{label} schema_version is invalid",
            details={"expected": schema_version, "observed": payload.get("schema_version")},
        )
    if payload.get("generated_by") != generated_by:
        raise InputError(
            f"{label} generated_by is invalid",
            details={"expected": generated_by, "observed": payload.get("generated_by")},
        )


def _verify_hub_plan_header(hub_plan: dict[str, Any]) -> None:
    _require_report_header(
        hub_plan,
        label="release candidate Hub plan",
        schema_version=HUB_RELEASE_SCHEMA_VERSION,
        generated_by=HUB_RELEASE_GENERATED_BY,
    )


def _verify_candidate_plan_identity(
    candidate: dict[str, Any],
    urls: dict[str, Any],
    hub_plan: dict[str, Any],
) -> None:
    for key in ("release_id", "model_id", "commit_sha", "repo_id"):
        if candidate.get(key) != hub_plan.get(key):
            raise InputError(
                "release candidate identity does not match embedded Hub plan",
                details={
                    "field": key,
                    "candidate": candidate.get(key),
                    "hub_plan": hub_plan.get(key),
                },
            )
    repo_id = _require_str(hub_plan.get("repo_id"), "hub_plan.repo_id")
    _require_equal(
        observed=urls.get("model"),
        expected=f"https://huggingface.co/{repo_id}",
        field="urls.model",
    )
    for url_key, plan_key in (
        ("dataset", "dataset_url"),
        ("demo", "demo_url"),
        ("paper", "paper_url"),
    ):
        expected = hub_plan.get(plan_key)
        observed = urls.get(url_key)
        if expected is None:
            if observed not in {None, ""}:
                raise InputError(
                    "release candidate URL does not match embedded Hub plan",
                    details={
                        "field": f"urls.{url_key}",
                        "expected": expected,
                        "observed": observed,
                    },
                )
            continue
        _require_equal(observed=observed, expected=expected, field=f"urls.{url_key}")


def _require_equal(*, observed: object, expected: object, field: str) -> None:
    if observed != expected:
        raise InputError(
            "release candidate URL does not match embedded Hub plan",
            details={"field": field, "expected": expected, "observed": observed},
        )


def _upload_files(payload: dict[str, Any], key: str) -> tuple[dict[str, Any], ...]:
    raw_files = payload.get(key)
    if not isinstance(raw_files, list) or not raw_files:
        raise InputError("release candidate Hub plan is missing upload files", details={"key": key})
    files: list[dict[str, Any]] = []
    seen_destinations: dict[str, int] = {}
    seen_demo_assets: dict[str, int] = {}
    for index, item in enumerate(raw_files):
        if not isinstance(item, dict):
            raise InputError("upload file entries must be objects", details={"key": key})
        destination = item.get("destination")
        if not isinstance(destination, str) or not destination.strip():
            raise InputError(
                "upload file entry is missing a required string",
                details={"key": key, "index": index, "field": "destination"},
            )
        _validate_upload_destination(destination, key=key, index=index)
        previous_index = seen_destinations.get(destination)
        if previous_index is not None:
            raise InputError(
                "release candidate Hub plan contains duplicate upload destinations",
                details={
                    "key": key,
                    "destination": destination,
                    "first_index": previous_index,
                    "duplicate_index": index,
                },
            )
        seen_destinations[destination] = index
        if key == "demo_files":
            asset_name = Path(destination).name
            previous_asset_index = seen_demo_assets.get(asset_name)
            if previous_asset_index is not None:
                raise InputError(
                    "release candidate Hub plan contains duplicate demo asset names",
                    details={
                        "asset": asset_name,
                        "first_index": previous_asset_index,
                        "duplicate_index": index,
                    },
                )
            seen_demo_assets[asset_name] = index
        expected_hash = item.get("sha256")
        if not isinstance(expected_hash, str) or not expected_hash:
            raise InputError(
                "upload file entry is missing a required string",
                details={"key": key, "index": index, "field": "sha256"},
            )
        if not looks_like_sha256(expected_hash):
            raise InputError(
                "upload file entry sha256 must be sha256:<64hex>",
                details={"key": key, "index": index, "sha256": expected_hash},
            )
        files.append(item)
    return tuple(files)


def _validate_upload_destination(destination: str, *, key: str, index: int) -> None:
    path = Path(destination)
    if "://" in destination or path.is_absolute() or ".." in path.parts or not path.parts:
        raise InputError(
            "upload file destinations must be package-relative",
            details={"key": key, "index": index, "destination": destination},
        )


def _unique_demo_files(files: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    by_name: dict[str, dict[str, Any]] = {}
    for file in files:
        by_name.setdefault(Path(file["destination"]).name, file)
    return tuple(by_name[name] for name in sorted(by_name))


def _download_checked(
    *,
    group: str,
    source_url: str,
    destination: Path,
    expected_sha256: str,
    timeout_seconds: float,
    token: str | None,
    report_root: Path,
) -> DownloadedArtifact:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = urllib_request.Request(
            source_url,
            method="GET",
            headers=_request_headers(token=token),
        )
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            destination.write_bytes(response.read())
    except (urllib_error.HTTPError, urllib_error.URLError, OSError) as exc:
        raise ResourceError(
            "failed to download public release artifact",
            details={"url": source_url, "path": str(destination)},
        ) from exc
    observed = sha256_file(destination)
    if observed != expected_sha256:
        raise InputError(
            "downloaded public artifact hash mismatch",
            details={
                "url": source_url,
                "path": str(destination),
                "expected": expected_sha256,
                "observed": observed,
            },
        )
    return DownloadedArtifact(
        group=group,
        source_url=source_url,
        path=_report_relative_path(destination, report_root),
        sha256=observed,
        size_bytes=destination.stat().st_size,
    )


def _collect_replay_artifacts(
    replay_dir: Path,
    *,
    report_root: Path,
) -> tuple[ReplayArtifact, ...]:
    expected = (
        ("terminal transcript", replay_dir / "terminal-demo-transcript.md"),
        ("terminal demo manifest", replay_dir / DEMO_MANIFEST_NAME),
        ("scores", replay_dir / "scores.jsonl"),
        ("receipts", replay_dir / "receipts.jsonl"),
        ("runtime preflight report", replay_dir / "runtime_preflight_report.json"),
        ("batch receipt report", replay_dir / "batch_receipt_report.json"),
    )
    return tuple(
        _inspect_replay_artifact(
            label, path, reported_path=_report_relative_path(path, report_root)
        )
        for label, path in expected
    )


def _verify_replayed_demo_manifest(
    manifest_path: Path,
    *,
    model_dir: Path,
    vcf: Path,
    fasta: Path,
    report_root: Path,
) -> None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(
            "replayed terminal demo manifest is missing",
            details={"path": str(manifest_path)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "replayed terminal demo manifest JSON is invalid",
            details={"path": str(manifest_path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("replayed terminal demo manifest must be a JSON object")
    _require_report_header(
        payload,
        label="replayed terminal demo manifest",
        schema_version=DEMO_MANIFEST_SCHEMA_VERSION,
        generated_by=DEMO_MANIFEST_GENERATED_BY,
    )
    if payload.get("status") != "passed":
        raise InputError(
            "replayed terminal demo manifest must have status=passed",
            details={"observed": payload.get("status")},
        )
    model = _require_dict(payload.get("model"), "model")
    downloaded_manifest_path = model_dir / "manifest.json"
    downloaded_manifest = load_manifest(downloaded_manifest_path)
    if model.get("model_id") != downloaded_manifest.model_id():
        raise InputError(
            "replayed terminal demo manifest model_id does not match downloaded model",
            details={
                "expected": downloaded_manifest.model_id(),
                "observed": model.get("model_id"),
            },
        )
    inputs = _require_dict(payload.get("inputs"), "inputs")
    _verify_manifest_file_identity(
        _require_dict(inputs.get("model_manifest"), "inputs.model_manifest"),
        expected_path=_report_relative_path(downloaded_manifest_path, report_root),
        actual_path=downloaded_manifest_path,
        label="downloaded model manifest",
    )
    _verify_manifest_file_identity(
        _require_dict(inputs.get("vcf"), "inputs.vcf"),
        expected_path=_report_relative_path(vcf, report_root),
        actual_path=vcf,
        label="replayed demo VCF",
    )
    _verify_manifest_file_identity(
        _require_dict(inputs.get("fasta"), "inputs.fasta"),
        expected_path=_report_relative_path(fasta, report_root),
        actual_path=fasta,
        label="replayed demo FASTA",
    )
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise InputError("replayed terminal demo manifest artifacts must be a non-empty list")
    artifacts_by_label = {
        item.get("label"): item for item in raw_artifacts if isinstance(item, dict)
    }
    for label, path in (
        ("terminal transcript", report_root / "replay" / "terminal-demo-transcript.md"),
        ("scores", report_root / "replay" / "scores.jsonl"),
        ("receipts", report_root / "replay" / "receipts.jsonl"),
        ("runtime preflight report", report_root / "replay" / "runtime_preflight_report.json"),
        ("batch receipt report", report_root / "replay" / "batch_receipt_report.json"),
    ):
        raw_identity = artifacts_by_label.get(label)
        if not isinstance(raw_identity, dict):
            raise InputError(
                "replayed terminal demo manifest is missing artifact identity",
                details={"label": label},
            )
        _verify_manifest_file_identity(
            raw_identity,
            expected_path=_report_relative_path(path, report_root),
            actual_path=path,
            label=label,
        )
    _verify_replayed_runtime_preflight_summary(
        payload.get("runtime_preflight"),
        report_root / "replay" / "runtime_preflight_report.json",
    )
    _verify_replayed_score_receipt_summary(
        payload.get("score_receipt_batch"),
        report_root / "replay" / "batch_receipt_report.json",
    )


def _verify_replayed_runtime_preflight_summary(raw: object, preflight_path: Path) -> None:
    if not isinstance(raw, dict):
        raise InputError("replayed terminal demo manifest must include runtime_preflight")
    try:
        payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(
            "replayed runtime preflight report is missing",
            details={"path": str(preflight_path)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "replayed runtime preflight report JSON is invalid",
            details={"path": str(preflight_path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("replayed runtime preflight report must be a JSON object")
    expected = _runtime_preflight_manifest_summary(payload)
    if expected is None:
        raise InputError("replayed runtime preflight report is missing summary fields")
    if raw != expected:
        raise InputError("replayed terminal demo manifest runtime_preflight summary mismatch")


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


def _verify_replayed_score_receipt_summary(raw: object, batch_report_path: Path) -> None:
    if not isinstance(raw, dict):
        raise InputError("replayed terminal demo manifest must include score_receipt_batch")
    try:
        payload = json.loads(batch_report_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(
            "replayed batch receipt report is missing",
            details={"path": str(batch_report_path)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "replayed batch receipt report JSON is invalid",
            details={"path": str(batch_report_path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("replayed batch receipt report must be a JSON object")
    expected = _score_receipt_manifest_summary(payload)
    if expected is None:
        raise InputError("replayed batch receipt report is missing summary fields")
    if raw != expected:
        raise InputError("replayed terminal demo manifest score_receipt_batch summary mismatch")


def _score_receipt_manifest_summary(payload: dict[str, Any]) -> dict[str, Any] | None:
    required = (
        "records",
        "model_id",
        "calibration_hash",
        "receipt_schema_version",
        "receipt_stream",
        "checked_score_fields",
        "runtime",
    )
    if any(field not in payload for field in required):
        return None
    return {field: payload[field] for field in required}


def _verify_manifest_file_identity(
    raw: dict[str, Any],
    *,
    expected_path: str,
    actual_path: Path,
    label: str,
) -> None:
    if raw.get("path") != expected_path:
        raise InputError(
            "replayed terminal demo manifest artifact path mismatch",
            details={"label": label, "expected": expected_path, "observed": raw.get("path")},
        )
    expected_hash = raw.get("sha256")
    if not isinstance(expected_hash, str) or not looks_like_sha256(expected_hash):
        raise InputError(
            "replayed terminal demo manifest artifact hash is invalid",
            details={"label": label, "observed": expected_hash},
        )
    observed_hash = sha256_file(actual_path)
    if observed_hash != expected_hash:
        raise InputError(
            "replayed terminal demo manifest artifact hash mismatch",
            details={
                "label": label,
                "path": expected_path,
                "expected": expected_hash,
                "observed": observed_hash,
            },
        )
    expected_size = _require_positive_int(raw.get("size_bytes"), f"{label}.size_bytes")
    observed_size = actual_path.stat().st_size
    if observed_size != expected_size:
        raise InputError(
            "replayed terminal demo manifest artifact size mismatch",
            details={
                "label": label,
                "path": expected_path,
                "expected": expected_size,
                "observed": observed_size,
            },
        )


def _inspect_replay_artifact(
    label: str,
    path: Path,
    *,
    reported_path: str | None = None,
) -> ReplayArtifact:
    if not path.is_file():
        raise InputError(
            "clean-machine replay artifact is missing",
            details={"label": label, "path": str(path)},
        )
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise InputError(
            "clean-machine replay artifact is empty",
            details={"label": label, "path": str(path)},
        )
    return ReplayArtifact(
        label=label,
        path=reported_path or str(path),
        sha256=sha256_file(path),
        size_bytes=size_bytes,
    )


def _report_relative_path(path: Path, report_root: Path) -> str:
    try:
        return path.resolve().relative_to(report_root.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError) as exc:
        raise InputError(
            "clean-machine report artifact path is outside the output directory",
            details={"path": str(path), "output_dir": str(report_root)},
        ) from exc


def _model_download_url(model_url: str, destination: str) -> str:
    parsed = urlparse(model_url)
    parts = tuple(part for part in parsed.path.strip("/").split("/") if part)
    if parsed.netloc != "huggingface.co" or len(parts) < 2:
        raise InputError("model URL must be a Hugging Face model URL", details={"url": model_url})
    return (
        f"https://huggingface.co/{parts[0]}/{parts[1]}/resolve/main/{quote(destination, safe='/')}"
    )


def _dataset_download_url(dataset_url: str, destination: str) -> str:
    parsed = urlparse(dataset_url)
    parts = tuple(part for part in parsed.path.strip("/").split("/") if part)
    if parsed.netloc != "huggingface.co" or len(parts) < 3 or parts[0] != "datasets":
        raise InputError(
            "dataset URL must be a Hugging Face dataset URL",
            details={"url": dataset_url},
        )
    return (
        f"https://huggingface.co/datasets/{parts[1]}/{parts[2]}"
        f"/resolve/main/{quote(destination, safe='/')}"
    )


def _github_release_asset_urls(
    demo_url: str,
    timeout_seconds: float,
    *,
    token: str | None,
) -> dict[str, str]:
    parsed = urlparse(demo_url)
    parts = tuple(part for part in parsed.path.strip("/").split("/") if part)
    if parsed.netloc != "github.com" or len(parts) < 5 or parts[2:4] != ("releases", "tag"):
        raise InputError(
            "demo URL must be a GitHub release tag URL",
            details={"demo_url": demo_url},
        )
    owner, repo, _releases, _tag, tag = parts[:5]
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    try:
        request = urllib_request.Request(
            api_url,
            method="GET",
            headers=_request_headers(token=token),
        )
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (json.JSONDecodeError, urllib_error.HTTPError, urllib_error.URLError, OSError) as exc:
        raise ResourceError(
            "failed to read GitHub release asset listing",
            details={"demo_url": demo_url, "api_url": api_url},
        ) from exc
    assets = payload.get("assets") if isinstance(payload, dict) else None
    if not isinstance(assets, list):
        raise InputError("GitHub release asset listing is invalid", details={"api_url": api_url})
    result: dict[str, str] = {}
    for item in assets:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        download_url = item.get("browser_download_url")
        if isinstance(name, str) and isinstance(download_url, str):
            result[name] = download_url
    return result


def _request_headers(*, token: str | None) -> dict[str, str]:
    headers = {"User-Agent": "GenoLeWM-clean-machine-demo/1.0"}
    normalized = _normalize_token(token)
    if normalized is not None:
        headers["Authorization"] = f"Bearer {normalized}"
    return headers


def _env_token(*names: str) -> str | None:
    for name in names:
        token = _normalize_token(os.environ.get(name))
        if token is not None:
            return token
    return None


def _normalize_token(token: str | None) -> str | None:
    if token is None:
        return None
    stripped = token.strip()
    return stripped or None


def _demo_inputs_from_manifest(manifest_path: Path, demo_dir: Path) -> tuple[Path, Path]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(
            "downloaded demo manifest is missing",
            details={"path": str(manifest_path)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "downloaded demo manifest JSON is invalid",
            details={"path": str(manifest_path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    inputs = _require_dict(payload.get("inputs"), "inputs")
    vcf = _downloaded_input_path(inputs, "vcf", demo_dir)
    fasta = _downloaded_input_path(inputs, "fasta", demo_dir)
    return vcf, fasta


def _downloaded_input_path(inputs: dict[str, Any], key: str, demo_dir: Path) -> Path:
    identity = _require_dict(inputs.get(key), f"inputs.{key}")
    raw_path = _require_str(identity.get("path"), f"inputs.{key}.path")
    _validate_demo_input_path(raw_path, key=key)
    path = demo_dir / Path(raw_path).name
    if not path.is_file():
        raise InputError(
            "downloaded demo input file is missing",
            details={"input": key, "path": str(path)},
        )
    expected_hash = _require_str(identity.get("sha256"), f"inputs.{key}.sha256")
    if not looks_like_sha256(expected_hash):
        raise InputError(
            "downloaded demo input hash must be sha256:<64hex>",
            details={"input": key, "sha256": expected_hash},
        )
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise InputError(
            "downloaded demo input hash mismatch",
            details={
                "input": key,
                "path": str(path),
                "expected": expected_hash,
                "observed": observed_hash,
            },
        )
    expected_size = _require_positive_int(identity.get("size_bytes"), f"inputs.{key}.size_bytes")
    observed_size = path.stat().st_size
    if observed_size != expected_size:
        raise InputError(
            "downloaded demo input size mismatch",
            details={
                "input": key,
                "path": str(path),
                "expected": expected_size,
                "observed": observed_size,
            },
        )
    return path


def _validate_demo_input_path(raw_path: str, *, key: str) -> None:
    path = Path(raw_path)
    if "://" in raw_path or path.is_absolute() or ".." in path.parts or not path.name:
        raise InputError(
            "downloaded demo input paths must be package-relative",
            details={"input": key, "path": raw_path},
        )


def _require_dict(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputError("release candidate field must be an object", details={"field": field})
    return value


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputError("release candidate field must be a string", details={"field": field})
    return value


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(
            "release candidate field must be a positive integer", details={"field": field}
        )
    return value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
