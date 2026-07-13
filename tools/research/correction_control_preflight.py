# SPDX-License-Identifier: Apache-2.0
"""Validate the immutable correction-control smoke-job contract."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import yaml

from geno_lewm.config import load_config
from geno_lewm.errors import GenoLeWMError
from geno_lewm.provenance import sha256_file

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.research.correction_control_preflight"

EXPECTED_CONFIG_PATH: Final = Path(
    "configs/correction_control/train-carbon-500m-snv-l2-smoke-v1.yaml"
)
EXPECTED_SNAPSHOT_PATH: Final = Path(
    "configs/correction_control/dataset-snapshot-snv-l2-smoke-v1.json"
)
EXPECTED_RUN_ID: Final = "correction-control-l2-p1-smoke-v1"
EXPECTED_SNAPSHOT_ID: Final = "geno-lewm-data-correction-control-l2-p1-proof-v1"
EXPECTED_SEED: Final = 104729

EXPECTED_CARBON_MODEL_DIR: Final = "/carbon"
EXPECTED_CARBON_REVISION: Final = "5d31d59b3c845b288a13aedb1358934196852eec"
EXPECTED_CARBON_CONFIG: Final = "eukaryote_generator_10B_subset"
EXPECTED_CARBON_SOURCE: Final = "eukaryotic_genes"
EXPECTED_CORPUS_ID: Final = "HuggingFaceBio/carbon-pretraining-corpus"
EXPECTED_CORPUS_REVISION: Final = "cb4c13a78102933b3a6ac65734d326f7b431d9b7"
EXPECTED_CONTAINER_IMAGE: Final = (
    "ghcr.io/astral-sh/uv@sha256:35b0aa516fbcf6f18624919cfc38fa02ab3458e0ffcd3c03e932051b37f315db"
)

EXPECTED_CLINVAR_URL: Final = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2026/clinvar_20260415.vcf.gz"
)
EXPECTED_GNOMAD_URL: Final = (
    "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/vcf/exomes/"
    "gnomad.exomes.v4.1.sites.chr22.vcf.bgz?generation=1713312296186865"
)
EXPECTED_SOURCE_INTEGRITY: Final[dict[str, object]] = {
    "clinvar_md5": "e63b5c3a046010c098cc70e81bebaa8d",
    "gnomad_generation": "1713312296186865",
    "gnomad_md5": "dcf191563e69054a71bd4dc77862799a",
    "gnomad_size_bytes": 5_060_347_554,
}

EXPECTED_STEPS: Final = 50
EXPECTED_MAX_WINDOWS: Final = 512
EXPECTED_CLINVAR_LINES: Final = 60_000
EXPECTED_GNOMAD_LINES: Final = 60_000
EXPECTED_TUPLE_THROUGHPUT_SAMPLES: Final = 400
EXPECTED_WINDOW_BP: Final = 4096
EXPECTED_HOLDOUT_CHROM: Final = 22
_COMMIT_SHA_RE: Final = re.compile(r"[0-9a-f]{40}")
_MISSING: Final = object()


@dataclass(frozen=True, slots=True)
class CorrectionControlRequest:
    """Inputs that define one correction-control smoke-job launch."""

    repo_root: Path
    config_path: Path
    snapshot_path: Path
    expected_commit_sha: str
    run_name: str
    run_attempt: int
    steps: int
    max_windows: int
    clinvar_lines: int
    gnomad_lines: int
    tuple_throughput_samples: int
    window_bp: int
    holdout_chrom: int
    carbon_model_dir: str
    carbon_config: str
    carbon_source: str
    corpus_revision: str
    container_image: str
    clinvar_url: str
    gnomad_url: str


@dataclass(frozen=True, slots=True)
class CorrectionControlIssue:
    """One mismatch against the immutable correction-control contract."""

    code: str
    path: str
    message: str
    expected: object | None = None
    observed: object | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "expected": _json_value(self.expected),
            "observed": _json_value(self.observed),
        }


@dataclass(frozen=True, slots=True)
class CorrectionControlPreflightReport:
    """Machine-readable result of validating a correction-control launch."""

    schema_version: str
    generated_by: str
    generated_at: str
    ok: bool
    repository: dict[str, object]
    job: dict[str, object]
    config: dict[str, object]
    snapshot: dict[str, object]
    issues: tuple[CorrectionControlIssue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "ok": self.ok,
            "repository": self.repository,
            "job": self.job,
            "config": self.config,
            "snapshot": self.snapshot,
            "issues": [issue.to_dict() for issue in self.issues],
            "claim_boundary": (
                "This preflight validates launch identity and immutable inputs only. It does not "
                "establish deterministic replay, model quality, benchmark performance, or "
                "clinical validity."
            ),
        }


def build_correction_control_preflight_report(
    request: CorrectionControlRequest,
    *,
    generated_at: str | None = None,
) -> CorrectionControlPreflightReport:
    """Validate a job request without network, accelerator, or output-directory access."""
    issues: list[CorrectionControlIssue] = []
    repo_root = request.repo_root.resolve()
    config_path = _resolve_repo_path(repo_root, request.config_path)
    snapshot_path = _resolve_repo_path(repo_root, request.snapshot_path)

    expected_sha_valid = _COMMIT_SHA_RE.fullmatch(request.expected_commit_sha) is not None
    if not expected_sha_valid:
        _issue(
            issues,
            "request.commit_sha_format",
            "expected_commit_sha",
            "expected commit must be a full lowercase Git SHA",
            "40 lowercase hexadecimal characters",
            request.expected_commit_sha,
        )
    observed_root, observed_head, git_error = _git_identity(repo_root)
    dirty_paths: tuple[str, ...] = ()
    worktree_error: str | None = None
    if git_error is not None:
        _issue(
            issues,
            "repository.git_unavailable",
            "repo_root",
            "repository identity could not be resolved",
            "a Git worktree with a committed HEAD",
            git_error,
        )
    else:
        if observed_root != repo_root:
            _issue(
                issues,
                "repository.root_mismatch",
                "repo_root",
                "repo_root must be the Git worktree root",
                str(repo_root),
                None if observed_root is None else str(observed_root),
            )
        if observed_head != request.expected_commit_sha:
            _issue(
                issues,
                "repository.head_mismatch",
                "expected_commit_sha",
                "checked-out HEAD does not match the requested correction commit",
                request.expected_commit_sha,
                observed_head,
            )
        dirty_paths, worktree_error = _git_dirty_paths(repo_root)
        if worktree_error is not None:
            _issue(
                issues,
                "repository.status_unavailable",
                "repo_root",
                "repository worktree status could not be resolved",
                "a readable clean Git worktree",
                worktree_error,
            )
        elif dirty_paths:
            _issue(
                issues,
                "repository.worktree_dirty",
                "repo_root",
                "correction-control execution requires a clean exact-commit worktree",
                [],
                list(dirty_paths),
            )

    attempt_valid = (
        isinstance(request.run_attempt, int)
        and not isinstance(request.run_attempt, bool)
        and request.run_attempt > 0
    )
    if not attempt_valid:
        _issue(
            issues,
            "request.run_attempt_invalid",
            "run_attempt",
            "run attempt must be a positive integer",
            "positive integer",
            request.run_attempt,
        )
    expected_run_name = (
        f"geno-lewm-l2-p1-smoke-{request.expected_commit_sha[:12]}-50-r{request.run_attempt}"
        if expected_sha_valid and attempt_valid
        else None
    )
    if request.run_name != expected_run_name:
        _issue(
            issues,
            "request.run_name_mismatch",
            "run_name",
            "run name must identify the correction-control lineage and exact short commit",
            expected_run_name,
            request.run_name,
        )

    _validate_request_values(request, issues)
    _expect_path(
        issues,
        code="request.config_path_mismatch",
        label="config_path",
        observed=config_path,
        expected=(repo_root / EXPECTED_CONFIG_PATH).resolve(),
    )
    _expect_path(
        issues,
        code="request.snapshot_path_mismatch",
        label="snapshot_path",
        observed=snapshot_path,
        expected=(repo_root / EXPECTED_SNAPSHOT_PATH).resolve(),
    )

    config_payload = _load_yaml_object(config_path, issues)
    if config_payload is not None:
        _validate_config(config_payload, config_path, issues)
    snapshot_payload = _load_json_object(snapshot_path, issues)
    if snapshot_payload is not None:
        _validate_snapshot(snapshot_payload, snapshot_path, issues)

    repository: dict[str, object] = {
        "root": ".",
        "expected_commit_sha": request.expected_commit_sha,
        "observed_commit_sha": observed_head,
        "observed_git_root": "." if observed_root == repo_root else None,
        "worktree_clean": worktree_error is None and not dirty_paths,
        "dirty_paths": list(dirty_paths),
    }
    job = {
        "run_name": request.run_name,
        "run_attempt": request.run_attempt,
        "steps": request.steps,
        "max_windows": request.max_windows,
        "clinvar_lines": request.clinvar_lines,
        "gnomad_lines": request.gnomad_lines,
        "tuple_throughput_samples": request.tuple_throughput_samples,
        "window_bp": request.window_bp,
        "holdout_chrom": request.holdout_chrom,
        "carbon_model_dir": request.carbon_model_dir,
        "carbon_config": request.carbon_config,
        "carbon_source": request.carbon_source,
        "corpus_revision": request.corpus_revision,
        "container_image": request.container_image,
        "sources": {
            "clinvar_url": request.clinvar_url,
            "gnomad_url": request.gnomad_url,
        },
    }
    config = _file_summary(
        config_path,
        config_payload,
        display_path=_display_repo_path(repo_root, config_path),
    )
    snapshot = _file_summary(
        snapshot_path,
        snapshot_payload,
        display_path=_display_repo_path(repo_root, snapshot_path),
    )
    return CorrectionControlPreflightReport(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        generated_at=generated_at or _utc_now(),
        ok=not issues,
        repository=repository,
        job=job,
        config=config,
        snapshot=snapshot,
        issues=tuple(issues),
    )


def write_correction_control_preflight_report(
    request: CorrectionControlRequest,
    output: Path,
    *,
    generated_at: str | None = None,
) -> CorrectionControlPreflightReport:
    """Validate the request and write its report to an explicitly requested path."""
    report = build_correction_control_preflight_report(request, generated_at=generated_at)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the correction-control preflight CLI."""
    args = _parser().parse_args(argv)
    request = CorrectionControlRequest(
        repo_root=args.repo_root,
        config_path=args.config,
        snapshot_path=args.snapshot,
        expected_commit_sha=args.expected_commit_sha,
        run_name=args.run_name,
        run_attempt=args.run_attempt,
        steps=args.steps,
        max_windows=args.max_windows,
        clinvar_lines=args.clinvar_lines,
        gnomad_lines=args.gnomad_lines,
        tuple_throughput_samples=args.tuple_throughput_samples,
        window_bp=args.window_bp,
        holdout_chrom=args.holdout_chrom,
        carbon_model_dir=args.carbon_model_dir,
        carbon_config=args.carbon_config,
        carbon_source=args.carbon_source,
        corpus_revision=args.corpus_revision,
        container_image=args.container_image,
        clinvar_url=args.clinvar_url,
        gnomad_url=args.gnomad_url,
    )
    report = (
        build_correction_control_preflight_report(request)
        if args.output is None
        else write_correction_control_preflight_report(request, args.output)
    )
    sys.stdout.write(json.dumps(report.to_dict(), sort_keys=True) + "\n")
    return 0 if report.ok else 2


def _validate_request_values(
    request: CorrectionControlRequest,
    issues: list[CorrectionControlIssue],
) -> None:
    expected: tuple[tuple[str, object, object], ...] = (
        ("steps", request.steps, EXPECTED_STEPS),
        ("max_windows", request.max_windows, EXPECTED_MAX_WINDOWS),
        ("clinvar_lines", request.clinvar_lines, EXPECTED_CLINVAR_LINES),
        ("gnomad_lines", request.gnomad_lines, EXPECTED_GNOMAD_LINES),
        (
            "tuple_throughput_samples",
            request.tuple_throughput_samples,
            EXPECTED_TUPLE_THROUGHPUT_SAMPLES,
        ),
        ("window_bp", request.window_bp, EXPECTED_WINDOW_BP),
        ("holdout_chrom", request.holdout_chrom, EXPECTED_HOLDOUT_CHROM),
        ("carbon_model_dir", request.carbon_model_dir, EXPECTED_CARBON_MODEL_DIR),
        ("carbon_config", request.carbon_config, EXPECTED_CARBON_CONFIG),
        ("carbon_source", request.carbon_source, EXPECTED_CARBON_SOURCE),
        ("corpus_revision", request.corpus_revision, EXPECTED_CORPUS_REVISION),
        ("container_image", request.container_image, EXPECTED_CONTAINER_IMAGE),
        ("clinvar_url", request.clinvar_url, EXPECTED_CLINVAR_URL),
        ("gnomad_url", request.gnomad_url, EXPECTED_GNOMAD_URL),
    )
    for label, observed, wanted in expected:
        _expect_equal(
            issues,
            code=f"request.{label}_mismatch",
            label=label,
            observed=observed,
            expected=wanted,
        )


def _validate_config(
    payload: dict[str, Any],
    path: Path,
    issues: list[CorrectionControlIssue],
) -> None:
    try:
        load_config(payload)
    except GenoLeWMError as exc:
        _issue(
            issues,
            "config.schema_invalid",
            str(path),
            "training config failed the typed GenoLeWM schema",
            "a valid schema-1.1 config",
            exc.to_dict(),
        )

    expected: tuple[tuple[tuple[str, ...], object], ...] = (
        (("run_id",), EXPECTED_RUN_ID),
        (("seed",), EXPECTED_SEED),
        (("phase",), "phase1"),
        (("deterministic",), True),
        (("schema_version",), "1.1.0"),
        (("encoder", "model_id"), EXPECTED_CARBON_MODEL_DIR),
        (("encoder", "revision"), EXPECTED_CARBON_REVISION),
        (("encoder", "dtype"), "bf16"),
        (("encoder", "state_layer"), 20),
        (("encoder", "pool_type"), "centered_mean"),
        (("encoder", "pool_radius"), 8),
        (("encoder", "normalize"), True),
        (("encoder", "state_contract_version"), "l2_normalized_v2"),
        (("encoder", "trust_remote_code"), False),
        (("predictor", "architecture"), "cross_attention"),
        (("predictor", "n_layers"), 6),
        (("predictor", "n_heads"), 8),
        (("predictor", "d_state"), 1024),
        (("predictor", "d_action"), 64),
        (("predictor", "dtype"), "fp32"),
        (("action", "d_action"), 64),
        (("action", "max_len"), 16),
        (("action", "sub_encoders"), ["snv"]),
        (("training", "max_steps"), EXPECTED_STEPS),
        (("training", "collapse_log_every_steps"), 10),
        (("optimizer", "name"), "adamw"),
        (("optimizer", "lr"), 3.0e-4),
        (("optimizer", "beta1"), 0.9),
        (("optimizer", "beta2"), 0.95),
        (("optimizer", "weight_decay"), 0.1),
        (("optimizer", "grad_clip"), 1.0),
        (("optimizer", "warmup_steps"), 10),
        (("optimizer", "schedule"), "wsd"),
        (("data", "corpus_id"), EXPECTED_CORPUS_ID),
        (("data", "corpus_revision"), EXPECTED_CORPUS_REVISION),
        (("data", "batch_size"), 8),
        (("data", "num_workers"), 0),
        (("data", "shuffle_buffer"), 0),
        (("runtime", "backend"), "torch"),
        (("runtime", "device"), "cuda"),
    )
    for keys, wanted in expected:
        _expect_nested(issues, "config", payload, keys, wanted)


def _validate_snapshot(
    payload: dict[str, Any],
    path: Path,
    issues: list[CorrectionControlIssue],
) -> None:
    expected: tuple[tuple[tuple[str, ...], object], ...] = (
        (("schema_version",), "1.0.0"),
        (("snapshot_id",), EXPECTED_SNAPSHOT_ID),
        (("clinvar", "release"), "2026-04-15"),
        (("clinvar", "max_allele_len"), 1),
        (("clinvar", "input_vcf"), "inputs/clinvar/clinvar-2026-04-15-snv.vcf.gz"),
        (("clinvar", "split"), "eval_clinvar"),
        (("gnomad", "release"), "v4.1"),
        (("gnomad", "max_allele_len"), 1),
        (("gnomad", "min_af"), 0.01),
        (("gnomad", "input_vcf"), "inputs/gnomad/gnomad-v4.1-snv.vcf.gz"),
        (("gnomad", "split"), "train_gnomad_common"),
    )
    for keys, wanted in expected:
        _expect_nested(issues, "snapshot", payload, keys, wanted)

    observed_integrity = payload.get("source_integrity", _MISSING)
    _expect_equal(
        issues,
        code="snapshot.source_integrity_mismatch",
        label="snapshot.source_integrity",
        observed=observed_integrity,
        expected=EXPECTED_SOURCE_INTEGRITY,
    )
    _validate_carbon_files(payload.get("carbon_files", _MISSING), issues)
    _validate_snapshot_sources(payload.get("sources", _MISSING), path, issues)


def _validate_carbon_files(
    value: object,
    issues: list[CorrectionControlIssue],
) -> None:
    expected = [
        {
            "path": "carbon/source-mix-windows.jsonl",
            "source_path": "inputs/carbon/source-mix-windows.jsonl",
            "split": "train_carbon",
        }
    ]
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        _issue(
            issues,
            "snapshot.carbon_files_mismatch",
            "snapshot.carbon_files",
            "snapshot must declare exactly one Carbon source-mix file",
            expected,
            value,
        )
        return
    observed = {key: value[0].get(key, _MISSING) for key in expected[0]}
    _expect_equal(
        issues,
        code="snapshot.carbon_files_mismatch",
        label="snapshot.carbon_files",
        observed=[observed],
        expected=expected,
    )


def _validate_snapshot_sources(
    value: object,
    path: Path,
    issues: list[CorrectionControlIssue],
) -> None:
    expected = {
        "Carbon pretraining corpus": {
            "revision": EXPECTED_CORPUS_REVISION,
            "url": "https://huggingface.co/datasets/HuggingFaceBio/carbon-pretraining-corpus",
        },
        "gnomAD": {
            "revision": "v4.1 chr22 generation 1713312296186865",
            "url": EXPECTED_GNOMAD_URL,
        },
        "ClinVar": {
            "revision": "2026-04-15 md5:e63b5c3a046010c098cc70e81bebaa8d",
            "url": EXPECTED_CLINVAR_URL,
        },
    }
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        _issue(
            issues,
            "snapshot.sources_invalid",
            str(path),
            "snapshot sources must be a list of objects",
            sorted(expected),
            value,
        )
        return
    by_name: dict[str, dict[str, Any]] = {}
    duplicate_names: list[str] = []
    for row in value:
        name = row.get("name")
        if not isinstance(name, str):
            _issue(
                issues,
                "snapshot.sources_invalid",
                "snapshot.sources.name",
                "each snapshot source requires a string name",
                sorted(expected),
                name,
            )
            continue
        if name in by_name:
            duplicate_names.append(name)
        by_name[name] = row
    if duplicate_names or set(by_name) != set(expected):
        _issue(
            issues,
            "snapshot.source_names_mismatch",
            "snapshot.sources",
            "snapshot source names must match the correction-control manifest",
            sorted(expected),
            {"names": sorted(by_name), "duplicates": sorted(duplicate_names)},
        )
    for name, fields in expected.items():
        row = by_name.get(name)
        if row is None:
            continue
        for field, wanted in fields.items():
            _expect_equal(
                issues,
                code=f"snapshot.sources.{_slug(name)}.{field}_mismatch",
                label=f"snapshot.sources[{name!r}].{field}",
                observed=row.get(field, _MISSING),
                expected=wanted,
            )


def _load_yaml_object(
    path: Path,
    issues: list[CorrectionControlIssue],
) -> dict[str, Any] | None:
    if not path.is_file():
        _issue(
            issues,
            "config.file_missing",
            str(path),
            "correction-control training config is missing",
            "an existing YAML file",
            None,
        )
        return None
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        _issue(
            issues,
            "config.parse_failed",
            str(path),
            "correction-control training config could not be parsed",
            "a UTF-8 YAML object",
            str(exc),
        )
        return None
    if not isinstance(value, dict):
        _issue(
            issues,
            "config.root_invalid",
            str(path),
            "correction-control training config must be a YAML object",
            "object",
            type(value).__name__,
        )
        return None
    return value


def _load_json_object(
    path: Path,
    issues: list[CorrectionControlIssue],
) -> dict[str, Any] | None:
    if not path.is_file():
        _issue(
            issues,
            "snapshot.file_missing",
            str(path),
            "correction-control dataset snapshot is missing",
            "an existing JSON file",
            None,
        )
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            "snapshot.parse_failed",
            str(path),
            "correction-control dataset snapshot could not be parsed",
            "a UTF-8 JSON object",
            str(exc),
        )
        return None
    if not isinstance(value, dict):
        _issue(
            issues,
            "snapshot.root_invalid",
            str(path),
            "correction-control dataset snapshot must be a JSON object",
            "object",
            type(value).__name__,
        )
        return None
    return value


def _git_identity(repo_root: Path) -> tuple[Path | None, str | None, str | None]:
    try:
        top = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, None, str(exc)
    return Path(top).resolve(), head, None


def _git_dirty_paths(repo_root: Path) -> tuple[tuple[str, ...], str | None]:
    try:
        output = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return (), str(exc)
    paths = tuple(line[3:] for line in output.splitlines() if len(line) >= 4)
    return paths, None


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _expect_path(
    issues: list[CorrectionControlIssue],
    *,
    code: str,
    label: str,
    observed: Path,
    expected: Path,
) -> None:
    _expect_equal(
        issues,
        code=code,
        label=label,
        observed=str(observed),
        expected=str(expected),
    )


def _expect_nested(
    issues: list[CorrectionControlIssue],
    prefix: str,
    payload: dict[str, Any],
    keys: tuple[str, ...],
    expected: object,
) -> None:
    observed: object = payload
    for key in keys:
        if not isinstance(observed, dict) or key not in observed:
            observed = _MISSING
            break
        observed = observed[key]
    label = ".".join((prefix, *keys))
    _expect_equal(
        issues,
        code=f"{label}_mismatch",
        label=label,
        observed=observed,
        expected=expected,
    )


def _expect_equal(
    issues: list[CorrectionControlIssue],
    *,
    code: str,
    label: str,
    observed: object,
    expected: object,
) -> None:
    if observed != expected or type(observed) is not type(expected):
        _issue(
            issues,
            code,
            label,
            f"{label} does not match the correction-control contract",
            expected,
            observed,
        )


def _issue(
    issues: list[CorrectionControlIssue],
    code: str,
    path: str,
    message: str,
    expected: object | None,
    observed: object | None,
) -> None:
    issues.append(
        CorrectionControlIssue(
            code=code,
            path=path,
            message=message,
            expected=expected,
            observed=observed,
        )
    )


def _file_summary(
    path: Path,
    payload: dict[str, Any] | None,
    *,
    display_path: str,
) -> dict[str, object]:
    summary: dict[str, object] = {"path": display_path, "exists": path.is_file()}
    if path.is_file():
        summary.update({"sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    if payload is not None:
        for key in ("run_id", "snapshot_id", "schema_version"):
            value = payload.get(key)
            if isinstance(value, str):
                summary[key] = value
    return summary


def _display_repo_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _json_value(value: object) -> object:
    if value is _MISSING:
        return "<missing>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--expected-commit-sha", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-attempt", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--max-windows", type=int, required=True)
    parser.add_argument("--clinvar-lines", type=int, required=True)
    parser.add_argument("--gnomad-lines", type=int, required=True)
    parser.add_argument("--tuple-throughput-samples", type=int, required=True)
    parser.add_argument("--window-bp", type=int, required=True)
    parser.add_argument("--holdout-chrom", type=int, required=True)
    parser.add_argument("--carbon-model-dir", required=True)
    parser.add_argument("--carbon-config", required=True)
    parser.add_argument("--carbon-source", required=True)
    parser.add_argument("--corpus-revision", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--clinvar-url", required=True)
    parser.add_argument("--gnomad-url", required=True)
    parser.add_argument("--output", type=Path)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
