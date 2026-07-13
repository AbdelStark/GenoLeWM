# SPDX-License-Identifier: Apache-2.0
"""Bind two completed correction-control runs into deterministic-pair evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file
from tools.release.training_reproducibility import (
    compare_deterministic_pair,
    load_training_run_evidence,
)
from tools.research.correction_control_postflight import (
    GENERATED_BY as CORRECTION_POSTFLIGHT_GENERATED_BY,
    SCHEMA_VERSION as CORRECTION_POSTFLIGHT_SCHEMA_VERSION,
)
from tools.research.correction_control_preflight import (
    GENERATED_BY as CORRECTION_PREFLIGHT_GENERATED_BY,
    SCHEMA_VERSION as CORRECTION_PREFLIGHT_SCHEMA_VERSION,
)

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.research.correction_control_replay"
REPORT_NAME: Final = "deterministic_replay_report.json"
_RUN_NAME_RE: Final = re.compile(
    r"^geno-lewm-l2-p1-smoke-(?P<commit>[0-9a-f]{12})-50-r(?P<attempt>[1-9][0-9]*)$"
)
_POSTFLIGHT_PATH: Final = Path("correction_control/correction_control_postflight.json")
_JOB_PREFLIGHT_PATH: Final = Path("correction_control/job_contract_preflight.json")
_SOURCE_IDENTITY_PATH: Final = Path("correction_control/source_identity_report.json")
_SOURCE_IDENTITY_SCHEMA_VERSION: Final = "1.0.0"
_SOURCE_IDENTITY_GENERATED_BY: Final = "tools.jobs.proof_run.source_identity"
_CLAIM_BOUNDARY: Final = (
    "This report establishes only whether two completed correction-control runs produced "
    "bit-exact dataset, config, and checkpoint artifacts under the same deterministic "
    "contract. It does not evaluate deterministic throughput, model quality, benchmark "
    "performance, planning utility, or clinical validity."
)


@dataclass(frozen=True, slots=True)
class CorrectionControlReplayRequest:
    """Inputs identifying one completed proof and its deterministic replay."""

    reference_run_dir: Path
    candidate_run_dir: Path
    reference_run_name: str
    candidate_run_name: str
    expected_commit_sha: str


def build_correction_control_replay_report(
    request: CorrectionControlReplayRequest,
    *,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Validate two completed proof archives and compare deterministic artifacts."""
    reference_attempt = _run_attempt(request.reference_run_name, request.expected_commit_sha)
    candidate_attempt = _run_attempt(request.candidate_run_name, request.expected_commit_sha)
    if candidate_attempt <= reference_attempt:
        raise InputError(
            "candidate replay attempt must be newer than the reference attempt",
            details={"reference": reference_attempt, "candidate": candidate_attempt},
        )

    try:
        reference_postflight = _load_completed_postflight(request.reference_run_dir)
    except InputError as exc:
        return _failed_report(
            request,
            reference_attempt=reference_attempt,
            candidate_attempt=candidate_attempt,
            blocker=_input_blocker("reference", exc),
            generated_at=generated_at,
        )
    try:
        candidate_postflight = _load_completed_postflight(request.candidate_run_dir)
    except InputError as exc:
        return _failed_report(
            request,
            reference_attempt=reference_attempt,
            candidate_attempt=candidate_attempt,
            blocker=_input_blocker("candidate", exc),
            generated_at=generated_at,
        )
    try:
        reference_job = _load_job_preflight(request.reference_run_dir)
        reference_source = _load_source_identity(
            request.reference_run_dir,
            reference_postflight,
        )
        _validate_run_contract(
            reference_postflight,
            reference_job,
            reference_source,
            run_name=request.reference_run_name,
            run_attempt=reference_attempt,
            expected_commit_sha=request.expected_commit_sha,
        )
    except InputError as exc:
        return _failed_report(
            request,
            reference_attempt=reference_attempt,
            candidate_attempt=candidate_attempt,
            blocker=_input_blocker("reference", exc),
            generated_at=generated_at,
            postflights_ok=True,
        )
    try:
        candidate_job = _load_job_preflight(request.candidate_run_dir)
        candidate_source = _load_source_identity(
            request.candidate_run_dir,
            candidate_postflight,
        )
        _validate_run_contract(
            candidate_postflight,
            candidate_job,
            candidate_source,
            run_name=request.candidate_run_name,
            run_attempt=candidate_attempt,
            expected_commit_sha=request.expected_commit_sha,
        )
    except InputError as exc:
        return _failed_report(
            request,
            reference_attempt=reference_attempt,
            candidate_attempt=candidate_attempt,
            blocker=_input_blocker("candidate", exc),
            generated_at=generated_at,
            postflights_ok=True,
        )
    try:
        _require_matching_contracts(
            reference_postflight,
            candidate_postflight,
            reference_job,
            candidate_job,
            reference_source,
            candidate_source,
        )
    except InputError as exc:
        return _failed_report(
            request,
            reference_attempt=reference_attempt,
            candidate_attempt=candidate_attempt,
            blocker=_input_blocker("pair", exc),
            generated_at=generated_at,
            postflights_ok=True,
        )

    try:
        reference = load_training_run_evidence(
            request.reference_run_dir,
            label="reference",
            require_preflight=True,
        )
    except GenoLeWMError as exc:
        return _failed_report(
            request,
            reference_attempt=reference_attempt,
            candidate_attempt=candidate_attempt,
            blocker=_training_archive_blocker("reference", exc),
            generated_at=generated_at,
            postflights_ok=True,
        )
    try:
        candidate = load_training_run_evidence(
            request.candidate_run_dir,
            label="candidate",
            require_preflight=True,
        )
    except GenoLeWMError as exc:
        return _failed_report(
            request,
            reference_attempt=reference_attempt,
            candidate_attempt=candidate_attempt,
            blocker=_training_archive_blocker("candidate", exc),
            generated_at=generated_at,
            postflights_ok=True,
        )
    pair = compare_deterministic_pair(reference, candidate)
    blockers = [blocker.to_dict() for blocker in pair.blockers]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": generated_at or _utc_now(),
        "scope": "deterministic_pair",
        "ok": pair.ok,
        "issue_47_complete": False,
        "throughput_evaluated": False,
        "postflights_ok": True,
        "reference": {
            "run_name": request.reference_run_name,
            "run_attempt": reference_attempt,
            "correction_control_postflight": _postflight_evidence(
                request.reference_run_dir,
                reference_postflight,
            ),
            "evidence": reference.to_dict(),
        },
        "candidate": {
            "run_name": request.candidate_run_name,
            "run_attempt": candidate_attempt,
            "correction_control_postflight": _postflight_evidence(
                request.candidate_run_dir,
                candidate_postflight,
            ),
            "evidence": candidate.to_dict(),
        },
        "deterministic_pair": pair.to_dict(),
        "blockers": blockers,
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def write_correction_control_replay_report(
    request: CorrectionControlReplayRequest,
    output: Path,
) -> dict[str, object]:
    """Build and persist a normalized deterministic-pair replay report."""
    report = build_correction_control_replay_report(request)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Validate a correction-control replay pair and write its evidence."""
    args = _parser().parse_args(argv)
    request = CorrectionControlReplayRequest(
        reference_run_dir=args.reference_run_dir,
        candidate_run_dir=args.candidate_run_dir,
        reference_run_name=args.reference_run_name,
        candidate_run_name=args.candidate_run_name,
        expected_commit_sha=args.expected_commit_sha,
    )
    try:
        report = write_correction_control_replay_report(request, args.output_json)
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    except OSError as exc:
        sys.stderr.write(f"error: failed to write correction-control replay report: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report["ok"] is True else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate bit-exact deterministic correction-control replay evidence."
    )
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--reference-run-name", required=True)
    parser.add_argument("--candidate-run-name", required=True)
    parser.add_argument("--expected-commit-sha", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _run_attempt(run_name: str, expected_commit_sha: str) -> int:
    match = _RUN_NAME_RE.fullmatch(run_name)
    if match is None or match.group("commit") != expected_commit_sha[:12]:
        raise InputError(
            "correction-control replay run name does not match the expected commit",
            details={"run_name": run_name, "expected_commit_sha": expected_commit_sha},
        )
    return int(match.group("attempt"))


def _load_completed_postflight(run_dir: Path) -> dict[str, Any]:
    _require_completed_run_dir(run_dir)
    payload = _load_json_object(run_dir / _POSTFLIGHT_PATH)
    if payload.get("schema_version") != CORRECTION_POSTFLIGHT_SCHEMA_VERSION:
        raise InputError("correction-control postflight schema version is invalid")
    if payload.get("generated_by") != CORRECTION_POSTFLIGHT_GENERATED_BY:
        raise InputError("correction-control postflight generator is invalid")
    if payload.get("ok") is not True:
        raise InputError(
            "correction-control replay requires postflight ok=true",
            details={"replay_code": "postflight_not_ok"},
        )
    return payload


def _load_job_preflight(run_dir: Path) -> dict[str, Any]:
    payload = _load_json_object(run_dir / _JOB_PREFLIGHT_PATH)
    if payload.get("schema_version") != CORRECTION_PREFLIGHT_SCHEMA_VERSION:
        raise InputError("correction-control job preflight schema version is invalid")
    if payload.get("generated_by") != CORRECTION_PREFLIGHT_GENERATED_BY:
        raise InputError("correction-control job preflight generator is invalid")
    if payload.get("ok") is not True:
        raise InputError("correction-control replay requires job preflight ok=true")
    return payload


def _load_source_identity(
    run_dir: Path,
    postflight: dict[str, Any],
) -> dict[str, Any]:
    path = run_dir / _SOURCE_IDENTITY_PATH
    _require_postflight_artifact_identity(
        postflight,
        artifact_name="source_identity_report",
        path=path,
    )
    payload = _load_json_object(path)
    if payload.get("schema_version") != _SOURCE_IDENTITY_SCHEMA_VERSION:
        raise InputError(
            "correction-control source identity schema version is invalid",
            details={"replay_code": "source_identity_invalid"},
        )
    if payload.get("generated_by") != _SOURCE_IDENTITY_GENERATED_BY:
        raise InputError(
            "correction-control source identity generator is invalid",
            details={"replay_code": "source_identity_invalid"},
        )
    if payload.get("ok") is not True:
        raise InputError(
            "correction-control replay requires source identity ok=true",
            details={"replay_code": "source_identity_invalid"},
        )
    return payload


def _require_postflight_artifact_identity(
    postflight: dict[str, Any],
    *,
    artifact_name: str,
    path: Path,
) -> None:
    artifacts = postflight.get("artifacts")
    expected = artifacts.get(artifact_name) if isinstance(artifacts, dict) else None
    observed: dict[str, object] = {"path": path.name, "exists": path.is_file()}
    if path.is_file():
        observed.update(
            {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if expected != observed:
        raise InputError(
            "correction-control source identity does not match its postflight receipt",
            details={"replay_code": "source_identity_artifact_mismatch"},
        )


def _validate_run_contract(
    postflight: dict[str, Any],
    job_preflight: dict[str, Any],
    source_identity: dict[str, Any],
    *,
    run_name: str,
    run_attempt: int,
    expected_commit_sha: str,
) -> None:
    expected = postflight.get("expected")
    repository = job_preflight.get("repository")
    job = job_preflight.get("job")
    if not isinstance(expected, dict) or expected.get("commit_sha") != expected_commit_sha:
        raise InputError("correction-control postflight commit identity is invalid")
    if not isinstance(repository, dict) or any(
        repository.get(field) != expected_commit_sha
        for field in ("expected_commit_sha", "observed_commit_sha")
    ):
        raise InputError("correction-control job repository identity is invalid")
    if not isinstance(job, dict):
        raise InputError("correction-control job preflight job contract is missing")
    if job.get("run_name") != run_name or job.get("run_attempt") != run_attempt:
        raise InputError("correction-control job run identity is invalid")
    if (
        source_identity.get("commit_sha") != expected_commit_sha
        or source_identity.get("run_name") != run_name
        or source_identity.get("dataset_snapshot_id") != expected.get("dataset_snapshot_id")
    ):
        raise InputError(
            "correction-control source identity is invalid",
            details={"replay_code": "source_identity_invalid"},
        )


def _require_matching_contracts(
    reference_postflight: dict[str, Any],
    candidate_postflight: dict[str, Any],
    reference_job: dict[str, Any],
    candidate_job: dict[str, Any],
    reference_source: dict[str, Any],
    candidate_source: dict[str, Any],
) -> None:
    if reference_postflight.get("expected") != candidate_postflight.get("expected"):
        raise InputError("correction-control postflight contracts do not match")
    for field in ("config", "snapshot"):
        if reference_job.get(field) != candidate_job.get(field):
            raise InputError(
                "correction-control job contracts do not match",
                details={"replay_code": "contract_mismatch", "field": field},
            )
    reference_job_payload = dict(reference_job.get("job", {}))
    candidate_job_payload = dict(candidate_job.get("job", {}))
    for field in ("run_name", "run_attempt"):
        reference_job_payload.pop(field, None)
        candidate_job_payload.pop(field, None)
    if reference_job_payload != candidate_job_payload:
        raise InputError(
            "correction-control launch contracts do not match",
            details={"replay_code": "contract_mismatch", "field": "job"},
        )
    for field in ("dataset_snapshot_id", "training_contract", "sources"):
        if reference_source.get(field) != candidate_source.get(field):
            raise InputError(
                "correction-control source contracts do not match",
                details={
                    "replay_code": "contract_mismatch",
                    "field": f"source_identity.{field}",
                },
            )


def _require_completed_run_dir(run_dir: Path) -> None:
    if run_dir.name != "run" or "run-partial" in run_dir.parts:
        raise InputError(
            "correction-control replay accepts only completed run directories",
            details={
                "replay_code": "incomplete_run_directory",
                "path": run_dir.name,
            },
        )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(
            "correction-control replay artifact is missing", details={"path": path.name}
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "correction-control replay artifact is invalid JSON",
            details={"path": path.name, "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError(
            "correction-control replay artifact must be a JSON object",
            details={"path": path.name},
        )
    return payload


def _input_blocker(label: str, error: InputError) -> dict[str, object]:
    code = error.details.get("replay_code", "invalid_evidence")
    return {
        "code": f"{label}.{code}",
        "message": error.message,
        "details": {key: value for key, value in error.details.items() if key != "replay_code"},
    }


def _training_archive_blocker(label: str, error: GenoLeWMError) -> dict[str, object]:
    return {
        "code": f"{label}.invalid_training_archive",
        "message": error.message,
        "details": {"error_code": error.code},
    }


def _postflight_evidence(
    run_dir: Path,
    postflight: dict[str, Any],
) -> dict[str, object]:
    path = run_dir / _POSTFLIGHT_PATH
    return {
        "path": _POSTFLIGHT_PATH.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "ok": postflight["ok"],
        "expected": postflight["expected"],
    }


def _failed_report(
    request: CorrectionControlReplayRequest,
    *,
    reference_attempt: int,
    candidate_attempt: int,
    blocker: dict[str, object],
    generated_at: str | None,
    postflights_ok: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": generated_at or _utc_now(),
        "scope": "deterministic_pair",
        "ok": False,
        "issue_47_complete": False,
        "throughput_evaluated": False,
        "postflights_ok": postflights_ok,
        "reference": {
            "run_name": request.reference_run_name,
            "run_attempt": reference_attempt,
            "correction_control_postflight": None,
            "evidence": None,
        },
        "candidate": {
            "run_name": request.candidate_run_name,
            "run_attempt": candidate_attempt,
            "correction_control_postflight": None,
            "evidence": None,
        },
        "deterministic_pair": {
            "status": "not_evaluated",
            "ok": False,
            "matched_fields": [],
            "matched_artifacts": [],
            "blockers": [],
        },
        "blockers": [blocker],
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
