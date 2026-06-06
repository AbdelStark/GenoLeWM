# SPDX-License-Identifier: Apache-2.0
"""Record an accepted RFC-0004 rollout-speed scope decision.

This tool does not turn failed rollout-speed measurements into speed
evidence. It binds a failing ``bench.rollout`` report to an explicit
project decision so release readiness can distinguish an accepted target
re-scope from an unreviewed benchmark miss.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.rollout_speed_scope"
DECISION: Final = "rescope_rfc0004_speed_target"
STATUS: Final = "accepted"
ROLLOUT_GENERATED_BY: Final = "bench.rollout"
ROLLOUT_SCHEMA_VERSION: Final = "1.0.0"
REQUIRED_ISSUE_REFS: Final = ("#42", "#197")
REQUIRED_HORIZONS: Final = (5, 20)
COMMAND_PATH_FLAGS: Final = frozenset(
    {
        "--rollout-speed-report",
        "--output-json",
        "--out-dir",
        "--output",
    }
)


def build_scope_report(
    *,
    rollout_speed_report: Path,
    accepted_by: str,
    accepted_at: str,
    decision_url: str,
    rationale: str,
    replacement_target: str,
    issue_refs: tuple[str, ...] = REQUIRED_ISSUE_REFS,
    command: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build a machine-readable accepted rollout-speed scope report."""
    payload = _load_rollout_report(rollout_speed_report)
    summary = _rollout_summary(payload)
    if not summary["failed_targets"]:
        raise InputError("rollout speed scope requires at least one failed target")
    _require_issue_refs(issue_refs)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": _utc_now(),
        "ok": True,
        "status": STATUS,
        "decision": DECISION,
        "accepted_by": _require_text_value(accepted_by, "accepted_by"),
        "accepted_at": _require_utc_timestamp(accepted_at, "accepted_at"),
        "decision_url": _require_url(decision_url, "decision_url"),
        "issue_refs": list(issue_refs),
        "rationale": _require_text_value(rationale, "rationale"),
        "replacement_target": _require_text_value(replacement_target, "replacement_target"),
        "command": _public_safe_command(command),
        "rollout_speed_report": _file_identity(rollout_speed_report),
        "rollout_speed_summary": summary,
        "negative_findings": [
            (
                "The RFC-0004 rollout-speed target was not met by the bound "
                "bench.rollout measurement."
            ),
            (
                "This report records an accepted scope decision only; it is not "
                "rollout-speed evidence."
            ),
        ],
        "claim_boundary": (
            "This report records an accepted RFC-0004 scope decision. It does not "
            "establish model quality, clinical utility, deployment readiness, privacy "
            "assurance, or that the original rollout-speed targets were met."
        ),
    }


def write_scope_report(
    *,
    rollout_speed_report: Path,
    output: Path,
    accepted_by: str,
    accepted_at: str,
    decision_url: str,
    rationale: str,
    replacement_target: str,
    issue_refs: tuple[str, ...] = REQUIRED_ISSUE_REFS,
    command: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build and write a rollout-speed scope report."""
    report = build_scope_report(
        rollout_speed_report=rollout_speed_report,
        accepted_by=accepted_by,
        accepted_at=accepted_at,
        decision_url=decision_url,
        rationale=rationale,
        replacement_target=replacement_target,
        issue_refs=issue_refs,
        command=command,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    command = _command_from_args(args)
    try:
        write_scope_report(
            rollout_speed_report=args.rollout_speed_report,
            output=args.output,
            accepted_by=args.accepted_by,
            accepted_at=args.accepted_at,
            decision_url=args.decision_url,
            rationale=args.rationale,
            replacement_target=args.replacement_target,
            issue_refs=tuple(args.issue_ref or REQUIRED_ISSUE_REFS),
            command=command,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(f"wrote {args.output}\n")
    return 0


def _load_rollout_report(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(
            "failed to read rollout speed report", details={"path": str(path)}
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "rollout speed report JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(raw, dict):
        raise InputError("rollout speed report payload must be a JSON object")
    schema_version = raw.get("schema_version")
    if schema_version != ROLLOUT_SCHEMA_VERSION:
        raise InputError(
            "rollout speed report schema_version is invalid",
            details={"expected": ROLLOUT_SCHEMA_VERSION, "observed": schema_version},
        )
    generated_by = raw.get("generated_by")
    if generated_by != ROLLOUT_GENERATED_BY:
        raise InputError(
            "rollout speed report generated_by is invalid",
            details={"expected": ROLLOUT_GENERATED_BY, "observed": generated_by},
        )
    return raw


def _rollout_summary(payload: dict[str, object]) -> dict[str, object]:
    command = _required_text_list(payload.get("command"), "rollout speed command")
    commit = _required_text(payload, "commit")
    report_ok = _required_bool(payload, "ok")
    _require_rollout_claim_boundary(payload.get("claim_boundary"))
    rows = _require_list(payload.get("rows"), "rollout speed rows")
    observed_values: dict[str, float] = {}
    failed_targets: list[dict[str, object]] = []
    observed_horizons: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise InputError("rollout speed rows must be objects")
        horizon = _required_int(row, "horizon")
        speedup = _required_number(row, "measured_speedup")
        target = _required_number(row, "target_speedup")
        observed_horizons.add(horizon)
        observed_values[f"k{horizon}_speedup"] = speedup
        if not _required_bool(row, "target_met"):
            failed_targets.append(
                {
                    "horizon": horizon,
                    "measured_speedup": speedup,
                    "target_speedup": target,
                    "shortfall": max(0.0, target - speedup),
                }
            )
    missing_horizons = [
        horizon for horizon in REQUIRED_HORIZONS if horizon not in observed_horizons
    ]
    if missing_horizons:
        raise InputError(
            "rollout speed scope requires K=5 and K=20 measurements",
            details={"missing_horizons": missing_horizons},
        )
    return {
        "commit": commit,
        "command": _public_safe_command(tuple(command)),
        "report_ok": report_ok,
        "observed_values": observed_values,
        "failed_targets": failed_targets,
    }


def _file_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise InputError("rollout speed report does not exist", details={"path": str(path)})
    return {
        "path": _public_safe_identity_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _public_safe_identity_path(path: Path) -> str:
    return _public_safe_path_text(str(path))


def _public_safe_path_text(value: str) -> str:
    if "://" in value:
        return value
    posix_path = PurePosixPath(value)
    if posix_path.is_absolute():
        return posix_path.name
    windows_path = PureWindowsPath(value)
    if windows_path.is_absolute():
        return windows_path.name
    return value


def _public_safe_command(command: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    sanitize_next = False
    for token in command:
        if sanitize_next:
            result.append(_public_safe_path_text(token))
            sanitize_next = False
            continue
        result.append(_public_safe_path_text(token))
        sanitize_next = token in COMMAND_PATH_FLAGS
    return result


def _require_issue_refs(issue_refs: tuple[str, ...]) -> None:
    missing = [ref for ref in REQUIRED_ISSUE_REFS if ref not in issue_refs]
    if missing:
        raise InputError(
            "rollout speed scope issue_refs must include #42 and #197",
            details={"missing_issue_refs": missing},
        )
    for ref in issue_refs:
        if not isinstance(ref, str) or not ref.startswith("#") or not ref[1:].isdigit():
            raise InputError("rollout speed scope issue_refs must be GitHub issue refs")


def _require_list(raw: object, label: str) -> list[object]:
    if not isinstance(raw, list):
        raise InputError(f"{label} must be a JSON list")
    return raw


def _required_int(raw: dict[str, object], field: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{field} must be an integer")
    return value


def _required_number(raw: dict[str, object], field: str) -> float:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputError(f"{field} must be a number")
    return float(value)


def _required_text(raw: dict[str, object], field: str) -> str:
    return _require_text_value(raw.get(field), field)


def _require_text_value(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{field} must be a non-empty string")
    return value.strip()


def _require_url(value: object, field: str) -> str:
    text = _require_text_value(value, field)
    if not text.startswith(("https://", "http://")):
        raise InputError(f"{field} must be an HTTP(S) URL")
    return text


def _require_rollout_claim_boundary(raw: object) -> None:
    text = _require_text_value(raw, "claim_boundary")
    normalized = text.lower()
    required_terms = (
        "rollout speed",
        "not",
        "model-quality",
        "clinical",
        "privacy",
        "release-readiness",
    )
    if any(term not in normalized for term in required_terms):
        raise InputError("rollout speed report claim_boundary must preserve benchmark limits")


def _required_text_list(raw: object, label: str) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise InputError(f"{label} must be a non-empty JSON list")
    if not all(isinstance(item, str) and item.strip() for item in raw):
        raise InputError(f"{label} entries must be non-empty strings")
    return [item.strip() for item in raw]


def _required_bool(raw: dict[str, object], field: str) -> bool:
    value = raw.get(field)
    if not isinstance(value, bool):
        raise InputError(f"{field} must be a boolean")
    return value


def _require_utc_timestamp(value: object, field: str) -> str:
    text = _require_text_value(value, field)
    if not text.endswith("Z"):
        raise InputError(f"{field} must be a UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise InputError(f"{field} must be an ISO-8601 UTC timestamp") from exc
    return text


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record an accepted RFC-0004 rollout-speed scope decision.",
    )
    parser.add_argument("--rollout-speed-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-by", required=True)
    parser.add_argument("--accepted-at", required=True)
    parser.add_argument("--decision-url", required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--replacement-target", required=True)
    parser.add_argument(
        "--issue-ref",
        action="append",
        help="GitHub issue ref such as #42. Defaults to #42 and #197.",
    )
    return parser


def _command_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "python",
        "-m",
        "tools.release.rollout_speed_scope",
        "--rollout-speed-report",
        _public_safe_identity_path(args.rollout_speed_report),
        "--output",
        _public_safe_identity_path(args.output),
        "--accepted-by",
        args.accepted_by,
        "--accepted-at",
        args.accepted_at,
        "--decision-url",
        args.decision_url,
        "--rationale",
        args.rationale,
        "--replacement-target",
        args.replacement_target,
    ]
    for ref in args.issue_ref or ():
        command.extend(("--issue-ref", ref))
    return tuple(command)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
