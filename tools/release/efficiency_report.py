# SPDX-License-Identifier: Apache-2.0
"""Validate and normalize first-release inference efficiency evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance.hashing import looks_like_sha256

REPORT_NAME: Final = "efficiency_report.json"
SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.efficiency_report"
COMMIT_RE: Final = re.compile(r"^[0-9a-fA-F]{7,40}$")
PLACEHOLDER_RE: Final = re.compile(
    r"\b(?:tbd|todo|placeholder|coming soon|fake|dummy|lorem ipsum)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class EfficiencyInputIdentity:
    """File identity for one benchmark input or artifact."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class EfficiencyMeasurements:
    """Measured latency, throughput, and peak-memory values."""

    single_variant_latency_ms: float
    batched_throughput_variants_per_s: float
    peak_memory_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "batched_throughput_variants_per_s": self.batched_throughput_variants_per_s,
            "peak_memory_bytes": self.peak_memory_bytes,
            "single_variant_latency_ms": self.single_variant_latency_ms,
        }


@dataclass(frozen=True, slots=True)
class EfficiencyReport:
    """Validated machine-readable inference efficiency evidence."""

    schema_version: str
    generated_by: str
    generated_at: str
    model_id: str
    model_release: str
    dataset_snapshot: str
    commit: str
    command: tuple[str, ...]
    hardware: str
    runtime: str
    warmup_batches: int
    samples: int
    measurements: EfficiencyMeasurements
    inputs: tuple[tuple[str, EfficiencyInputIdentity], ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "model_id": self.model_id,
            "model_release": self.model_release,
            "dataset_snapshot": self.dataset_snapshot,
            "commit": self.commit,
            "command": list(self.command),
            "hardware": self.hardware,
            "runtime": self.runtime,
            "warmup_batches": self.warmup_batches,
            "samples": self.samples,
            "measurements": self.measurements.to_dict(),
            "inputs": {key: identity.to_dict() for key, identity in self.inputs},
            "limitations": list(self.limitations),
        }


def load_efficiency_report(
    path: Path,
    *,
    allow_placeholders: bool = False,
) -> EfficiencyReport:
    """Load and validate an efficiency report JSON file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(
            "failed to read efficiency report JSON", details={"path": str(path)}
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "efficiency report JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    return parse_efficiency_report(payload, allow_placeholders=allow_placeholders)


def parse_efficiency_report(
    payload: Any,
    *,
    allow_placeholders: bool = False,
) -> EfficiencyReport:
    """Validate a decoded efficiency report payload."""
    if not isinstance(payload, dict):
        raise InputError("efficiency report payload must be a JSON object")
    schema_version = _required_text(payload, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise InputError(
            "unsupported efficiency-report schema version",
            details={"expected": SCHEMA_VERSION, "observed": schema_version},
        )
    generated_by = _required_text(payload, "generated_by")
    if generated_by != GENERATED_BY:
        raise InputError(
            "efficiency report generated_by is invalid",
            details={"expected": GENERATED_BY, "observed": generated_by},
        )
    model_id = _required_text(payload, "model_id")
    if not looks_like_sha256(model_id):
        raise InputError("model_id must be a sha256:<64hex> identifier")
    commit = _required_text(payload, "commit")
    if COMMIT_RE.fullmatch(commit) is None:
        raise InputError("commit must be a 7-40 character hexadecimal SHA")

    command = _parse_command(payload.get("command"))
    measurements = _parse_measurements(payload.get("measurements"))
    inputs = _parse_inputs(payload.get("inputs"))
    limitations = _parse_text_list(payload.get("limitations"), field="limitations")
    report = EfficiencyReport(
        schema_version=schema_version,
        generated_by=generated_by,
        generated_at=_optional_text(payload, "generated_at") or _utc_now(),
        model_id=model_id,
        model_release=_required_text(payload, "model_release"),
        dataset_snapshot=_required_text(payload, "dataset_snapshot"),
        commit=commit.lower(),
        command=command,
        hardware=_required_text(payload, "hardware"),
        runtime=_required_text(payload, "runtime"),
        warmup_batches=_required_non_negative_int(payload, "warmup_batches"),
        samples=_required_positive_int(payload, "samples"),
        measurements=measurements,
        inputs=inputs,
        limitations=limitations,
    )
    if not allow_placeholders:
        _reject_placeholders(_text_fields(report))
    return report


def write_efficiency_report(
    input_json: Path,
    output: Path,
    *,
    allow_placeholders: bool = False,
) -> EfficiencyReport:
    """Validate ``input_json`` and write normalized ``efficiency_report.json``."""
    report = load_efficiency_report(input_json, allow_placeholders=allow_placeholders)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        write_efficiency_report(
            args.input_json,
            args.output,
            allow_placeholders=args.allow_placeholders,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(f"wrote {args.output}\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate measured inference efficiency JSON for a release package.",
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow placeholder wording for local drafts. Do not use for releases.",
    )
    return parser


def _parse_measurements(raw: Any) -> EfficiencyMeasurements:
    if not isinstance(raw, dict):
        raise InputError("measurements must be a JSON object")
    return EfficiencyMeasurements(
        single_variant_latency_ms=_required_positive_number(
            raw,
            "single_variant_latency_ms",
            prefix="measurements.",
        ),
        batched_throughput_variants_per_s=_required_positive_number(
            raw,
            "batched_throughput_variants_per_s",
            prefix="measurements.",
        ),
        peak_memory_bytes=_required_positive_int(raw, "peak_memory_bytes", prefix="measurements."),
    )


def _parse_inputs(raw: Any) -> tuple[tuple[str, EfficiencyInputIdentity], ...]:
    if not isinstance(raw, dict) or not raw:
        raise InputError("inputs must be a non-empty object")
    inputs: list[tuple[str, EfficiencyInputIdentity]] = []
    for key in sorted(raw):
        if not isinstance(key, str) or not key:
            raise InputError("input keys must be non-empty strings")
        item = raw[key]
        if not isinstance(item, dict):
            raise InputError("input identities must be objects", details={"field": key})
        sha256 = _required_text(item, "sha256", prefix=f"inputs.{key}.")
        if not looks_like_sha256(sha256):
            raise InputError(
                "input sha256 must be a sha256:<64hex> identifier",
                details={"field": f"inputs.{key}.sha256"},
            )
        inputs.append(
            (
                key,
                EfficiencyInputIdentity(
                    path=_parse_input_path(item, key),
                    sha256=sha256,
                    size_bytes=_required_non_negative_int(
                        item,
                        "size_bytes",
                        prefix=f"inputs.{key}.",
                    ),
                ),
            )
        )
    return tuple(inputs)


def _parse_input_path(item: dict[str, Any], key: str) -> str:
    path = _required_text(item, "path", prefix=f"inputs.{key}.")
    if path.startswith("inline:"):
        label = path.removeprefix("inline:")
        if not label or "/" in label or "\\" in label or label in {".", ".."}:
            raise InputError(
                "inline input paths must use inline:<label>",
                details={"field": f"inputs.{key}.path"},
            )
        return path
    if "://" in path:
        raise InputError(
            "input paths must be package-relative or inline labels",
            details={"field": f"inputs.{key}.path"},
        )
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise InputError(
            "input paths must be package-relative or inline labels",
            details={"field": f"inputs.{key}.path"},
        )
    return path


def _parse_command(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise InputError("command must be a non-empty string list")
    command: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise InputError(
                "command entries must be non-empty strings",
                details={"field": f"command[{index}]"},
            )
        command.append(item.strip())
    return tuple(command)


def _parse_text_list(raw: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise InputError(f"{field} must be a non-empty list")
    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise InputError(
                f"{field} entries must be non-empty strings",
                details={"field": f"{field}[{index}]"},
            )
        values.append(item.strip())
    return tuple(values)


def _required_text(payload: dict[str, Any], key: str, *, prefix: str = "") -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{prefix}{key} must be a non-empty string")
    return value.strip()


def _optional_text(payload: dict[str, Any], key: str, *, prefix: str = "") -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{prefix}{key} must be a non-empty string when supplied")
    return value.strip()


def _required_positive_number(payload: dict[str, Any], key: str, *, prefix: str = "") -> float:
    value = payload.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise InputError(f"{prefix}{key} must be a finite positive number")
    return float(value)


def _required_positive_int(payload: dict[str, Any], key: str, *, prefix: str = "") -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(f"{prefix}{key} must be a positive integer")
    return value


def _required_non_negative_int(payload: dict[str, Any], key: str, *, prefix: str = "") -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InputError(f"{prefix}{key} must be a non-negative integer")
    return value


def _text_fields(report: EfficiencyReport) -> dict[str, str]:
    fields = {
        "generated_by": report.generated_by,
        "generated_at": report.generated_at,
        "model_id": report.model_id,
        "model_release": report.model_release,
        "dataset_snapshot": report.dataset_snapshot,
        "commit": report.commit,
        "hardware": report.hardware,
        "runtime": report.runtime,
    }
    for index, value in enumerate(report.command):
        fields[f"command[{index}]"] = value
    for key, identity in report.inputs:
        fields[f"inputs.{key}.path"] = identity.path
        fields[f"inputs.{key}.sha256"] = identity.sha256
    for index, value in enumerate(report.limitations):
        fields[f"limitations[{index}]"] = value
    return fields


def _reject_placeholders(values: dict[str, str]) -> None:
    for key, value in values.items():
        if PLACEHOLDER_RE.search(value):
            raise InputError(
                "placeholder text is not allowed in efficiency reports",
                details={"field": key},
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
