# SPDX-License-Identifier: Apache-2.0
"""Build and verify scalable v0.3 membership stores."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath

from geno_lewm.data.membership_store import (
    MembershipSourceInput,
    build_membership_store,
    verify_membership_store,
)
from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for

BUILD_SPEC_SCHEMA_VERSION = "geno-lewm.membership-build-spec.v1"


def main(argv: list[str] | None = None) -> int:
    """Run the membership build or verification command."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="build one immutable membership store")
    build_parser.add_argument("--spec-json", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify", help="full-scan one membership store")
    verify_parser.add_argument("--store-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            artifact_id, lineage_path, sources = _load_build_spec(args.spec_json)
            manifest = build_membership_store(
                artifact_id=artifact_id,
                snapshot_lineage_path=lineage_path,
                sources=sources,
                output_dir=args.output_dir,
            )
            payload: Mapping[str, object] = {
                "ok": True,
                "artifact_id": manifest.artifact_id,
                "content_identity": manifest.content_identity,
                "row_count": manifest.row_count,
                "output_dir": str(args.output_dir),
            }
        else:
            payload = verify_membership_store(args.store_dir).to_dict()
    except (GenoLeWMError, OSError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc) if isinstance(exc, GenoLeWMError) else 2
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _load_build_spec(
    path: Path,
) -> tuple[str, Path, tuple[MembershipSourceInput, ...]]:
    try:
        raw: object = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except OSError as exc:
        raise InputError(
            "failed to read membership build spec", details={"path": str(path)}
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "membership build spec JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    payload = _require_mapping(raw, "membership build spec")
    _require_exact_keys(
        payload,
        {"$schema", "schema_version", "artifact_id", "snapshot_lineage", "sources"},
        "membership build spec",
    )
    if payload.get("$schema") != "./membership-build-spec.schema.json":
        raise InputError("membership build spec $schema is not recognized")
    if payload.get("schema_version") != BUILD_SPEC_SCHEMA_VERSION:
        raise InputError("membership build spec schema version is not recognized")
    artifact_id = _require_text(payload.get("artifact_id"), "membership artifact_id")
    lineage_path = _resolve_relative(
        path.parent, payload.get("snapshot_lineage"), "snapshot_lineage"
    )
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise InputError("membership build sources must be a non-empty array")
    sources: list[MembershipSourceInput] = []
    for index, raw_source in enumerate(raw_sources):
        source = _require_mapping(raw_source, f"membership sources[{index}]")
        kind = _require_text(source.get("kind"), f"membership sources[{index}].kind")
        if kind == "gnomad":
            _require_exact_keys(
                source,
                {"kind", "chromosome", "path"},
                f"membership sources[{index}]",
            )
            chromosome = _require_text(
                source.get("chromosome"), f"membership sources[{index}].chromosome"
            )
        elif kind == "clinvar":
            _require_exact_keys(source, {"kind", "path"}, f"membership sources[{index}]")
            chromosome = None
        else:
            raise InputError("membership build source kind must be gnomad or clinvar")
        sources.append(
            MembershipSourceInput(
                kind=kind,
                path=_resolve_relative(path.parent, source.get("path"), f"sources[{index}].path"),
                chromosome=chromosome,
            )
        )
    return artifact_id, lineage_path, tuple(sources)


def _resolve_relative(root: Path, value: object, field: str) -> Path:
    text = _require_text(value, field)
    candidate = Path(text)
    windows = PureWindowsPath(text)
    if candidate.is_absolute() or windows.is_absolute() or windows.drive or ".." in candidate.parts:
        raise InputError(f"membership build {field} must be a relative path without '..'")
    root_resolved = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise InputError(f"membership build {field} resolves outside the spec directory") from exc
    return resolved


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InputError(f"{field} must be an object")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise InputError("duplicate JSON key is not allowed", details={"key": key})
        payload[key] = value
    return payload


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputError(f"{field} must be a non-empty string")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise InputError(
            f"{field} keys do not match the closed schema",
            details={
                "missing": sorted(expected - set(value)),
                "unexpected": sorted(set(value) - expected),
            },
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
