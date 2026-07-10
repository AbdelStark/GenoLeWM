# SPDX-License-Identifier: Apache-2.0
"""Build rollout-state examples from cache-backed latent state specs.

This tool consumes explicit benchmark example specs that reference measured
latent states in the documented window-embedding cache. It reads source,
target, and candidate embeddings by cache key, then writes the
``tools.release.rollout_state_examples`` JSONL accepted by
``tools.release.rollout_state_rows``.

It does not run Carbon encoding, choose held-out haplotypes, or synthesize
edit chains; those remain upstream benchmark inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from geno_lewm.action import EditType, RelEdit
from geno_lewm.cli._artifact_paths import package_relative_artifact_path
from geno_lewm.encoder._normalization import l2_normalize_state
from geno_lewm.encoder.cache import (
    CACHE_SCHEMA_VERSION,
    INDEX_DB_NAME,
    WindowCacheKey,
    read_embedding,
)
from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file

SCHEMA_VERSION: Final = "1.0.0"
SPEC_SCHEMA_VERSION: Final = "1.2.0"
EXAMPLE_SCHEMA_VERSION: Final = "1.2.0"
GENERATED_BY: Final = "tools.release.rollout_state_examples"
SPEC_GENERATED_BY: Final = "tools.release.rollout_state_example_specs"
ISSUE_REFS: Final = ("#57", "#197")


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One candidate state reference in a rollout example spec."""

    candidate_id: str
    state_key: WindowCacheKey


@dataclass(frozen=True, slots=True)
class RolloutStateExampleSpec:
    """One cache-keyed rollout example before embeddings are loaded."""

    row_id: str
    split: str
    normalize: bool
    source_state_key: WindowCacheKey
    target_state_key: WindowCacheKey
    edits: tuple[RelEdit, ...]
    candidates: tuple[CandidateSpec, ...]
    target_candidate_id: str

    @property
    def horizon(self) -> int:
        return len(self.edits)


def load_rollout_state_example_specs(path: Path) -> tuple[RolloutStateExampleSpec, ...]:
    """Load cache-keyed rollout-state example specs from JSONL."""
    rows: list[RolloutStateExampleSpec] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputError("failed to read rollout-state example spec JSONL") from exc
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(
                "rollout-state example spec JSONL row is invalid",
                details={"path": str(path), "line": line_no, "column": exc.colno},
            ) from exc
        rows.append(_parse_spec(payload, line_no=line_no))
    if not rows:
        raise InputError("rollout-state example spec JSONL must contain at least one row")
    duplicates = _duplicates(spec.row_id for spec in rows)
    if duplicates:
        raise InputError(
            "rollout-state example spec ids must be unique",
            details={"duplicates": duplicates},
        )
    return tuple(rows)


def generate_rollout_state_examples(
    specs: tuple[RolloutStateExampleSpec, ...],
    *,
    cache_dir: Path,
) -> tuple[dict[str, object], ...]:
    """Resolve cache keys and return normalized rollout-state example rows."""
    rows: list[dict[str, object]] = []
    for spec in specs:
        _require_matching_state_key(
            spec.source_state_key,
            spec.target_state_key,
            field="target_state_key",
            row_id=spec.row_id,
        )
        source_state = _read_state(
            cache_dir,
            spec.source_state_key,
            label="source_state",
            normalize=spec.normalize,
        )
        target_state = _read_state(
            cache_dir,
            spec.target_state_key,
            label="target_state",
            normalize=spec.normalize,
        )
        _require_state_dim(
            target_state,
            expected_dim=len(source_state),
            field="target_state",
        )
        target_candidates = [
            candidate
            for candidate in spec.candidates
            if candidate.candidate_id == spec.target_candidate_id
        ]
        if len(target_candidates) != 1:
            raise InputError(
                "target_candidate_id must identify exactly one candidate",
                details={"id": spec.row_id, "target_candidate_id": spec.target_candidate_id},
            )
        if target_candidates[0].state_key != spec.target_state_key:
            raise InputError(
                "target candidate key must match target_state_key",
                details={"id": spec.row_id, "target_candidate_id": spec.target_candidate_id},
            )
        candidate_rows: list[dict[str, object]] = []
        for candidate in spec.candidates:
            _require_matching_state_key(
                spec.source_state_key,
                candidate.state_key,
                field=f"candidate:{candidate.candidate_id}.state_key",
                row_id=spec.row_id,
            )
            state = _read_state(
                cache_dir,
                candidate.state_key,
                label=f"candidate:{candidate.candidate_id}",
                normalize=spec.normalize,
            )
            _require_state_dim(
                state,
                expected_dim=len(source_state),
                field=f"candidate:{candidate.candidate_id}",
            )
            candidate_rows.append(
                {
                    "id": candidate.candidate_id,
                    "state": list(state),
                    "state_key": _state_key_dict(candidate.state_key),
                }
            )
        rows.append(
            {
                "schema_version": EXAMPLE_SCHEMA_VERSION,
                "generated_by": GENERATED_BY,
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "cached_state_value_contract": "raw_pooled_v1",
                "materialized_state_contract": (
                    "l2_normalized_v2" if spec.normalize else "legacy_raw_v1"
                ),
                "id": spec.row_id,
                "split": spec.split,
                "normalize": spec.normalize,
                "source_state": list(source_state),
                "source_state_key": _state_key_dict(spec.source_state_key),
                "target_state": list(target_state),
                "target_state_key": _state_key_dict(spec.target_state_key),
                "target_candidate_id": spec.target_candidate_id,
                "edits": [_edit_dict(edit) for edit in spec.edits],
                "candidates": candidate_rows,
            }
        )
    return tuple(rows)


def write_rollout_state_example_artifacts(
    *,
    spec_jsonl: Path,
    cache_dir: Path,
    artifact_root: Path,
    output_jsonl: Path,
    output_report: Path,
    command: tuple[str, ...] = (),
) -> dict[str, object]:
    """Write rollout-state examples JSONL and a companion provenance report."""
    _require_cache_index(cache_dir)
    specs = load_rollout_state_example_specs(spec_jsonl)
    rows = generate_rollout_state_examples(specs, cache_dir=cache_dir)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    report = _build_report(
        specs=specs,
        rows=rows,
        spec_jsonl=spec_jsonl,
        cache_dir=cache_dir,
        artifact_root=artifact_root,
        output_jsonl=output_jsonl,
        output_report=output_report,
        command=command,
    )
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    command = _command_from_args(args)
    try:
        write_rollout_state_example_artifacts(
            spec_jsonl=args.spec_jsonl,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            output_jsonl=args.output_jsonl,
            output_report=args.output_report,
            command=command,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(f"wrote {args.output_jsonl}\n")
    sys.stdout.write(f"wrote {args.output_report}\n")
    return 0


def _parse_spec(payload: Any, *, line_no: int) -> RolloutStateExampleSpec:
    if not isinstance(payload, dict):
        raise InputError(
            "rollout-state example specs must be JSON objects",
            details={"line": line_no},
        )
    schema_version = _required_text(payload, "schema_version", line_no=line_no)
    if schema_version != SPEC_SCHEMA_VERSION:
        raise InputError(
            "unsupported rollout-state example spec schema_version",
            details={
                "line": line_no,
                "schema_version": schema_version,
                "supported": SPEC_SCHEMA_VERSION,
            },
        )
    _require_contract_field(
        payload,
        "cache_schema_version",
        CACHE_SCHEMA_VERSION,
        line_no=line_no,
    )
    _require_contract_field(
        payload,
        "cached_state_value_contract",
        "raw_pooled_v1",
        line_no=line_no,
    )
    generated_by = _required_text(payload, "generated_by", line_no=line_no)
    if generated_by != SPEC_GENERATED_BY:
        raise InputError(
            "rollout-state example spec generated_by is invalid",
            details={"line": line_no, "expected": SPEC_GENERATED_BY, "observed": generated_by},
        )
    candidates = _candidate_specs(payload.get("candidates"), line_no=line_no)
    normalize = _spec_normalization(payload, schema_version=schema_version, line_no=line_no)
    return RolloutStateExampleSpec(
        row_id=_required_text(payload, "id", line_no=line_no),
        split=_required_text(payload, "split", line_no=line_no),
        normalize=normalize,
        source_state_key=_state_key(payload.get("source_state_key"), line_no=line_no),
        target_state_key=_state_key(payload.get("target_state_key"), line_no=line_no),
        edits=_edits(payload.get("edits"), line_no=line_no),
        candidates=candidates,
        target_candidate_id=_required_text(payload, "target_candidate_id", line_no=line_no),
    )


def _candidate_specs(raw: object, *, line_no: int) -> tuple[CandidateSpec, ...]:
    if not isinstance(raw, list) or len(raw) < 2:
        raise InputError(
            "candidates must contain at least two candidate state keys",
            details={"line": line_no},
        )
    candidates: list[CandidateSpec] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InputError(
                "candidate entries must be JSON objects",
                details={"line": line_no, "index": index},
            )
        candidates.append(
            CandidateSpec(
                candidate_id=_required_text(item, "id", line_no=line_no),
                state_key=_state_key(item.get("state_key"), line_no=line_no),
            )
        )
    duplicates = _duplicates(candidate.candidate_id for candidate in candidates)
    if duplicates:
        raise InputError(
            "candidate ids must be unique",
            details={"line": line_no, "duplicates": duplicates},
        )
    return tuple(candidates)


def _edits(raw: object, *, line_no: int) -> tuple[RelEdit, ...]:
    if not isinstance(raw, list) or not raw:
        raise InputError("edits must be a non-empty JSON list", details={"line": line_no})
    edits: list[RelEdit] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InputError(
                "edit entries must be JSON objects",
                details={"line": line_no, "index": index},
            )
        edit_type_value = _required_int(item, "edit_type", line_no=line_no)
        try:
            edit_type = EditType(edit_type_value)
        except ValueError as exc:
            raise InputError(
                "edit_type is not a supported EditType",
                details={"line": line_no, "index": index, "edit_type": edit_type_value},
            ) from exc
        edits.append(
            RelEdit(
                rel_pos=_required_int(item, "rel_pos", line_no=line_no),
                edit_type=edit_type,
                ref_bases=_required_text(item, "ref_bases", line_no=line_no),
                alt_bases=_required_text(item, "alt_bases", line_no=line_no),
            )
        )
    return tuple(edits)


def _state_key(raw: object, *, line_no: int) -> WindowCacheKey:
    if not isinstance(raw, dict):
        raise InputError("state keys must be JSON objects", details={"line": line_no})
    return WindowCacheKey(
        window_hash=_required_hex32(raw, "window_hash", line_no=line_no),
        encoder_hash=_required_hex32(raw, "encoder_hash", line_no=line_no),
        state_layer=_required_int(raw, "state_layer", line_no=line_no),
        pool_type=_required_text(raw, "pool_type", line_no=line_no),
        pool_radius=_required_int(raw, "pool_radius", line_no=line_no),
        center_token=_optional_int(raw, "center_token", line_no=line_no),
        dtype=_required_text(raw, "dtype", line_no=line_no),
    )


def _read_state(
    cache_dir: Path,
    key: WindowCacheKey,
    *,
    label: str,
    normalize: bool,
) -> tuple[float, ...]:
    value = read_embedding(cache_dir, key)
    if value is None:
        raise InputError(
            "rollout-state example references a missing cache embedding",
            details={"field": label, "state_key": _state_key_dict(key)},
        )
    state = _state_vector(value, field=label)
    return l2_normalize_state(state) if normalize else state


def _require_matching_state_key(
    expected: WindowCacheKey,
    observed: WindowCacheKey,
    *,
    field: str,
    row_id: str,
) -> None:
    expected_fields = _state_representation(expected)
    observed_fields = _state_representation(observed)
    if observed_fields != expected_fields:
        raise InputError(
            "rollout-state spec keys must share one state representation",
            details={
                "id": row_id,
                "field": field,
                "expected": expected_fields,
                "observed": observed_fields,
            },
        )


def _state_representation(key: WindowCacheKey) -> dict[str, object]:
    return {
        "encoder_hash": key.encoder_hash.hex(),
        "state_layer": key.state_layer,
        "pool_type": key.pool_type,
        "pool_radius": key.pool_radius,
        "center_token": key.center_token,
        "dtype": key.dtype,
    }


def _state_vector(raw: object, *, field: str) -> tuple[float, ...]:
    if not isinstance(raw, list | tuple) or not raw:
        raise InputError(f"{field} must be a non-empty numeric vector")
    values: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise InputError(
                f"{field} entries must be numeric",
                details={"index": index, "type": type(value).__name__},
            )
        number = float(value)
        if not math.isfinite(number):
            raise InputError(f"{field} entries must be finite", details={"index": index})
        values.append(number)
    return tuple(values)


def _require_state_dim(values: tuple[float, ...], *, expected_dim: int, field: str) -> None:
    if len(values) != expected_dim:
        raise InputError(
            "rollout-state example vectors must share the same dimension",
            details={
                "field": field,
                "expected_dim": expected_dim,
                "observed_dim": len(values),
            },
        )


def _require_cache_index(cache_dir: Path) -> None:
    index = cache_dir / "embeddings" / INDEX_DB_NAME
    if not index.is_file():
        raise InputError(
            "rollout-state example cache index is missing",
            details={"cache_dir": str(cache_dir), "index": str(index)},
            remediation="run geno-lewm-cache-windows --reindex for the staged cache",
        )


def _build_report(
    *,
    specs: tuple[RolloutStateExampleSpec, ...],
    rows: tuple[dict[str, object], ...],
    spec_jsonl: Path,
    cache_dir: Path,
    artifact_root: Path,
    output_jsonl: Path,
    output_report: Path,
    command: tuple[str, ...],
) -> dict[str, object]:
    splits = sorted({spec.split for spec in specs})
    horizons = sorted({spec.horizon for spec in specs})
    unique_state_keys = (
        {_state_key_id(spec.source_state_key) for spec in specs}
        | {_state_key_id(spec.target_state_key) for spec in specs}
        | {_state_key_id(candidate.state_key) for spec in specs for candidate in spec.candidates}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": _utc_now(),
        "ok": True,
        "command": list(command),
        "issue_refs": list(ISSUE_REFS),
        "inputs": {
            "spec_jsonl": _file_identity(
                spec_jsonl,
                artifact_root=artifact_root,
                label="spec_jsonl",
            ),
            "cache_dir": {
                "path": _artifact_path(
                    cache_dir,
                    artifact_root=artifact_root,
                    label="cache_dir",
                )
            },
            "cache_index": _file_identity(
                cache_dir / "embeddings" / INDEX_DB_NAME,
                artifact_root=artifact_root,
                label="cache_index",
            ),
        },
        "outputs": {
            "examples_jsonl": _file_identity(
                output_jsonl,
                artifact_root=artifact_root,
                label="examples_jsonl",
            ),
            "report": {
                "path": _artifact_path(
                    output_report,
                    artifact_root=artifact_root,
                    label="output_report",
                )
            },
        },
        "rows": len(rows),
        "splits": splits,
        "horizons": horizons,
        "normalization_views": sorted({spec.normalize for spec in specs}),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cached_state_value_contract": "raw_pooled_v1",
        "unique_cache_state_keys": len(unique_state_keys),
        "negative_findings": [
            (
                "This report does not establish clinical utility, privacy assurance, "
                "deployment readiness, rollout-speed target closure, or model-quality claims."
            )
        ],
        "limitations": [
            (
                "Input specs must already identify held-out benchmark membership and "
                "candidate sets. This tool does not construct held-out haplotypes."
            ),
            (
                "The cache must contain raw pooled latent states from a compatible encoder run. "
                "Normalization is an explicit spec-level view applied after cache lookup."
            ),
        ],
        "claim_boundary": (
            "This report only proves that cache-backed latent states were resolved into "
            "rollout-state examples for downstream manifest-backed predictor execution."
        ),
    }


def _file_identity(path: Path, *, artifact_root: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise InputError("artifact file is missing", details={"artifact": label, "path": str(path)})
    return {
        "path": _artifact_path(path, artifact_root=artifact_root, label=label),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _artifact_path(path: Path, *, artifact_root: Path, label: str) -> str:
    return package_relative_artifact_path(
        path,
        root_dir=artifact_root,
        label=label,
        outside_message="rollout-state example artifact path must stay under artifact_root",
        root_detail="artifact_root",
        remediation="stage rollout example inputs and outputs under the release artifact root",
    )


def _state_key_dict(key: WindowCacheKey) -> dict[str, object]:
    return {
        "window_hash": key.window_hash.hex(),
        "encoder_hash": key.encoder_hash.hex(),
        "state_layer": key.state_layer,
        "pool_type": key.pool_type,
        "pool_radius": key.pool_radius,
        "center_token": key.center_token,
        "dtype": key.dtype,
    }


def _state_key_id(key: WindowCacheKey) -> str:
    return json.dumps(_state_key_dict(key), sort_keys=True, separators=(",", ":"))


def _edit_dict(edit: RelEdit) -> dict[str, object]:
    return {
        "rel_pos": edit.rel_pos,
        "edit_type": int(edit.edit_type),
        "ref_bases": edit.ref_bases,
        "alt_bases": edit.alt_bases,
    }


def _required_hex32(payload: dict[str, object], field: str, *, line_no: int) -> bytes:
    value = _required_text(payload, field, line_no=line_no)
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise InputError(
            f"{field} must be a 32-byte hex digest",
            details={"line": line_no, "field": field},
        ) from exc
    if len(decoded) != 32:
        raise InputError(
            f"{field} must be a 32-byte hex digest",
            details={"line": line_no, "field": field, "observed_bytes": len(decoded)},
        )
    return decoded


def _required_text(payload: dict[str, object], field: str, *, line_no: int) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise InputError(f"{field} must be a non-empty string", details={"line": line_no})
    return value


def _required_int(payload: dict[str, object], field: str, *, line_no: int) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{field} must be an integer", details={"line": line_no})
    return value


def _optional_int(payload: dict[str, object], field: str, *, line_no: int) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{field} must be an integer or null", details={"line": line_no})
    return value


def _spec_normalization(
    payload: dict[str, object],
    *,
    schema_version: str,
    line_no: int,
) -> bool:
    del schema_version
    value = payload.get("normalize")
    if not isinstance(value, bool):
        raise InputError("normalize must be boolean", details={"line": line_no})
    return value


def _require_contract_field(
    payload: dict[str, object],
    field: str,
    expected: str,
    *,
    line_no: int,
) -> None:
    observed = payload.get(field)
    if observed != expected:
        raise InputError(
            f"{field} does not match the rollout-state contract",
            details={"line": line_no, "expected": expected, "observed": observed},
        )


def _duplicates(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            duplicates.add(text)
        seen.add(text)
    return sorted(duplicates)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-jsonl", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    return parser


def _command_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    return (
        "python",
        "-m",
        "tools.release.rollout_state_examples",
        "--spec-jsonl",
        str(args.spec_jsonl),
        "--cache-dir",
        str(args.cache_dir),
        "--artifact-root",
        str(args.artifact_root),
        "--output-jsonl",
        str(args.output_jsonl),
        "--output-report",
        str(args.output_report),
    )


if __name__ == "__main__":
    raise SystemExit(main())
