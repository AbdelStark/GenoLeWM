# SPDX-License-Identifier: Apache-2.0
"""Generate measured rollout-state rows from latent rollout examples.

This tool consumes precomputed latent examples: source state, target
state, candidate states for target-rank computation, and a sequence of
window-relative edits. It loads the manifest-backed action encoder and
predictor from a local model directory, runs autoregressive rollout, and
writes ``geno-lewm-rollout-states`` JSONL accepted by
``geno-lewm-rollout``.

It does not run Carbon encoding or construct held-out haplotypes; those
remain upstream benchmark inputs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

from geno_lewm._artifact_sources import (
    ROLLOUT_STATES_GENERATED_BY,
    ROLLOUT_STATES_SCHEMA_VERSION,
)
from geno_lewm._inference import torch_inference_context
from geno_lewm.action import EditType, RelEdit
from geno_lewm.cli._artifact_paths import package_relative_artifact_path
from geno_lewm.encoder import WindowCacheKey
from geno_lewm.encoder.cache import CACHE_SCHEMA_VERSION
from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import Manifest, load_manifest, sha256_file

SCHEMA_VERSION: Final = "1.0.0"
EXAMPLE_SCHEMA_VERSION: Final = "1.3.0"
GENERATED_BY: Final = "tools.release.rollout_state_rows"
INPUT_GENERATED_BY: Final = "tools.release.rollout_state_examples"
ISSUE_REFS: Final = ("#57", "#197")

PredictorFn: TypeAlias = Callable[["RolloutStateExample"], tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class CandidateState:
    """One candidate latent state used for target-rank computation."""

    candidate_id: str
    state: tuple[float, ...]
    state_key: WindowCacheKey


@dataclass(frozen=True, slots=True)
class StateRepresentation:
    """Latent representation fields that must match a checkpoint exactly."""

    encoder_hash: bytes
    state_layer: int
    pool_type: str
    pool_radius: int
    dtype: str
    normalize: bool
    d_state: int

    def to_dict(self) -> dict[str, object]:
        return {
            "encoder_hash": self.encoder_hash.hex(),
            "state_layer": self.state_layer,
            "pool_type": self.pool_type,
            "pool_radius": self.pool_radius,
            "dtype": self.dtype,
            "normalize": self.normalize,
            "d_state": self.d_state,
        }


@dataclass(frozen=True, slots=True)
class RolloutStateExample:
    """One latent rollout example before predictor execution."""

    row_id: str
    split: str
    normalize: bool
    state_representation: StateRepresentation
    source_state: tuple[float, ...]
    target_state: tuple[float, ...]
    edits: tuple[RelEdit, ...]
    candidates: tuple[CandidateState, ...]
    target_candidate_id: str

    @property
    def horizon(self) -> int:
        return len(self.edits)


def load_rollout_state_examples(path: Path) -> tuple[RolloutStateExample, ...]:
    """Load release-shaped latent rollout examples from JSONL."""
    rows: list[RolloutStateExample] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputError("failed to read rollout-state examples JSONL") from exc
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(
                "rollout-state example JSONL row is invalid",
                details={"path": str(path), "line": line_no, "column": exc.colno},
            ) from exc
        rows.append(_parse_example(payload, line_no=line_no))
    if not rows:
        raise InputError("rollout-state examples JSONL must contain at least one row")
    duplicates = _duplicates(example.row_id for example in rows)
    if duplicates:
        raise InputError(
            "rollout-state example ids must be unique",
            details={"duplicates": duplicates},
        )
    return tuple(rows)


def generate_rollout_state_rows(
    examples: tuple[RolloutStateExample, ...],
    *,
    predictor_fn: PredictorFn,
) -> tuple[dict[str, object], ...]:
    """Generate ``geno-lewm-rollout-states`` rows from latent examples."""
    output: list[dict[str, object]] = []
    for example in examples:
        predicted_state = _state_vector(
            predictor_fn(example),
            field="predicted_state",
            line_no=None,
        )
        _require_state_dim(
            predicted_state,
            expected_dim=len(example.target_state),
            field="predicted_state",
            line_no=None,
        )
        if example.normalize:
            _require_unit_state(predicted_state, field="predicted_state", line_no=None)
        target_rank = _rank_target(
            predicted_state,
            candidates=example.candidates,
            target_candidate_id=example.target_candidate_id,
        )
        baseline_rank = _rank_target(
            example.source_state,
            candidates=example.candidates,
            target_candidate_id=example.target_candidate_id,
        )
        output.append(
            {
                "schema_version": ROLLOUT_STATES_SCHEMA_VERSION,
                "generated_by": ROLLOUT_STATES_GENERATED_BY,
                "id": example.row_id,
                "split": example.split,
                "k": example.horizon,
                "source_state": list(example.source_state),
                "predicted_state": list(predicted_state),
                "target_state": list(example.target_state),
                "target_rank": target_rank,
                "baseline_target_rank": baseline_rank,
            }
        )
    return tuple(output)


def write_rollout_state_artifacts(
    *,
    examples_jsonl: Path,
    model_dir: Path,
    artifact_root: Path,
    output_jsonl: Path,
    output_report: Path,
    command: tuple[str, ...] = (),
) -> dict[str, object]:
    """Generate rollout-state JSONL and a companion provenance report."""
    manifest = _load_and_verify_model_manifest(model_dir)
    examples = load_rollout_state_examples(examples_jsonl)
    state_contract_version, model_representation = _model_state_contract(model_dir, manifest)
    _validate_example_state_contract(
        examples,
        state_contract_version=state_contract_version,
        model_representation=model_representation,
    )
    resolved_predictor_fn = _predictor_fn_from_model_dir(model_dir)
    rows = generate_rollout_state_rows(examples, predictor_fn=resolved_predictor_fn)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    report = _build_report(
        examples=examples,
        rows=rows,
        manifest=manifest,
        model_dir=model_dir,
        examples_jsonl=examples_jsonl,
        output_jsonl=output_jsonl,
        output_report=output_report,
        artifact_root=artifact_root,
        command=command,
        state_contract_version=state_contract_version,
        model_representation=model_representation,
    )
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    command = _command_from_args(args)
    try:
        write_rollout_state_artifacts(
            examples_jsonl=args.examples_jsonl,
            model_dir=args.model_dir,
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


def _parse_example(payload: Any, *, line_no: int) -> RolloutStateExample:
    if not isinstance(payload, dict):
        raise InputError("rollout-state examples must be JSON objects", details={"line": line_no})
    schema_version = _required_text(payload, "schema_version", line_no=line_no)
    if schema_version != EXAMPLE_SCHEMA_VERSION:
        raise InputError(
            "unsupported rollout-state example schema_version",
            details={
                "line": line_no,
                "schema_version": schema_version,
                "supported": EXAMPLE_SCHEMA_VERSION,
            },
        )
    _require_example_contract_field(
        payload,
        "cache_schema_version",
        CACHE_SCHEMA_VERSION,
        line_no=line_no,
    )
    _require_example_contract_field(
        payload,
        "cache_physical_encoding",
        "fixed_size_list<float32>",
        line_no=line_no,
    )
    _require_example_contract_field(
        payload,
        "cached_state_value_contract",
        "raw_pooled_v1",
        line_no=line_no,
    )
    generated_by = _required_text(payload, "generated_by", line_no=line_no)
    if generated_by != INPUT_GENERATED_BY:
        raise InputError(
            "rollout-state example generated_by is invalid",
            details={"line": line_no, "expected": INPUT_GENERATED_BY, "observed": generated_by},
        )
    normalize = _example_normalization(payload, schema_version=schema_version, line_no=line_no)
    _require_example_contract_field(
        payload,
        "materialized_state_contract",
        "l2_normalized_v2" if normalize else "legacy_raw_v1",
        line_no=line_no,
    )
    source_state_key = _state_key(
        payload.get("source_state_key"), field="source_state_key", line_no=line_no
    )
    target_state_key = _state_key(
        payload.get("target_state_key"), field="target_state_key", line_no=line_no
    )
    _require_same_state_representation(
        source_state_key,
        target_state_key,
        field="target_state_key",
        line_no=line_no,
    )
    source_state = _state_vector(payload.get("source_state"), field="source_state", line_no=line_no)
    target_state = _state_vector(payload.get("target_state"), field="target_state", line_no=line_no)
    _require_state_dim(
        target_state, expected_dim=len(source_state), field="target_state", line_no=line_no
    )
    edits = _edits(payload.get("edits"), line_no=line_no)
    candidates = _candidate_states(payload.get("candidates"), line_no=line_no)
    target_candidate_id = _required_text(payload, "target_candidate_id", line_no=line_no)
    target_candidates = [
        candidate for candidate in candidates if candidate.candidate_id == target_candidate_id
    ]
    if len(target_candidates) != 1:
        raise InputError(
            "target_candidate_id must identify exactly one candidate",
            details={"line": line_no, "target_candidate_id": target_candidate_id},
        )
    for candidate in candidates:
        _require_same_state_representation(
            source_state_key,
            candidate.state_key,
            field=f"candidate:{candidate.candidate_id}.state_key",
            line_no=line_no,
        )
        _require_state_dim(
            candidate.state,
            expected_dim=len(source_state),
            field=f"candidate:{candidate.candidate_id}",
            line_no=line_no,
        )
    if target_candidates[0].state != target_state:
        raise InputError(
            "target candidate state must match target_state",
            details={"line": line_no, "target_candidate_id": target_candidate_id},
        )
    if target_candidates[0].state_key != target_state_key:
        raise InputError(
            "target candidate state_key must match target_state_key",
            details={"line": line_no, "target_candidate_id": target_candidate_id},
        )
    _validate_state_value_contract(
        normalize=normalize,
        source_state=source_state,
        target_state=target_state,
        candidates=candidates,
        line_no=line_no,
    )
    return RolloutStateExample(
        row_id=_required_text(payload, "id", line_no=line_no),
        split=_required_text(payload, "split", line_no=line_no),
        normalize=normalize,
        state_representation=StateRepresentation(
            encoder_hash=source_state_key.encoder_hash,
            state_layer=source_state_key.state_layer,
            pool_type=source_state_key.pool_type,
            pool_radius=source_state_key.pool_radius,
            dtype=source_state_key.dtype,
            normalize=normalize,
            d_state=len(source_state),
        ),
        source_state=source_state,
        target_state=target_state,
        edits=edits,
        candidates=candidates,
        target_candidate_id=target_candidate_id,
    )


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


def _candidate_states(raw: object, *, line_no: int) -> tuple[CandidateState, ...]:
    if not isinstance(raw, list) or len(raw) < 2:
        raise InputError(
            "candidates must contain at least two candidate states",
            details={"line": line_no},
        )
    candidates: list[CandidateState] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InputError(
                "candidate entries must be JSON objects",
                details={"line": line_no, "index": index},
            )
        candidates.append(
            CandidateState(
                candidate_id=_required_text(item, "id", line_no=line_no),
                state=_state_vector(item.get("state"), field="candidate.state", line_no=line_no),
                state_key=_state_key(
                    item.get("state_key"),
                    field="candidate.state_key",
                    line_no=line_no,
                ),
            )
        )
    duplicates = _duplicates(candidate.candidate_id for candidate in candidates)
    if duplicates:
        raise InputError(
            "candidate ids must be unique",
            details={"line": line_no, "duplicates": duplicates},
        )
    return tuple(candidates)


def _predictor_fn_from_model_dir(model_dir: Path) -> PredictorFn:
    from geno_lewm.deploy.runtime import load_action_predictor_modules
    from geno_lewm.predictor import ARPredictor

    action_encoder, predictor = load_action_predictor_modules(model_dir)
    rollout = cast(Any, ARPredictor(predictor))

    def predict(example: RolloutStateExample) -> tuple[float, ...]:
        action_method = getattr(action_encoder, "encode", None)
        if callable(action_method):
            actions = cast(Callable[[object], object], action_method)([list(example.edits)])
        elif callable(action_encoder):
            actions = cast(Callable[[object], object], action_encoder)([list(example.edits)])
        else:
            raise InputError(
                "action_encoder must be callable or expose encode(edits)",
                details={"type": type(action_encoder).__name__},
            )
        state = _state_tensor_like_actions(example.source_state, actions)
        with torch_inference_context():
            predictions = rollout.rollout_tensor(state, actions)
        final = predictions[0, example.horizon - 1]
        return _state_vector(final, field="predicted_state", line_no=None)

    return predict


def _state_tensor_like_actions(state: tuple[float, ...], actions: object) -> object:
    new_tensor = getattr(actions, "new_tensor", None)
    if not callable(new_tensor):
        torch = importlib.import_module("torch")
        return torch.tensor([list(state)], dtype=torch.float32)
    tensor = cast(Callable[[object], object], new_tensor)([list(state)])
    return tensor


def _load_and_verify_model_manifest(model_dir: Path) -> Manifest:
    manifest_path = model_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    _verify_manifest_artifact(
        model_dir, "predictor", manifest.predictor.file, manifest.predictor.hash
    )
    _verify_manifest_artifact(
        model_dir,
        "action_encoder",
        manifest.action_encoder.file,
        manifest.action_encoder.hash,
    )
    return manifest


def _model_state_contract(
    model_dir: Path,
    manifest: Manifest,
) -> tuple[str, StateRepresentation]:
    from geno_lewm.config import load_config
    from geno_lewm.config._state_contract import encoder_uses_normalized_states

    config_file = manifest.training.config_file
    _verify_manifest_artifact(
        model_dir,
        "training_config",
        config_file,
        manifest.training.hash,
    )
    config = load_config(model_dir / config_file)
    version = config.encoder.state_contract_version
    return version, StateRepresentation(
        encoder_hash=_manifest_hash_bytes(manifest.encoder.hash),
        state_layer=config.encoder.state_layer,
        pool_type=config.encoder.pool_type,
        pool_radius=config.encoder.pool_radius,
        dtype=config.encoder.dtype,
        normalize=encoder_uses_normalized_states(config.encoder),
        d_state=config.predictor.d_state,
    )


def _validate_example_state_contract(
    examples: tuple[RolloutStateExample, ...],
    *,
    state_contract_version: str,
    model_representation: StateRepresentation,
) -> None:
    observed = {example.state_representation for example in examples}
    if len(observed) != 1:
        raise InputError(
            "rollout-state examples must use one state representation",
            details={"observed": [item.to_dict() for item in observed]},
        )
    example_representation = next(iter(observed))
    if example_representation != model_representation:
        raise InputError(
            "rollout-state example representation does not match model state contract",
            details={
                "model_state_contract_version": state_contract_version,
                "example": example_representation.to_dict(),
                "model": model_representation.to_dict(),
            },
        )


def _verify_manifest_artifact(
    model_dir: Path,
    artifact: str,
    file_name: str,
    expected_hash: str,
) -> None:
    path = (model_dir / file_name).resolve()
    try:
        path.relative_to(model_dir.resolve())
    except ValueError as exc:
        raise InputError(
            "manifest artifact paths must stay inside model_dir",
            details={"artifact": artifact, "path": file_name, "model_dir": str(model_dir)},
        ) from exc
    if not path.is_file():
        raise InputError(
            "manifest artifact is missing",
            details={"artifact": artifact, "path": str(path)},
        )
    observed = sha256_file(path)
    if observed != expected_hash:
        raise InputError(
            "manifest artifact hash mismatch",
            details={"artifact": artifact, "expected": expected_hash, "observed": observed},
        )


def _build_report(
    *,
    examples: tuple[RolloutStateExample, ...],
    rows: tuple[dict[str, object], ...],
    manifest: Manifest,
    model_dir: Path,
    examples_jsonl: Path,
    output_jsonl: Path,
    output_report: Path,
    artifact_root: Path,
    command: tuple[str, ...],
    state_contract_version: str,
    model_representation: StateRepresentation,
) -> dict[str, object]:
    splits = sorted({example.split for example in examples})
    horizons = sorted({example.horizon for example in examples})
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": _utc_now(),
        "ok": True,
        "model_id": manifest.model_id(),
        "model_release": manifest.release_id,
        "state_contract": {
            "version": state_contract_version,
            **model_representation.to_dict(),
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_physical_encoding": "fixed_size_list<float32>",
            "cached_state_value_contract": "raw_pooled_v1",
            "validated_against_examples": True,
        },
        "command": list(command),
        "issue_refs": list(ISSUE_REFS),
        "inputs": {
            "model_manifest": _file_identity(
                model_dir / "manifest.json",
                artifact_root=artifact_root,
                label="model_manifest",
            ),
            "examples_jsonl": _file_identity(
                examples_jsonl,
                artifact_root=artifact_root,
                label="examples_jsonl",
            ),
        },
        "outputs": {
            "rollout_states_jsonl": _file_identity(
                output_jsonl,
                artifact_root=artifact_root,
                label="rollout_states_jsonl",
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
        "negative_findings": [
            (
                "This report does not establish clinical utility, privacy assurance, "
                "deployment readiness, or rollout-speed target closure."
            )
        ],
        "limitations": [
            (
                "Input examples must already contain measured source, target, and "
                "candidate latent states. This tool does not run Carbon encoding."
            ),
            (
                "Input examples must already encode held-out membership and benchmark "
                "split assignment. This tool does not construct held-out haplotypes."
            ),
        ],
        "claim_boundary": (
            "This report only proves that manifest-backed action encoder and predictor "
            "execution produced rollout-state rows from supplied measured latent examples."
        ),
    }


def _example_normalization(
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


def _require_example_contract_field(
    payload: dict[str, object],
    field: str,
    expected: str,
    *,
    line_no: int,
) -> None:
    observed = payload.get(field)
    if observed != expected:
        raise InputError(
            f"{field} does not match the rollout-state example contract",
            details={"line": line_no, "expected": expected, "observed": observed},
        )


def _state_key(raw: object, *, field: str, line_no: int) -> WindowCacheKey:
    if not isinstance(raw, dict):
        raise InputError(f"{field} must be a state-key object", details={"line": line_no})
    return WindowCacheKey(
        window_hash=_hex32(raw.get("window_hash"), field=f"{field}.window_hash", line_no=line_no),
        encoder_hash=_hex32(
            raw.get("encoder_hash"), field=f"{field}.encoder_hash", line_no=line_no
        ),
        state_layer=_mapping_int(raw, "state_layer", field=field, line_no=line_no),
        pool_type=_mapping_text(raw, "pool_type", field=field, line_no=line_no),
        pool_radius=_mapping_int(raw, "pool_radius", field=field, line_no=line_no),
        center_token=_mapping_optional_int(
            raw,
            "center_token",
            field=field,
            line_no=line_no,
        ),
        dtype=_mapping_text(raw, "dtype", field=field, line_no=line_no),
    )


def _require_same_state_representation(
    expected: WindowCacheKey,
    observed: WindowCacheKey,
    *,
    field: str,
    line_no: int,
) -> None:
    expected_fields = _key_representation(expected)
    observed_fields = _key_representation(observed)
    if observed_fields != expected_fields:
        raise InputError(
            "rollout-state example keys must share one state representation",
            details={
                "line": line_no,
                "field": field,
                "expected": expected_fields,
                "observed": observed_fields,
            },
        )


def _key_representation(key: WindowCacheKey) -> dict[str, object]:
    return {
        "encoder_hash": key.encoder_hash.hex(),
        "state_layer": key.state_layer,
        "pool_type": key.pool_type,
        "pool_radius": key.pool_radius,
        "center_token": key.center_token,
        "dtype": key.dtype,
    }


def _hex32(raw: object, *, field: str, line_no: int) -> bytes:
    if not isinstance(raw, str):
        raise InputError(f"{field} must be a 64-character hex string", details={"line": line_no})
    try:
        value = bytes.fromhex(raw)
    except ValueError as exc:
        raise InputError(
            f"{field} must be a 64-character hex string", details={"line": line_no}
        ) from exc
    if len(value) != 32:
        raise InputError(f"{field} must be a 64-character hex string", details={"line": line_no})
    return value


def _mapping_text(raw: dict[object, object], name: str, *, field: str, line_no: int) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{field}.{name} must be a non-empty string", details={"line": line_no})
    return value.strip()


def _mapping_int(raw: dict[object, object], name: str, *, field: str, line_no: int) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{field}.{name} must be an integer", details={"line": line_no})
    return value


def _mapping_optional_int(
    raw: dict[object, object],
    name: str,
    *,
    field: str,
    line_no: int,
) -> int | None:
    value = raw.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{field}.{name} must be an integer or null", details={"line": line_no})
    return value


def _manifest_hash_bytes(value: str) -> bytes:
    raw = value.removeprefix("sha256:")
    try:
        parsed = bytes.fromhex(raw)
    except ValueError as exc:
        raise InputError("model manifest encoder hash must be SHA-256 hex") from exc
    if len(parsed) != 32:
        raise InputError("model manifest encoder hash must be SHA-256 hex")
    return parsed


def _validate_state_value_contract(
    *,
    normalize: bool,
    source_state: tuple[float, ...],
    target_state: tuple[float, ...],
    candidates: tuple[CandidateState, ...],
    line_no: int,
) -> None:
    if not normalize:
        return
    values = [
        ("source_state", source_state),
        ("target_state", target_state),
        *((f"candidate:{candidate.candidate_id}", candidate.state) for candidate in candidates),
    ]
    for field, state in values:
        _require_unit_state(state, field=field, line_no=line_no)


def _require_unit_state(
    state: tuple[float, ...],
    *,
    field: str,
    line_no: int | None,
) -> None:
    norm = math.hypot(*state)
    if abs(norm - 1.0) > 1.0e-5:
        raise InputError(
            f"{field} must be unit norm under the normalized rollout-state contract",
            details={**_line(line_no), "field": field, "norm": norm},
        )


def _rank_target(
    query_state: tuple[float, ...],
    *,
    candidates: tuple[CandidateState, ...],
    target_candidate_id: str,
) -> int:
    ranked = sorted(
        ((_l2(query_state, candidate.state), candidate.candidate_id) for candidate in candidates),
        key=lambda item: (item[0], item[1]),
    )
    for index, (_distance, candidate_id) in enumerate(ranked, start=1):
        if candidate_id == target_candidate_id:
            return index
    raise InputError("target candidate is missing from candidate states")


def _state_vector(raw: object, *, field: str, line_no: int | None) -> tuple[float, ...]:
    materialized = raw
    for attr in ("detach", "cpu", "flatten"):
        method = getattr(materialized, attr, None)
        if callable(method):
            materialized = cast(Callable[[], object], method)()
    tolist = getattr(materialized, "tolist", None)
    if callable(tolist):
        materialized = cast(Callable[[], object], tolist)()
    if not isinstance(materialized, list | tuple) or not materialized:
        raise InputError(f"{field} must be a non-empty numeric vector", details=_line(line_no))
    values: list[float] = []
    for index, value in enumerate(materialized):
        if isinstance(value, bool) or not isinstance(value, int | float):
            details = {"index": index, "type": type(value).__name__, **_line(line_no)}
            raise InputError(f"{field} entries must be numeric", details=details)
        number = float(value)
        if not math.isfinite(number):
            raise InputError(
                f"{field} entries must be finite", details={"index": index, **_line(line_no)}
            )
        values.append(number)
    return tuple(values)


def _require_state_dim(
    values: tuple[float, ...],
    *,
    expected_dim: int,
    field: str,
    line_no: int | None,
) -> None:
    if len(values) != expected_dim:
        raise InputError(
            "rollout-state vectors must share the same dimension",
            details={
                "field": field,
                "expected_dim": expected_dim,
                "observed_dim": len(values),
                **_line(line_no),
            },
        )


def _l2(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    _require_state_dim(b, expected_dim=len(a), field="candidate_state", line_no=None)
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b, strict=True)))


def _file_identity(path: Path, *, artifact_root: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise InputError("rollout-state artifact does not exist", details={"path": str(path)})
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
        outside_message="rollout-state artifact path must stay inside artifact_root",
        root_detail="artifact_root",
        remediation="place rollout-state artifacts under the release package root",
    )


def _required_text(raw: dict[str, object], field: str, *, line_no: int) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{field} must be a non-empty string", details=_line(line_no))
    return value.strip()


def _required_int(raw: dict[str, object], field: str, *, line_no: int) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{field} must be an integer", details=_line(line_no))
    return value


def _duplicates(values: Any) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(str(value))
        seen.add(str(value))
    return sorted(duplicates)


def _line(line_no: int | None) -> dict[str, int]:
    return {} if line_no is None else {"line": line_no}


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate rollout-state JSONL from measured latent rollout examples.",
    )
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def _command_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    return (
        "python",
        "-m",
        "tools.release.rollout_state_rows",
        "--examples-jsonl",
        str(args.examples_jsonl),
        "--model-dir",
        str(args.model_dir),
        "--artifact-root",
        str(args.artifact_root),
        "--output-jsonl",
        str(args.output_jsonl),
        "--output-report",
        str(args.output_report),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
