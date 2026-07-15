# SPDX-License-Identifier: Apache-2.0
"""Verify the pinned Carbon runtime content lock without model downloads."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from geno_lewm.provenance import canonical_json_sha256, sha256_file

SCHEMA_VERSION: Final = "geno-lewm.carbon-runtime-content-lock.v1"
RUNTIME_HASH_CONTRACT: Final = "geno_lewm.encoder_runtime.v1"
CARBON_REPOSITORY: Final = "HuggingFaceBio/Carbon-500M"
CARBON_REVISION: Final = "5d31d59b3c845b288a13aedb1358934196852eec"
RUNTIME_FILE_PATHS: Final = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.py",
    "added_tokens.json",
    "dna_config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "vocab.json",
)
IMPLEMENTATION_FILE_PATHS: Final = (
    "geno_lewm/_inference.py",
    "geno_lewm/encoder/_canonical.py",
    "geno_lewm/encoder/_dna_tokenizer.py",
    "geno_lewm/encoder/_identity.py",
    "geno_lewm/encoder/_normalization.py",
    "geno_lewm/encoder/carbon.py",
    "geno_lewm/encoder/pooling.py",
    "geno_lewm/encoder/windowing.py",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_TOP_LEVEL_KEYS = frozenset(
    {
        "$schema",
        "schema_version",
        "model",
        "weights",
        "runtime_hash_contract",
        "runtime_files",
        "implementation_files",
        "runtime_hash",
        "correction_receipt",
        "claim_boundary",
    }
)


class CarbonRuntimeLockError(ValueError):
    """Raised when the offline Carbon content lock is incomplete or drifts."""


def verify_carbon_runtime_lock(
    lock_path: Path,
    *,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Recompute implementation identities and the canonical runtime hash offline."""
    root = (repository_root or _REPOSITORY_ROOT).resolve()
    payload = _read_lock(lock_path)
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "content lock")
    _require_equal(
        payload.get("$schema"), "./carbon-500m-runtime-content-lock.schema.json", "$schema"
    )
    _require_equal(payload.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require_equal(
        payload.get("runtime_hash_contract"), RUNTIME_HASH_CONTRACT, "runtime_hash_contract"
    )

    model = _require_mapping(payload.get("model"), "model")
    _require_exact_keys(model, frozenset({"repository", "revision"}), "model")
    _require_equal(model.get("repository"), CARBON_REPOSITORY, "model.repository")
    _require_equal(model.get("revision"), CARBON_REVISION, "model.revision")

    weights = _require_mapping(payload.get("weights"), "weights")
    _require_exact_keys(weights, frozenset({"artifact", "sha256"}), "weights")
    _require_equal(weights.get("artifact"), "model.safetensors", "weights.artifact")
    weights_hash = _require_sha256(weights.get("sha256"), "weights.sha256")

    runtime_files = _require_identity_entries(
        payload.get("runtime_files"),
        expected_paths=RUNTIME_FILE_PATHS,
        field="runtime_files",
    )
    locked_implementations = _require_identity_entries(
        payload.get("implementation_files"),
        expected_paths=IMPLEMENTATION_FILE_PATHS,
        field="implementation_files",
    )
    observed_implementations: list[dict[str, str]] = []
    for locked in locked_implementations:
        relative = locked["path"]
        candidate = _resolve_repository_file(root, relative)
        observed = sha256_file(candidate)
        if observed != locked["sha256"]:
            raise CarbonRuntimeLockError(
                f"implementation file hash drifted for {relative}: "
                f"expected {locked['sha256']}, observed {observed}"
            )
        observed_implementations.append({"path": relative, "sha256": observed})

    observed_runtime_hash = canonical_json_sha256(
        {
            "contract": RUNTIME_HASH_CONTRACT,
            "weights_hash": weights_hash,
            "runtime_files": runtime_files,
            "implementation_files": observed_implementations,
        }
    )
    expected_runtime_hash = _require_sha256(payload.get("runtime_hash"), "runtime_hash")
    if observed_runtime_hash != expected_runtime_hash:
        raise CarbonRuntimeLockError(
            "canonical runtime hash drifted: "
            f"expected {expected_runtime_hash}, observed {observed_runtime_hash}"
        )

    return {
        "implementation_file_count": len(observed_implementations),
        "model_revision": CARBON_REVISION,
        "runtime_file_count": len(runtime_files),
        "runtime_hash": observed_runtime_hash,
        "schema_version": SCHEMA_VERSION,
        "verified": True,
        "weights_hash": weights_hash,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "lock",
        nargs="?",
        type=Path,
        default=Path("configs/data_v03/carbon-500m-runtime-content-lock.json"),
    )
    parser.add_argument("--repository-root", type=Path, default=_REPOSITORY_ROOT)
    args = parser.parse_args(argv)
    try:
        report = verify_carbon_runtime_lock(
            args.lock,
            repository_root=args.repository_root,
        )
    except (OSError, json.JSONDecodeError, CarbonRuntimeLockError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


def _read_lock(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_bytes(), object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise CarbonRuntimeLockError(f"content lock is invalid JSON: {exc}") from exc
    return _require_mapping(payload, "content lock")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise CarbonRuntimeLockError(f"content lock contains duplicate key {key!r}")
        payload[key] = value
    return payload


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CarbonRuntimeLockError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise CarbonRuntimeLockError(f"{field} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    field: str,
) -> None:
    observed = frozenset(value)
    if observed != expected:
        raise CarbonRuntimeLockError(
            f"{field} must use the closed key set; "
            f"expected {sorted(expected)}, observed {sorted(observed)}"
        )


def _require_equal(observed: object, expected: object, field: str) -> None:
    if observed != expected:
        raise CarbonRuntimeLockError(
            f"{field} drifted: expected {expected!r}, observed {observed!r}"
        )


def _require_sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise CarbonRuntimeLockError(f"{field} must be a lowercase sha256:<64hex> digest")
    return value


def _require_identity_entries(
    value: object,
    *,
    expected_paths: Sequence[str],
    field: str,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise CarbonRuntimeLockError(f"{field} must be an array")
    entries: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        entry = _require_mapping(raw, f"{field}[{index}]")
        _require_exact_keys(entry, frozenset({"path", "sha256"}), f"{field}[{index}]")
        path = entry.get("path")
        if type(path) is not str or not path:
            raise CarbonRuntimeLockError(f"{field}[{index}].path must be non-empty text")
        entries.append(
            {
                "path": path,
                "sha256": _require_sha256(entry.get("sha256"), f"{field}[{index}].sha256"),
            }
        )
    observed_paths = tuple(entry["path"] for entry in entries)
    if observed_paths != tuple(expected_paths):
        raise CarbonRuntimeLockError(
            f"{field} inventory drifted: expected {tuple(expected_paths)!r}, "
            f"observed {observed_paths!r}"
        )
    return entries


def _resolve_repository_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CarbonRuntimeLockError(
            f"implementation file escapes repository root: {relative!r}"
        ) from exc
    if not candidate.is_file():
        raise CarbonRuntimeLockError(f"implementation file is missing: {relative!r}")
    return candidate


if __name__ == "__main__":
    raise SystemExit(main())
