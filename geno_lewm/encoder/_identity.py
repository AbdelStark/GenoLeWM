# SPDX-License-Identifier: Apache-2.0
"""Content identity for local encoder weight artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from geno_lewm.errors import InputError
from geno_lewm.provenance import canonical_json_sha256, sha256_file

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_ENCODER_IMPLEMENTATION_FILES = (
    ("geno_lewm/_inference.py", _PACKAGE_ROOT / "_inference.py"),
    ("geno_lewm/encoder/_canonical.py", Path(__file__).with_name("_canonical.py")),
    ("geno_lewm/encoder/_dna_tokenizer.py", Path(__file__).with_name("_dna_tokenizer.py")),
    ("geno_lewm/encoder/_identity.py", Path(__file__)),
    ("geno_lewm/encoder/_normalization.py", Path(__file__).with_name("_normalization.py")),
    ("geno_lewm/encoder/carbon.py", Path(__file__).with_name("carbon.py")),
    ("geno_lewm/encoder/pooling.py", Path(__file__).with_name("pooling.py")),
    ("geno_lewm/encoder/windowing.py", Path(__file__).with_name("windowing.py")),
)
_MONOLITHIC_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")
_SHARDED_INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
# Match Transformers' default local loader precedence (use_safetensors=None).
_WEIGHT_CANDIDATES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
_REQUIRED_RUNTIME_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.py",
    "dna_config.json",
)
# Keep this stable ordering to preserve the published runtime hash contract.
_RUNTIME_IDENTITY_FILES = (
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


def encoder_weights_hash(encoder_weights: Path) -> str:
    """Hash one weight file or every shard referenced by a local index."""
    path = Path(encoder_weights)
    if path.is_dir():
        for name in _WEIGHT_CANDIDATES:
            candidate = path / name
            if candidate.is_file():
                if name in _SHARDED_INDEX_NAMES:
                    return _sharded_weights_hash(candidate)
                return sha256_file(candidate)
        raise InputError(
            "encoder weights directory has no recognized weight file",
            details={
                "path": str(path),
                "expected": [*_MONOLITHIC_WEIGHT_NAMES, *_SHARDED_INDEX_NAMES],
            },
        )
    if path.is_file():
        if path.name in _SHARDED_INDEX_NAMES:
            return _sharded_weights_hash(path)
        return sha256_file(path)
    raise InputError("encoder weights path does not exist", details={"path": str(path)})


def encoder_runtime_hash(model_dir: Path) -> str:
    """Hash weights plus every Carbon file that can change encoded states."""
    root = Path(model_dir)
    if not root.is_dir():
        raise InputError("encoder runtime path must be a directory", details={"path": str(root)})
    missing = [name for name in _REQUIRED_RUNTIME_FILES if not (root / name).is_file()]
    if missing:
        raise InputError(
            "encoder runtime is missing required identity files",
            details={"path": str(root), "missing": missing},
        )
    files = [
        {"path": name, "sha256": sha256_file(root / name)}
        for name in _RUNTIME_IDENTITY_FILES
        if (root / name).is_file()
    ]
    return canonical_json_sha256(
        {
            "contract": "geno_lewm.encoder_runtime.v1",
            "weights_hash": encoder_weights_hash(root),
            "runtime_files": files,
            "implementation_files": [
                {"path": name, "sha256": sha256_file(path)}
                for name, path in _ENCODER_IMPLEMENTATION_FILES
            ],
        }
    )


def encoder_identity_hash(model_dir: Path, *, state_contract_version: str) -> str:
    """Resolve the identity rule for a legacy or corrected checkpoint lineage."""
    if state_contract_version == "legacy_raw_v1":
        return encoder_weights_hash(model_dir)
    if state_contract_version == "l2_normalized_v2":
        return encoder_runtime_hash(model_dir)
    raise InputError(
        "unsupported encoder state contract for identity hashing",
        details={"state_contract_version": state_contract_version},
    )


def _sharded_weights_hash(index_path: Path) -> str:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(
            "encoder weight index is not readable JSON",
            details={"path": str(index_path)},
        ) from exc
    if not isinstance(payload, Mapping):
        raise InputError("encoder weight index must be a JSON object")
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise InputError("encoder weight index must contain a non-empty weight_map")

    shard_names: set[str] = set()
    for value in weight_map.values():
        if not isinstance(value, str) or not value.strip():
            raise InputError("encoder weight_map values must be non-empty shard paths")
        shard_names.add(value)

    root = index_path.parent.resolve()
    files: list[dict[str, str]] = []
    for name in sorted(shard_names):
        candidate = (root / name).resolve()
        try:
            relative = candidate.relative_to(root)
        except ValueError as exc:
            raise InputError(
                "encoder shard paths must stay beside the weight index",
                details={"index": str(index_path), "shard": name},
            ) from exc
        if not candidate.is_file():
            raise InputError(
                "encoder weight shard is missing",
                details={"index": str(index_path), "shard": relative.as_posix()},
            )
        files.append({"path": relative.as_posix(), "sha256": sha256_file(candidate)})

    return canonical_json_sha256(
        {
            "contract": "geno_lewm.encoder_weight_bundle.v1",
            "index": {"path": index_path.name, "sha256": sha256_file(index_path)},
            "files": files,
        }
    )
