# SPDX-License-Identifier: Apache-2.0
"""Content-addressing primitives for GenoLeWM artifacts (artifact-provenance contract).

Two functions:

- :func:`canonical_json_sha256` — SHA-256 of the canonical JSON
  serialization of a value. Canonical JSON per artifact-provenance contract: keys
  sorted lexicographically, no whitespace, UTF-8, NaN / Infinity
  rejected. Byte-stable across platforms and Python releases.
- :func:`sha256_file` / :func:`sha256_bytes` — stream-friendly file /
  in-memory hashing used to compute the per-artifact ``hash`` fields
  in :class:`Manifest`.

All outputs are returned as ``"sha256:<hex>"`` strings to match the
on-disk manifest convention.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from geno_lewm.errors import InputError

__all__ = ["canonical_json_bytes", "canonical_json_sha256", "sha256_bytes", "sha256_file"]


_PREFIX = "sha256:"
_HASH_HEX_LEN = 64
_CHUNK = 1 << 20  # 1 MiB stream chunks for file hashing


def _canonical_default(obj: Any) -> Any:
    """Reject obviously-unstable JSON inputs.

     The canonical encoding refuses ``NaN`` / ``Infinity`` (RFC 8785
    ) because they have multiple non-stable spellings. We also
     refuse ``bytes`` — manifests are pure JSON; binary content goes
     into ``sha256_file`` instead.
    """
    if isinstance(obj, bytes | bytearray | memoryview):
        raise InputError(
            "bytes are not allowed in canonical JSON; hash the bytes separately",
            details={"type": type(obj).__name__},
        )
    raise InputError(
        f"unsupported type for canonical JSON: {type(obj).__name__}",
        details={"type": type(obj).__name__},
    )


def _check_floats(value: Any) -> None:
    """Walk the value and reject NaN / Infinity floats."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise InputError(
                "canonical JSON does not allow NaN / Infinity floats",
                details={"value": str(value)},
            )
        return
    if isinstance(value, Mapping):
        for v in value.values():
            _check_floats(v)
    elif isinstance(value, list | tuple):
        for v in value:
            _check_floats(v)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical-JSON byte string of ``value``.

    Canonical form (artifact-provenance contract, similar to RFC 8785):

    - Keys are sorted lexicographically at every level.
    - No whitespace (compact ``separators=(",", ":")``).
    - UTF-8 encoded.
    - NaN / Infinity rejected.
    - ``bytes`` rejected (must be hashed separately and embedded by
      reference).
    """
    _check_floats(value)
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_canonical_default,
    )
    return text.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Return ``"sha256:<hex>"`` for the canonical JSON of ``value``."""
    return _PREFIX + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    """Return ``"sha256:<hex>"`` for ``data``."""
    return _PREFIX + hashlib.sha256(bytes(data)).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return ``"sha256:<hex>"`` for the file at ``path``.

    Streams the file in 1 MiB chunks; safe for arbitrarily large
    artifacts (weights files can be multi-GB).
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return _PREFIX + h.hexdigest()


def looks_like_sha256(s: str) -> bool:
    """Return True iff ``s`` matches the ``"sha256:<64hex>"`` shape."""
    if not isinstance(s, str) or not s.startswith(_PREFIX):
        return False
    rest = s[len(_PREFIX) :]
    if len(rest) != _HASH_HEX_LEN:
        return False
    return all(c in "0123456789abcdef" for c in rest)
