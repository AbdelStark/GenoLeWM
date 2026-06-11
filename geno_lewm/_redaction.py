# SPDX-License-Identifier: Apache-2.0
"""Redaction filter for the GenoLeWM structured logger.

Single chokepoint between callers and the JSONL sink. Defined by
observability contract and ``metrics registry docs``. The four rules:

1. **Per-event allowlist** — keys not listed in
   :attr:`EventSpec.allowed_keys` for the event are dropped and counted
   in :data:`RedactionStats.dropped_keys`. Soft drop (registry drift).
2. **Type allowlist** — only ``None``, ``bool``, ``int``, ``float``,
   ``str`` and shallow containers thereof are allowed. ``bytes``,
   torch tensors, numpy arrays, callables, sets — dropped. Strict
   mode raises :class:`InvariantViolation`.
3. **Pattern filter** — any string matching ``^[ACGTNacgtn]{20,}$``
   is dropped regardless of key. Strict mode raises.
4. **Deny list** — exact field-name match against
   :data:`PERSONAL_DATA_FIELDS` always drops and, in strict mode, raises.

Strict mode is on by default (``GENO_LEWM_REDACTION_STRICT`` != ``"0"``).
The metric ``geno_lewm.observability.redacted_keys`` (observability contract) is
served by :data:`STATS` and will be wired into the metrics registry by
#25; today it is observable via :func:`redaction_stats`.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from geno_lewm.errors import InvariantViolation

__all__ = [
    "DNA_RE",
    "PERSONAL_DATA_FIELDS",
    "STATS",
    "RedactionStats",
    "is_strict",
    "redact",
    "redaction_stats",
]


#: Field-name deny list — matched against keys exactly. Drops always; in
#: strict mode (default) we raise.
PERSONAL_DATA_FIELDS: frozenset[str] = frozenset(
    {
        "vcf_content",
        "genotype",
        "sample_id",
        "user_email",
        "email",
        "phone",
        "address",
        "dob",
        "birthdate",
    }
)


#: Pattern for raw DNA-like strings. Matches ACGTN ≥ 20 bp, case-insensitive,
#: full-string anchor on each value. Per the spec, bypass is a strict-mode
#: error: callers must hash variant bases before logging them.
DNA_RE: re.Pattern[str] = re.compile(r"^[ACGTNacgtn]{20,}$")


_SCALAR_TYPES: tuple[type, ...] = (type(None), bool, int, float, str)


def _is_strict_env() -> bool:
    """Return whether redaction strict-mode is active."""
    raw = os.environ.get("GENO_LEWM_REDACTION_STRICT", "1")
    return raw != "0"


def is_strict() -> bool:
    """Public alias for :func:`_is_strict_env` so callers can branch."""
    return _is_strict_env()


@dataclass
class RedactionStats:
    """Counters incremented by the filter.

    The metrics registry (#25) will export these under
    ``geno_lewm.observability.redacted_keys`` and friends.
    """

    dropped_keys: int = 0  # not in per-event allowlist
    dropped_denied: int = 0  # in PERSONAL_DATA_FIELDS
    dropped_dna: int = 0  # matched DNA_RE
    dropped_type: int = 0  # bytes / tensor / set / …
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def reset(self) -> None:
        with self._lock:
            self.dropped_keys = 0
            self.dropped_denied = 0
            self.dropped_dna = 0
            self.dropped_type = 0

    def total(self) -> int:
        return self.dropped_keys + self.dropped_denied + self.dropped_dna + self.dropped_type

    def as_dict(self) -> dict[str, int]:
        return {
            "dropped_keys": self.dropped_keys,
            "dropped_denied": self.dropped_denied,
            "dropped_dna": self.dropped_dna,
            "dropped_type": self.dropped_type,
        }

    def _inc(self, attr: str) -> None:
        with self._lock:
            setattr(self, attr, getattr(self, attr) + 1)


STATS = RedactionStats()


def redaction_stats() -> RedactionStats:
    """Return the module-level counter object."""
    return STATS


def _scalars_only(seq: Iterable[Any]) -> bool:
    return all(isinstance(x, _SCALAR_TYPES) for x in seq)


def _violate(reason: str, *, event: str, key: str, kind: str) -> None:
    """Raise an InvariantViolation under strict mode.

    Strict mode treats deny-list / type-violation / DNA-pattern bypass as
    a programmer bug: the calling code is leaking what it must not.
    """
    raise InvariantViolation(
        reason,
        details={"event": event, "key": key, "kind": kind},
        remediation=(
            "Hash personal data or restrict the payload to the per-event "
            "allowlist (see observability contract)."
        ),
    )


def _check_dna(value: Any, *, event: str, key: str, strict: bool) -> bool:
    """Return True iff the value should be dropped because it looks like DNA."""
    if isinstance(value, str) and DNA_RE.match(value):
        if strict:
            _violate("DNA-like string in log payload", event=event, key=key, kind="dna")
        STATS._inc("dropped_dna")
        return True
    return False


def _check_type(value: Any, *, event: str, key: str, strict: bool) -> bool:
    """Return True iff the value should be dropped because its type is forbidden."""
    if isinstance(value, _SCALAR_TYPES):
        return False
    if isinstance(value, list | tuple):
        if _scalars_only(value):
            return False
        if strict:
            _violate("non-scalar element in list value", event=event, key=key, kind="type")
        STATS._inc("dropped_type")
        return True
    if isinstance(value, dict):
        # Only shallow dicts of scalars are allowed.
        if all(isinstance(k, str) for k in value) and _scalars_only(value.values()):
            return False
        if strict:
            _violate("nested or non-scalar dict value", event=event, key=key, kind="type")
        STATS._inc("dropped_type")
        return True
    # bytes, bytearray, memoryview, set, frozenset, tensors, ndarrays, callables…
    if strict:
        _violate(
            f"unsupported value type {type(value).__name__}", event=event, key=key, kind="type"
        )
    STATS._inc("dropped_type")
    return True


def _check_dna_in_container(value: Any, *, event: str, key: str, strict: bool) -> bool:
    """Run the DNA pattern over the strings nested inside a list/dict value."""
    if isinstance(value, list | tuple):
        for v in value:
            if isinstance(v, str) and DNA_RE.match(v):
                if strict:
                    _violate("DNA-like string in list value", event=event, key=key, kind="dna")
                STATS._inc("dropped_dna")
                return True
    elif isinstance(value, dict):
        for k, v in value.items():
            if k in PERSONAL_DATA_FIELDS:
                if strict:
                    _violate(
                        "deny-listed key inside nested dict",
                        event=event,
                        key=f"{key}.{k}",
                        kind="denied",
                    )
                STATS._inc("dropped_denied")
                return True
            if isinstance(v, str) and DNA_RE.match(v):
                if strict:
                    _violate("DNA-like string in dict value", event=event, key=key, kind="dna")
                STATS._inc("dropped_dna")
                return True
    return False


def redact(
    event: str,
    data: Mapping[str, Any],
    *,
    allowed_keys: frozenset[str],
    strict: bool | None = None,
) -> dict[str, Any]:
    """Return a redacted copy of ``data`` suitable for the sink.

    ``allowed_keys`` is the per-event allowlist (typically pulled from
    :attr:`EventSpec.allowed_keys`). When the event is not registered,
    pass an empty set — every key will be soft-dropped under
    rule (1), preserving the no-leak guarantee.
    """
    if strict is None:
        strict = _is_strict_env()

    out: dict[str, Any] = {}
    for key, value in data.items():
        # Rule 4: deny-list (always raises in strict mode).
        if key in PERSONAL_DATA_FIELDS:
            if strict:
                _violate("personal-data field in log payload", event=event, key=key, kind="denied")
            STATS._inc("dropped_denied")
            continue

        # Rule 3: DNA pattern on raw string value.
        if _check_dna(value, event=event, key=key, strict=strict):
            continue

        # Rule 2: type allowlist.
        if _check_type(value, event=event, key=key, strict=strict):
            continue

        # Rule 3+4 inside containers: DNA / deny-listed key nested under
        # an otherwise-allowed dict or list.
        if _check_dna_in_container(value, event=event, key=key, strict=strict):
            continue

        # Rule 1: per-event allowlist. Soft drop — not a strict-mode raise.
        if key not in allowed_keys:
            STATS._inc("dropped_keys")
            continue

        # Defensive copy so the caller cannot mutate the record post-hoc
        # via shared references (matches test_data_is_independent_copy).
        if isinstance(value, dict):
            out[key] = dict(value)
        elif isinstance(value, list | tuple):
            out[key] = list(value)
        else:
            out[key] = value

    return out
