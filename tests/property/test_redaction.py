"""Property test for the redaction filter.

Acceptance criterion (issue #24):

> Property test in ``tests/property/test_redaction.py`` over 10k random
> payloads — zero leaks.

A "leak" means any of the following appears in the filter's output:

1. A deny-listed field name.
2. A DNA-like string (matches ``DNA_RE``) at any depth.
3. A value of a forbidden type (bytes, set, deep nested dict, etc.).
4. A key not present in the event's allowlist.

The test uses ``random.Random(seed=…)`` directly rather than Hypothesis
to keep this suite zero-dependency for the bootstrap phase. The seed is
fixed so failures are reproducible.
"""

from __future__ import annotations

import random
import string
from collections.abc import Sequence
from typing import Any

import pytest

from geno_lewm import _redaction as red, observability as obs
from geno_lewm.errors import InvariantViolation

# How many random payloads to throw at the filter.
N_TRIALS = 10_000

# Universe of keys: half are allowed for some event, half are denied,
# half are gibberish. We sample from this set for each trial.
ALL_ALLOWED_KEYS: list[str] = sorted({k for e in obs.EVENTS for k in e.allowed_keys})


def _rand_dna(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGTNacgtn") for _ in range(n))


def _rand_string(rng: random.Random, n: int) -> str:
    # Strings outside ACGTN — must NOT be flagged as DNA.
    alphabet = string.ascii_letters + string.digits + " ._-"
    return "".join(rng.choice(alphabet) for _ in range(n))


def _rand_value(rng: random.Random, depth: int = 0) -> Any:
    choice = rng.randint(0, 9)
    if choice == 0:
        return None
    if choice == 1:
        return rng.choice([True, False])
    if choice == 2:
        return rng.randint(-(2**31), 2**31 - 1)
    if choice == 3:
        return rng.uniform(-1e6, 1e6)
    if choice == 4:
        return _rand_string(rng, rng.randint(0, 32))
    if choice == 5:
        return _rand_dna(rng, rng.randint(20, 64))  # explicit DNA, must drop
    if choice == 6:
        return _rand_string(rng, rng.randint(5, 19))  # too short to match DNA
    if choice == 7 and depth < 1:
        n = rng.randint(0, 4)
        return [_rand_value(rng, depth=depth + 1) for _ in range(n)]
    if choice == 8 and depth < 1:
        n = rng.randint(0, 4)
        return {_rand_string(rng, 4): _rand_value(rng, depth=depth + 1) for _ in range(n)}
    if choice == 9:
        # Forbidden types — must drop.
        return rng.choice(
            [
                b"\xde\xad\xbe\xef",
                {1, 2, 3},
                bytearray(b"x"),
            ]
        )
    return None


def _rand_key(rng: random.Random) -> str:
    pool: list[str] = (
        list(red.PERSONAL_DATA_FIELDS) + ALL_ALLOWED_KEYS + [_rand_string(rng, 8) for _ in range(5)]
    )
    return rng.choice(pool)


def _make_payload(rng: random.Random) -> dict[str, Any]:
    n_fields = rng.randint(0, 8)
    return {_rand_key(rng): _rand_value(rng) for _ in range(n_fields)}


def _walk_strings(value: Any) -> Sequence[str]:
    """Yield every string nested anywhere in ``value``."""
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list | tuple):
        for v in value:
            out.extend(_walk_strings(v))
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(_walk_strings(v))
    return out


def _check_no_leak(event: obs.EventSpec, payload: dict[str, Any]) -> None:
    """Assert no leakage rule was breached."""
    for k, v in payload.items():
        # 1. Deny-listed keys never present.
        assert k not in red.PERSONAL_DATA_FIELDS, f"deny-listed key in output: {k}"
        # 4. Key must be in the event allowlist.
        assert k in event.allowed_keys, (
            f"key {k!r} not in allowlist for {event.name}: {sorted(event.allowed_keys)}"
        )
        # 2. No DNA-like string at any depth.
        for s in _walk_strings(v):
            assert not red.DNA_RE.match(s), f"DNA string leaked: {s!r}"
        # 3. Type allowlist.
        assert isinstance(v, (type(None), bool, int, float, str, list, dict))
        if isinstance(v, list):
            for item in v:
                assert isinstance(item, (type(None), bool, int, float, str))
        if isinstance(v, dict):
            for kk, vv in v.items():
                assert isinstance(kk, str)
                assert isinstance(vv, (type(None), bool, int, float, str))
                assert kk not in red.PERSONAL_DATA_FIELDS


@pytest.fixture(autouse=True)
def _stats_reset() -> Any:
    red.STATS.reset()
    yield
    red.STATS.reset()


def test_no_leaks_over_random_payloads_permissive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GENO_LEWM_REDACTION_STRICT", "0")
    rng = random.Random(0xC0FFEE)
    events = obs.EVENTS

    for _ in range(N_TRIALS):
        event = rng.choice(events)
        payload = _make_payload(rng)
        out = red.redact(event.name, payload, allowed_keys=event.allowed_keys)
        _check_no_leak(event, out)


def test_strict_mode_either_raises_or_does_not_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under strict mode either the payload passes cleanly, or the filter raises.

    In neither outcome is a leakage rule allowed to escape.
    """
    monkeypatch.setenv("GENO_LEWM_REDACTION_STRICT", "1")
    rng = random.Random(0xBADBEEF)
    events = obs.EVENTS

    # Smaller batch — strict mode raises on most trials, so the
    # interesting work is whether the filter ever returns a leaky dict.
    for _ in range(N_TRIALS // 5):
        event = rng.choice(events)
        payload = _make_payload(rng)
        try:
            out = red.redact(event.name, payload, allowed_keys=event.allowed_keys)
        except InvariantViolation:
            continue
        _check_no_leak(event, out)
