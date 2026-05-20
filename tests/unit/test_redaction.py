"""Unit tests for ``geno_lewm._redaction``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from geno_lewm import _redaction as red
from geno_lewm import observability as obs
from geno_lewm.errors import InvariantViolation


@pytest.fixture(autouse=True)
def _reset() -> Any:
    red.STATS.reset()
    for sink in list(obs._SINKS.values()):  # noqa: SLF001
        sink.close()
    obs._SINKS.clear()  # noqa: SLF001
    obs._LOGGERS.clear()  # noqa: SLF001
    yield
    red.STATS.reset()
    for sink in list(obs._SINKS.values()):  # noqa: SLF001
        sink.close()
    obs._SINKS.clear()  # noqa: SLF001
    obs._LOGGERS.clear()  # noqa: SLF001


@pytest.fixture
def strict_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENO_LEWM_REDACTION_STRICT", "0")


@pytest.fixture
def strict_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENO_LEWM_REDACTION_STRICT", "1")


# ---------------------------------------------------------------------------
# Strict-mode behaviour: deny-list, DNA pattern, type violations all raise.


def test_strict_mode_raises_on_denied_field(strict_on: None) -> None:
    with pytest.raises(InvariantViolation):
        red.redact(
            "training.run.start",
            {"user_email": "x@y.z"},
            allowed_keys=frozenset(),
        )


def test_strict_mode_raises_on_dna_value(strict_on: None) -> None:
    with pytest.raises(InvariantViolation):
        red.redact(
            "training.run.start",
            {"config_path": "ACGTACGTACGTACGTACGT"},
            allowed_keys=frozenset({"config_path"}),
        )


def test_strict_mode_raises_on_bytes(strict_on: None) -> None:
    with pytest.raises(InvariantViolation):
        red.redact(
            "training.run.start",
            {"config_path": b"deadbeef"},
            allowed_keys=frozenset({"config_path"}),
        )


def test_strict_mode_raises_on_nested_dna(strict_on: None) -> None:
    with pytest.raises(InvariantViolation):
        red.redact(
            "training.metric",
            {"value": ["ACGTACGTACGTACGTACGTA"]},
            allowed_keys=frozenset({"value"}),
        )


def test_strict_mode_raises_on_nested_denied(strict_on: None) -> None:
    with pytest.raises(InvariantViolation):
        red.redact(
            "training.metric",
            {"value": {"user_email": "x@y.z"}},
            allowed_keys=frozenset({"value"}),
        )


def test_strict_mode_raises_on_set_value(strict_on: None) -> None:
    with pytest.raises(InvariantViolation):
        red.redact(
            "training.metric",
            {"value": {1, 2, 3}},
            allowed_keys=frozenset({"value"}),
        )


# ---------------------------------------------------------------------------
# Permissive mode: drop and count, no raise.


def test_permissive_drops_denied_field(strict_off: None) -> None:
    out = red.redact(
        "training.run.start",
        {"user_email": "x@y.z", "config_path": "ok"},
        allowed_keys=frozenset({"config_path"}),
    )
    assert out == {"config_path": "ok"}
    assert red.STATS.dropped_denied == 1


def test_permissive_drops_dna(strict_off: None) -> None:
    out = red.redact(
        "training.run.start",
        {"config_path": "ACGTACGTACGTACGTACGT"},
        allowed_keys=frozenset({"config_path"}),
    )
    assert out == {}
    assert red.STATS.dropped_dna == 1


def test_permissive_drops_bytes(strict_off: None) -> None:
    out = red.redact(
        "training.run.start",
        {"config_path": b"deadbeef"},
        allowed_keys=frozenset({"config_path"}),
    )
    assert out == {}
    assert red.STATS.dropped_type == 1


# ---------------------------------------------------------------------------
# Per-event allowlist: keys outside the allowlist are SOFT-dropped, no raise
# even in strict mode (this is registry drift, not a bypass).


def test_unallowlisted_key_is_soft_drop_even_in_strict_mode(strict_on: None) -> None:
    out = red.redact(
        "training.run.start",
        {"surprise_field": 1},
        allowed_keys=frozenset({"config_path"}),
    )
    assert out == {}
    assert red.STATS.dropped_keys == 1


# ---------------------------------------------------------------------------
# Type allowlist: scalars, list of scalars, shallow dict of scalars.


def test_scalar_values_pass(strict_on: None) -> None:
    out = red.redact(
        "training.metric",
        {"name": "loss", "value": 1.5, "kind": "gauge"},
        allowed_keys=frozenset({"name", "value", "kind"}),
    )
    assert out == {"name": "loss", "value": 1.5, "kind": "gauge"}


def test_list_of_scalars_passes(strict_on: None) -> None:
    out = red.redact(
        "training.metric",
        {"value": [1, 2, 3]},
        allowed_keys=frozenset({"value"}),
    )
    assert out == {"value": [1, 2, 3]}


def test_shallow_dict_of_scalars_passes(strict_on: None) -> None:
    out = red.redact(
        "training.metric",
        {"value": {"k": 1}},
        allowed_keys=frozenset({"value"}),
    )
    assert out == {"value": {"k": 1}}


def test_defensive_copy_breaks_aliasing(strict_on: None) -> None:
    src = {"k": 1}
    out = red.redact(
        "training.metric",
        {"value": src},
        allowed_keys=frozenset({"value"}),
    )
    src["k"] = 999
    assert out["value"] == {"k": 1}


# ---------------------------------------------------------------------------
# Stats counter — sums correctly.


def test_stats_total_and_as_dict(strict_off: None) -> None:
    red.redact(
        "training.run.start",
        {
            "user_email": "x@y.z",  # denied
            "dna": "ACGTACGTACGTACGTACGT",  # dna
            "bytes_field": b"x",  # type
            "unknown_key": 1,  # soft drop
        },
        allowed_keys=frozenset(),
    )
    assert red.STATS.dropped_denied == 1
    assert red.STATS.dropped_dna == 1
    assert red.STATS.dropped_type == 1
    assert red.STATS.dropped_keys == 1
    assert red.STATS.total() == 4
    assert red.STATS.as_dict() == {
        "dropped_keys": 1,
        "dropped_denied": 1,
        "dropped_dna": 1,
        "dropped_type": 1,
    }


# ---------------------------------------------------------------------------
# Coverage check: every event's allowlist is internally consistent.


@pytest.mark.parametrize("event_spec", obs.EVENTS, ids=lambda e: e.name)
def test_each_event_allowlist_round_trips(event_spec: obs.EventSpec, strict_on: None) -> None:
    # Build a sample payload using only the event's allowlist; every key
    # gets a benign scalar. The filter MUST let everything through.
    sample = {k: "v" for k in event_spec.allowed_keys}
    out = red.redact(event_spec.name, sample, allowed_keys=event_spec.allowed_keys)
    assert out == sample


# ---------------------------------------------------------------------------
# Wired through the logger end-to-end.


def test_logger_path_drops_denied_fields_in_strict_mode(
    tmp_path: Path, strict_on: None
) -> None:
    lg = obs.get_logger("runtime", run_id="rrd", log_dir=tmp_path)
    with pytest.raises(InvariantViolation):
        lg.info("training.run.start", user_email="x@y.z")
    obs.shutdown_run("rrd", tmp_path)


def test_logger_path_drops_unallowed_keys_silently(
    tmp_path: Path, strict_on: None
) -> None:
    lg = obs.get_logger("runtime", run_id="rru", log_dir=tmp_path)
    lg.info("training.run.start", config_path="cfg", bogus_field=1)
    obs.shutdown_run("rru", tmp_path)
    [rec] = [json.loads(line) for line in (tmp_path / "rru.jsonl").read_text().splitlines()]
    assert rec["data"] == {"config_path": "cfg"}
    assert red.STATS.dropped_keys >= 1
