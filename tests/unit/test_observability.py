"""Tests for ``geno_lewm.observability``.

Pins the record shape and required fields from
``docs/spec/05-observability.md`` (RFC-0013) and verifies that records
survive a crash via the wrapping ``logged_run`` context manager.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from geno_lewm import observability as obs
from geno_lewm.errors import InvalidEditError

# The full canonical list from docs/spec/05-observability.md. Editing
# this set is a public-API change and intentionally locked here.
CANONICAL_EVENT_NAMES = {
    "training.run.start",
    "training.run.end",
    "training.step",
    "training.epoch.end",
    "training.checkpoint.write",
    "training.collapse.alert",
    "training.metric",
    "eval.run.start",
    "eval.run.end",
    "eval.regression",
    "data.cache.hit",
    "data.cache.miss",
    "data.shard.write",
    "inference.score.start",
    "inference.score.end",
    "inference.batch.end",
    "inference.network.blocked",
    "attestation.receipt.write",
    "attestation.verify.start",
    "attestation.verify.end",
    "attestation.verify.mismatch",
    "error",
}


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


def _reset_global_state() -> None:
    for sink in list(obs._SINKS.values()):
        sink.close()
    obs._SINKS.clear()
    obs._LOGGERS.clear()


@pytest.fixture(autouse=True)
def reset_state() -> Any:
    _reset_global_state()
    yield
    _reset_global_state()


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_events_registry_covers_canonical_list() -> None:
    names = {e.name for e in obs.EVENTS}
    missing = CANONICAL_EVENT_NAMES - names
    extra = names - CANONICAL_EVENT_NAMES
    assert not missing, f"missing events: {missing}"
    assert not extra, f"unexpected events: {extra}"


def test_events_registry_unique_and_severities_valid() -> None:
    names = [e.name for e in obs.EVENTS]
    assert len(names) == len(set(names))
    valid = {"debug", "info", "warn", "error"}
    for e in obs.EVENTS:
        assert e.severity in valid, f"{e.name} has invalid severity {e.severity!r}"
        assert e.summary, f"{e.name} missing summary"


def test_record_shape_has_required_fields(tmp_log_dir: Path) -> None:
    lg = obs.get_logger("training", run_id="run-test", log_dir=tmp_log_dir, level="debug")
    lg.info("training.run.start", config_path="cfg.yaml")
    obs.shutdown_run("run-test", tmp_log_dir)

    [rec] = _read_records(tmp_log_dir / "run-test.jsonl")
    for key in ("ts", "severity", "event", "run_id", "data", "component"):
        assert key in rec, f"missing required key {key!r}"
    assert rec["severity"] == "info"
    assert rec["event"] == "training.run.start"
    assert rec["run_id"] == "run-test"
    assert rec["component"] == "training"
    assert rec["data"] == {"config_path": "cfg.yaml"}


def test_iso8601_utc_with_millisecond_resolution(tmp_log_dir: Path) -> None:
    lg = obs.get_logger("c", run_id="r1", log_dir=tmp_log_dir)
    lg.info("eval.run.start")
    obs.shutdown_run("r1", tmp_log_dir)
    [rec] = _read_records(tmp_log_dir / "r1.jsonl")
    # YYYY-MM-DDTHH:MM:SS.mmmZ — millisecond precision, UTC indicator.
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$", rec["ts"]), rec["ts"]


def test_severity_threshold_drops_lower_records(tmp_log_dir: Path) -> None:
    lg = obs.get_logger("c", run_id="r2", log_dir=tmp_log_dir, level="warn")
    assert lg.debug("training.step") is None
    assert lg.info("training.run.start") is None
    assert lg.warn("training.collapse.alert") is not None
    obs.shutdown_run("r2", tmp_log_dir)
    recs = _read_records(tmp_log_dir / "r2.jsonl")
    assert [r["event"] for r in recs] == ["training.collapse.alert"]


def test_standardized_fields_are_promoted_out_of_data(tmp_log_dir: Path) -> None:
    lg = obs.get_logger("c", run_id="r3", log_dir=tmp_log_dir)
    lg.info("training.epoch.end", step=42, epoch=3, phase="phase1", duration_ms=412, loss=1.23)
    obs.shutdown_run("r3", tmp_log_dir)
    [rec] = _read_records(tmp_log_dir / "r3.jsonl")
    assert rec["step"] == 42
    assert rec["epoch"] == 3
    assert rec["phase"] == "phase1"
    assert rec["duration_ms"] == 412
    # ``loss`` is event-specific and stays inside ``data``.
    assert rec["data"] == {"loss": 1.23}


def test_trace_context_attaches_when_set(tmp_log_dir: Path) -> None:
    lg = obs.get_logger("c", run_id="rt", log_dir=tmp_log_dir)
    with obs.set_trace_context(trace_id="t" * 32, span_id="s" * 16):
        lg.info("inference.batch.end")
    obs.shutdown_run("rt", tmp_log_dir)
    [rec] = _read_records(tmp_log_dir / "rt.jsonl")
    assert rec["trace_id"] == "t" * 32
    assert rec["span_id"] == "s" * 16


def test_trace_context_absent_by_default(tmp_log_dir: Path) -> None:
    lg = obs.get_logger("c", run_id="rt2", log_dir=tmp_log_dir)
    lg.info("inference.batch.end")
    obs.shutdown_run("rt2", tmp_log_dir)
    [rec] = _read_records(tmp_log_dir / "rt2.jsonl")
    assert "trace_id" not in rec
    assert "span_id" not in rec


def test_error_code_field_present_only_when_supplied(tmp_log_dir: Path) -> None:
    lg = obs.get_logger("c", run_id="re", log_dir=tmp_log_dir)
    lg.info("eval.run.start")
    lg.error("error", error_code="INPUT.INVALID_EDIT", message="bad")
    obs.shutdown_run("re", tmp_log_dir)
    recs = _read_records(tmp_log_dir / "re.jsonl")
    assert "error_code" not in recs[0]
    assert recs[1]["error_code"] == "INPUT.INVALID_EDIT"


def test_logger_factory_is_cached(tmp_log_dir: Path) -> None:
    a = obs.get_logger("eval", run_id="rc", log_dir=tmp_log_dir)
    b = obs.get_logger("eval", run_id="rc", log_dir=tmp_log_dir)
    assert a is b


def test_default_log_dir_resolves_env_or_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GENO_LEWM_LOG_DIR", str(tmp_path / "envlogs"))
    assert obs._resolve_log_dir(None) == tmp_path / "envlogs"
    monkeypatch.delenv("GENO_LEWM_LOG_DIR")
    home_default = obs._resolve_log_dir(None)
    assert home_default.name == "logs"
    assert home_default.parent.name == ".geno-lewm"


def test_run_id_default_is_unique(tmp_log_dir: Path) -> None:
    a = obs.get_logger("c1", log_dir=tmp_log_dir)
    b = obs.get_logger("c2", log_dir=tmp_log_dir)
    assert a.run_id != b.run_id
    assert a.run_id.startswith("run-")


def test_jsonl_file_path_matches_spec(tmp_log_dir: Path) -> None:
    obs.get_logger("c", run_id="run-abc", log_dir=tmp_log_dir).info("eval.run.start")
    obs.shutdown_run("run-abc", tmp_log_dir)
    assert (tmp_log_dir / "run-abc.jsonl").is_file()


def test_logged_run_book_ends_emit(tmp_log_dir: Path) -> None:
    with obs.logged_run(
        "runtime",
        run_id="rb",
        log_dir=tmp_log_dir,
        start_event="training.run.start",
        end_event="training.run.end",
        start_data={"config_path": "x"},
    ) as lg:
        lg.info("training.step", step=1)
    obs.shutdown_run("rb", tmp_log_dir)
    recs = _read_records(tmp_log_dir / "rb.jsonl")
    # debug is below info threshold by default, so 'training.step' may be dropped.
    events = [r["event"] for r in recs]
    assert events[0] == "training.run.start"
    assert events[-1] == "training.run.end"
    assert recs[0]["data"] == {"config_path": "x"}


def test_records_survive_crash(tmp_log_dir: Path) -> None:
    with pytest.raises(InvalidEditError):
        with obs.logged_run("runtime", run_id="rc", log_dir=tmp_log_dir) as lg:
            lg.info("eval.run.start", x=1)
            lg.info("inference.batch.end", n=10)
            raise InvalidEditError("boom", details={"why": "test"})

    obs.shutdown_run("rc", tmp_log_dir)
    recs = _read_records(tmp_log_dir / "rc.jsonl")
    events = [r["event"] for r in recs]
    # The pre-crash records made it to disk, and the wrapper emitted
    # an ``error`` record carrying the GenoLeWM error code.
    assert "eval.run.start" in events
    assert "inference.batch.end" in events
    assert events[-1] == "error"
    assert recs[-1]["error_code"] == "INPUT.INVALID_EDIT"
    assert recs[-1]["data"]["message"] == "boom"


def test_records_survive_crash_for_non_geno_lewm_exception(tmp_log_dir: Path) -> None:
    # A non-typed exception still flushes the pre-crash records — we
    # just don't emit a synthetic ``error`` record because that path
    # is reserved for typed errors.
    with pytest.raises(RuntimeError):
        with obs.logged_run("runtime", run_id="rcx", log_dir=tmp_log_dir) as lg:
            lg.info("eval.run.start")
            raise RuntimeError("not ours")
    obs.shutdown_run("rcx", tmp_log_dir)
    recs = _read_records(tmp_log_dir / "rcx.jsonl")
    assert any(r["event"] == "eval.run.start" for r in recs)
    assert not any(r["event"] == "error" for r in recs)


def test_set_level_validates_input(tmp_log_dir: Path) -> None:
    from geno_lewm.errors import InputError

    lg = obs.get_logger("c", run_id="rlv", log_dir=tmp_log_dir)
    with pytest.raises(InputError):
        lg.set_level("loud")  # type: ignore[arg-type]


def test_thread_safety_smoke(tmp_log_dir: Path) -> None:
    import threading

    lg = obs.get_logger("c", run_id="rth", log_dir=tmp_log_dir, level="debug")

    def burn() -> None:
        for i in range(50):
            lg.info("training.step", step=i)

    threads = [threading.Thread(target=burn) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    obs.shutdown_run("rth", tmp_log_dir)
    recs = _read_records(tmp_log_dir / "rth.jsonl")
    assert len(recs) == 200
    for r in recs:
        assert r["event"] == "training.step"


def test_unknown_event_still_emitted_but_not_crash(tmp_log_dir: Path) -> None:
    # Runtime is forgiving; the AST linter (#27) is the enforcement.
    lg = obs.get_logger("c", run_id="ru", log_dir=tmp_log_dir)
    lg.info("not.a.registered.event", k=1)
    obs.shutdown_run("ru", tmp_log_dir)
    [rec] = _read_records(tmp_log_dir / "ru.jsonl")
    assert rec["event"] == "not.a.registered.event"


def test_data_is_independent_copy(tmp_log_dir: Path) -> None:
    lg = obs.get_logger("c", run_id="rd", log_dir=tmp_log_dir)
    payload = {"k": 1}
    lg.info("training.metric", value=payload)
    payload["k"] = 999  # mutate after logging
    obs.shutdown_run("rd", tmp_log_dir)
    [rec] = _read_records(tmp_log_dir / "rd.jsonl")
    # The record went through JSON serialization, so post-call mutation
    # cannot affect the on-disk record.
    assert rec["data"]["value"] == {"k": 1}
