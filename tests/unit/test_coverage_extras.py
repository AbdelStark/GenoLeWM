"""Targeted tests covering the small branches the main test modules
intentionally do not exercise.

This file exists to keep the coverage gate honest without polluting
the per-module suites with edge-case noise.
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Any

import pytest

from geno_lewm import _redaction as red, metrics as m, observability as obs
from geno_lewm.errors import InputError
from geno_lewm.provenance.hashing import looks_like_sha256

# ---------------------------------------------------------------------------
# Metric primitive reset paths.


def test_counter_reset() -> None:
    c = m.get_counter("geno_lewm.data.cache.hit")
    c.inc(5)
    assert c.value() == 5
    c.reset()
    assert c.value() == 0


def test_gauge_reset() -> None:
    g = m.get_gauge("geno_lewm.training.loss.pred")
    g.set(1.5)
    g.reset()
    assert g.value() == 0


def test_histogram_reset() -> None:
    h = m.get_histogram("geno_lewm.training.step.duration")
    h.observe(1.0)
    h.observe(2.0)
    h.reset()
    snap = h.snapshot()
    assert snap["count"] == 0
    assert snap["sum"] == 0
    assert all(c == 0 for c in snap["counts"])


def test_histogram_explicit_empty_buckets_rejected() -> None:
    spec = m.MetricSpec("test.histogram.empty", "histogram", "ms", "")
    with pytest.raises(InputError):
        m.Histogram(spec, buckets=[])


# ---------------------------------------------------------------------------
# Snapshot-all view.


def test_snapshot_all_returns_kind_per_metric() -> None:
    m._reset_for_tests()
    m.get_counter("geno_lewm.errors.raised").inc(2)
    m.get_gauge("geno_lewm.training.loss.pred").set(0.1)
    m.get_histogram("geno_lewm.training.step.duration").observe(5)
    snap = m.snapshot_all()
    assert snap["geno_lewm.errors.raised"]["kind"] == "counter"
    assert snap["geno_lewm.training.loss.pred"]["kind"] == "gauge"
    assert snap["geno_lewm.training.step.duration"]["kind"] == "histogram"


# ---------------------------------------------------------------------------
# Exporter formatting of special floats.


def test_exporter_handles_special_floats(tmp_path: Path) -> None:
    g = m.get_gauge("geno_lewm.inference.batch.throughput")
    g.set(float("inf"))
    dest = m.export_prometheus_textfile(tmp_path / "m.prom")
    body = dest.read_text(encoding="utf-8")
    assert "+Inf" in body


def test_exporter_default_path_uses_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GENO_LEWM_LOG_DIR", str(tmp_path))
    m._reset_for_tests()
    m.get_counter("geno_lewm.data.cache.hit").inc()
    dest = m.export_prometheus_textfile()
    assert dest == tmp_path / "metrics.prom"
    assert dest.is_file()


def test_metrics_path_falls_back_to_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GENO_LEWM_LOG_DIR", raising=False)
    p = m.metrics_path()
    # Compare path components so the test holds on Windows
    # (backslashes) as well as POSIX (forward slashes).
    assert p.name == "metrics.prom"
    assert p.parts[-3:] == (".geno-lewm", "logs", "metrics.prom")


# ---------------------------------------------------------------------------
# Logger error / pretty-format paths.


def _shutdown_obs() -> None:
    for sink_key in list(obs._SINKS):
        with contextlib.suppress(Exception):
            obs._SINKS[sink_key].close()
    obs._SINKS.clear()
    obs._LOGGERS.clear()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("GENO_LEWM_LOG_DIR", str(tmp_path))
    _shutdown_obs()
    yield
    _shutdown_obs()


def test_logger_set_level_rejects_unknown() -> None:
    log = obs.get_logger("test")
    with pytest.raises(InputError):
        log.set_level("trace")  # type: ignore[arg-type]


def test_logger_below_level_returns_none() -> None:
    log = obs.get_logger("test", level="warn")
    assert log.debug("training.step", loss=0.1) is None
    assert log.info("training.metric", name="x", value=1) is None


def test_logger_unknown_event_still_emits(tmp_path: Path) -> None:
    log = obs.get_logger("test", level="debug")
    rec = log.info("not.in.registry.event", arbitrary="ignored")
    assert rec is not None
    assert rec.event == "not.in.registry.event"


def test_logger_pretty_writes_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = obs.get_logger("test", pretty=True)
    log.info("training.run.start", config_path="cfg.yaml", git_sha="deadbeef")
    err = capsys.readouterr().err
    assert "training.run.start" in err


def test_logged_run_emits_error_on_geno_lewm_exception() -> None:
    from geno_lewm.errors import InputError as _input_error  # noqa: N813

    with pytest.raises(_input_error):
        with obs.logged_run(
            "runtime",
            start_event="training.run.start",
            end_event="training.run.end",
            start_data={"config_path": "cfg.yaml"},
        ):
            raise _input_error("boom", details={"reason": "test"}, remediation="ignore")


def test_logged_run_passes_unknown_exceptions_through() -> None:
    with pytest.raises(RuntimeError):
        with obs.logged_run("runtime"):
            raise RuntimeError("not a GenoLeWMError")


def test_get_logger_returns_cached_instance() -> None:
    a = obs.get_logger("component-x", run_id="run-test")
    b = obs.get_logger("component-x", run_id="run-test")
    assert a is b


def test_trace_context_round_trip() -> None:
    with obs.set_trace_context(trace_id="t-1", span_id="s-1"):
        assert obs.current_trace_context() == ("t-1", "s-1")
    # After the context exits, the IDs are gone.
    assert obs.current_trace_context() == (None, None)


def test_logger_format_pretty_includes_step_and_error_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = obs.get_logger("test", pretty=True)
    log.error("error", step=5, error_code="INPUT.GENERIC", message="boom")
    err = capsys.readouterr().err
    assert "step=5" in err
    assert "INPUT.GENERIC" in err


def test_shutdown_run_closes_sink(tmp_path: Path) -> None:
    log = obs.get_logger("test", run_id="run-x", log_dir=tmp_path)
    log.info("training.run.start")
    obs.shutdown_run("run-x", log_dir=tmp_path)
    # Re-opening creates a fresh logger.
    fresh = obs.get_logger("test", run_id="run-x", log_dir=tmp_path)
    assert fresh is not log


# ---------------------------------------------------------------------------
# Hashing edge cases.


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not-a-hash",
        "sha256:short",
        "sha256:" + "z" * 64,
        "sha256:" + "0" * 63,
        None,
        123,
    ],
)
def test_looks_like_sha256_rejects(value: object) -> None:
    assert looks_like_sha256(value) is False  # type: ignore[arg-type]


def test_looks_like_sha256_accepts() -> None:
    h = "sha256:" + "0" * 64
    assert looks_like_sha256(h) is True


# ---------------------------------------------------------------------------
# Redaction stats public path.


def test_redaction_stats_helpers() -> None:
    red.STATS.reset()
    assert red.redaction_stats().total() == 0
    red.STATS._inc("dropped_keys")
    assert red.STATS.total() == 1
    red.STATS.reset()
    assert red.STATS.total() == 0


# ---------------------------------------------------------------------------
# Histogram NaN / Inf in exporter.


def test_exporter_negative_inf_in_gauge(tmp_path: Path) -> None:
    g = m.get_gauge("geno_lewm.inference.batch.throughput")
    g.set(float("-inf"))
    dest = m.export_prometheus_textfile(tmp_path / "neg.prom")
    body = dest.read_text(encoding="utf-8")
    assert "-Inf" in body


def test_exporter_nan_in_gauge(tmp_path: Path) -> None:
    g = m.get_gauge("geno_lewm.inference.batch.throughput")
    g.set(math.nan)
    dest = m.export_prometheus_textfile(tmp_path / "nan.prom")
    body = dest.read_text(encoding="utf-8")
    assert "NaN" in body


# ---------------------------------------------------------------------------
# Synthetic samplers — error paths.


def test_uniform_snv_n_must_be_non_negative() -> None:
    import random

    from geno_lewm.action import uniform_snv

    with pytest.raises(InputError):
        uniform_snv("ACGT" * 64, -1, rng=random.Random(0))


def test_indel_rejects_negative_n() -> None:
    import random

    from geno_lewm.action import indel

    with pytest.raises(InputError):
        indel("ACGT" * 64, -1, rng=random.Random(0))


def test_indel_rejects_zero_type_mix() -> None:
    import random

    from geno_lewm.action import indel

    with pytest.raises(InputError):
        indel("ACGT" * 64, 1, rng=random.Random(0), type_mix=(0.0, 0.0))


def test_indel_rejects_negative_type_mix() -> None:
    import random

    from geno_lewm.action import indel

    with pytest.raises(InputError):
        indel("ACGT" * 64, 1, rng=random.Random(0), type_mix=(-1.0, 1.0))


def test_mnv_rejects_negative_n() -> None:
    import random

    from geno_lewm.action import mnv

    with pytest.raises(InputError):
        mnv("ACGT" * 64, -1, rng=random.Random(0))


def test_validate_window_rejects_negative_margin() -> None:
    import random

    from geno_lewm.action import uniform_snv

    with pytest.raises(InputError):
        uniform_snv("ACGT" * 64, 1, rng=random.Random(0), edge_margin=-1)


def test_validate_window_rejects_oversized_margin() -> None:
    import random

    from geno_lewm.action import uniform_snv

    with pytest.raises(InputError):
        uniform_snv("ACGT", 1, rng=random.Random(0), edge_margin=10)


def test_draw_indel_length_rejects_empty_mapping() -> None:
    import random

    from geno_lewm.action import indel

    with pytest.raises(InputError):
        indel("ACGT" * 64, 1, rng=random.Random(0), length_dist={})


def test_draw_indel_length_rejects_negative_weight() -> None:
    import random

    from geno_lewm.action import indel

    with pytest.raises(InputError):
        indel("ACGT" * 64, 1, rng=random.Random(0), length_dist={1: -0.5, 2: 1.0})


def test_draw_indel_length_rejects_out_of_range_keys() -> None:
    import random

    from geno_lewm.action import indel

    with pytest.raises(InputError):
        indel("ACGT" * 64, 1, rng=random.Random(0), length_dist={0: 1.0})


def test_draw_indel_length_rejects_zero_total_weight() -> None:
    import random

    from geno_lewm.action import indel

    with pytest.raises(InputError):
        indel("ACGT" * 64, 1, rng=random.Random(0), length_dist={1: 0.0, 2: 0.0})


def test_synthetic_samplers_skip_n_bases() -> None:
    """Smoke check: a window dominated by `N`s yields fewer edits than
    requested, because non-ACGT anchors are skipped."""
    import random

    from geno_lewm.action import indel, mnv

    n_window = "ACGN" * 64
    edits = indel(n_window, 16, rng=random.Random(0))
    assert isinstance(edits, list)
    edits2 = mnv(n_window, 16, rng=random.Random(0))
    assert isinstance(edits2, list)


# ---------------------------------------------------------------------------
# Receipt + verify path coverage.


def test_receipt_provenance_rejects_unknown_kind() -> None:
    from geno_lewm.errors import InputError as _input_error  # noqa: N813
    from geno_lewm.provenance import ReceiptProvenance

    with pytest.raises(_input_error):
        ReceiptProvenance(kind="not-a-kind")


def test_receipt_output_rejects_non_float() -> None:
    from geno_lewm.errors import InputError as _input_error  # noqa: N813
    from geno_lewm.provenance import ReceiptOutput

    with pytest.raises(_input_error):
        ReceiptOutput(
            sigma_raw="x",  # type: ignore[arg-type]
            sigma_calibrated=0.0,
            bucket_id="b",
            confidence=0.5,
            low_confidence=False,
        )


def test_action_apply_truncate_left_no_change_for_same_length() -> None:
    from geno_lewm.action import EditType, RelEdit, apply_edit

    w = "ACGTACGTACGT"
    out = apply_edit(
        w,
        RelEdit(rel_pos=5, edit_type=EditType.SNV, ref_bases="C", alt_bases="T"),
        preserve_length=True,
    )
    assert len(out) == len(w)


# ---------------------------------------------------------------------------
# Errors.to_json carries the timestamp.


def test_error_to_json_includes_timestamp() -> None:
    from geno_lewm.errors import InputError as _input_error  # noqa: N813

    payload = _input_error("boom", details={"k": "v"}).to_json()
    assert '"code":' in payload
    assert '"ts":' in payload
