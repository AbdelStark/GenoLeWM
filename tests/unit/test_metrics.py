"""Unit tests for ``geno_lewm.metrics``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from geno_lewm import metrics as m
from geno_lewm.errors import InputError

# Canonical names from docs/spec/05-observability.md — locking the public
# contract here. Editing this set is a public-API change.
CANONICAL_METRIC_NAMES = {
    "geno_lewm.training.step.duration",
    "geno_lewm.training.loss.pred",
    "geno_lewm.training.loss.reg",
    "geno_lewm.training.collapse.alert",
    "geno_lewm.training.collapse.pred_cos_mean",
    "geno_lewm.training.collapse.pred_l2_mean",
    "geno_lewm.training.collapse.target_var_per_dim",
    "geno_lewm.training.collapse.pred_var_per_dim",
    "geno_lewm.training.collapse.pred_target_corr",
    "geno_lewm.training.collapse.pairwise_pred_dist_mean",
    "geno_lewm.training.collapse.kl_reg",
    "geno_lewm.data.cache.hit",
    "geno_lewm.data.cache.miss",
    "geno_lewm.data.encode.duration",
    "geno_lewm.inference.score.duration",
    "geno_lewm.inference.batch.throughput",
    "geno_lewm.inference.memory.peak_bytes",
    "geno_lewm.planning.cem.iter.duration",
    "geno_lewm.planning.cem.calls",
    "geno_lewm.attestation.verify.duration",
    "geno_lewm.observability.redacted_keys",
    "geno_lewm.errors.raised",
}


@pytest.fixture(autouse=True)
def _reset() -> Any:
    m._reset_for_tests()
    yield
    m._reset_for_tests()


# ---------------------------------------------------------------------------
# Registry shape.


def test_registry_covers_canonical_list() -> None:
    names = {x.name for x in m.METRICS}
    missing = CANONICAL_METRIC_NAMES - names
    extra = names - CANONICAL_METRIC_NAMES
    assert not missing, f"missing metrics: {missing}"
    assert not extra, f"unexpected metrics: {extra}"


def test_registry_unique_and_kinds_valid() -> None:
    names = [x.name for x in m.METRICS]
    assert len(names) == len(set(names))
    valid_kinds = {"counter", "gauge", "histogram"}
    for spec in m.METRICS:
        assert spec.kind in valid_kinds
        assert spec.unit
        assert spec.summary


# ---------------------------------------------------------------------------
# Counter behaviour.


def test_counter_inc() -> None:
    c = m.get_counter("geno_lewm.data.cache.hit")
    assert c.value() == 0
    c.inc()
    c.inc(2)
    assert c.value() == 3


def test_counter_rejects_negative() -> None:
    c = m.get_counter("geno_lewm.data.cache.hit")
    with pytest.raises(InputError):
        c.inc(-1)


def test_counter_is_cached() -> None:
    a = m.get_counter("geno_lewm.errors.raised")
    b = m.get_counter("geno_lewm.errors.raised")
    assert a is b


# ---------------------------------------------------------------------------
# Gauge.


def test_gauge_set_and_inc() -> None:
    g = m.get_gauge("geno_lewm.training.loss.pred")
    g.set(1.5)
    assert g.value() == 1.5
    g.inc(0.5)
    assert g.value() == 2.0
    g.dec(1.0)
    assert g.value() == 1.0


# ---------------------------------------------------------------------------
# Histogram.


def test_histogram_observe_buckets() -> None:
    h = m.get_histogram("geno_lewm.training.step.duration")
    h.observe(0.5)
    h.observe(100)
    h.observe(10000)
    snap = h.snapshot()
    assert snap["count"] == 3
    assert snap["sum"] == 10100.5
    # Buckets are cumulative; the +Inf bucket holds the total count.
    assert snap["counts"][-1] == 3


def test_histogram_nan_rejected() -> None:
    h = m.get_histogram("geno_lewm.training.step.duration")
    import math

    with pytest.raises(InputError):
        h.observe(math.nan)


def test_histogram_bytes_unit_uses_bytes_buckets() -> None:
    # None of the canonical histogram metrics use the bytes unit, so we
    # exercise the default-bucket chooser via a synthetic spec rather
    # than through the public accessor.
    spec = m.MetricSpec("test.histogram.bytes", "histogram", "bytes", "")
    th = m.Histogram(spec)
    assert th._buckets[-1] == float("inf")
    assert th._buckets[0] == m.DEFAULT_HISTOGRAM_BUCKETS_BYTES[0]


# ---------------------------------------------------------------------------
# Accessor validation.


def test_unregistered_metric_rejected() -> None:
    with pytest.raises(InputError):
        m.get_counter("not.a.metric")


def test_kind_mismatch_rejected() -> None:
    # training.loss.pred is a gauge; asking for counter raises.
    with pytest.raises(InputError):
        m.get_counter("geno_lewm.training.loss.pred")
    with pytest.raises(InputError):
        m.get_histogram("geno_lewm.training.loss.pred")


# ---------------------------------------------------------------------------
# Prometheus textfile exporter.


def test_exporter_emits_help_type_and_value_lines(tmp_path: Path) -> None:
    m.get_counter("geno_lewm.data.cache.hit").inc(7)
    m.get_gauge("geno_lewm.training.loss.pred").set(0.42)
    h = m.get_histogram("geno_lewm.training.step.duration")
    h.observe(10)
    h.observe(100)

    dest = m.export_prometheus_textfile(tmp_path / "metrics.prom")
    body = dest.read_text(encoding="utf-8")

    # Names map dots → underscores for Prometheus name-validity.
    assert "# HELP geno_lewm_data_cache_hit per-call increment (unit: ops)" in body
    assert "# TYPE geno_lewm_data_cache_hit counter" in body
    assert "geno_lewm_data_cache_hit 7" in body.replace(".0", "")

    assert "# TYPE geno_lewm_training_loss_pred gauge" in body
    assert "geno_lewm_training_loss_pred 0.42" in body

    # Histogram emits _bucket / _sum / _count rows.
    assert "geno_lewm_training_step_duration_bucket{" in body
    assert "geno_lewm_training_step_duration_sum 110" in body.replace(".0", "")
    assert "geno_lewm_training_step_duration_count 2" in body


def test_exporter_textfile_is_well_formed(tmp_path: Path) -> None:
    """Validate the format against the Prometheus textfile grammar.

    We do not depend on ``promtool`` — that would need a Go toolchain
    fetch in CI. Instead we parse line-by-line against the spec from
    https://prometheus.io/docs/instrumenting/exposition_formats/.
    """
    m.get_counter("geno_lewm.errors.raised").inc(3)
    m.get_histogram("geno_lewm.inference.score.duration").observe(5)
    dest = m.export_prometheus_textfile(tmp_path / "metrics.prom")
    body = dest.read_text(encoding="utf-8")

    metric_re = re.compile(
        r"^[a-zA-Z_:][a-zA-Z0-9_:]*"  # metric name
        r"(\{[^}]*\})?"  # optional labels
        r"\s+"
        r"([0-9eE.+\-]+|\+Inf|-Inf|NaN)\s*$"
    )
    type_re = re.compile(
        r"^# TYPE [a-zA-Z_:][a-zA-Z0-9_:]*\s+(counter|gauge|histogram|summary|untyped)\s*$"
    )
    help_re = re.compile(r"^# HELP [a-zA-Z_:][a-zA-Z0-9_:]*\s+.+$")

    for line in body.splitlines():
        if not line:
            continue
        if line.startswith("# HELP"):
            assert help_re.match(line), line
        elif line.startswith("# TYPE"):
            assert type_re.match(line), line
        else:
            assert metric_re.match(line), line


def test_exporter_write_is_atomic(tmp_path: Path) -> None:
    # The exporter writes to .tmp and renames; ensure no .tmp file is
    # left behind after a successful run.
    m.get_counter("geno_lewm.data.cache.miss").inc(1)
    dest = m.export_prometheus_textfile(tmp_path / "m.prom")
    assert dest.is_file()
    assert not (tmp_path / "m.prom.tmp").exists()


def test_default_metrics_path_resolves_env_or_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GENO_LEWM_LOG_DIR", str(tmp_path))
    p = m.metrics_path()
    assert p == tmp_path / "metrics.prom"


def test_sync_redaction_counter_brings_metric_up(tmp_path: Path) -> None:
    from geno_lewm import _redaction as red

    red.STATS.reset()
    # Trigger 3 soft drops via the filter directly.
    red.redact(
        "training.run.start",
        {"unknown_a": 1, "unknown_b": 2, "unknown_c": 3},
        allowed_keys=frozenset(),
        strict=False,
    )

    c = m.get_counter("geno_lewm.observability.redacted_keys")
    assert c.value() == 0
    m.sync_redaction_counter()
    assert c.value() == 3


# ---------------------------------------------------------------------------
# Thread safety smoke.


def test_counter_thread_safety() -> None:
    import threading

    c = m.get_counter("geno_lewm.data.cache.hit")

    def burn() -> None:
        for _ in range(1000):
            c.inc()

    threads = [threading.Thread(target=burn) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.value() == 4000
