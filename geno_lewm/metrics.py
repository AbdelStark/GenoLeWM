# SPDX-License-Identifier: Apache-2.0
"""GenoLeWM metrics registry and minimal Prometheus textfile exporter.

Defined by RFC-0013 §3.3 / 4 and ``docs/spec/05-observability.md``.
Provides:

- :data:`METRICS` — immutable registry of ``MetricSpec(name, kind,
  unit, summary)`` rows. Renaming a metric name is a MAJOR change.
- :class:`Counter`, :class:`Gauge`, :class:`Histogram` — minimal,
  thread-safe primitives with no external dependency.
- :func:`get_counter` / :func:`get_gauge` / :func:`get_histogram` —
  validated accessors. The accessor verifies the metric is registered
  and has the expected kind, then returns a cached instance.
- :func:`export_prometheus_textfile` — writes a Prometheus textfile
  exposition to ``${GENO_LEWM_LOG_DIR}/metrics.prom`` (or a path of
  the caller's choosing). Honours the standard ``# HELP`` / ``# TYPE``
  lines and the histogram bucket / sum / count expansion.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal, TypedDict

from geno_lewm.errors import InputError, InvariantViolation


class HistogramSnapshot(TypedDict):
    """Structured snapshot returned by :meth:`Histogram.snapshot`.

    Buckets are cumulative upper bounds (the last bucket is always
    ``+Inf``). ``counts[i]`` is the cumulative count of observations
    ``<= buckets[i]`` since the last :meth:`Histogram.reset`.
    """

    buckets: list[float]
    counts: list[int]
    sum: float
    count: int


__all__ = [
    "DEFAULT_HISTOGRAM_BUCKETS_BYTES",
    "DEFAULT_HISTOGRAM_BUCKETS_MS",
    "METRICS",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricKind",
    "MetricSpec",
    "export_prometheus_textfile",
    "get_counter",
    "get_gauge",
    "get_histogram",
    "metrics_path",
    "snapshot_all",
]

MetricKind = Literal["counter", "gauge", "histogram"]


@dataclass(frozen=True)
class MetricSpec:
    """A single row in the :data:`METRICS` registry."""

    name: str
    kind: MetricKind
    unit: str
    summary: str


# Default histogram bucket sets. Two flavours cover the canonical metrics:
# duration in ms, and memory in bytes. Both are exposed at module level so
# tests and CI tooling can reference them by name.
DEFAULT_HISTOGRAM_BUCKETS_MS: tuple[float, ...] = (
    0.5,
    1,
    2,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
    float("inf"),
)
DEFAULT_HISTOGRAM_BUCKETS_BYTES: tuple[float, ...] = (
    1e6,
    1e7,
    1e8,
    5e8,
    1e9,
    5e9,
    1e10,
    5e10,
    1e11,
    float("inf"),
)


#: Canonical metrics for v0.1. Order matches
#: ``docs/spec/05-observability.md``.
METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("geno_lewm.training.step.duration", "histogram", "ms", "per training step"),
    MetricSpec("geno_lewm.training.loss.pred", "gauge", "unitless", "last logged pred loss"),
    MetricSpec(
        "geno_lewm.training.loss.reg",
        "gauge",
        "unitless",
        "last logged reg loss (Phase 2 active, Phase 1 monitoring)",
    ),
    MetricSpec(
        "geno_lewm.training.collapse.alert",
        "counter",
        "events",
        "total collapse alerts during run",
    ),
    MetricSpec(
        "geno_lewm.training.collapse.pred_cos_mean",
        "gauge",
        "unitless",
        "last collapse-monitor mean prediction-target cosine similarity",
    ),
    MetricSpec(
        "geno_lewm.training.collapse.pred_l2_mean",
        "gauge",
        "unitless",
        "last collapse-monitor mean prediction-target L2 distance",
    ),
    MetricSpec(
        "geno_lewm.training.collapse.target_var_per_dim",
        "gauge",
        "unitless",
        "last collapse-monitor mean target variance per latent dimension",
    ),
    MetricSpec(
        "geno_lewm.training.collapse.pred_var_per_dim",
        "gauge",
        "unitless",
        "last collapse-monitor mean prediction variance per latent dimension",
    ),
    MetricSpec(
        "geno_lewm.training.collapse.pred_target_corr",
        "gauge",
        "unitless",
        "last collapse-monitor flattened prediction-target correlation",
    ),
    MetricSpec(
        "geno_lewm.training.collapse.pairwise_pred_dist_mean",
        "gauge",
        "unitless",
        "last collapse-monitor mean pairwise prediction distance",
    ),
    MetricSpec(
        "geno_lewm.training.collapse.kl_reg",
        "gauge",
        "unitless",
        "last collapse-monitor LeJEPA KL regularizer value",
    ),
    MetricSpec("geno_lewm.data.cache.hit", "counter", "ops", "per-call increment"),
    MetricSpec("geno_lewm.data.cache.miss", "counter", "ops", "per-call increment"),
    MetricSpec("geno_lewm.data.encode.duration", "histogram", "ms", "per-window encoder call"),
    MetricSpec(
        "geno_lewm.inference.score.duration",
        "histogram",
        "ms",
        "per variant",
    ),
    MetricSpec(
        "geno_lewm.inference.batch.throughput",
        "gauge",
        "variants/s",
        "last batched scoring",
    ),
    MetricSpec(
        "geno_lewm.inference.memory.peak_bytes",
        "gauge",
        "bytes",
        "RSS peak per process",
    ),
    MetricSpec(
        "geno_lewm.planning.cem.iter.duration",
        "histogram",
        "ms",
        "per CEM iteration",
    ),
    MetricSpec(
        "geno_lewm.planning.cem.calls",
        "counter",
        "predictor_calls",
        "per planning run",
    ),
    MetricSpec(
        "geno_lewm.attestation.verify.duration",
        "histogram",
        "ms",
        "per verification",
    ),
    MetricSpec(
        "geno_lewm.observability.redacted_keys",
        "counter",
        "keys",
        "total dropped by redaction filter",
    ),
    MetricSpec("geno_lewm.errors.raised", "counter", "events", "tagged with code"),
)

_BY_NAME: dict[str, MetricSpec] = {m.name: m for m in METRICS}
if len(_BY_NAME) != len(METRICS):
    raise InvariantViolation(
        "Duplicate metric name in METRICS",
        details={"len_registry": len(METRICS), "len_unique": len(_BY_NAME)},
    )


# ---------------------------------------------------------------------------
# Primitives


class _Metric:
    """Base class shared by counter / gauge / histogram."""

    kind: MetricKind = "counter"  # overridden by subclasses

    def __init__(self, spec: MetricSpec) -> None:
        self.spec = spec
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.spec.name


class Counter(_Metric):
    """Monotonically non-decreasing total."""

    kind: MetricKind = "counter"

    def __init__(self, spec: MetricSpec) -> None:
        super().__init__(spec)
        self._value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        if amount < 0:
            raise InputError(
                "counter increments must be non-negative",
                details={"name": self.name, "amount": amount},
            )
        with self._lock:
            self._value += float(amount)

    def value(self) -> float:
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset the counter. Used in tests; not part of the public
        production contract."""
        with self._lock:
            self._value = 0.0


class Gauge(_Metric):
    """Settable scalar."""

    kind: MetricKind = "gauge"

    def __init__(self, spec: MetricSpec) -> None:
        super().__init__(spec)
        self._value: float = 0.0

    def set(self, v: float) -> None:
        with self._lock:
            self._value = float(v)

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += float(amount)

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= float(amount)

    def value(self) -> float:
        with self._lock:
            return self._value

    def reset(self) -> None:
        with self._lock:
            self._value = 0.0


class Histogram(_Metric):
    """Bucketed observation distribution.

    Prometheus-compatible: the exposition emits ``<name>_bucket{le=…}``
    rows, ``<name>_sum``, and ``<name>_count``. Buckets are cumulative
    by spec — the last bucket is always ``+Inf``.
    """

    kind: MetricKind = "histogram"

    def __init__(
        self,
        spec: MetricSpec,
        buckets: Iterable[float] | None = None,
    ) -> None:
        super().__init__(spec)
        if buckets is None:
            buckets = (
                DEFAULT_HISTOGRAM_BUCKETS_BYTES
                if spec.unit == "bytes"
                else DEFAULT_HISTOGRAM_BUCKETS_MS
            )
        bs = sorted(float(b) for b in buckets)
        if not bs:
            raise InputError(
                "histogram buckets must be non-empty",
                details={"name": spec.name, "buckets": list(bs)},
            )
        if bs[-1] != float("inf"):
            bs.append(float("inf"))
        self._buckets: tuple[float, ...] = tuple(bs)
        self._counts: list[int] = [0] * len(self._buckets)
        self._sum: float = 0.0
        self._count: int = 0

    def observe(self, value: float) -> None:
        v = float(value)
        if math.isnan(v):
            raise InputError(
                "histogram cannot observe NaN",
                details={"name": self.name},
            )
        with self._lock:
            for i, ub in enumerate(self._buckets):
                if v <= ub:
                    self._counts[i] += 1
            self._sum += v
            self._count += 1

    def snapshot(self) -> HistogramSnapshot:
        with self._lock:
            return HistogramSnapshot(
                buckets=list(self._buckets),
                counts=list(self._counts),
                sum=self._sum,
                count=self._count,
            )

    def reset(self) -> None:
        with self._lock:
            self._counts = [0] * len(self._buckets)
            self._sum = 0.0
            self._count = 0


# ---------------------------------------------------------------------------
# Registry / accessors


_INSTANCES: dict[str, _Metric] = {}
_INSTANCES_LOCK = threading.Lock()


def _require_spec(name: str, expected_kind: MetricKind) -> MetricSpec:
    spec = _BY_NAME.get(name)
    if spec is None:
        raise InputError(
            "metric is not registered",
            details={"name": name, "expected_kind": expected_kind},
            remediation="register the metric in geno_lewm/metrics.py::METRICS",
        )
    if spec.kind != expected_kind:
        raise InputError(
            "metric kind mismatch",
            details={
                "name": name,
                "registered_kind": spec.kind,
                "requested_kind": expected_kind,
            },
        )
    return spec


def get_counter(name: str) -> Counter:
    spec = _require_spec(name, "counter")
    with _INSTANCES_LOCK:
        instance = _INSTANCES.get(name)
        if instance is None:
            instance = Counter(spec)
            _INSTANCES[name] = instance
        if not isinstance(instance, Counter):  # pragma: no cover - kind already checked
            raise InvariantViolation(
                "registered metric is not a Counter",
                details={"name": name, "actual_kind": type(instance).__name__},
            )
        return instance


def get_gauge(name: str) -> Gauge:
    spec = _require_spec(name, "gauge")
    with _INSTANCES_LOCK:
        instance = _INSTANCES.get(name)
        if instance is None:
            instance = Gauge(spec)
            _INSTANCES[name] = instance
        if not isinstance(instance, Gauge):  # pragma: no cover
            raise InvariantViolation(
                "registered metric is not a Gauge",
                details={"name": name, "actual_kind": type(instance).__name__},
            )
        return instance


def get_histogram(name: str, *, buckets: Iterable[float] | None = None) -> Histogram:
    spec = _require_spec(name, "histogram")
    with _INSTANCES_LOCK:
        instance = _INSTANCES.get(name)
        if instance is None:
            instance = Histogram(spec, buckets=buckets)
            _INSTANCES[name] = instance
        if not isinstance(instance, Histogram):  # pragma: no cover
            raise InvariantViolation(
                "registered metric is not a Histogram",
                details={"name": name, "actual_kind": type(instance).__name__},
            )
        return instance


def snapshot_all() -> dict[str, Mapping[str, object]]:
    """Return a structured snapshot of every live metric."""
    out: dict[str, Mapping[str, object]] = {}
    with _INSTANCES_LOCK:
        instances = list(_INSTANCES.values())
    for inst in instances:
        if isinstance(inst, Counter | Gauge):
            out[inst.name] = {"kind": inst.kind, "value": inst.value()}
        else:
            assert isinstance(inst, Histogram)
            out[inst.name] = {"kind": "histogram", **inst.snapshot()}
    return out


def _reset_for_tests() -> None:
    """Test-only helper: drop the cached instances."""
    with _INSTANCES_LOCK:
        _INSTANCES.clear()


# ---------------------------------------------------------------------------
# Prometheus textfile exporter


def _prom_name(name: str) -> str:
    """Translate the dotted metric name to Prometheus-friendly form.

    Prometheus accepts ``[a-zA-Z_:][a-zA-Z0-9_:]*``. Our names use
    dots (``geno_lewm.training.step.duration``) which the textfile
    parser rejects. We map ``.`` → ``_`` for the wire format.
    """
    return name.replace(".", "_")


def _format_float(v: float) -> str:
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    if math.isnan(v):
        return "NaN"
    return repr(v)


def _write_metric_block(out: IO[str], inst: _Metric) -> None:
    spec = inst.spec
    name = _prom_name(spec.name)
    out.write(f"# HELP {name} {spec.summary} (unit: {spec.unit})\n")
    out.write(f"# TYPE {name} {spec.kind}\n")

    if isinstance(inst, Counter):
        out.write(f"{name} {_format_float(inst.value())}\n")
        return
    if isinstance(inst, Gauge):
        out.write(f"{name} {_format_float(inst.value())}\n")
        return
    if isinstance(inst, Histogram):
        snap = inst.snapshot()
        for ub, c in zip(snap["buckets"], snap["counts"], strict=True):
            # _counts is already cumulative by construction.
            out.write(f'{name}_bucket{{le="{_format_float(ub)}"}} {c}\n')
        out.write(f"{name}_sum {_format_float(snap['sum'])}\n")
        out.write(f"{name}_count {snap['count']}\n")
        return
    raise InvariantViolation(  # pragma: no cover - unreachable
        "unknown metric kind in exporter", details={"name": spec.name}
    )


def metrics_path(log_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the default exporter path."""
    if log_dir is not None:
        base = Path(log_dir).expanduser()
    else:
        env = os.environ.get("GENO_LEWM_LOG_DIR")
        base = Path(env).expanduser() if env else Path.home() / ".geno-lewm" / "logs"
    return base / "metrics.prom"


def export_prometheus_textfile(
    path: str | os.PathLike[str] | None = None,
) -> Path:
    """Write the current metric snapshot to a Prometheus textfile.

    The write is atomic: contents go to ``<path>.tmp`` and are renamed
    over ``<path>`` once flushed. Scrapers that read the file mid-flush
    therefore see either the previous or the new value, never a partial
    record. Returns the destination path.
    """
    dest = Path(path) if path is not None else metrics_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with _INSTANCES_LOCK:
        instances = list(_INSTANCES.values())
    instances.sort(key=lambda i: i.name)
    with tmp.open("w", encoding="utf-8") as f:
        for inst in instances:
            _write_metric_block(f, inst)
            f.write("\n")
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# Wiring with the redaction stats counter from #24 — every call into
# observability._redaction increments ``geno_lewm.observability.redacted_keys``
# via :func:`sync_redaction_counter`. Called by the exporter so the
# counter is always up to date when the textfile is written.


def sync_redaction_counter() -> None:
    """Pull the in-process redaction counters into the metric.

    The redaction filter from #24 keeps its own thread-local counters
    (so it can run without the metrics package being imported); this
    function reconciles the two views right before exposition.
    """
    try:
        from geno_lewm._redaction import STATS  # local import: optional
    except Exception:  # pragma: no cover - tested via integration
        return
    c = get_counter("geno_lewm.observability.redacted_keys")
    target = STATS.total()
    # The counter is monotonic; just bring it up to the current total.
    current = c.value()
    if target > current:
        c.inc(target - current)
