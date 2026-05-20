"""Structured logging for GenoLeWM.

This module is the single source of truth for the JSONL logging format
defined in ``docs/spec/05-observability.md`` and RFC-0013.

What it provides:

- :data:`EVENTS` — the immutable registry of event names, severities,
  and summaries. Renaming an event name is a MAJOR change.
- :func:`get_logger` — factory returning a :class:`GenoLeWMLogger`
  bound to a component (subsystem). Loggers share the same sinks per
  ``run_id`` so concurrent components write to one ordered stream.
- :func:`logged_run` — context manager that opens / closes the per-run
  sink, emits ``run.start`` / ``run.end`` style book-ends if asked,
  and flushes the buffer on any exception so records survive a crash.

What it does NOT provide (deferred to follow-up issues):

- Redaction filter (#24). The logger currently accepts whatever payload
  it is given; the filter will plug into :class:`GenoLeWMLogger._emit`.
- Metrics registry / Prometheus exporter (#25).
- ``registered_event_name`` AST linter (#27).
- wandb / OpenTelemetry sinks (RFC-0013 §"Sinks").

The interface ships first so dependent subsystems can take a hard
dependency on the event registry today.
"""

from __future__ import annotations

import contextlib
import contextvars
import io
import os
import sys
import threading
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Literal

try:  # stdlib in 3.11+
    import json
except ImportError:  # pragma: no cover - 3.10 still ships json
    raise

__all__ = [
    "EventSpec",
    "EVENTS",
    "Severity",
    "GenoLeWMLogger",
    "LogRecord",
    "get_logger",
    "logged_run",
    "current_trace_context",
    "set_trace_context",
]

Severity = Literal["debug", "info", "warn", "error"]
_SEVERITY_ORDER: dict[Severity, int] = {"debug": 0, "info": 1, "warn": 2, "error": 3}
_VALID_SEVERITIES: frozenset[str] = frozenset(_SEVERITY_ORDER)


# ---------------------------------------------------------------------------
# Event registry
# Source of truth for ``docs/api/log-events.md`` (regenerated at release).


@dataclass(frozen=True)
class EventSpec:
    """A single row in the :data:`EVENTS` registry.

    ``allowed_keys`` lists the ``data`` keys that the per-event redaction
    allowlist permits (RFC-0013 §3.5). Standardized fields (``step``,
    ``epoch``, ``phase``, ``duration_ms``, ``trace_id``, ``span_id``,
    ``error_code``) are promoted out of ``data`` before redaction and
    are always allowed at the top level — they need not appear here.
    """

    name: str
    severity: Severity
    summary: str
    allowed_keys: frozenset[str] = frozenset()


#: Canonical events for v0.1. Order matches ``docs/spec/05-observability.md``.
#:
#: Each ``allowed_keys`` enumerates the per-event redaction allowlist.
#: New keys for an existing event are a MINOR change; tightening the set
#: is MAJOR.
EVENTS: tuple[EventSpec, ...] = (
    EventSpec(
        "training.run.start",
        "info",
        "trainer initialized",
        frozenset({"config_path", "model_id", "device", "world_size", "git_sha"}),
    ),
    EventSpec(
        "training.run.end",
        "info",
        "trainer exited",
        frozenset({"reason", "best_step", "best_loss"}),
    ),
    EventSpec(
        "training.step",
        "debug",
        "every training step (sampled)",
        frozenset({"loss", "loss_pred", "loss_reg", "lr", "grad_norm"}),
    ),
    EventSpec(
        "training.epoch.end",
        "info",
        "every epoch",
        frozenset({"loss", "loss_pred", "loss_reg", "lr", "samples"}),
    ),
    EventSpec(
        "training.checkpoint.write",
        "info",
        "checkpoint saved",
        frozenset({"path", "size_bytes", "tag"}),
    ),
    EventSpec(
        "training.collapse.alert",
        "warn",
        "collapse alert criterion tripped",
        frozenset({"criterion", "value", "threshold"}),
    ),
    EventSpec(
        "training.metric",
        "info",
        "scalar metric logged",
        frozenset({"name", "value", "unit", "kind"}),
    ),
    EventSpec(
        "eval.run.start",
        "info",
        "benchmark started",
        frozenset({"benchmark", "dataset", "n_items"}),
    ),
    EventSpec(
        "eval.run.end",
        "info",
        "benchmark finished",
        frozenset({"benchmark", "metrics", "elapsed_s"}),
    ),
    EventSpec(
        "eval.regression",
        "error",
        "smoke eval regression",
        frozenset({"metric", "baseline", "current", "delta", "threshold"}),
    ),
    EventSpec(
        "data.cache.hit",
        "debug",
        "cache hit",
        frozenset({"shard_id", "key"}),
    ),
    EventSpec(
        "data.cache.miss",
        "debug",
        "cache miss",
        frozenset({"shard_id", "key"}),
    ),
    EventSpec(
        "data.shard.write",
        "info",
        "new shard written",
        frozenset({"shard_id", "path", "n_rows", "size_bytes"}),
    ),
    EventSpec(
        "inference.score.start",
        "debug",
        "scoring call entered",
        frozenset({"variant_id", "model_id"}),
    ),
    EventSpec(
        "inference.score.end",
        "debug",
        "scoring call returned",
        frozenset({"variant_id", "score", "category"}),
    ),
    EventSpec(
        "inference.batch.end",
        "info",
        "batch finished",
        frozenset({"n", "batch_id", "throughput_per_s"}),
    ),
    EventSpec(
        "inference.network.blocked",
        "error",
        "fail-closed network guard tripped",
        frozenset({"url_host", "operation"}),
    ),
    EventSpec(
        "attestation.receipt.write",
        "info",
        "receipt written",
        frozenset({"path", "model_id", "n_outputs"}),
    ),
    EventSpec(
        "attestation.verify.start",
        "info",
        "verifier started",
        frozenset({"path", "model_id"}),
    ),
    EventSpec(
        "attestation.verify.end",
        "info",
        "verifier returned",
        frozenset({"path", "ok"}),
    ),
    EventSpec(
        "attestation.verify.mismatch",
        "error",
        "a hash check failed",
        frozenset({"kind", "expected", "got"}),
    ),
    EventSpec(
        "error",
        "error",
        "a GenoLeWMError raised inside a logged span",
        frozenset({"message", "details", "remediation"}),
    ),
)

_EVENTS_BY_NAME: dict[str, EventSpec] = {e.name: e for e in EVENTS}
if len(_EVENTS_BY_NAME) != len(EVENTS):  # pragma: no cover - tested via registry test
    # Same defensive check as the errors registry: catch duplicates at import.
    from geno_lewm.errors import InvariantViolation

    raise InvariantViolation(
        "Duplicate event name in EVENTS registry",
        details={"len_registry": len(EVENTS), "len_unique_names": len(_EVENTS_BY_NAME)},
    )


def _is_registered_event(name: str) -> bool:
    return name in _EVENTS_BY_NAME


# ---------------------------------------------------------------------------
# Trace context — a contextvar pair carrying OTel-shaped IDs.
# Tracing is optional in v1 (RFC-0013 §"Tracing"); the logger just attaches
# the IDs to records when they are set so downstream sinks can join.


_TRACE_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
_SPAN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("span_id", default=None)


def current_trace_context() -> tuple[str | None, str | None]:
    """Return ``(trace_id, span_id)`` from the current context."""
    return _TRACE_ID.get(), _SPAN_ID.get()


@contextlib.contextmanager
def set_trace_context(*, trace_id: str | None, span_id: str | None) -> Iterator[None]:
    """Push ``(trace_id, span_id)`` into the contextvar for the block."""
    t_tok = _TRACE_ID.set(trace_id)
    s_tok = _SPAN_ID.set(span_id)
    try:
        yield
    finally:
        _TRACE_ID.reset(t_tok)
        _SPAN_ID.reset(s_tok)


# ---------------------------------------------------------------------------
# Record


@dataclass
class LogRecord:
    """One row written by the logger.

    The record carries the spec-required fields directly and stashes
    event-specific structured fields under :attr:`data`. ``to_dict``
    returns the exact wire shape — keys are stable across versions.
    """

    ts: str
    severity: Severity
    event: str
    run_id: str
    component: str
    data: dict[str, Any] = field(default_factory=dict)
    step: int | None = None
    epoch: int | None = None
    phase: str | None = None
    duration_ms: int | None = None
    trace_id: str | None = None
    span_id: str | None = None
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ts": self.ts,
            "severity": self.severity,
            "event": self.event,
            "run_id": self.run_id,
            "component": self.component,
            "data": self.data,
        }
        for k in ("step", "epoch", "phase", "duration_ms", "trace_id", "span_id", "error_code"):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        return out


# ---------------------------------------------------------------------------
# Sink — per-run shared writer
# Each (run_id, log_dir) tuple gets exactly one sink so concurrent
# components write to a single ordered JSONL stream. Append-only.


@dataclass
class _Sink:
    path: Path
    stream: IO[str]
    lock: threading.Lock = field(default_factory=threading.Lock)

    def write_line(self, line: str) -> None:
        with self.lock:
            self.stream.write(line)
            if not line.endswith("\n"):
                self.stream.write("\n")
            self.stream.flush()

    def close(self) -> None:
        with self.lock:
            try:
                self.stream.flush()
            finally:
                if not isinstance(self.stream, io.StringIO):
                    self.stream.close()


_SINKS: dict[tuple[str, str], _Sink] = {}
_SINKS_LOCK = threading.Lock()


def _resolve_log_dir(explicit: str | os.PathLike[str] | None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    env = os.environ.get("GENO_LEWM_LOG_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".geno-lewm" / "logs"


def _open_sink(run_id: str, log_dir: Path) -> _Sink:
    key = (str(log_dir.resolve()), run_id)
    with _SINKS_LOCK:
        sink = _SINKS.get(key)
        if sink is not None:
            return sink
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{run_id}.jsonl"
        # Line-buffered text mode: every ``\n`` triggers a flush so a
        # crash that bypasses our explicit ``flush`` still preserves
        # everything up to the last newline.
        stream = path.open("a", buffering=1, encoding="utf-8")
        sink = _Sink(path=path, stream=stream)
        _SINKS[key] = sink
        return sink


def _close_sink(run_id: str, log_dir: Path) -> None:
    key = (str(log_dir.resolve()), run_id)
    with _SINKS_LOCK:
        sink = _SINKS.pop(key, None)
    if sink is not None:
        sink.close()


# ---------------------------------------------------------------------------
# Pretty stderr formatter — only used when stderr is a TTY.


def _format_pretty(rec: LogRecord) -> str:
    parts = [rec.ts, rec.severity.upper().ljust(5), rec.event, rec.component]
    if rec.step is not None:
        parts.append(f"step={rec.step}")
    if rec.error_code:
        parts.append(rec.error_code)
    if rec.data:
        parts.append(json.dumps(rec.data, sort_keys=True, default=str))
    return " | ".join(parts)


def _stderr_is_tty() -> bool:
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


# ---------------------------------------------------------------------------
# Logger


class GenoLeWMLogger:
    """Component-scoped structured logger.

    Loggers are cheap to construct (cached by ``(component, run_id)``)
    and thread-safe: the underlying sink serializes writes.
    """

    def __init__(
        self,
        component: str,
        *,
        run_id: str,
        log_dir: Path,
        sink: _Sink,
        level: Severity = "info",
        pretty: bool = False,
    ) -> None:
        self.component = component
        self.run_id = run_id
        self.log_dir = log_dir
        self._sink = sink
        self._level = level
        self._pretty = pretty

    @property
    def level(self) -> Severity:
        return self._level

    def set_level(self, level: Severity) -> None:
        if level not in _VALID_SEVERITIES:
            from geno_lewm.errors import InputError

            raise InputError(
                f"unknown severity {level!r}",
                details={"valid": sorted(_VALID_SEVERITIES)},
                remediation="pass one of debug | info | warn | error",
            )
        self._level = level

    # ---- public API -----------------------------------------------------

    def debug(self, event: str, **fields: Any) -> LogRecord | None:
        return self._log("debug", event, fields)

    def info(self, event: str, **fields: Any) -> LogRecord | None:
        return self._log("info", event, fields)

    def warn(self, event: str, **fields: Any) -> LogRecord | None:
        return self._log("warn", event, fields)

    def error(self, event: str, **fields: Any) -> LogRecord | None:
        return self._log("error", event, fields)

    # ---- internals ------------------------------------------------------

    def _log(self, severity: Severity, event: str, fields: Mapping[str, Any]) -> LogRecord | None:
        if _SEVERITY_ORDER[severity] < _SEVERITY_ORDER[self._level]:
            return None

        spec = _EVENTS_BY_NAME.get(event)
        if spec is None:
            # Unknown event names are a contract violation but the
            # AST linter (#27) catches them at PR time. At runtime we
            # still emit, but tag the record so a downstream sink can
            # alert. INV-OBS-1 is enforced by the linter; this code
            # path remains best-effort to avoid crashing user runs.
            pass

        # Pull spec-standardized fields out of the kwargs.
        step = fields.get("step")
        epoch = fields.get("epoch")
        phase = fields.get("phase")
        duration_ms = fields.get("duration_ms")
        error_code = fields.get("error_code")

        # Everything else goes into ``data``. We copy so callers can
        # reuse their kwarg dict freely without aliasing the record.
        reserved = {"step", "epoch", "phase", "duration_ms", "error_code"}
        raw_data = {k: v for k, v in fields.items() if k not in reserved}

        # The redaction filter is the single chokepoint between callers
        # and the sink (RFC-0013 §3.5, INV-OBS-3). Unknown events get an
        # empty allowlist → every payload key is soft-dropped.
        from geno_lewm._redaction import redact as _redact

        allowed = spec.allowed_keys if spec is not None else frozenset()
        data = _redact(event, raw_data, allowed_keys=allowed)

        trace_id, span_id = current_trace_context()

        rec = LogRecord(
            ts=datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            severity=severity,
            event=event,
            run_id=self.run_id,
            component=self.component,
            data=data,
            step=step,
            epoch=epoch,
            phase=phase,
            duration_ms=duration_ms,
            trace_id=trace_id,
            span_id=span_id,
            error_code=error_code,
        )

        line = json.dumps(rec.to_dict(), sort_keys=True, default=str)
        self._sink.write_line(line)

        if self._pretty:
            sys.stderr.write(_format_pretty(rec) + "\n")

        return rec


# ---------------------------------------------------------------------------
# Factory


_LOGGERS: dict[tuple[str, str, str], GenoLeWMLogger] = {}
_LOGGERS_LOCK = threading.Lock()


def _env_level() -> Severity:
    raw = os.environ.get("GENO_LEWM_LOG_LEVEL", "info").lower()
    if raw in _VALID_SEVERITIES:
        return raw  # type: ignore[return-value]
    return "info"


def _env_pretty() -> bool:
    fmt = os.environ.get("GENO_LEWM_LOG_FORMAT")
    if fmt is not None:
        return fmt.lower() == "pretty"
    return _stderr_is_tty()


def _new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


def get_logger(
    component: str,
    *,
    run_id: str | None = None,
    log_dir: str | os.PathLike[str] | None = None,
    level: Severity | None = None,
    pretty: bool | None = None,
) -> GenoLeWMLogger:
    """Return a logger bound to ``component``.

    Loggers are cached by ``(component, run_id, log_dir)``; calling
    ``get_logger`` twice with the same arguments returns the same
    instance, so independent subsystems share one ordered stream per
    run.

    Defaults:

    - ``run_id``: ``$GENO_LEWM_RUN_ID`` or a random ``run-<hex>``.
    - ``log_dir``: ``$GENO_LEWM_LOG_DIR`` or ``~/.geno-lewm/logs``.
    - ``level``: ``$GENO_LEWM_LOG_LEVEL`` (default ``info``).
    - ``pretty``: TTY-detected, overridable by ``$GENO_LEWM_LOG_FORMAT``.
    """
    rid = run_id or os.environ.get("GENO_LEWM_RUN_ID") or _new_run_id()
    ldir = _resolve_log_dir(log_dir)
    sink = _open_sink(rid, ldir)
    lvl: Severity = level if level is not None else _env_level()
    pp = pretty if pretty is not None else _env_pretty()

    key = (component, rid, str(ldir.resolve()))
    with _LOGGERS_LOCK:
        existing = _LOGGERS.get(key)
        if existing is not None:
            return existing
        logger = GenoLeWMLogger(
            component=component,
            run_id=rid,
            log_dir=ldir,
            sink=sink,
            level=lvl,
            pretty=pp,
        )
        _LOGGERS[key] = logger
        return logger


# ---------------------------------------------------------------------------
# Crash-safe run wrapper


@contextlib.contextmanager
def logged_run(
    component: str = "runtime",
    *,
    run_id: str | None = None,
    log_dir: str | os.PathLike[str] | None = None,
    start_event: str | None = None,
    end_event: str | None = None,
    start_data: Mapping[str, Any] | None = None,
) -> Iterator[GenoLeWMLogger]:
    """Open a sink for the run; flush on exit; never swallow exceptions.

    The wrapper guarantees that any records emitted up to a crash are
    flushed to disk (INV-OBS-6: "a crash before logger init still
    produces a sanitized minimal record"). Optional ``start_event`` /
    ``end_event`` book-end the run. If the block raises and the
    exception is a ``geno_lewm.errors.GenoLeWMError``, an ``error``
    record is emitted before the exception propagates.
    """
    logger = get_logger(component, run_id=run_id, log_dir=log_dir)
    rid = logger.run_id
    ldir = logger.log_dir
    if start_event:
        logger.info(start_event, **(dict(start_data) if start_data else {}))
    try:
        yield logger
    except BaseException as exc:
        from geno_lewm.errors import GenoLeWMError  # local import to avoid cycle

        if isinstance(exc, GenoLeWMError):
            logger.error(
                "error",
                error_code=exc.code,
                message=exc.message,
                details=exc.details,
                remediation=exc.remediation,
            )
        # Always flush the sink before the exception unwinds the stack.
        try:
            logger._sink.stream.flush()
        except Exception:  # pragma: no cover - flush is best effort on crash
            pass
        raise
    else:
        if end_event:
            logger.info(end_event)
    finally:
        # Close the sink iff no other logger is still bound to it.
        # We do not aggressively close so concurrent components can
        # share the run; explicit teardown is via ``shutdown_run``.
        pass


def shutdown_run(run_id: str, log_dir: str | os.PathLike[str] | None = None) -> None:
    """Flush and close the sink for ``run_id``.

    Primarily used in tests; production callers can leave sinks open
    for the process lifetime.
    """
    ldir = _resolve_log_dir(log_dir)
    _close_sink(run_id, ldir)
    # Also drop any cached loggers bound to this run.
    with _LOGGERS_LOCK:
        for k in [k for k in _LOGGERS if k[1] == run_id]:
            _LOGGERS.pop(k, None)
