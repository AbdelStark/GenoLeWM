# SPDX-License-Identifier: Apache-2.0
"""Enforce the observability contract at PR time.

Two AST checks (RFC-0013 INV-OBS-1/2, RFC-0015 §3.4):

1. ``registered_event_name`` — every ``logger.{debug,info,warn,error}(
   "<event>", …)`` call site passes a literal that appears in
   ``geno_lewm/observability.py::EVENTS``.

2. ``registered_metric_name`` — every ``counter.inc(NAME)`` /
   ``histogram.observe(NAME)`` call site passes a literal registered in
   the (forthcoming) ``METRICS`` tuple. The metrics registry ships with
   #25; until then the check skips when ``METRICS`` is absent.

Parses ``observability.py`` with the ``ast`` module to discover both
registries so the linter has no runtime dependency on the package.

Usage:

    python -m tools.lint.check_event_names [PATH ...]

Exit code is 0 on success, 1 on any violation.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "geno_lewm"
OBSERVABILITY_MODULE = PACKAGE_DIR / "observability.py"
METRICS_MODULE = PACKAGE_DIR / "metrics.py"

_LOGGER_LEVEL_METHODS: frozenset[str] = frozenset({"debug", "info", "warn", "warning", "error"})
_METRIC_METHODS: frozenset[str] = frozenset({"inc", "observe", "set"})


@dataclass(frozen=True)
class Violation:
    """A single linter offence."""

    path: Path
    line: int
    col: int
    check: str
    message: str

    def format(self, root: Path) -> str:
        try:
            rel = self.path.relative_to(root)
        except ValueError:
            rel = self.path
        return f"{rel}:{self.line}:{self.col}: error: [{self.check}] {self.message}"


# ---------------------------------------------------------------------------
# Registry discovery


def _read_tuple_of_calls(tree: ast.AST, assign_name: str, ctor_name: str) -> set[str]:
    """Return the set of first-arg string literals from ``CTOR(...)``
    elements appearing in a tuple assigned to ``ASSIGN_NAME``."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == assign_name for t in targets):
            continue
        value = node.value
        if value is None or not isinstance(value, ast.Tuple):
            continue
        for entry in value.elts:
            if not (
                isinstance(entry, ast.Call)
                and isinstance(entry.func, ast.Name)
                and entry.func.id == ctor_name
                and entry.args
                and isinstance(entry.args[0], ast.Constant)
                and isinstance(entry.args[0].value, str)
            ):
                continue
            out.add(entry.args[0].value)
    return out


def discover_registered_events(module: Path = OBSERVABILITY_MODULE) -> set[str]:
    """Return the set of registered event-name string literals."""
    if not module.is_file():
        return set()
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    return _read_tuple_of_calls(tree, "EVENTS", "EventSpec")


def discover_registered_metrics(module: Path = METRICS_MODULE) -> set[str] | None:
    """Return registered metric names, or ``None`` if METRICS is absent.

    Returning ``None`` lets the caller skip the metric check entirely
    until the metrics module ships (#25). When the module lands, the
    check arms automatically.
    """
    if not module.is_file():
        return None
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    # METRICS is expected to be a tuple of MetricSpec(name, ...) — same shape.
    found = _read_tuple_of_calls(tree, "METRICS", "MetricSpec")
    if not found:
        # METRICS variable might exist as something other than a tuple of
        # MetricSpec(...) — at registry-bootstrap time treat the check as
        # not-yet-armed.
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign | ast.AnnAssign):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == "METRICS" for t in targets):
                    return found
        return None
    return found


# ---------------------------------------------------------------------------
# AST walks


def _is_logger_method_call(node: ast.Call) -> ast.Attribute | None:
    """Return the Attribute node iff ``node`` looks like
    ``logger.debug/info/warn/error(...)``."""
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in _LOGGER_LEVEL_METHODS:
        return func
    return None


def _is_metric_method_call(node: ast.Call) -> ast.Attribute | None:
    """Return the Attribute node iff ``node`` looks like
    ``<x>.inc(...)`` / ``observe(...)`` / ``set(...)``.

    Name-based: we cannot prove the callee is actually a counter
    without type inference, but the convention in this codebase is
    explicit (`counter.inc`, `histogram.observe`). False positives are
    fine — every spelling of those method names must take a registered
    metric literal anyway.
    """
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in _METRIC_METHODS:
        # Be conservative: only flag when the callee resembles a metric
        # handle by name — ``foo.inc`` / ``foo_counter.inc`` /
        # ``foo_histogram.observe``. This excludes ``Counter.inc(self,…)``
        # type definitions and unrelated ``.inc`` calls in vector math.
        value = func.value
        receiver_name: str | None = None
        if isinstance(value, ast.Name):
            receiver_name = value.id
        elif isinstance(value, ast.Attribute):
            receiver_name = value.attr
        if receiver_name and (
            "counter" in receiver_name.lower()
            or "histogram" in receiver_name.lower()
            or "gauge" in receiver_name.lower()
            or "metric" in receiver_name.lower()
        ):
            return func
    return None


def _const_str(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def check_file(
    path: Path,
    *,
    events: set[str],
    metrics: set[str] | None,
) -> list[Violation]:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        return [
            Violation(
                path=path,
                line=e.lineno or 0,
                col=(e.offset or 0) - 1,
                check="registered_event_name",
                message=f"could not parse file: {e.msg}",
            )
        ]

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Logger call?
        attr = _is_logger_method_call(node)
        if attr is not None:
            # First positional argument is the event name.
            if not node.args:
                continue
            literal = _const_str(node.args[0])
            if literal is None:
                # Dynamic event name (variable / f-string). Cannot
                # statically check; defer to runtime.
                continue
            if literal not in events:
                violations.append(
                    Violation(
                        path=path,
                        line=node.lineno,
                        col=node.col_offset,
                        check="registered_event_name",
                        message=(
                            f"event name {literal!r} is not in EVENTS. "
                            "Register it in geno_lewm/observability.py::EVENTS "
                            "(RFC-0013 §3.3, INV-OBS-1)."
                        ),
                    )
                )
            continue

        # Metric call?
        m_attr = _is_metric_method_call(node)
        if m_attr is not None and metrics is not None:
            if not node.args:
                continue
            literal = _const_str(node.args[0])
            if literal is None:
                continue
            if literal not in metrics:
                violations.append(
                    Violation(
                        path=path,
                        line=node.lineno,
                        col=node.col_offset,
                        check="registered_metric_name",
                        message=(
                            f"metric name {literal!r} is not in METRICS. "
                            "Register it in geno_lewm/observability.py::METRICS "
                            "(RFC-0013 §4, INV-OBS-2)."
                        ),
                    )
                )
    return violations


def _walk_python_files(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            yield p
        elif p.is_dir():
            yield from sorted(p.rglob("*.py"))


def run(paths: Sequence[Path] | None = None) -> list[Violation]:
    targets = list(paths) if paths else [PACKAGE_DIR]
    events = discover_registered_events()
    metrics = discover_registered_metrics()

    violations: list[Violation] = []
    for file in _walk_python_files(targets):
        # Skip the registry modules themselves — their own EventSpec /
        # MetricSpec literals would otherwise self-trigger.
        if file.resolve() in {OBSERVABILITY_MODULE.resolve(), METRICS_MODULE.resolve()}:
            continue
        violations.extend(check_file(file, events=events, metrics=metrics))
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    paths = [Path(a) for a in args] if args else [PACKAGE_DIR]
    violations = run(paths)
    for v in violations:
        print(v.format(REPO_ROOT), file=sys.stderr)
    if violations:
        print(
            f"check_event_names: {len(violations)} violation(s)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
