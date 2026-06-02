"""Unit tests for ``tools.lint.check_event_names``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.lint import check_event_names as linter


def write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_registry_discovery_finds_canonical_events() -> None:
    events = linter.discover_registered_events()
    # Spot-check across families.
    for n in ("training.run.start", "eval.regression", "provenance.verify.end", "error"):
        assert n in events


def test_registered_event_passes(tmp_path: Path) -> None:
    f = write(
        tmp_path,
        "ok.py",
        "def f(logger):\n    logger.info('training.run.start', config_path='x')\n",
    )
    events = linter.discover_registered_events()
    assert linter.check_file(f, events=events, metrics=None) == []


def test_unregistered_event_is_flagged(tmp_path: Path) -> None:
    f = write(
        tmp_path,
        "bad.py",
        "def f(logger):\n    logger.info('totally.not.an.event', x=1)\n",
    )
    events = linter.discover_registered_events()
    v = linter.check_file(f, events=events, metrics=None)
    assert len(v) == 1
    assert v[0].check == "registered_event_name"
    assert "'totally.not.an.event'" in v[0].message


def test_dynamic_event_name_is_skipped(tmp_path: Path) -> None:
    # Variable event name — cannot statically check.
    f = write(
        tmp_path,
        "dyn.py",
        "def f(logger, name):\n    logger.info(name, x=1)\n",
    )
    events = linter.discover_registered_events()
    assert linter.check_file(f, events=events, metrics=None) == []


def test_f_string_event_name_is_skipped(tmp_path: Path) -> None:
    f = write(
        tmp_path,
        "fs.py",
        "def f(logger, t):\n    logger.info(f'training.{t}', x=1)\n",
    )
    events = linter.discover_registered_events()
    assert linter.check_file(f, events=events, metrics=None) == []


@pytest.mark.parametrize("method", ["debug", "info", "warn", "warning", "error"])
def test_every_logger_method_is_checked(tmp_path: Path, method: str) -> None:
    f = write(
        tmp_path,
        "m.py",
        f"def f(logger):\n    logger.{method}('not.registered.event')\n",
    )
    events = linter.discover_registered_events()
    v = linter.check_file(f, events=events, metrics=None)
    assert len(v) == 1
    assert v[0].check == "registered_event_name"


def test_metric_check_skipped_when_metrics_absent(tmp_path: Path) -> None:
    # METRICS hasn't shipped — the registry is None and the metric
    # branch never fires.
    f = write(
        tmp_path,
        "mm.py",
        "def f(counter):\n    counter.inc('imaginary.metric')\n",
    )
    events = linter.discover_registered_events()
    assert linter.check_file(f, events=events, metrics=None) == []


def test_metric_check_armed_when_registry_present(tmp_path: Path) -> None:
    f = write(
        tmp_path,
        "mm.py",
        "def f(counter):\n    counter.inc('imaginary.metric')\n",
    )
    events = linter.discover_registered_events()
    metrics = {"real.metric"}
    v = linter.check_file(f, events=events, metrics=metrics)
    assert len(v) == 1
    assert v[0].check == "registered_metric_name"


def test_metric_check_passes_with_registered_name(tmp_path: Path) -> None:
    f = write(
        tmp_path,
        "mm.py",
        "def f(counter):\n    counter.inc('real.metric', 1)\n",
    )
    events = linter.discover_registered_events()
    metrics = {"real.metric"}
    assert linter.check_file(f, events=events, metrics=metrics) == []


def test_metric_check_ignores_unrelated_inc_calls(tmp_path: Path) -> None:
    # ``foo.inc(1)`` on a non-metric object should not trigger — the
    # heuristic looks at the receiver name.
    f = write(
        tmp_path,
        "u.py",
        "def f(box):\n    box.inc(1)\n",
    )
    events = linter.discover_registered_events()
    assert linter.check_file(f, events=events, metrics={"x"}) == []


def test_main_returns_zero_for_clean_package() -> None:
    assert linter.main([str(linter.PACKAGE_DIR)]) == 0


def test_main_returns_one_with_violations(tmp_path: Path) -> None:
    write(
        tmp_path,
        "bad.py",
        "def f(logger):\n    logger.info('not.registered.event')\n",
    )
    assert linter.main([str(tmp_path)]) == 1


def test_violation_format_includes_path_line_col(tmp_path: Path) -> None:
    bad = write(tmp_path, "fmt.py", "\n\ndef f(l):\n    l.info('bogus')\n")
    events = linter.discover_registered_events()
    [v] = linter.check_file(bad, events=events, metrics=None)
    formatted = v.format(tmp_path)
    assert formatted.startswith("fmt.py:4:")
    assert "[registered_event_name]" in formatted


def test_observability_module_itself_is_skipped() -> None:
    # The module defines EventSpec literals; without the skip it would
    # appear to be raising event names of every spec.
    rc = linter.main([str(linter.OBSERVABILITY_MODULE)])
    assert rc == 0


def test_logger_call_without_event_arg_is_skipped(tmp_path: Path) -> None:
    # ``logger.info()`` with no args is malformed but not the linter's
    # job to catch.
    f = write(tmp_path, "noargs.py", "def f(l):\n    l.info()\n")
    events = linter.discover_registered_events()
    assert linter.check_file(f, events=events, metrics=None) == []
