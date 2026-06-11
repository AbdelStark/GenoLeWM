# SPDX-License-Identifier: Apache-2.0
"""pytest-benchmark microbenchmark suite (performance budget).

Every test in this package carries the ``bench`` marker so the default
test run (``pytest``) skips it. The nightly perf job runs::

    pytest tests/benchmark/ -m bench --benchmark-only --benchmark-json=...

with ``--benchmark-only`` (overrides the default ``--benchmark-disable``
in ``pyproject.toml``) so the timing data lands in the JSON file the
regression detector consumes.
"""
