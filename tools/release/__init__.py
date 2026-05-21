# SPDX-License-Identifier: Apache-2.0
"""Maintainer-side release tooling.

Two mechanical helpers used by the release checklist in
``docs/spec/09-release-and-versioning.md#release-process``:

* :mod:`tools.release.bump` updates the single source of truth for
  the package version (``__version__`` in :mod:`geno_lewm`) and
  cross-checks that ``pyproject.toml`` consumes it dynamically.
* :mod:`tools.release.changelog` rewrites ``CHANGELOG.md`` by lifting
  the ``[Unreleased]`` section to a versioned heading (or by emitting a
  fresh section synthesised from ``git log``) following the
  Keep-a-Changelog 1.1.0 grammar.

Both helpers are pure stdlib and run as ``python -m tools.release.bump``
/ ``python -m tools.release.changelog`` so the release runner does
not need optional dependencies installed.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
