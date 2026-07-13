# SPDX-License-Identifier: Apache-2.0
"""Compatibility wrapper for the installable calibration command.

The implementation lives inside :mod:`geno_lewm` so the wheel console script
does not depend on the source-only top-level ``tools`` package.  This wrapper
keeps existing ``python -m tools.release.build_calibration`` workflows stable.
"""

from __future__ import annotations

from geno_lewm.cli.calibrate import (
    GENERATED_BY,
    REPORT_NAME,
    SCHEMA_VERSION,
    build_calibration,
    main,
)

__all__ = [
    "GENERATED_BY",
    "REPORT_NAME",
    "SCHEMA_VERSION",
    "build_calibration",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
