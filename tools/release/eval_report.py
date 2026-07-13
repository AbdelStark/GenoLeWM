# SPDX-License-Identifier: Apache-2.0
"""Compatibility wrapper for the installable evaluation-report core."""

from __future__ import annotations

from geno_lewm._evaluation_report import (
    ALLOWED_GENERATORS,
    EVAL_ALL_GENERATED_BY,
    EVAL_GENERATED_BY,
    PLACEHOLDER_RE,
    REQUIRED_ARTIFACTS,
    SCHEMA_VERSION,
    EvalReportInput,
    MetricResult,
    load_report_input,
    main,
    parse_report_input,
    render_report,
)

__all__ = [
    "ALLOWED_GENERATORS",
    "EVAL_ALL_GENERATED_BY",
    "EVAL_GENERATED_BY",
    "PLACEHOLDER_RE",
    "REQUIRED_ARTIFACTS",
    "SCHEMA_VERSION",
    "EvalReportInput",
    "MetricResult",
    "load_report_input",
    "main",
    "parse_report_input",
    "render_report",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
