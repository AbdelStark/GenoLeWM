# SPDX-License-Identifier: Apache-2.0
"""Compatibility wrapper for installable training-run evidence packaging.

The implementation lives inside :mod:`geno_lewm` so
``geno-lewm-train --package-release-run`` remains functional from a wheel
without the source-only top-level ``tools`` package.
"""

from __future__ import annotations

from geno_lewm._training_run_package import (
    ACCEPTED_STATUSES,
    BOUND_SCHEMA_VERSION,
    CARD_NAME,
    CHECKSUMS_NAME,
    COMMIT_RE,
    GENERATED_BY,
    GENERATED_FILES,
    MANIFEST_NAME,
    PLACEHOLDER_RE,
    REQUIRED_PREFLIGHT_DATASET_CORE_FILES,
    SCHEMA_VERSION,
    TRAINING_PREFLIGHT_KIND,
    TrainingArtifact,
    TrainingRunManifest,
    TrainingRunPackageReport,
    build_training_run_package,
    main,
    parse_training_run_metadata,
    render_training_run_card,
    verify_training_run_manifest,
)

__all__ = [
    "ACCEPTED_STATUSES",
    "BOUND_SCHEMA_VERSION",
    "CARD_NAME",
    "CHECKSUMS_NAME",
    "COMMIT_RE",
    "GENERATED_BY",
    "GENERATED_FILES",
    "MANIFEST_NAME",
    "PLACEHOLDER_RE",
    "REQUIRED_PREFLIGHT_DATASET_CORE_FILES",
    "SCHEMA_VERSION",
    "TRAINING_PREFLIGHT_KIND",
    "TrainingArtifact",
    "TrainingRunManifest",
    "TrainingRunPackageReport",
    "build_training_run_package",
    "main",
    "parse_training_run_metadata",
    "render_training_run_card",
    "verify_training_run_manifest",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
