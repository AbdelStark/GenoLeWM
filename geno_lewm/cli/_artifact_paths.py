# SPDX-License-Identifier: Apache-2.0
"""Shared CLI helpers for release-portable artifact paths."""

from __future__ import annotations

from pathlib import Path

from geno_lewm.errors import InputError


def package_relative_artifact_path(
    path: Path,
    *,
    root_dir: Path,
    label: str,
    outside_message: str,
    root_detail: str,
    remediation: str,
) -> str:
    """Return a package-relative artifact path, rejecting private host paths."""
    raw = str(path).strip()
    if not raw:
        raise InputError(
            "artifact path must be non-empty",
            details={"artifact": label},
        )
    if "://" in raw:
        raise InputError(
            "artifact paths must be package-relative",
            details={"artifact": label, "path": raw},
        )
    if path.is_absolute():
        try:
            relative = path.resolve().relative_to(root_dir.resolve())
        except (OSError, RuntimeError, ValueError) as exc:
            raise InputError(
                outside_message,
                details={"artifact": label, "path": str(path), root_detail: str(root_dir)},
                remediation=remediation,
            ) from exc
    else:
        relative = path
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise InputError(
            "artifact paths must be package-relative",
            details={"artifact": label, "path": raw},
        )
    return relative.as_posix()


def require_package_relative_artifacts(
    artifacts: dict[str, str],
    *,
    input_index: int | None = None,
) -> None:
    """Reject artifact table values that cannot survive package extraction."""
    for key, value in artifacts.items():
        details: dict[str, object] = {"artifact": key, "path": value}
        if input_index is not None:
            details["input_index"] = input_index
        if "://" in value:
            raise InputError(
                "metrics JSON artifact paths must be package-relative",
                details=details,
            )
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise InputError(
                "metrics JSON artifact paths must be package-relative",
                details=details,
            )
