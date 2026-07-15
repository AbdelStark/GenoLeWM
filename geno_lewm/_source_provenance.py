# SPDX-License-Identifier: Apache-2.0
"""Immutable source identity for production package execution."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from geno_lewm.errors import InputError

_FULL_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Source and package identity attached to a production checkpoint."""

    commit_sha: str
    tree_sha: str
    package_version: str

    def to_dict(self) -> dict[str, str]:
        return {
            "commit_sha": self.commit_sha,
            "tree_sha": self.tree_sha,
            "package_version": self.package_version,
        }


def resolve_package_source(
    *,
    package_version: str,
    package_root: Path | None = None,
) -> SourceProvenance:
    """Resolve source identity from the imported package, never the caller CWD."""
    root = (package_root or Path(__file__).resolve().parent).resolve()
    repository = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if root != (repository / "geno_lewm").resolve():
        raise InputError("production source must resolve from the tracked package root")
    for relative in ("geno_lewm/__init__.py", "geno_lewm/cli/train.py"):
        _git(repository, "ls-files", "--error-unmatch", relative)
    staged_package_files = _git(
        repository,
        "ls-files",
        "--stage",
        "--",
        "geno_lewm",
    ).splitlines()
    non_regular = sorted(
        line.split("\t", maxsplit=1)[-1]
        for line in staged_package_files
        if not line.startswith(("100644 ", "100755 "))
    )
    if non_regular:
        raise InputError(
            "production package source must contain only regular tracked files",
            details={"non_regular_paths": non_regular},
        )
    status = _git(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "geno_lewm",
    )
    if status:
        raise InputError(
            "production package source checkout must be clean",
            details={"dirty_paths": status.splitlines()},
        )
    ignored = _git(
        repository,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        "geno_lewm",
    ).splitlines()
    if ignored:
        raise InputError(
            "production package source contains ignored executable or data artifacts",
            details={"ignored_paths": sorted(ignored)},
            remediation=(
                "remove ignored bytecode/native artifacts and disable bytecode writes for the "
                "production launch"
            ),
        )
    commit_sha = _git(repository, "rev-parse", "HEAD").lower()
    tree_sha = _git(repository, "rev-parse", "HEAD^{tree}").lower()
    if _FULL_SHA.fullmatch(commit_sha) is None or _FULL_SHA.fullmatch(tree_sha) is None:
        raise InputError("production package source requires full lowercase Git commit/tree SHAs")
    if not isinstance(package_version, str) or not package_version.strip():
        raise InputError("production package source requires a package version")
    return SourceProvenance(
        commit_sha=commit_sha,
        tree_sha=tree_sha,
        package_version=package_version,
    )


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InputError(
            "production package source provenance could not be resolved",
            remediation="run the production Carbon trainer from a clean immutable source package",
        ) from exc
    return result.stdout.strip()
