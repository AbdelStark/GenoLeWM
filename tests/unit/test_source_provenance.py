"""Behavioral tests for production package/source provenance."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from geno_lewm._source_provenance import resolve_package_source
from geno_lewm.errors import InputError


def test_package_source_identity_does_not_depend_on_non_git_caller_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, commit_sha, tree_sha = _write_package_repo(tmp_path / "source")
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)

    source = resolve_package_source(
        package_root=repo / "geno_lewm",
        package_version="0.2.1",
    )

    assert source.commit_sha == commit_sha
    assert source.tree_sha == tree_sha
    assert source.package_version == "0.2.1"


def test_package_source_identity_ignores_unrelated_git_caller_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, commit_sha, tree_sha = _write_package_repo(tmp_path / "source")
    caller, _, _ = _write_package_repo(tmp_path / "unrelated")
    monkeypatch.chdir(caller)

    source = resolve_package_source(
        package_root=repo / "geno_lewm",
        package_version="0.2.1",
    )

    assert source.commit_sha == commit_sha
    assert source.tree_sha == tree_sha


def test_package_source_identity_rejects_dirty_runtime_source(tmp_path: Path) -> None:
    repo, _, _ = _write_package_repo(tmp_path / "source")
    (repo / "geno_lewm" / "cli" / "train.py").write_text("# dirty\n", encoding="utf-8")

    with pytest.raises(InputError, match="must be clean"):
        resolve_package_source(
            package_root=repo / "geno_lewm",
            package_version="0.2.1",
        )


def test_package_source_identity_allows_run_outputs_outside_package_tree(
    tmp_path: Path,
) -> None:
    repo, commit_sha, tree_sha = _write_package_repo(tmp_path / "source")
    (repo / "runs").mkdir()
    (repo / "runs" / "metrics.json").write_text("{}\n", encoding="utf-8")

    source = resolve_package_source(
        package_root=repo / "geno_lewm",
        package_version="0.2.1",
    )

    assert (source.commit_sha, source.tree_sha) == (commit_sha, tree_sha)


def test_package_source_identity_rejects_ignored_untracked_package_copy(
    tmp_path: Path,
) -> None:
    repo, _, _ = _write_package_repo(tmp_path / "source")
    (repo / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "--quiet", "-m", "ignore runtime copy")
    ignored = repo / "ignored" / "geno_lewm"
    (ignored / "cli").mkdir(parents=True)
    (ignored / "__init__.py").write_text("", encoding="utf-8")
    (ignored / "cli" / "train.py").write_text("", encoding="utf-8")

    with pytest.raises(InputError, match="tracked package root"):
        resolve_package_source(
            package_root=ignored,
            package_version="0.2.1",
        )


def test_package_source_identity_rejects_ignored_executable_in_tracked_package(
    tmp_path: Path,
) -> None:
    repo, _, _ = _write_package_repo(tmp_path / "source")
    (repo / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "--quiet", "-m", "ignore bytecode")
    ignored_bytecode = repo / "geno_lewm" / "runtime.pyc"
    ignored_bytecode.write_bytes(b"loadable-but-unbound")

    with pytest.raises(InputError, match="ignored executable"):
        resolve_package_source(
            package_root=repo / "geno_lewm",
            package_version="0.2.1",
        )


def test_package_source_identity_rejects_tracked_package_symlink(tmp_path: Path) -> None:
    repo, _, _ = _write_package_repo(tmp_path / "source")
    (repo / "outside.py").write_text("VALUE = 'outside'\n", encoding="utf-8")
    try:
        (repo / "geno_lewm" / "runtime.py").symlink_to(repo / "outside.py")
    except OSError as exc:  # pragma: no cover - platform capability boundary.
        pytest.skip(f"symlinks unavailable: {exc}")
    _git(repo, "add", "outside.py", "geno_lewm/runtime.py")
    _git(repo, "commit", "--quiet", "-m", "tracked package symlink")

    with pytest.raises(InputError, match="regular tracked files"):
        resolve_package_source(
            package_root=repo / "geno_lewm",
            package_version="0.2.1",
        )


def test_package_source_identity_fails_closed_without_embedded_wheel_provenance(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "site-packages" / "geno_lewm"
    (package_root / "cli").mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "cli" / "train.py").write_text("", encoding="utf-8")

    with pytest.raises(InputError, match="could not be resolved"):
        resolve_package_source(
            package_root=package_root,
            package_version="0.2.1",
        )


def _write_package_repo(root: Path) -> tuple[Path, str, str]:
    (root / "geno_lewm" / "cli").mkdir(parents=True)
    (root / "geno_lewm" / "__init__.py").write_text("", encoding="utf-8")
    (root / "geno_lewm" / "cli" / "train.py").write_text("", encoding="utf-8")
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "GenoLeWM test")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "add", "geno_lewm")
    _git(root, "commit", "--quiet", "-m", "fixture")
    return root, _git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", "HEAD^{tree}")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
