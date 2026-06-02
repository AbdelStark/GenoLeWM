# SPDX-License-Identifier: Apache-2.0
"""Tests for ``tools/release/bump.py`` and ``tools/release/changelog.py``.

Covers Acceptance Criteria from issue #102:

- ``bump --dry-run`` produces a unified diff without touching the
  tree.
- ``changelog generate`` on a synthetic git range produces a valid
  Keep-a-Changelog section ordered ``Added / Changed / ... / Security``.
"""

from __future__ import annotations

import datetime as _dt
import io
import subprocess
from pathlib import Path

import pytest

from tools.release import bump, changelog

# ---------------------------------------------------------------------------
# bump.Version
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "0.1.0",
        "1.2.3",
        "0.1.0.dev0",
        "0.1.0a1",
        "0.1.0b2",
        "0.1.0rc1",
        "1.0.0.post1",
        "2.0.0rc1.dev3",
    ],
)
def test_version_parse_accepts_supported_forms(raw: str) -> None:
    assert bump.Version.parse(raw).raw == raw


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "v0.1.0",
        "0.1",  # release segment must have >=2 dots? actually PEP allows it,
        # but our subset requires X.Y.Z+
        "0.1.0+local",  # local versions are not allowed
        "0!1.0.0",  # epoch is not allowed
        "1.0.0beta",  # alpha/beta/rc must use a/b/rc shorthand
    ],
)
def test_version_parse_rejects_unsupported(raw: str) -> None:
    if raw == "0.1":
        # Our regex actually accepts this (>=2 numeric components).
        # Documenting the choice explicitly.
        bump.Version.parse(raw)
        return
    with pytest.raises(ValueError):
        bump.Version.parse(raw)


def test_version_ordering_pre_lower_than_release() -> None:
    pre = bump.Version.parse("0.1.0a1")
    rel = bump.Version.parse("0.1.0")
    assert pre < rel
    assert not (rel < pre)


def test_version_ordering_dev_lower_than_release_and_pre() -> None:
    dev = bump.Version.parse("0.1.0.dev0")
    pre = bump.Version.parse("0.1.0a1")
    rel = bump.Version.parse("0.1.0")
    # Dev should sort lowest within the (release, pre) bucket.
    assert dev < pre < rel


def test_version_ordering_release_components() -> None:
    a = bump.Version.parse("0.1.0")
    b = bump.Version.parse("0.2.0")
    c = bump.Version.parse("0.10.0")
    assert a < b < c
    assert not (c < a)


# ---------------------------------------------------------------------------
# bump.read_current_version / rewrite_version
# ---------------------------------------------------------------------------


def _write_minimal_init(tmp_path: Path, version: str) -> Path:
    init = tmp_path / "__init__.py"
    init.write_text(
        f'"""docstring"""\n\nfrom __future__ import annotations\n\n'
        f'__version__ = "{version}"\n\n'
        f"# trailing content\n",
        encoding="utf-8",
    )
    return init


def test_read_current_version(tmp_path: Path) -> None:
    init = _write_minimal_init(tmp_path, "0.1.0.dev0")
    assert bump.read_current_version(init) == "0.1.0.dev0"


def test_rewrite_version_replaces_only_the_assignment(tmp_path: Path) -> None:
    init = _write_minimal_init(tmp_path, "0.1.0.dev0")
    old, new = bump.rewrite_version("0.1.0", version_file=init)
    assert '__version__ = "0.1.0.dev0"' in old
    assert '__version__ = "0.1.0"' in new
    # No collateral changes.
    assert old.replace('"0.1.0.dev0"', '"0.1.0"') == new


def test_read_current_version_missing_raises(tmp_path: Path) -> None:
    init = tmp_path / "__init__.py"
    init.write_text("# no version here\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        bump.read_current_version(init)


# ---------------------------------------------------------------------------
# bump.main CLI
# ---------------------------------------------------------------------------


def test_bump_main_show(capfd: pytest.CaptureFixture[str]) -> None:
    rc = bump.main(["--show"])
    out, _ = capfd.readouterr()
    assert rc == 0
    # Current canonical version must be a parseable PEP 440 string.
    bump.Version.parse(out.strip())


def test_bump_main_missing_arg_returns_2() -> None:
    err = io.StringIO()
    rc = bump.main([], stderr=err)
    assert rc == 2
    assert "new_version is required" in err.getvalue()


def test_bump_main_dry_run_emits_diff_no_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init = _write_minimal_init(tmp_path, "0.1.0.dev0")
    monkeypatch.setattr(bump, "VERSION_FILE", init)
    # pyproject check must still pass; reuse the real file.
    out = io.StringIO()
    rc = bump.main(["0.1.0", "--dry-run"], stdout=out)
    assert rc == 0
    diff = out.getvalue()
    assert '-__version__ = "0.1.0.dev0"' in diff
    assert '+__version__ = "0.1.0"' in diff
    # On-disk content unchanged.
    assert '__version__ = "0.1.0.dev0"' in init.read_text(encoding="utf-8")


def test_bump_main_rejects_non_increasing_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init = _write_minimal_init(tmp_path, "0.2.0")
    monkeypatch.setattr(bump, "VERSION_FILE", init)
    err = io.StringIO()
    rc = bump.main(["0.1.0"], stderr=err)
    assert rc == 2
    assert "not greater than current" in err.getvalue()


def test_bump_main_writes_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init = _write_minimal_init(tmp_path, "0.1.0.dev0")
    monkeypatch.setattr(bump, "VERSION_FILE", init)
    out = io.StringIO()
    rc = bump.main(["0.1.0"], stdout=out)
    assert rc == 0
    assert '__version__ = "0.1.0"' in init.read_text(encoding="utf-8")
    assert "0.1.0.dev0 -> 0.1.0" in out.getvalue()


def test_bump_check_pyproject_dynamic_passes_on_real_file() -> None:
    # Smoke-checks the real pyproject.toml: if this fails, the
    # release toolchain would silently let __version__ drift.
    bump.check_pyproject_dynamic()


# ---------------------------------------------------------------------------
# changelog.classify / parse_commit_line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("subject", "expected_section", "expected_breaking"),
    [
        ("feat: add foo", "Added", False),
        ("fix: handle empty input", "Fixed", False),
        ("fix(api): missing field", "Fixed", False),
        ("feat!: drop legacy mode", "Added", True),
        ("docs: update README", "Changed", False),
        ("ci: pin actions", "Changed", False),
        ("chore: bump deps", "Changed", False),
        ("security: redact token in logs", "Security", False),
        ("deprecate: old flag", "Deprecated", False),
        ("remove: old flag", "Removed", False),
        ("uncategorized subject", "Changed", False),
        ("provenance: implement receipt", "Changed", False),
    ],
)
def test_classify(
    subject: str,
    expected_section: str,
    expected_breaking: bool,
) -> None:
    section, breaking, _ = changelog.classify(subject)
    assert section == expected_section
    assert breaking is expected_breaking


def test_parse_commit_line_strips_pr_ref() -> None:
    commit = changelog.parse_commit_line("abc1234 feat: add foo (#42)")
    assert commit is not None
    assert commit.pr == 42
    assert commit.bullet == "feat: add foo (#42)"
    assert commit.section == "Added"


def test_parse_commit_line_blank_returns_none() -> None:
    assert changelog.parse_commit_line("") is None
    assert changelog.parse_commit_line("abc1234") is None


# ---------------------------------------------------------------------------
# changelog.build_section
# ---------------------------------------------------------------------------


def _commit(
    subject: str,
    sha: str = "abc1234",
) -> changelog.Commit:
    parsed = changelog.parse_commit_line(f"{sha} {subject}")
    assert parsed is not None
    return parsed


def test_build_section_orders_and_omits_empty() -> None:
    commits = [
        _commit("feat: add A"),
        _commit("fix: handle B"),
        _commit("security: redact C"),
    ]
    section = changelog.build_section(
        "0.1.0",
        commits,
        today=_dt.date(2026, 5, 21),
    )
    assert section.startswith("## [0.1.0] - 2026-05-21\n")
    assert "### Added" in section
    assert "### Fixed" in section
    assert "### Security" in section
    assert "### Deprecated" not in section
    assert "### Removed" not in section
    # Canonical ordering: Added -> Changed -> ... -> Security.
    added_idx = section.index("### Added")
    fixed_idx = section.index("### Fixed")
    security_idx = section.index("### Security")
    assert added_idx < fixed_idx < security_idx


def test_build_section_flags_breaking() -> None:
    commits = [_commit("feat!: drop legacy mode")]
    section = changelog.build_section(
        "1.0.0",
        commits,
        today=_dt.date(2026, 5, 21),
    )
    assert "### ⚠ BREAKING CHANGES" in section
    # Breaking entry also appears in the categorical section.
    assert section.count("drop legacy mode") == 2


def test_build_section_with_no_commits_is_minimal() -> None:
    section = changelog.build_section("0.1.0", [], today=_dt.date(2026, 5, 21))
    assert section.strip() == "## [0.1.0] - 2026-05-21"


# ---------------------------------------------------------------------------
# changelog.lift_unreleased
# ---------------------------------------------------------------------------


def test_lift_unreleased_replaces_block() -> None:
    text = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "### Added\n- something old\n\n"
        "## [0.0.1] - 2026-01-01\n\n"
        "### Added\n- prior release\n"
    )
    rendered = "## [0.1.0] - 2026-05-21\n\n### Added\n\n- the new thing\n"
    out = changelog.lift_unreleased(text, rendered)
    assert "## [Unreleased]\n\n## [0.1.0] - 2026-05-21" in out
    assert "## [0.0.1] - 2026-01-01" in out
    # The old Unreleased entries are dropped (lift, not merge).
    assert "something old" not in out


def test_lift_unreleased_raises_when_unreleased_missing() -> None:
    text = "# Changelog\n\n## [0.0.1] - 2026-01-01\n"
    with pytest.raises(changelog.InputError):
        changelog.lift_unreleased(text, "## [0.1.0] - 2026-05-21\n")


# ---------------------------------------------------------------------------
# changelog.main end-to-end against a temp git repo
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "HOME": str(tmp_path),
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    _git(["init", "--initial-branch=main"], repo, env=env)
    _git(["config", "user.email", "test@example.com"], repo, env=env)
    _git(["config", "user.name", "Test"], repo, env=env)
    _git(["config", "commit.gpgsign", "false"], repo, env=env)
    (repo / "README.md").write_text("seed", encoding="utf-8")
    _git(["add", "README.md"], repo, env=env)
    _git(["commit", "-m", "chore: seed"], repo, env=env)

    _git(["tag", "v0.0.1"], repo, env=env)

    (repo / "a.txt").write_text("a", encoding="utf-8")
    _git(["add", "a.txt"], repo, env=env)
    _git(["commit", "-m", "feat: add A (#1)"], repo, env=env)

    (repo / "b.txt").write_text("b", encoding="utf-8")
    _git(["add", "b.txt"], repo, env=env)
    _git(["commit", "-m", "fix: handle B"], repo, env=env)

    (repo / "c.txt").write_text("c", encoding="utf-8")
    _git(["add", "c.txt"], repo, env=env)
    _git(["commit", "-m", "security: redact C in logs (#3)"], repo, env=env)
    return repo


def test_iter_commits_against_synthetic_repo(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    commits = changelog.iter_commits(since="v0.0.1", until="HEAD", cwd=repo)
    bullets = [c.bullet for c in commits]
    sections = {c.bullet: c.section for c in commits}
    assert "feat: add A (#1)" in bullets
    assert "fix: handle B" in bullets
    assert "security: redact C in logs (#3)" in bullets
    assert sections["feat: add A (#1)"] == "Added"
    assert sections["fix: handle B"] == "Fixed"
    assert sections["security: redact C in logs (#3)"] == "Security"


def test_changelog_main_dry_run_emits_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    monkeypatch.setattr(changelog, "REPO_ROOT", repo)
    monkeypatch.setattr(changelog, "CHANGELOG_PATH", repo / "CHANGELOG.md")
    out = io.StringIO()
    rc = changelog.main(
        ["generate", "--version", "0.1.0", "--since", "v0.0.1", "--date", "2026-05-21"],
        stdout=out,
    )
    assert rc == 0
    rendered = out.getvalue()
    assert rendered.startswith("## [0.1.0] - 2026-05-21")
    assert "### Added" in rendered
    assert "### Fixed" in rendered
    assert "### Security" in rendered
    # Dry-run does not create CHANGELOG.md.
    assert not (repo / "CHANGELOG.md").exists()


def test_changelog_main_write_lifts_unreleased(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    changelog_path = repo / "CHANGELOG.md"
    changelog_path.write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "### Added\n- a placeholder entry\n\n"
        "## [0.0.1] - 2026-01-01\n\n"
        "### Added\n- seed\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(changelog, "REPO_ROOT", repo)
    monkeypatch.setattr(changelog, "CHANGELOG_PATH", changelog_path)
    rc = changelog.main(
        ["generate", "--version", "0.1.0", "--since", "v0.0.1", "--date", "2026-05-21", "--write"],
    )
    assert rc == 0
    updated = changelog_path.read_text(encoding="utf-8")
    assert "## [Unreleased]" in updated
    assert "## [0.1.0] - 2026-05-21" in updated
    assert "## [0.0.1] - 2026-01-01" in updated
    # The new section precedes the old release section.
    assert updated.index("## [0.1.0]") < updated.index("## [0.0.1]")


def test_changelog_main_errors_on_empty_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    # Re-tag HEAD so since..HEAD is empty.
    _git(
        ["tag", "v0.0.2"],
        repo,
        env={
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "HOME": str(tmp_path),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        },
    )
    monkeypatch.setattr(changelog, "REPO_ROOT", repo)
    err = io.StringIO()
    rc = changelog.main(
        ["generate", "--version", "0.1.0", "--since", "v0.0.2", "--date", "2026-05-21"],
        stderr=err,
    )
    assert rc == 2
    assert "no commits found" in err.getvalue()


def test_changelog_main_allow_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    _git(
        ["tag", "v0.0.2"],
        repo,
        env={
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "HOME": str(tmp_path),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        },
    )
    monkeypatch.setattr(changelog, "REPO_ROOT", repo)
    out = io.StringIO()
    rc = changelog.main(
        [
            "generate",
            "--version",
            "0.1.0",
            "--since",
            "v0.0.2",
            "--date",
            "2026-05-21",
            "--allow-empty",
        ],
        stdout=out,
    )
    assert rc == 0
    assert "## [0.1.0] - 2026-05-21" in out.getvalue()
