# SPDX-License-Identifier: Apache-2.0
"""Keep-a-Changelog section generator for the release checklist.

The release flow in ``CONTRIBUTING.md`` requires
every release to carry a CHANGELOG entry. This helper emits a
candidate section synthesised from ``git log`` so the maintainer's
manual curation has a stable starting point.

The grammar follows Keep a Changelog 1.1.0:

* One ``## [VERSION] - YYYY-MM-DD`` heading.
* Subsections drawn from ``{Added, Changed, Deprecated, Removed,
  Fixed, Security}``. Empty subsections are omitted.
* One bullet per matching commit subject, with the merge PR number
  (``(#NNN)``) preserved if present.

Commit-subject classification follows the project's conventional /
area-prefixed style. The mapping is deliberately conservative — the
human curates the result, not the other way around.

Usage::

    # Dry-run: print the section to stdout, do not touch CHANGELOG.md.
    python -m tools.release.changelog generate --version 0.1.0

    # Same, but bound to an explicit range.
    python -m tools.release.changelog generate \\
        --version 0.1.0 --since v0.0.1 --until HEAD

    # Write mode: replace the existing ``[Unreleased]`` block with the
    # versioned section and re-open an empty ``[Unreleased]`` above it.
    python -m tools.release.changelog generate --version 0.1.0 --write

Exit codes follow the project's CLI conventions:

- 0 — success.
- 2 — ``InputError`` (bad args, no commits in range, missing CHANGELOG).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Final

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
CHANGELOG_PATH: Final = REPO_ROOT / "CHANGELOG.md"

# Keep-a-Changelog section names, in canonical order.
SECTIONS: Final = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")

# Conventional-commit / area-prefix -> KAC section.
_TYPE_TO_SECTION: Final[dict[str, str]] = {
    # Standard conventional types.
    "feat": "Added",
    "fix": "Fixed",
    "perf": "Changed",
    "refactor": "Changed",
    "docs": "Changed",
    "test": "Changed",
    "ci": "Changed",
    "build": "Changed",
    "style": "Changed",
    "chore": "Changed",
    "revert": "Removed",
    # Project-specific markers used in commit subjects.
    "security": "Security",
    "deprecate": "Deprecated",
    "deprecation": "Deprecated",
    "remove": "Removed",
    "removal": "Removed",
}

# Pattern captures ``<type>(<scope>)?: ...`` or ``<area>: ...`` from
# the start of the subject line.
_SUBJECT_PREFIX_RE: Final = re.compile(
    r"^(?P<type>[A-Za-z][A-Za-z0-9_-]*)"
    r"(?P<scope>\([^)]*\))?"
    r"(?P<bang>!)?"
    r":\s*(?P<rest>.+)$"
)

# Pattern captures a trailing PR reference like ``(#123)``.
_PR_REF_RE: Final = re.compile(r"\s*\(#(?P<num>\d+)\)\s*$")


@dataclass(frozen=True)
class Commit:
    """One commit subject parsed into a structured form."""

    sha: str
    subject: str
    breaking: bool
    pr: int | None
    bullet: str
    section: str


@dataclass
class ChangelogSection:
    """Accumulator for one Keep-a-Changelog subsection."""

    title: str
    entries: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"### {self.title}", ""]
        lines.extend(f"- {entry}" for entry in self.entries)
        lines.append("")
        return "\n".join(lines)


def classify(subject: str) -> tuple[str, bool, str]:
    """Return ``(section, is_breaking, bullet_text)`` for a subject."""
    breaking = False
    match = _SUBJECT_PREFIX_RE.match(subject)
    if match is None:
        return "Changed", False, subject

    type_token = match.group("type").lower()
    breaking = match.group("bang") == "!"
    rest = match.group("rest").strip()

    # ``feat!: ...`` / ``fix!: ...`` keep their semantic mapping but
    # are flagged as breaking for the release manager to review.
    section = _TYPE_TO_SECTION.get(type_token, "Changed")

    # Reattach the scope/type so the bullet remains greppable.
    scope = match.group("scope") or ""
    bullet = f"{type_token}{scope}: {rest}" if type_token else rest
    return section, breaking, bullet


def parse_commit_line(line: str) -> Commit | None:
    """Parse a single ``%H %s`` line into a :class:`Commit`."""
    line = line.rstrip("\n")
    if not line:
        return None
    sha, _, subject = line.partition(" ")
    if not subject:
        return None
    section, breaking, body = classify(subject)
    pr_match = _PR_REF_RE.search(body)
    if pr_match:
        pr_num: int | None = int(pr_match.group("num"))
        body = _PR_REF_RE.sub("", body).rstrip()
    else:
        pr_num = None
    bullet = f"{body} (#{pr_num})" if pr_num is not None else body
    return Commit(
        sha=sha,
        subject=subject,
        breaking=breaking,
        pr=pr_num,
        bullet=bullet,
        section=section,
    )


def _run_git(args: Sequence[str], *, cwd: Path = REPO_ROOT) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def latest_tag(*, cwd: Path = REPO_ROOT) -> str | None:
    """Return the most recent ``v*`` tag reachable from HEAD, or ``None``."""
    try:
        out = _run_git(
            ["describe", "--tags", "--abbrev=0", "--match", "v*"],
            cwd=cwd,
        )
    except subprocess.CalledProcessError:
        return None
    tag = out.strip()
    return tag or None


def iter_commits(
    *,
    since: str | None,
    until: str,
    cwd: Path = REPO_ROOT,
) -> list[Commit]:
    """Return commits in ``since..until`` (exclusive..inclusive).

    Merge commits are skipped — they don't carry user-visible content.
    """
    if since is None:
        revspec = until
    else:
        revspec = f"{since}..{until}"
    raw = _run_git(
        ["log", "--no-merges", "--pretty=format:%H %s", revspec],
        cwd=cwd,
    )
    commits: list[Commit] = []
    for line in raw.splitlines():
        commit = parse_commit_line(line)
        if commit is not None:
            commits.append(commit)
    return commits


def build_section(
    version: str,
    commits: Iterable[Commit],
    *,
    today: _dt.date | None = None,
) -> str:
    """Render a Keep-a-Changelog section for ``version``."""
    date = today or _dt.date.today()
    buckets: dict[str, ChangelogSection] = {
        title: ChangelogSection(title=title) for title in SECTIONS
    }
    breaking: list[str] = []
    for commit in commits:
        section = buckets[commit.section]
        section.entries.append(commit.bullet)
        if commit.breaking:
            breaking.append(commit.bullet)

    parts = [f"## [{version}] - {date.isoformat()}", ""]
    if breaking:
        parts.append("### ⚠ BREAKING CHANGES")
        parts.append("")
        parts.extend(f"- {entry}" for entry in breaking)
        parts.append("")
    for title in SECTIONS:
        bucket = buckets[title]
        if bucket.entries:
            parts.append(bucket.render())
    rendered = "\n".join(parts).rstrip() + "\n"
    return rendered


class InputError(Exception):
    """Raised on invalid arguments or missing prerequisites."""


# Marker used by ``--write`` to lift the existing ``[Unreleased]``
# entries into the new version section.
_UNRELEASED_HEADER_RE: Final = re.compile(r"^## \[Unreleased\].*$", re.MULTILINE)
_NEXT_SECTION_RE: Final = re.compile(r"^## \[", re.MULTILINE)


def lift_unreleased(
    changelog_text: str,
    rendered_section: str,
) -> str:
    """Replace the ``[Unreleased]`` block with the rendered section.

    The new file keeps an empty ``[Unreleased]`` placeholder above the
    just-released section so the next change can land there
    immediately.
    """
    match = _UNRELEASED_HEADER_RE.search(changelog_text)
    if match is None:
        raise InputError(
            "CHANGELOG.md is missing an `## [Unreleased]` heading; cannot lift in --write mode"
        )
    start = match.start()
    next_match = _NEXT_SECTION_RE.search(changelog_text, match.end())
    end = next_match.start() if next_match else len(changelog_text)

    head = changelog_text[:start]
    tail = changelog_text[end:]
    placeholder = "## [Unreleased]\n\n"
    return f"{head}{placeholder}{rendered_section}\n{tail}"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tools.release.changelog",
        description="Generate a Keep-a-Changelog section from git log.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="Emit a candidate Keep-a-Changelog section.",
    )
    generate.add_argument(
        "--version",
        required=True,
        help="Version string for the section heading (e.g. 0.1.0).",
    )
    generate.add_argument(
        "--since",
        default=None,
        help=(
            "Git ref to start the log from (exclusive). "
            "Defaults to the most recent ``v*`` tag reachable from "
            "the until-ref, or to project history if there is none."
        ),
    )
    generate.add_argument(
        "--until",
        default="HEAD",
        help="Git ref to stop the log at (inclusive). Defaults to HEAD.",
    )
    generate.add_argument(
        "--write",
        action="store_true",
        help="Rewrite CHANGELOG.md in place (default: print to stdout).",
    )
    generate.add_argument(
        "--date",
        default=None,
        help=(
            "ISO date for the section heading. Defaults to today. "
            "Overridable for reproducible test output."
        ),
    )
    generate.add_argument(
        "--allow-empty",
        action="store_true",
        help=(
            "Do not error out when no commits are found in the range; "
            "emit a heading with no subsections instead."
        ),
    )
    return parser.parse_args(argv)


def _resolve_since(value: str | None, *, cwd: Path) -> str | None:
    if value is not None:
        return value
    return latest_tag(cwd=cwd)


def _parse_date(value: str | None) -> _dt.date | None:
    if value is None:
        return None
    try:
        return _dt.date.fromisoformat(value)
    except ValueError as exc:
        raise InputError(f"--date must be ISO YYYY-MM-DD: {exc}") from exc


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    args = _parse_args(argv)

    try:
        # Re-read module attributes so test monkeypatches take effect.
        repo_root = REPO_ROOT
        changelog_path = CHANGELOG_PATH
        date = _parse_date(args.date)
        since = _resolve_since(args.since, cwd=repo_root)
        commits = iter_commits(since=since, until=args.until, cwd=repo_root)
        if not commits and not args.allow_empty:
            raise InputError(f"no commits found in range {since or '<root>'}..{args.until}")
        section = build_section(args.version, commits, today=date)
        if args.write:
            if not changelog_path.exists():
                raise InputError(f"CHANGELOG.md not found at {changelog_path}")
            original = changelog_path.read_text(encoding="utf-8")
            updated = lift_unreleased(original, section)
            changelog_path.write_text(updated, encoding="utf-8")
            try:
                rel = changelog_path.relative_to(repo_root)
            except ValueError:
                rel = Path(changelog_path.name)
            print(
                f"updated {rel} with {len(commits)} entries for {args.version}",
                file=out,
            )
        else:
            print(section, file=out, end="")
    except InputError as exc:
        print(f"error: {exc}", file=err)
        return 2
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.strip() if exc.stderr else str(exc)
        print(f"error: git failed: {msg}", file=err)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - thin shim
    raise SystemExit(main())
