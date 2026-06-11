# SPDX-License-Identifier: Apache-2.0
"""Version-bump helper for the release checklist.

``geno_lewm/__init__.py`` carries the canonical ``__version__``
assignment; ``pyproject.toml`` consumes it via Hatch's
``regex_commit`` source so the runtime constant and the package
metadata cannot drift (configuration contract; see also
``CONTRIBUTING.md``). This module rewrites that
single assignment after validating the new version against PEP 440 and
the project's pre-1.0 ordering policy.

Usage::

    python -m tools.release.bump <new-version> [--dry-run]
    python -m tools.release.bump --show

``--show`` prints the current version and exits 0.

``--dry-run`` prints the unified diff that would be applied and exits 0
without touching the working tree. This is the form invoked by the
release-PR template's acceptance check.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Final, NamedTuple

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
VERSION_FILE: Final = REPO_ROOT / "geno_lewm" / "__init__.py"
PYPROJECT_FILE: Final = REPO_ROOT / "pyproject.toml"

# Single canonical assignment. Hatch's ``regex_commit`` source is
# pinned to the same pattern in ``pyproject.toml``.
_VERSION_LINE_RE: Final = re.compile(
    r'^(?P<prefix>__version__\s*=\s*")(?P<version>[^"]+)(?P<suffix>"\s*)$',
    re.MULTILINE,
)

# Permissive PEP 440 release segment matcher. We deliberately accept
# only the subset the project commits to publishing (release, alpha,
# beta, release-candidate, post, dev); local version identifiers and
# epoch are intentionally excluded — releases must be globally
# orderable.
_PEP440_RE: Final = re.compile(
    r"^(?P<release>\d+(?:\.\d+)+)"
    r"(?:(?P<pre_tag>a|b|rc)(?P<pre_n>\d+))?"
    r"(?:\.post(?P<post_n>\d+))?"
    r"(?:\.dev(?P<dev_n>\d+))?$"
)


class Version(NamedTuple):
    """A parsed, orderable PEP 440 version."""

    release: tuple[int, ...]
    pre: tuple[str, int] | None
    post: int | None
    dev: int | None
    raw: str

    @classmethod
    def parse(cls, raw: str) -> Version:
        match = _PEP440_RE.match(raw)
        if match is None:
            raise ValueError(
                f"{raw!r} is not a supported PEP 440 version "
                "(supported segments: release[.aN|.bN|.rcN][.postN][.devN])"
            )
        release = tuple(int(p) for p in match.group("release").split("."))
        pre_tag = match.group("pre_tag")
        pre = (pre_tag, int(match.group("pre_n"))) if pre_tag else None
        post_n = match.group("post_n")
        dev_n = match.group("dev_n")
        return cls(
            release=release,
            pre=pre,
            post=int(post_n) if post_n is not None else None,
            dev=int(dev_n) if dev_n is not None else None,
            raw=raw,
        )

    def _ordering_key(self) -> tuple[object, ...]:
        # PEP 440 ordering, mirroring ``packaging.version._cmpkey`` for
        # our supported subset:
        #
        # * ``X.Y.Z.devN`` (dev with no pre) sorts BEFORE any pre-release.
        # * ``X.Y.ZaN.devN`` sorts before ``X.Y.ZaN``.
        # * ``X.Y.Z`` (final) sorts after any pre-release.
        # * ``X.Y.Z.postN`` sorts after the bare release.
        #
        # We encode pre as a tuple so the three regimes are
        # lexicographically separable:
        #   (-1,)       — dev-only (no pre)             ; sorts first
        #   (rank, n)   — explicit "a"/"b"/"rc" pre     ; sorts middle
        #   (3, 0)      — no pre, no dev (final / post) ; sorts last
        pre_rank: tuple[int, ...]
        if self.pre is None and self.dev is not None:
            pre_rank = (-1,)
        elif self.pre is None:
            pre_rank = (3, 0)
        else:
            letter, n = self.pre
            letter_rank = {"a": 0, "b": 1, "rc": 2}[letter]
            pre_rank = (letter_rank, n)
        post_rank = self.post if self.post is not None else -1
        dev_rank: float = self.dev if self.dev is not None else float("inf")
        return (self.release, pre_rank, post_rank, dev_rank)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._ordering_key() < other._ordering_key()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._ordering_key() <= other._ordering_key()


def read_current_version(version_file: Path | None = None) -> str:
    """Return the value of ``__version__`` in ``version_file``."""
    target = version_file if version_file is not None else VERSION_FILE
    text = target.read_text(encoding="utf-8")
    match = _VERSION_LINE_RE.search(text)
    if match is None:
        raise RuntimeError(f"Could not locate __version__ assignment in {target}")
    return match.group("version")


def rewrite_version(
    new_version: str,
    *,
    version_file: Path | None = None,
) -> tuple[str, str]:
    """Return ``(old_text, new_text)`` after substituting the version.

    Does not write to disk; callers decide whether to commit or to
    print the diff for ``--dry-run``.
    """
    target = version_file if version_file is not None else VERSION_FILE
    old_text = target.read_text(encoding="utf-8")
    match = _VERSION_LINE_RE.search(old_text)
    if match is None:
        raise RuntimeError(f"Could not locate __version__ assignment in {target}")
    new_line = f"{match.group('prefix')}{new_version}{match.group('suffix')}"
    new_text = old_text[: match.start()] + new_line + old_text[match.end() :]
    return old_text, new_text


def check_pyproject_dynamic(
    pyproject_file: Path = PYPROJECT_FILE,
) -> None:
    """Validate ``pyproject.toml`` keeps ``version`` dynamic.

    A hardcoded ``version = "X"`` line would silently drift from
    ``__version__``. ``[tool.hatch.version]`` must point back at the
    Python file so the release checklist's single edit lands in both
    surfaces.
    """
    text = pyproject_file.read_text(encoding="utf-8")
    if 'dynamic = ["version"]' not in text:
        raise RuntimeError(
            'pyproject.toml must declare ``dynamic = ["version"]`` '
            "so Hatch sources the version from geno_lewm/__init__.py"
        )
    if "[tool.hatch.version]" not in text:
        raise RuntimeError("pyproject.toml is missing the [tool.hatch.version] table")


def _diff(old_text: str, new_text: str, path: Path) -> str:
    try:
        rel: Path = path.relative_to(REPO_ROOT)
    except ValueError:
        rel = Path(path.name)
    return "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tools.release.bump",
        description="Bump GenoLeWM's canonical __version__ assignment.",
    )
    parser.add_argument(
        "new_version",
        nargs="?",
        help="New PEP 440 version (e.g. 0.1.0, 0.2.0.dev0, 0.3.0rc1).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the current version and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the diff that would be applied and exit.",
    )
    parser.add_argument(
        "--allow-equal",
        action="store_true",
        help="Allow the new version to equal the current one (useful "
        "for idempotent re-runs from CI).",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """Entry point. Returns process exit code."""
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    args = _parse_args(argv)

    current_raw = read_current_version()
    if args.show:
        print(current_raw, file=out)
        return 0

    if args.new_version is None:
        print(
            "error: new_version is required (or pass --show)",
            file=err,
        )
        return 2

    try:
        new = Version.parse(args.new_version)
        current = Version.parse(current_raw)
    except ValueError as exc:
        print(f"error: {exc}", file=err)
        return 2

    if new <= current and not args.allow_equal:
        print(
            f"error: new version {new.raw} is not greater than current {current.raw}",
            file=err,
        )
        return 2
    if new.raw == current.raw and not args.allow_equal:
        # ``__le__`` already covers strict; this only catches the
        # explicit equality case (``__le__`` returns True at equality
        # too). Keep both messages distinct so CI logs are precise.
        print(
            f"error: new version {new.raw} equals current "
            f"{current.raw}; pass --allow-equal to allow",
            file=err,
        )
        return 2

    check_pyproject_dynamic()

    old_text, new_text = rewrite_version(new.raw)
    diff = _diff(old_text, new_text, VERSION_FILE)

    if args.dry_run:
        print(diff, file=out, end="")
        return 0

    VERSION_FILE.write_text(new_text, encoding="utf-8")
    print(
        f"bumped __version__: {current.raw} -> {new.raw}",
        file=out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin shim
    raise SystemExit(main())
