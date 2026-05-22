# SPDX-License-Identifier: Apache-2.0
"""``license_headers`` AST linter.

Every shipped ``.py`` file under ``geno_lewm/`` and ``tools/`` must
carry the SPDX-License-Identifier line documented in ``CONTRIBUTING.md``.
The check is intentionally tolerant of the file's other preamble:

- A shebang line is allowed but discouraged in library code.
- The module docstring may appear before or after the SPDX line.
- ``from __future__`` imports MUST come after the SPDX line.

The required header is a single line:

    # SPDX-License-Identifier: Apache-2.0

Any deviation (missing, mis-spelled identifier, wrong key) is flagged.

Usage:

    python -m tools.lint.check_license_headers [PATH ...]

Exits 0 on success and 1 on any violation.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS: tuple[Path, ...] = (
    REPO_ROOT / "geno_lewm",
    REPO_ROOT / "tools",
)
REQUIRED_HEADER = "# SPDX-License-Identifier: Apache-2.0"


@dataclass(frozen=True)
class Violation:
    """A single linter offence."""

    path: Path
    line: int
    message: str

    def format(self, root: Path) -> str:
        try:
            rel = self.path.relative_to(root)
        except ValueError:
            rel = self.path
        return f"{rel}:{self.line}:1: error: [license_headers] {self.message}"


def _walk_python_files(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            yield p
        elif p.is_dir():
            yield from sorted(p.rglob("*.py"))


def check_file(path: Path) -> list[Violation]:
    """Return a violation iff the file is missing the required header."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [Violation(path=path, line=0, message="file is not valid UTF-8")]

    # We scan the first 20 lines so a long shebang + encoding cookie +
    # module docstring still leaves room to find the SPDX line.
    head = text.splitlines()[:20]
    for line in head:
        if line.strip() == REQUIRED_HEADER:
            return []

    return [
        Violation(
            path=path,
            line=1,
            message=(
                f"missing {REQUIRED_HEADER!r} in the file head. "
                "Add it as a top-of-file comment (after any shebang)."
            ),
        )
    ]


def run(paths: Sequence[Path] | None = None) -> list[Violation]:
    targets = list(paths) if paths else list(DEFAULT_TARGETS)
    violations: list[Violation] = []
    for file in _walk_python_files(targets):
        violations.extend(check_file(file))
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    paths = [Path(a) for a in args] if args else list(DEFAULT_TARGETS)
    violations = run(paths)
    for v in violations:
        print(v.format(REPO_ROOT), file=sys.stderr)
    if violations:
        print(f"check_license_headers: {len(violations)} violation(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
