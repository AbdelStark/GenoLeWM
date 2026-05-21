# SPDX-License-Identifier: Apache-2.0
"""``no_print`` AST linter (RFC-0015 §3.4).

Disallow bare ``print(...)`` calls in ``geno_lewm/``. Production code
emits records through :mod:`geno_lewm.observability` and the typed
error layer; ``print`` is reserved for CLI entry points (which live
under ``geno_lewm/cli/`` and pass an explicit ``file=`` to make their
intent visible) and tests.

The check fires on any ``ast.Call`` whose function is the bare name
``print``. ``print(..., file=sys.stderr)`` is still flagged unless
the file is explicitly allowlisted, because CLI files belong here.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "geno_lewm"

#: Paths (relative to ``geno_lewm/``) where ``print`` is allowed. CLI
#: dispatch is the only place the package writes to stdout / stderr
#: directly; the structured logger handles everything else.
_ALLOWED_PRINT_PATHS: frozenset[Path] = frozenset(
    {
        Path("cli"),  # every file under geno_lewm/cli/
    }
)


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    col: int
    message: str

    def format(self, root: Path) -> str:
        try:
            rel = self.path.relative_to(root)
        except ValueError:
            rel = self.path
        return f"{rel}:{self.line}:{self.col}: error: [no_print] {self.message}"


def _is_allowlisted(file: Path) -> bool:
    try:
        rel = file.resolve().relative_to(PACKAGE_DIR.resolve())
    except ValueError:
        return False  # outside the canonical package — still scanned (test fixtures use tmp dirs)
    return any(str(rel).startswith(str(p) + "/") or rel == p for p in _ALLOWED_PRINT_PATHS)


def check_file(path: Path) -> list[Violation]:
    if _is_allowlisted(path):
        return []
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        return [
            Violation(
                path=path,
                line=e.lineno or 0,
                col=(e.offset or 0) - 1,
                message=f"could not parse file: {e.msg}",
            )
        ]
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "print":
            out.append(
                Violation(
                    path=path,
                    line=node.lineno,
                    col=node.col_offset,
                    message=(
                        "bare print() is not allowed in geno_lewm/; "
                        "use geno_lewm.observability.get_logger(...) instead "
                        "(RFC-0013 / RFC-0015 §3.4)."
                    ),
                )
            )
    return out


def _walk_python_files(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            yield p
        elif p.is_dir():
            yield from sorted(p.rglob("*.py"))


def run(paths: Sequence[Path] | None = None) -> list[Violation]:
    targets = list(paths) if paths else [PACKAGE_DIR]
    violations: list[Violation] = []
    for file in _walk_python_files(targets):
        violations.extend(check_file(file))
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    paths = [Path(a) for a in args] if args else [PACKAGE_DIR]
    violations = run(paths)
    for v in violations:
        print(v.format(REPO_ROOT), file=sys.stderr)
    if violations:
        print(f"check_no_print: {len(violations)} violation(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
