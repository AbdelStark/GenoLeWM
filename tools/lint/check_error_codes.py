# SPDX-License-Identifier: Apache-2.0
"""Enforce the GenoLeWM error contract at PR time.

Two AST checks, both required by error taxonomy and testing contract:

1. ``raise_geno_lewm_error`` — every ``raise <Class>(...)`` in
   ``geno_lewm/`` raises a ``GenoLeWMError`` subclass (or is a bare
   re-raise).

2. ``registered_error_code`` — the raised class is one of the leaf
   classes registered in ``geno_lewm/errors.py::ERROR_CODES``.

The linter is import-free: it parses ``geno_lewm/errors.py`` with the
``ast`` module to discover the legal class set, then walks every other
file under ``geno_lewm/`` to inspect ``raise`` statements. That keeps
the check fast, dependency-free, and runnable in environments where the
package cannot be imported (e.g. during early CI bootstrap).

Usage:

    python -m tools.lint.check_error_codes [PATH ...]

If no paths are given, ``geno_lewm/`` is scanned. Exit code is 0 on
success and 1 on any violation (one ``error:`` line per offence,
emitted to stderr in a GitHub-Actions-friendly ``file:line:col:`` form).
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "geno_lewm"
ERRORS_MODULE = PACKAGE_DIR / "errors.py"

# Builtin or stdlib exceptions intentionally NEVER raised in ``geno_lewm/``.
# Listed here so the diagnostic for an accidental ``raise ValueError`` is
# unambiguous. The full check rejects anything not in the registry, but
# this list lets us produce a clearer message for the common cases.
_COMMON_BUILTIN_EXCS: frozenset[str] = frozenset(
    {
        "Exception",
        "BaseException",
        "ValueError",
        "TypeError",
        "RuntimeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "NotImplementedError",
        "AssertionError",
        "OSError",
        "IOError",
        "FileNotFoundError",
        "PermissionError",
        "TimeoutError",
        "ImportError",
        "LookupError",
        "ArithmeticError",
        "ZeroDivisionError",
        "OverflowError",
        "MemoryError",
        "StopIteration",
        "StopAsyncIteration",
    }
)


@dataclass(frozen=True)
class Violation:
    """A single linter offence."""

    path: Path
    line: int
    col: int
    check: str  # ``raise_geno_lewm_error`` | ``registered_error_code``
    message: str

    def format(self, root: Path) -> str:
        try:
            rel = self.path.relative_to(root)
        except ValueError:
            rel = self.path
        return f"{rel}:{self.line}:{self.col}: error: [{self.check}] {self.message}"


def discover_registered_classes(errors_module: Path = ERRORS_MODULE) -> set[str]:
    """Return the set of class names registered as raisable error types.

    A class is "registered" iff:

    - it is defined in ``errors.py`` (directly or by inheritance chain
      whose root is ``GenoLeWMError``), AND
    - it has an entry in ``ERROR_CODES``.

    We parse the AST instead of importing so the linter has no runtime
    dependency on the package being installable.
    """
    if not errors_module.is_file():
        raise FileNotFoundError(f"errors module not found: {errors_module}")

    tree = ast.parse(errors_module.read_text(encoding="utf-8"), filename=str(errors_module))

    # 1) Find every class transitively inheriting from GenoLeWMError.
    classes: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
            classes[node.name] = bases

    subclasses_of_root: set[str] = set()

    def _collect(name: str, stack: tuple[str, ...] = ()) -> bool:
        if name == "GenoLeWMError":
            return True
        if name in stack or name not in classes:
            return False
        return any(_collect(base, (*stack, name)) for base in classes[name])

    for name in classes:
        if _collect(name):
            subclasses_of_root.add(name)

    # 2) Intersect with the names appearing in ERROR_CODES entries. We
    #    look for the assignment ``ERROR_CODES: ... = (...)`` and pull the
    #    second positional argument of each ``ErrorCodeEntry(...)`` call.
    registered: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == "ERROR_CODES" for t in targets):
            continue
        value = node.value
        if value is None or not isinstance(value, ast.Tuple):
            continue
        for entry in value.elts:
            if not (
                isinstance(entry, ast.Call)
                and isinstance(entry.func, ast.Name)
                and entry.func.id == "ErrorCodeEntry"
                and len(entry.args) >= 2
                and isinstance(entry.args[1], ast.Name)
            ):
                continue
            registered.add(entry.args[1].id)

    # The intersection is the final answer: a name in ERROR_CODES that
    # somehow isn't a GenoLeWMError subclass is a separate registry bug
    # that the unit tests for #21 catch.
    return registered & subclasses_of_root


def _raised_class_name(node: ast.Raise) -> str | None:
    """Return the short name of the class being raised, or ``None`` for
    a bare ``raise``."""
    exc = node.exc
    if exc is None:
        return None  # bare re-raise

    # ``raise X(...)`` — Call
    if isinstance(exc, ast.Call):
        func = exc.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    # ``raise X`` — Name (uncommon but legal)
    if isinstance(exc, ast.Name):
        return exc.id
    if isinstance(exc, ast.Attribute):
        return exc.attr

    return None


def _walk_python_files(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            yield p
        elif p.is_dir():
            yield from sorted(p.rglob("*.py"))


def check_file(path: Path, registered: set[str]) -> list[Violation]:
    """Return all violations in ``path``."""
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:  # surface parse errors as their own violation
        return [
            Violation(
                path=path,
                line=e.lineno or 0,
                col=(e.offset or 0) - 1,
                check="raise_geno_lewm_error",
                message=f"could not parse file: {e.msg}",
            )
        ]

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        name = _raised_class_name(node)
        if name is None:
            continue  # bare re-raise: always allowed

        if name in registered:
            continue

        if name in _COMMON_BUILTIN_EXCS or name.endswith(("Error", "Exception")):
            violations.append(
                Violation(
                    path=path,
                    line=node.lineno,
                    col=node.col_offset,
                    check="raise_geno_lewm_error",
                    message=(
                        f"raises {name!r}, which is not a registered GenoLeWMError "
                        "subclass. Add a leaf class to geno_lewm/errors.py and an "
                        "entry to ERROR_CODES (error taxonomy)."
                    ),
                )
            )
        else:
            # Almost certainly a typo or an unregistered subclass.
            violations.append(
                Violation(
                    path=path,
                    line=node.lineno,
                    col=node.col_offset,
                    check="registered_error_code",
                    message=(
                        f"raises {name!r} but it is not in ERROR_CODES. "
                        "Register the class in geno_lewm/errors.py::ERROR_CODES "
                        "before raising it (INV-ERR-2)."
                    ),
                )
            )
    return violations


def run(paths: Sequence[Path] | None = None) -> list[Violation]:
    """Run the linter and return all violations."""
    targets = list(paths) if paths else [PACKAGE_DIR]
    registered = discover_registered_classes()

    violations: list[Violation] = []
    for file in _walk_python_files(targets):
        # Never lint the errors module against itself: it raises
        # ``InvariantViolation`` while *defining* it, which the
        # AST cannot resolve as a forward reference.
        if file.resolve() == ERRORS_MODULE.resolve():
            continue
        violations.extend(check_file(file, registered))
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    paths = [Path(a) for a in args] if args else [PACKAGE_DIR]
    violations = run(paths)
    for v in violations:
        print(v.format(REPO_ROOT), file=sys.stderr)
    if violations:
        print(
            f"check_error_codes: {len(violations)} violation(s)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
