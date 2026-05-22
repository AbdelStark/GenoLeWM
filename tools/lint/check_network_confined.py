# SPDX-License-Identifier: Apache-2.0
"""``network_confined`` AST linter (RFC-0015 §3.4).

Restrict network-capable imports to the two files that legitimately
need them:

- ``geno_lewm/deploy/runtime.py`` — first-run model download (RFC-0010 §3.7).
- ``geno_lewm/cli/update.py`` — explicit user-initiated update (RFC-0010 §3.3).

Anywhere else in ``geno_lewm/`` importing ``urllib``, ``urllib3``,
``urllib.request``, ``httpx``, ``requests``, ``aiohttp``, ``socket``,
or similar networking modules is a contract violation: the runtime
must remain fail-closed off-network by default (INV-OBS-5,
RFC-0010 §3.7).

The check fires on:
- ``import <forbidden>``
- ``from <forbidden> import …``
- ``import <forbidden>.<submodule>``
- ``from <forbidden>.<submodule> import …``
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "geno_lewm"

#: Top-level module names whose import is restricted. Submodule
#: imports (``urllib.request``) are caught because we compare against
#: the first ``.``-component of the import target.
_FORBIDDEN_MODULES: frozenset[str] = frozenset(
    {
        "urllib",
        "urllib3",
        "httpx",
        "requests",
        "aiohttp",
        "socket",
        "ssl",
        "asyncio",  # asyncio enables many networking patterns; explicit allowlist
    }
)

#: Files (relative to ``geno_lewm/``) allowed to import networking modules.
_ALLOWED_PATHS: frozenset[Path] = frozenset(
    {
        Path("deploy/runtime.py"),
        Path("cli/update.py"),
    }
)


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    col: int
    module: str
    message: str

    def format(self, root: Path) -> str:
        try:
            rel = self.path.relative_to(root)
        except ValueError:
            rel = self.path
        return f"{rel}:{self.line}:{self.col}: error: [network_confined] {self.message}"


def _is_allowlisted(file: Path) -> bool:
    try:
        rel = file.resolve().relative_to(PACKAGE_DIR.resolve())
    except ValueError:
        return False
    return rel in _ALLOWED_PATHS


def _first_part(name: str) -> str:
    return name.split(".", 1)[0]


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
                module="",
                message=f"could not parse file: {e.msg}",
            )
        ]

    out: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_name = _first_part(alias.name)
                if root_name in _FORBIDDEN_MODULES:
                    out.append(
                        Violation(
                            path=path,
                            line=node.lineno,
                            col=node.col_offset,
                            module=alias.name,
                            message=(
                                f"import of network-capable module {alias.name!r} "
                                "is not allowed outside deploy/runtime.py or cli/update.py "
                                "(RFC-0010 §3.7, INV-OBS-5)."
                            ),
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative imports never refer to the forbidden top-level set
            mod = node.module or ""
            root_name = _first_part(mod)
            if root_name in _FORBIDDEN_MODULES:
                out.append(
                    Violation(
                        path=path,
                        line=node.lineno,
                        col=node.col_offset,
                        module=mod,
                        message=(
                            f"from-import of network-capable module {mod!r} "
                            "is not allowed outside deploy/runtime.py or cli/update.py "
                            "(RFC-0010 §3.7, INV-OBS-5)."
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
        print(f"check_network_confined: {len(violations)} violation(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
