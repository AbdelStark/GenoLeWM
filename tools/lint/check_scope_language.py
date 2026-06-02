# SPDX-License-Identifier: Apache-2.0
"""Reject de-scoped trust claims in repo text."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_CHECKED_SUFFIXES = frozenset(
    {
        ".cfg",
        ".json",
        ".ipynb",
        ".md",
        ".py",
        ".rst",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
_SKIPPED_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "site",
        "target",
    }
)
_ALLOWLISTED_PATHS = frozenset(
    {
        Path("tools/lint/check_scope_language.py"),
        Path("tests/lint/test_check_scope_language.py"),
    }
)
_BANNED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "unsupported_trust_scope",
        re.compile(r"\b" + "sta" + r"rk(?:s|[-_\s]proof|[-_\s]proven)?\b", re.IGNORECASE),
    ),
    (
        "unsupported_inference_scope",
        re.compile(r"\bverifiable[-\s]+" + "inference" + r"\b", re.IGNORECASE),
    ),
    (
        "proof_of_inference_scope",
        re.compile(r"\bproof(?:s)?[-\s]+of[-\s]+inference\b", re.IGNORECASE),
    ),
    (
        "inference_certification_scope",
        re.compile(
            r"\binference[-\s]+certification\b"
            r"|\bcertification[-\s]+of[-\s]+inference\b"
            r"|\bindependent[-\s]+certification\b"
            r"|\bindependently[-\s]+certified\b",
            re.IGNORECASE,
        ),
    ),
    (
        "legacy_receipt_field_scope",
        re.compile(
            r"[\"']attestation[\"']\s*:"
            r"|\battestation\.kind\b"
            r"|\[[\"']attestation[\"']\]",
            re.IGNORECASE,
        ),
    ),
    (
        "unsupported_provenance_kind_scope",
        re.compile(r"\bhardware_signed\b|\bexternal_certified\b", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class Violation:
    """One de-scoped scope-language violation."""

    path: Path
    line: int
    col: int
    check: str
    snippet: str

    def format(self, root: Path) -> str:
        try:
            rel = self.path.relative_to(root)
        except ValueError:
            rel = self.path
        return (
            f"{rel}:{self.line}:{self.col}: error: [scope_language] "
            f"{self.check} is de-scoped; use checksum/provenance wording instead: "
            f"{self.snippet!r}"
        )


def check_file(path: Path) -> list[Violation]:
    """Return de-scoped scope-language violations in ``path``."""
    if _is_allowlisted(path) or not _is_text_target(path):
        return []
    try:
        text = _read_text(path)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    violations: list[Violation] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for check, pattern in _BANNED_PATTERNS:
            match = pattern.search(line)
            if match is None:
                continue
            violations.append(
                Violation(
                    path=path,
                    line=line_no,
                    col=match.start(),
                    check=check,
                    snippet=line.strip(),
                )
            )
    return violations


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".ipynb":
        return _notebook_text(path)
    return path.read_text(encoding="utf-8")


def _notebook_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ""
    cells = payload.get("cells")
    if not isinstance(cells, list):
        return ""
    chunks: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        chunks.extend(_source_text(cell.get("source")))
        outputs = cell.get("outputs")
        if isinstance(outputs, list):
            for output in outputs:
                if isinstance(output, dict):
                    chunks.extend(_notebook_output_text(output))
    return "\n".join(chunks)


def _notebook_output_text(output: dict[str, object]) -> tuple[str, ...]:
    chunks: list[str] = []
    chunks.extend(_source_text(output.get("text")))
    data = output.get("data")
    if isinstance(data, dict):
        for key in ("text/plain", "text/markdown"):
            chunks.extend(_source_text(data.get(key)))
    return tuple(chunks)


def _source_text(raw: object) -> tuple[str, ...]:
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return tuple(raw)
    return ()


def run(paths: Sequence[Path] | None = None) -> list[Violation]:
    """Run the scope-language check over repo text files."""
    targets = list(paths) if paths else [REPO_ROOT]
    violations: list[Violation] = []
    for file in _walk_files(targets):
        violations.extend(check_file(file))
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    paths = [Path(arg) for arg in args] if args else [REPO_ROOT]
    violations = run(paths)
    for violation in violations:
        print(violation.format(REPO_ROOT), file=sys.stderr)
    if violations:
        print(f"check_scope_language: {len(violations)} violation(s)", file=sys.stderr)
        return 1
    return 0


def _walk_files(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_file():
            yield path
            continue
        if not path.is_dir():
            continue
        for child in sorted(path.rglob("*")):
            if child.is_dir():
                continue
            if any(part in _SKIPPED_DIRS or part.startswith(".venv") for part in child.parts):
                continue
            yield child


def _is_text_target(path: Path) -> bool:
    return path.suffix.lower() in _CHECKED_SUFFIXES


def _is_allowlisted(path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return False
    return rel in _ALLOWLISTED_PATHS


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
