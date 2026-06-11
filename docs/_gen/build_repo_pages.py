"""Mirror project-root markdown into the docs tree with rewritten links.

The project-root files (``README.md``, ``CHANGELOG.md``,
``CONTRIBUTING.md``, ``SECURITY.md``, ``PRIVACY.md``,
``CODE_OF_CONDUCT.md``, and ``ARCHITECTURE.md``) reference paths from
the repo root, so links need docs-relative rewrites when rendered
inside MkDocs.

This script emits the docs-tree copies through ``mkdocs-gen-files``;
nothing on disk is touched and the canonical repo-root files stay
authoritative.
"""

from __future__ import annotations

import re
from pathlib import Path

import mkdocs_gen_files  # type: ignore[import-not-found]

REPO_ROOT = Path(__file__).resolve().parents[2]

# (source -> destination) for top-level docs that map 1:1 into MkDocs.
_SOURCES: tuple[tuple[str, str], ...] = (
    ("CHANGELOG.md", "changelog.md"),
    ("CONTRIBUTING.md", "contributing.md"),
    ("CODE_OF_CONDUCT.md", "code-of-conduct.md"),
    ("SECURITY.md", "security.md"),
    ("PRIVACY.md", "privacy.md"),
    ("ARCHITECTURE.md", "architecture.md"),
)

# Link rewrites applied to the rendered copies. Order is important —
# longer patterns first so they aren't masked by shorter prefixes.
_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\]\(docs/api/"), "](api/"),
    (re.compile(r"\]\(docs/"), "]("),
    (re.compile(r"\]\(\.\./docs/"), "]("),
    (re.compile(r"\]\(CHANGELOG\.md"), "](changelog.md"),
    (re.compile(r"\]\(CONTRIBUTING\.md"), "](contributing.md"),
    (re.compile(r"\]\(CODE_OF_CONDUCT\.md"), "](code-of-conduct.md"),
    (re.compile(r"\]\(SECURITY\.md"), "](security.md"),
    (re.compile(r"\]\(PRIVACY\.md"), "](privacy.md"),
    (re.compile(r"\]\(ARCHITECTURE\.md"), "](architecture.md"),
    (re.compile(r"\]\(README\.md"), "](index.md"),
    (re.compile(r"\]\(LICENSE\)"), "](https://github.com/AbdelStark/GenoLeWM/blob/main/LICENSE)"),
)


def _rewrite(text: str) -> str:
    for pattern, repl in _REPLACEMENTS:
        text = pattern.sub(repl, text)
    return text


def main() -> None:
    for src, dest in _SOURCES:
        content = (REPO_ROOT / src).read_text(encoding="utf-8")
        with mkdocs_gen_files.open(dest, "w") as fd:
            fd.write(_rewrite(content))


main()
