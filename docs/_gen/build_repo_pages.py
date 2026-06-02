"""Mirror project-root markdown into the docs tree with rewritten links.

The project-root files (``README.md``, ``CHANGELOG.md``,
``CONTRIBUTING.md``, ``SECURITY.md``, ``PRIVACY.md``,
``CODE_OF_CONDUCT.md``) reference paths from the repo root —
``docs/spec/04-error-model.md``, ``../rfcs/0001-…`` etc. — so they
browse correctly on GitHub. When rendered inside MkDocs we need
docs-relative paths.

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
)

# Link rewrites applied to the rendered copies. Order is important —
# longer patterns first so they aren't masked by shorter prefixes.
_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\]\(docs/spec/"), "](spec/"),
    (re.compile(r"\]\(docs/api/"), "](api/"),
    (re.compile(r"\]\(docs/roadmap/"), "](roadmap/"),
    (re.compile(r"\]\(docs/"), "]("),
    (re.compile(r"\]\(\.\./docs/spec/"), "](spec/"),
    (re.compile(r"\]\(\.\./docs/"), "]("),
    (re.compile(r"\]\(\.\./rfcs/"), "](rfcs/"),
    (re.compile(r"\]\(rfcs/"), "](rfcs/"),  # idempotent
    (re.compile(r"\]\(CHANGELOG\.md"), "](changelog.md"),
    (re.compile(r"\]\(CONTRIBUTING\.md"), "](contributing.md"),
    (re.compile(r"\]\(CODE_OF_CONDUCT\.md"), "](code-of-conduct.md"),
    (re.compile(r"\]\(SECURITY\.md"), "](security.md"),
    (re.compile(r"\]\(PRIVACY\.md"), "](privacy.md"),
    (re.compile(r"\]\(SPEC\.md"), "](spec/index.md"),
    (
        re.compile(r"\]\(SPECIFICATION\.md"),
        "](https://github.com/AbdelStark/GenoLeWM/blob/main/SPECIFICATION.md",
    ),
    (
        re.compile(r"\]\(ARCHITECTURE\.md"),
        "](https://github.com/AbdelStark/GenoLeWM/blob/main/ARCHITECTURE.md",
    ),
    (
        re.compile(r"\]\(ROADMAP\.md"),
        "](https://github.com/AbdelStark/GenoLeWM/blob/main/ROADMAP.md",
    ),
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
