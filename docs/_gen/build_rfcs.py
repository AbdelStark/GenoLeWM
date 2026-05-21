"""Mirror ``rfcs/*.md`` into ``docs/rfcs/`` at build time.

The RFC corpus is authored at repo root (``rfcs/``) and references
``../docs/spec/…`` / ``../docs/api/…`` so it browses correctly on
GitHub. When rendered under MkDocs the rfcs sit at ``docs/rfcs/`` and
the links must point at ``../spec/…`` / ``../api/…``.

We rewrite the inbound link patterns and emit one virtual page per
RFC. ``mkdocs-gen-files`` ships these to MkDocs without touching the
repository on disk.
"""

from __future__ import annotations

import re
from pathlib import Path

import mkdocs_gen_files  # type: ignore[import-not-found]

REPO_ROOT = Path(__file__).resolve().parents[2]
RFCS_DIR = REPO_ROOT / "rfcs"

# Order matters: longer patterns first to avoid prefix collisions.
_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\.\./docs/spec/"), "../spec/"),
    (re.compile(r"\.\./docs/api/"), "../api/"),
    (re.compile(r"\.\./docs/roadmap/"), "../roadmap/"),
    (re.compile(r"\.\./docs/"), "../"),
    (
        re.compile(r"\.\./SPECIFICATION\.md"),
        "https://github.com/AbdelStark/GenoLeWM/blob/main/SPECIFICATION.md",
    ),
    (
        re.compile(r"\.\./ARCHITECTURE\.md"),
        "https://github.com/AbdelStark/GenoLeWM/blob/main/ARCHITECTURE.md",
    ),
    (
        re.compile(r"\.\./ROADMAP\.md"),
        "https://github.com/AbdelStark/GenoLeWM/blob/main/ROADMAP.md",
    ),
    (
        re.compile(r"\.\./README\.md"),
        "https://github.com/AbdelStark/GenoLeWM/blob/main/README.md",
    ),
    (
        re.compile(r"\.\./LICENSE"),
        "https://github.com/AbdelStark/GenoLeWM/blob/main/LICENSE",
    ),
    (
        re.compile(r"\.\./SECURITY\.md"),
        "../security.md",
    ),
    (
        re.compile(r"\.\./PRIVACY\.md"),
        "../privacy.md",
    ),
    (
        re.compile(r"\.\./CHANGELOG\.md"),
        "../changelog.md",
    ),
    (
        re.compile(r"\.\./CONTRIBUTING\.md"),
        "../contributing.md",
    ),
    (
        re.compile(r"\.\./CODE_OF_CONDUCT\.md"),
        "../code-of-conduct.md",
    ),
)


def _rewrite(text: str) -> str:
    for pattern, repl in _REPLACEMENTS:
        text = pattern.sub(repl, text)
    return text


def main() -> None:
    for src in sorted(RFCS_DIR.glob("*.md")):
        rewritten = _rewrite(src.read_text(encoding="utf-8"))
        with mkdocs_gen_files.open(f"rfcs/{src.name}", "w") as fd:
            fd.write(rewritten)


main()
