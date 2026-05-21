"""Generate the API reference pages for mkdocstrings.

Run by ``mkdocs-gen-files`` at build time. For every public submodule
under :mod:`geno_lewm` we emit a ``reference/<module>.md`` file
containing a single ``::: <module>`` directive; mkdocstrings then
renders the docstring tree. A ``SUMMARY.md`` is generated so
``mkdocs-literate-nav`` can pick up the structure without a hand-edited
``mkdocs.yml`` nav entry per module.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files  # type: ignore[import-not-found]

PACKAGE_DIR = Path(__file__).resolve().parents[2] / "geno_lewm"
REFERENCE_DIR = Path("reference")


def _iter_modules() -> list[tuple[str, Path]]:
    """Return ``(dotted_module, source_path)`` pairs for every public
    submodule, sorted by dotted name."""
    found: list[tuple[str, Path]] = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if any(part.startswith("_") and part != "__init__.py" for part in path.relative_to(PACKAGE_DIR).parts):
            continue
        rel = path.relative_to(PACKAGE_DIR.parent)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            continue
        dotted = ".".join(parts)
        found.append((dotted, path))
    return found


def main() -> None:
    nav = mkdocs_gen_files.Nav()

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Index page for the reference section.
    index_path = REFERENCE_DIR / "index.md"
    with mkdocs_gen_files.open(index_path, "w") as fd:
        fd.write(
            "# API reference\n\n"
            "Auto-generated from the source docstrings. Every public\n"
            "submodule has its own page; private modules (anything\n"
            "matching ``_*``) are excluded.\n\n"
            "Stability of each symbol follows RFC-0014. The public\n"
            "surface is committed at ``tests/api/public_surface.json``\n"
            "and any change is gated by a CI snapshot.\n"
        )
    nav["Overview"] = "index.md"

    for dotted, _src in _iter_modules():
        page_path = REFERENCE_DIR / f"{dotted}.md"
        with mkdocs_gen_files.open(page_path, "w") as fd:
            fd.write(f"# `{dotted}`\n\n::: {dotted}\n")
        # Build a nested nav entry from the dotted parts. The first
        # part is always "geno_lewm" — keep it as a single nav entry
        # so the sidebar renders the package as the root.
        nav[tuple(dotted.split("."))] = f"{dotted}.md"

    with mkdocs_gen_files.open(REFERENCE_DIR / "SUMMARY.md", "w") as fd:
        fd.writelines(nav.build_literate_nav())


main()
