# SPDX-License-Identifier: Apache-2.0
"""Build the checked TeX paper and bind source/PDF artifact identities."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from geno_lewm.errors import GenoLeWMError, InputError, RuntimeSetupError, exit_code_for
from geno_lewm.provenance import sha256_file

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.paper_tex"
REPORT_NAME: Final = "paper_tex_build_report.json"
DEFAULT_PAPER_DIR: Final = Path("paper")
DEFAULT_TEX_NAME: Final = "main.tex"
DEFAULT_PDF_NAME: Final = "main.pdf"
SOURCE_FILES: Final = (
    "main.tex",
    "figures.tex",
    "tables.tex",
    "refs.bib",
    "neurips.sty",
)
CLAIM_BOUNDARY: Final = (
    "The TeX build report binds paper source and PDF artifact identity only; "
    "it is not model-quality, clinical, privacy, or deployment evidence."
)
Runner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Public-safe file identity for a paper build input or output."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class PaperTexBuildReport:
    """Machine-readable evidence for a checked TeX paper build."""

    schema_version: str
    generated_by: str
    generated_at: str
    paper_dir: str
    tex_path: str
    pdf: FileIdentity
    sources: tuple[FileIdentity, ...]
    commands: tuple[tuple[str, ...], ...]
    build_passes: int
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "paper_dir": self.paper_dir,
            "tex_path": self.tex_path,
            "pdf": self.pdf.to_dict(),
            "sources": [source.to_dict() for source in self.sources],
            "commands": [list(command) for command in self.commands],
            "build_passes": self.build_passes,
            "claim_boundary": self.claim_boundary,
        }


def build_paper_tex_report(
    *,
    paper_dir: Path = DEFAULT_PAPER_DIR,
    tex_name: str = DEFAULT_TEX_NAME,
    pdf_name: str = DEFAULT_PDF_NAME,
    passes: int = 2,
    runner: Runner | None = None,
    generated_at: str | None = None,
) -> PaperTexBuildReport:
    """Run Tectonic and return source/PDF artifact identity evidence."""
    _require_positive_int("passes", passes)
    paper_dir = Path(paper_dir)
    tex_path = _safe_paper_file(paper_dir, tex_name)
    pdf_path = _safe_paper_file(paper_dir, pdf_name)
    source_paths = tuple(_safe_paper_file(paper_dir, name) for name in SOURCE_FILES)
    if tex_path not in source_paths:
        source_paths = (tex_path, *source_paths)
    _require_files(source_paths)

    command = ("tectonic", "--keep-intermediates", tex_path.name)
    commands: list[tuple[str, ...]] = []
    selected_runner = _run_command if runner is None else runner
    for _index in range(passes):
        _run_build_pass(command, paper_dir=paper_dir, runner=selected_runner)
        commands.append(command)
    if not pdf_path.is_file():
        raise InputError(
            "TeX paper build did not produce the expected PDF",
            details={"path": _relative_to_paper_dir(paper_dir, pdf_path)},
            remediation="check tectonic output and paper/main.tex",
        )
    return PaperTexBuildReport(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        generated_at=generated_at or _utc_now(),
        paper_dir=_public_path(paper_dir),
        tex_path=tex_path.name,
        pdf=_identity(paper_dir, pdf_path),
        sources=tuple(_identity(paper_dir, path) for path in source_paths),
        commands=tuple(commands),
        build_passes=passes,
        claim_boundary=CLAIM_BOUNDARY,
    )


def write_paper_tex_report(
    *,
    paper_dir: Path,
    output: Path,
    tex_name: str = DEFAULT_TEX_NAME,
    pdf_name: str = DEFAULT_PDF_NAME,
    passes: int = 2,
) -> PaperTexBuildReport:
    """Build the TeX paper and write ``paper_tex_build_report.json``."""
    report = build_paper_tex_report(
        paper_dir=paper_dir,
        tex_name=tex_name,
        pdf_name=pdf_name,
        passes=passes,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = write_paper_tex_report(
            paper_dir=args.paper_dir,
            output=args.output,
            tex_name=args.tex_name,
            pdf_name=args.pdf_name,
            passes=args.passes,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the checked TeX paper and write source/PDF identity evidence.",
    )
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_PAPER_DIR / REPORT_NAME)
    parser.add_argument("--tex-name", default=DEFAULT_TEX_NAME)
    parser.add_argument("--pdf-name", default=DEFAULT_PDF_NAME)
    parser.add_argument("--passes", type=int, default=2)
    return parser


def _run_build_pass(
    command: tuple[str, ...],
    *,
    paper_dir: Path,
    runner: Runner,
) -> None:
    try:
        result = runner(command, paper_dir)
    except FileNotFoundError as exc:
        raise RuntimeSetupError(
            "Tectonic is required to build the TeX paper",
            remediation="install tectonic or use the Markdown paper package path",
        ) from exc
    if result.returncode != 0:
        raise InputError(
            "TeX paper build failed",
            details={
                "command": list(command),
                "returncode": result.returncode,
                "stderr": _trim_process_output(result.stderr),
            },
            remediation="fix paper/main.tex or the local TeX toolchain",
        )


def _run_command(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _identity(paper_dir: Path, path: Path) -> FileIdentity:
    return FileIdentity(
        path=_relative_to_paper_dir(paper_dir, path),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _require_files(paths: Sequence[Path]) -> None:
    missing = [_public_path(path) for path in paths if not path.is_file()]
    if missing:
        raise InputError(
            "TeX paper source files are missing",
            details={"missing": missing},
            remediation="restore the checked paper source tree before building the PDF",
        )


def _safe_paper_file(paper_dir: Path, name: str) -> Path:
    if not isinstance(name, str) or not name:
        raise InputError("paper file name must be non-empty")
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise InputError(
            "paper file names must stay inside paper_dir",
            details={"name": name},
        )
    return paper_dir / path


def _relative_to_paper_dir(paper_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(paper_dir).as_posix()
    except ValueError:
        return path.name


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(
            f"{name} must be a positive integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _public_path(path: Path) -> str:
    if path.is_absolute():
        return path.name
    if ".." in path.parts or not path.parts:
        return path.name
    return path.as_posix()


def _trim_process_output(text: str | None) -> str:
    if not text:
        return ""
    collapsed = " ".join(text.split())
    return collapsed[:500]


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
