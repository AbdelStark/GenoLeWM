# SPDX-License-Identifier: Apache-2.0
"""Build a typeset paper PDF from a generated Markdown paper draft."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from geno_lewm.errors import GenoLeWMError, InputError, RuntimeSetupError, exit_code_for
from geno_lewm.provenance import sha256_file

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.paper_pdf"
DEFAULT_PDF_ENGINE: Final = "xelatex"
_MAX_CAPTURED_OUTPUT: Final = 4000

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class PaperPdfReport:
    """Identity report for a Markdown-to-PDF paper build."""

    paper_markdown: Path
    output_pdf: Path
    paper_markdown_sha256: str
    paper_markdown_size_bytes: int
    output_pdf_sha256: str
    output_pdf_size_bytes: int
    pandoc: str
    pdf_engine: str
    command: tuple[str, ...]
    bibliography: Path | None = None
    bibliography_sha256: str | None = None
    csl: Path | None = None
    csl_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_by": GENERATED_BY,
            "paper_markdown": self.paper_markdown.name,
            "paper_markdown_sha256": self.paper_markdown_sha256,
            "paper_markdown_size_bytes": self.paper_markdown_size_bytes,
            "output_pdf": self.output_pdf.name,
            "output_pdf_sha256": self.output_pdf_sha256,
            "output_pdf_size_bytes": self.output_pdf_size_bytes,
            "pandoc": self.pandoc,
            "pdf_engine": self.pdf_engine,
            "bibliography": None if self.bibliography is None else self.bibliography.name,
            "bibliography_sha256": self.bibliography_sha256,
            "csl": None if self.csl is None else self.csl.name,
            "csl_sha256": self.csl_sha256,
            "command": list(self.command),
        }


def build_paper_pdf(
    *,
    paper_md: str | Path,
    output_pdf: str | Path,
    report_json: str | Path | None = None,
    pandoc: str = "pandoc",
    pdf_engine: str = DEFAULT_PDF_ENGINE,
    bibliography: str | Path | None = None,
    csl: str | Path | None = None,
    runner: Runner | None = None,
) -> PaperPdfReport:
    """Convert a generated Markdown paper to PDF with Pandoc."""
    paper_path = Path(paper_md)
    output_path = Path(output_pdf)
    bibliography_path = None if bibliography is None else Path(bibliography)
    csl_path = None if csl is None else Path(csl)

    _require_tool_name(pandoc, field="pandoc")
    _require_tool_name(pdf_engine, field="pdf_engine")
    _require_nonempty_input_file(paper_path, label="paper Markdown")
    if output_path.resolve() == paper_path.resolve():
        raise InputError(
            "output PDF must differ from paper Markdown", details={"path": str(output_path)}
        )
    if bibliography_path is not None:
        _require_nonempty_input_file(bibliography_path, label="bibliography")
    if csl_path is not None:
        _require_nonempty_input_file(csl_path, label="CSL style")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = _build_pandoc_command(
        paper_path=paper_path,
        output_path=output_path,
        pandoc=pandoc,
        pdf_engine=pdf_engine,
        bibliography_path=bibliography_path,
        csl_path=csl_path,
    )
    completed = _invoke_pandoc(command, runner or _run_pandoc, pandoc=pandoc)
    if completed.returncode != 0:
        raise RuntimeSetupError(
            "pandoc paper PDF build failed",
            details={
                "returncode": completed.returncode,
                "stderr": _truncate(completed.stderr),
                "stdout": _truncate(completed.stdout),
            },
            remediation="install Pandoc and the configured LaTeX engine, then rerun paper_pdf",
        )
    _require_nonempty_output_file(output_path)

    report = PaperPdfReport(
        paper_markdown=paper_path,
        output_pdf=output_path,
        paper_markdown_sha256=sha256_file(paper_path),
        paper_markdown_size_bytes=paper_path.stat().st_size,
        output_pdf_sha256=sha256_file(output_path),
        output_pdf_size_bytes=output_path.stat().st_size,
        pandoc=pandoc,
        pdf_engine=pdf_engine,
        command=_public_command(
            command,
            paper_path=paper_path,
            output_path=output_path,
            bibliography_path=bibliography_path,
            csl_path=csl_path,
        ),
        bibliography=bibliography_path,
        bibliography_sha256=None if bibliography_path is None else sha256_file(bibliography_path),
        csl=csl_path,
        csl_sha256=None if csl_path is None else sha256_file(csl_path),
    )
    if report_json is not None:
        report_path = Path(report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = build_paper_pdf(
            paper_md=args.paper_md,
            output_pdf=args.output_pdf,
            report_json=args.report_json,
            pandoc=args.pandoc,
            pdf_engine=args.pdf_engine,
            bibliography=args.bibliography,
            csl=args.csl,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a PDF from a generated paper Markdown file."
    )
    parser.add_argument("--paper-md", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--pandoc", default="pandoc")
    parser.add_argument("--pdf-engine", default=DEFAULT_PDF_ENGINE)
    parser.add_argument("--bibliography", type=Path)
    parser.add_argument("--csl", type=Path)
    return parser


def _build_pandoc_command(
    *,
    paper_path: Path,
    output_path: Path,
    pandoc: str,
    pdf_engine: str,
    bibliography_path: Path | None,
    csl_path: Path | None,
) -> tuple[str, ...]:
    command = [
        pandoc,
        str(paper_path),
        "--from",
        "gfm",
        "--standalone",
        "--pdf-engine",
        pdf_engine,
        "--output",
        str(output_path),
    ]
    if bibliography_path is not None or csl_path is not None:
        command.append("--citeproc")
    if bibliography_path is not None:
        command.extend(["--bibliography", str(bibliography_path)])
    if csl_path is not None:
        command.extend(["--csl", str(csl_path)])
    return tuple(command)


def _invoke_pandoc(
    command: Sequence[str],
    runner: Runner,
    *,
    pandoc: str,
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(command)
    except FileNotFoundError as exc:
        raise RuntimeSetupError(
            "paper PDF build requires pandoc",
            details={"pandoc": pandoc},
            remediation="install Pandoc and a LaTeX engine such as xelatex",
        ) from exc
    except OSError as exc:
        raise RuntimeSetupError(
            "paper PDF build could not launch pandoc",
            details={"pandoc": pandoc, "error": str(exc)},
            remediation="check the Pandoc executable path and local permissions",
        ) from exc


def _run_pandoc(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )


def _public_command(
    command: Sequence[str],
    *,
    paper_path: Path,
    output_path: Path,
    bibliography_path: Path | None,
    csl_path: Path | None,
) -> tuple[str, ...]:
    replacements = {
        str(paper_path): paper_path.name,
        str(output_path): output_path.name,
    }
    if bibliography_path is not None:
        replacements[str(bibliography_path)] = bibliography_path.name
    if csl_path is not None:
        replacements[str(csl_path)] = csl_path.name
    return tuple(replacements.get(part, part) for part in command)


def _require_tool_name(value: str, *, field: str) -> None:
    if not value.strip():
        raise InputError(f"{field} must be non-empty")


def _require_nonempty_input_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise InputError(f"{label} file is required", details={"path": str(path)})
    if path.stat().st_size <= 0:
        raise InputError(f"{label} file must be non-empty", details={"path": str(path)})


def _require_nonempty_output_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeSetupError(
            "pandoc did not produce the output PDF",
            details={"path": str(path)},
            remediation="check the Pandoc and LaTeX engine logs",
        )
    if path.stat().st_size <= 0:
        raise RuntimeSetupError(
            "pandoc produced an empty output PDF",
            details={"path": str(path)},
            remediation="check the Pandoc and LaTeX engine logs",
        )


def _truncate(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= _MAX_CAPTURED_OUTPUT:
        return value
    return value[:_MAX_CAPTURED_OUTPUT] + "...[truncated]"


if __name__ == "__main__":
    raise SystemExit(main())
