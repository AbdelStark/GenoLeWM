"""Tests for the optional paper PDF build helper."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from geno_lewm.errors import InputError, RuntimeSetupError
from tools.release import paper_pdf
from tools.release.paper_pdf import build_paper_pdf


def test_build_paper_pdf_runs_pandoc_and_writes_public_report(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Results\n\nMeasured limitations are preserved.\n", encoding="utf-8")
    bibliography = tmp_path / "references.bib"
    bibliography.write_text("@article{demo,title={Demo}}\n", encoding="utf-8")
    csl = tmp_path / "style.csl"
    csl.write_text("<style></style>\n", encoding="utf-8")
    output_pdf = tmp_path / "paper.pdf"
    report_json = tmp_path / "paper_pdf_report.json"
    commands: list[tuple[str, ...]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        commands.append(tuple(command))
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(b"%PDF-1.7\nfixture\n")
        return subprocess.CompletedProcess(list(command), 0, stdout="", stderr="")

    report = build_paper_pdf(
        paper_md=paper,
        output_pdf=output_pdf,
        report_json=report_json,
        bibliography=bibliography,
        csl=csl,
        runner=runner,
    )
    payload = json.loads(report_json.read_text(encoding="utf-8"))

    assert commands == [
        (
            "pandoc",
            str(paper),
            "--from",
            "gfm",
            "--standalone",
            "--pdf-engine",
            "xelatex",
            "--output",
            str(output_pdf),
            "--citeproc",
            "--bibliography",
            str(bibliography),
            "--csl",
            str(csl),
        )
    ]
    assert report.to_dict() == payload
    assert payload["generated_by"] == "tools.release.paper_pdf"
    assert payload["paper_markdown"] == "paper.md"
    assert payload["output_pdf"] == "paper.pdf"
    assert payload["paper_markdown_sha256"].startswith("sha256:")
    assert payload["output_pdf_sha256"].startswith("sha256:")
    assert payload["bibliography"] == "references.bib"
    assert payload["bibliography_sha256"].startswith("sha256:")
    assert payload["csl"] == "style.csl"
    assert payload["csl_sha256"].startswith("sha256:")
    assert payload["command"] == [
        "pandoc",
        "paper.md",
        "--from",
        "gfm",
        "--standalone",
        "--pdf-engine",
        "xelatex",
        "--output",
        "paper.pdf",
        "--citeproc",
        "--bibliography",
        "references.bib",
        "--csl",
        "style.csl",
    ]
    assert str(tmp_path) not in json.dumps(payload)


def test_build_paper_pdf_rejects_empty_markdown(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("", encoding="utf-8")

    with pytest.raises(InputError, match="paper Markdown file must be non-empty"):
        build_paper_pdf(
            paper_md=paper,
            output_pdf=tmp_path / "paper.pdf",
            runner=_unexpected_runner,
        )


def test_build_paper_pdf_reports_missing_pandoc(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Paper\n", encoding="utf-8")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(command[0])

    with pytest.raises(RuntimeSetupError, match="requires pandoc"):
        build_paper_pdf(
            paper_md=paper,
            output_pdf=tmp_path / "paper.pdf",
            runner=runner,
        )


def test_build_paper_pdf_reports_pandoc_failure(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Paper\n", encoding="utf-8")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(command),
            2,
            stdout="",
            stderr="latex engine failed",
        )

    with pytest.raises(RuntimeSetupError, match="pandoc paper PDF build failed") as excinfo:
        build_paper_pdf(
            paper_md=paper,
            output_pdf=tmp_path / "paper.pdf",
            runner=runner,
        )

    assert excinfo.value.details["returncode"] == 2
    assert excinfo.value.details["stderr"] == "latex engine failed"


def test_build_paper_pdf_requires_pandoc_output(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Paper\n", encoding="utf-8")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(command), 0, stdout="", stderr="")

    with pytest.raises(RuntimeSetupError, match="did not produce"):
        build_paper_pdf(
            paper_md=paper,
            output_pdf=tmp_path / "paper.pdf",
            runner=runner,
        )


def test_paper_pdf_main_outputs_json_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Paper\n", encoding="utf-8")
    output_pdf = tmp_path / "paper.pdf"

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(b"%PDF-1.7\nfixture\n")
        return subprocess.CompletedProcess(list(command), 0, stdout="", stderr="")

    monkeypatch.setattr(paper_pdf, "_run_pandoc", runner)

    rc = paper_pdf.main(
        [
            "--paper-md",
            str(paper),
            "--output-pdf",
            str(output_pdf),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["paper_markdown"] == "paper.md"
    assert payload["output_pdf"] == "paper.pdf"
    assert str(tmp_path) not in json.dumps(payload)


def _unexpected_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    raise AssertionError(f"unexpected command: {command}")
