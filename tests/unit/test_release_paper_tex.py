"""Tests for the checked TeX paper build report."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from geno_lewm.provenance import sha256_file
from tools.release.paper_tex import (
    CLAIM_BOUNDARY,
    GENERATED_BY,
    REPORT_NAME,
    build_paper_tex_report,
    main,
)


def test_build_paper_tex_report_records_sources_and_pdf_identity(tmp_path: Path) -> None:
    paper_dir = _write_paper_tree(tmp_path / "paper")
    calls: list[tuple[str, ...]] = []

    report = build_paper_tex_report(
        paper_dir=paper_dir,
        runner=_fake_runner(calls, pdf_bytes=b"pdf bytes\n"),
        generated_at="2026-06-11T00:00:00Z",
    )

    assert report.generated_by == GENERATED_BY
    assert report.generated_at == "2026-06-11T00:00:00Z"
    assert report.pdf.path == "main.pdf"
    assert report.pdf.sha256 == sha256_file(paper_dir / "main.pdf")
    assert [source.path for source in report.sources] == [
        "main.tex",
        "figures.tex",
        "tables.tex",
        "refs.bib",
        "neurips.sty",
    ]
    assert calls == [
        ("tectonic", "--keep-intermediates", "main.tex"),
        ("tectonic", "--keep-intermediates", "main.tex"),
    ]
    payload = report.to_dict()
    assert payload["claim_boundary"] == CLAIM_BOUNDARY
    assert payload["commands"] == [list(calls[0]), list(calls[1])]


def test_build_paper_tex_report_rejects_missing_source(tmp_path: Path) -> None:
    paper_dir = _write_paper_tree(tmp_path / "paper")
    (paper_dir / "refs.bib").unlink()

    with pytest.raises(InputError, match="source files are missing"):
        build_paper_tex_report(
            paper_dir=paper_dir,
            runner=_fake_runner([], pdf_bytes=b"pdf bytes\n"),
        )


def test_build_paper_tex_report_rejects_failed_tectonic_run(tmp_path: Path) -> None:
    paper_dir = _write_paper_tree(tmp_path / "paper")

    with pytest.raises(InputError, match="TeX paper build failed"):
        build_paper_tex_report(
            paper_dir=paper_dir,
            runner=lambda command, cwd: subprocess.CompletedProcess(
                list(command),
                returncode=1,
                stdout="",
                stderr="undefined control sequence",
            ),
        )


def test_paper_tex_main_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paper_dir = _write_paper_tree(tmp_path / "paper")
    output = tmp_path / REPORT_NAME

    monkeypatch.setattr(
        "tools.release.paper_tex._run_command",
        _fake_runner([], pdf_bytes=b"pdf bytes\n"),
    )

    rc = main(["--paper-dir", str(paper_dir), "--output", str(output)])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["generated_by"] == GENERATED_BY
    assert payload["pdf"]["sha256"] == sha256_file(paper_dir / "main.pdf")


def _write_paper_tree(root: Path) -> Path:
    root.mkdir(parents=True)
    for name in ("figures.tex", "tables.tex", "refs.bib", "neurips.sty"):
        (root / name).write_text(f"% {name}\n", encoding="utf-8")
    (root / "main.tex").write_text(
        "\n".join(
            [
                r"\documentclass{article}",
                r"\begin{document}",
                "GenoLeWM fixture paper",
                r"\end{document}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _fake_runner(
    calls: list[tuple[str, ...]],
    *,
    pdf_bytes: bytes,
) -> Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]:
    def _run(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(command))
        (cwd / "main.pdf").write_bytes(pdf_bytes)
        return subprocess.CompletedProcess(list(command), returncode=0, stdout="", stderr="")

    return _run
