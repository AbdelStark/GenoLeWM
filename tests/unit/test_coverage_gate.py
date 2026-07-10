# SPDX-License-Identifier: Apache-2.0
"""Tests for ``tools.ci.coverage_gate`` (testing contract).

Covers Acceptance Criteria from issue #88:

- Pass case: PR touches a fully-covered file → exit 0.
- Fail case: PR touches a file with new uncovered lines → exit 1.

The tests construct synthetic Cobertura XML and unified diffs so the
gate is exercised without requiring a real git repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.ci import coverage_gate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coverage_xml(
    files: dict[str, dict[int, int]],
    *,
    source: str | None = None,
) -> str:
    """Build a minimal Cobertura XML from ``{filename: {line: hits}}``."""
    classes = []
    for fn, lines in files.items():
        line_xml = "\n".join(
            f'          <line number="{n}" hits="{h}"/>' for n, h in sorted(lines.items())
        )
        classes.append(
            f"""      <class filename="{fn}">
        <lines>
{line_xml}
        </lines>
      </class>"""
        )
    classes_xml = "\n".join(classes)
    sources_xml = "" if source is None else f"  <sources><source>{source}</source></sources>\n"
    return f"""<?xml version="1.0" ?>
<coverage version="7.0">
{sources_xml}\
  <packages>
    <package name="geno_lewm">
      <classes>
{classes_xml}
      </classes>
    </package>
  </packages>
</coverage>
"""


def _diff(file_lines: dict[str, list[int]]) -> str:
    """Build a unified diff that *adds* the given line numbers per file.

    The hunks claim each added line at its new-side line number with a
    single line of body. The result is parsed by ``parse_added_lines``.
    """
    parts: list[str] = []
    for fn, lines in file_lines.items():
        parts.append(f"diff --git a/{fn} b/{fn}")
        parts.append(f"--- a/{fn}")
        parts.append(f"+++ b/{fn}")
        for n in lines:
            parts.append(f"@@ -{n},0 +{n},1 @@")
            parts.append(f"+added line at {n}")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# parse_coverage_xml
# ---------------------------------------------------------------------------


def test_parse_coverage_xml_partitions_tracked_and_covered(tmp_path: Path) -> None:
    xml = _coverage_xml({"geno_lewm/foo.py": {1: 1, 2: 0, 3: 5, 4: 0}})
    p = tmp_path / "coverage.xml"
    p.write_text(xml, encoding="utf-8")

    tracked, covered = coverage_gate.parse_coverage_xml(p)

    assert tracked == {"geno_lewm/foo.py": frozenset({1, 2, 3, 4})}
    assert covered == {"geno_lewm/foo.py": frozenset({1, 3})}


def test_parse_coverage_xml_normalizes_paths(tmp_path: Path) -> None:
    # Cobertura writers sometimes emit absolute or Windows-style paths.
    xml = _coverage_xml({"./geno_lewm/foo.py": {10: 1}})
    p = tmp_path / "coverage.xml"
    p.write_text(xml, encoding="utf-8")
    tracked, _ = coverage_gate.parse_coverage_xml(p)
    assert "geno_lewm/foo.py" in tracked


def test_parse_coverage_xml_resolves_package_relative_source_path(tmp_path: Path) -> None:
    xml = _coverage_xml(
        {"encoder/carbon.py": {10: 1}},
        source=str(coverage_gate.REPO_ROOT / "geno_lewm"),
    )
    p = tmp_path / "coverage.xml"
    p.write_text(xml, encoding="utf-8")

    tracked, covered = coverage_gate.parse_coverage_xml(p)

    assert tracked == {"geno_lewm/encoder/carbon.py": frozenset({10})}
    assert covered == {"geno_lewm/encoder/carbon.py": frozenset({10})}


@pytest.mark.parametrize(
    "filename",
    [
        "geno_lewm/encoder/carbon.py",
        str(coverage_gate.REPO_ROOT / "geno_lewm" / "encoder" / "carbon.py"),
    ],
)
def test_parse_coverage_xml_preserves_prefixed_and_absolute_paths(
    tmp_path: Path,
    filename: str,
) -> None:
    xml = _coverage_xml(
        {filename: {10: 1}},
        source=str(coverage_gate.REPO_ROOT / "geno_lewm"),
    )
    p = tmp_path / "coverage.xml"
    p.write_text(xml, encoding="utf-8")

    tracked, _ = coverage_gate.parse_coverage_xml(p)

    assert tracked == {"geno_lewm/encoder/carbon.py": frozenset({10})}


# ---------------------------------------------------------------------------
# parse_added_lines
# ---------------------------------------------------------------------------


def test_parse_added_lines_single_file_single_hunk() -> None:
    diff = _diff({"geno_lewm/foo.py": [12, 13, 14]})
    added = coverage_gate.parse_added_lines(diff)
    assert added == {"geno_lewm/foo.py": frozenset({12, 13, 14})}


def test_parse_added_lines_ignores_deletion_only_files() -> None:
    diff = "diff --git a/old.py b/old.py\n--- a/old.py\n+++ /dev/null\n"
    assert coverage_gate.parse_added_lines(diff) == {}


def test_parse_added_lines_handles_multi_hunk() -> None:
    diff = (
        "diff --git a/geno_lewm/x.py b/geno_lewm/x.py\n"
        "--- a/geno_lewm/x.py\n"
        "+++ b/geno_lewm/x.py\n"
        "@@ -1,0 +1,2 @@\n"
        "+first\n"
        "+second\n"
        "@@ -50,0 +60,1 @@\n"
        "+later\n"
    )
    added = coverage_gate.parse_added_lines(diff)
    assert added == {"geno_lewm/x.py": frozenset({1, 2, 60})}


def test_parse_added_lines_strips_b_prefix() -> None:
    diff = "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n@@ -1,0 +1,1 @@\n+x\n"
    added = coverage_gate.parse_added_lines(diff)
    assert "foo.py" in added


# ---------------------------------------------------------------------------
# compute_results
# ---------------------------------------------------------------------------


def test_compute_results_skips_non_gated_files() -> None:
    changed = {"docs/index.md": frozenset({1}), "geno_lewm/y.py": frozenset({5})}
    tracked = {"geno_lewm/y.py": frozenset({5})}
    covered = {"geno_lewm/y.py": frozenset({5})}
    out = coverage_gate.compute_results(
        changed=changed,
        tracked=tracked,
        covered=covered,
        prefix="geno_lewm/",
    )
    assert [r.path for r in out] == ["geno_lewm/y.py"]


def test_compute_results_ignores_changed_lines_outside_tracked_universe() -> None:
    # Touching a comment line (not in the tracked set) should not count.
    changed = {"geno_lewm/y.py": frozenset({99})}
    tracked = {"geno_lewm/y.py": frozenset({1, 2, 3})}
    covered = {"geno_lewm/y.py": frozenset({1, 2, 3})}
    out = coverage_gate.compute_results(
        changed=changed,
        tracked=tracked,
        covered=covered,
        prefix="geno_lewm/",
    )
    assert out == []


def test_compute_results_partial_coverage_reported() -> None:
    changed = {"geno_lewm/y.py": frozenset({1, 2, 3, 4})}
    tracked = {"geno_lewm/y.py": frozenset({1, 2, 3, 4})}
    covered = {"geno_lewm/y.py": frozenset({1, 2})}
    out = coverage_gate.compute_results(
        changed=changed,
        tracked=tracked,
        covered=covered,
        prefix="geno_lewm/",
    )
    assert len(out) == 1
    assert out[0].total_changed == 4
    assert out[0].covered == 2
    assert out[0].ratio == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# FileResult
# ---------------------------------------------------------------------------


def test_file_result_zero_changed_is_full_credit() -> None:
    r = coverage_gate.FileResult(path="x", total_changed=0, covered=0)
    assert r.ratio == 1.0


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


def test_format_report_marks_failures() -> None:
    rows = [
        coverage_gate.FileResult(path="geno_lewm/a.py", total_changed=10, covered=10),
        coverage_gate.FileResult(path="geno_lewm/b.py", total_changed=10, covered=5),
    ]
    report = coverage_gate.format_report(rows, threshold=0.9)
    assert "PASS" in report
    assert "FAIL" in report
    assert "geno_lewm/a.py" in report
    assert "geno_lewm/b.py" in report


def test_format_report_empty_message() -> None:
    out = coverage_gate.format_report([], threshold=0.9)
    assert "no measurable changed lines" in out


# ---------------------------------------------------------------------------
# build_json_report
# ---------------------------------------------------------------------------


def test_build_json_report_summarizes_changed_file_coverage(tmp_path: Path) -> None:
    rows = [
        coverage_gate.FileResult(path="geno_lewm/a.py", total_changed=10, covered=10),
        coverage_gate.FileResult(path="geno_lewm/b.py", total_changed=10, covered=5),
    ]

    report = coverage_gate.build_json_report(
        rows,
        threshold=0.9,
        base="origin/main",
        prefix="geno_lewm/",
        coverage_xml=tmp_path / "coverage.xml",
        diff_file=tmp_path / "diff.patch",
    )

    assert report["generated_by"] == "tools.ci.coverage_gate"
    assert report["base"] == "origin/main"
    assert report["threshold"] == 0.9
    assert report["passed"] is False
    assert report["summary"] == {
        "measured_files": 2,
        "total_changed_lines": 20,
        "total_covered_lines": 15,
        "overall_changed_line_ratio": 0.75,
        "minimum_file_ratio": 0.5,
        "failing_files": 1,
    }
    assert report["files"] == [
        {
            "path": "geno_lewm/a.py",
            "changed_lines": 10,
            "covered_lines": 10,
            "coverage_ratio": 1.0,
            "status": "pass",
        },
        {
            "path": "geno_lewm/b.py",
            "changed_lines": 10,
            "covered_lines": 5,
            "coverage_ratio": 0.5,
            "status": "fail",
        },
    ]


def test_write_json_report_creates_parent_dirs(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "coverage-gate-report.json"

    coverage_gate.write_json_report({"b": 1, "a": 2}, output)

    assert output.read_text(encoding="utf-8") == '{\n  "a": 2,\n  "b": 1\n}\n'


# ---------------------------------------------------------------------------
# main() — end-to-end with synthetic diff
# ---------------------------------------------------------------------------


def test_main_pass_case(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Changed lines all covered → exit 0."""
    xml = tmp_path / "coverage.xml"
    diff = tmp_path / "diff.patch"
    xml.write_text(_coverage_xml({"geno_lewm/foo.py": {1: 1, 2: 1, 3: 1}}), encoding="utf-8")
    diff.write_text(_diff({"geno_lewm/foo.py": [1, 2, 3]}), encoding="utf-8")

    code = coverage_gate.main(
        ["--coverage-xml", str(xml), "--diff-file", str(diff), "--threshold", "0.9"]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "PASS" in captured.out
    assert "FAIL" not in captured.out


def test_main_fail_case(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Changed lines partially uncovered → exit 1."""
    xml = tmp_path / "coverage.xml"
    diff = tmp_path / "diff.patch"
    xml.write_text(
        _coverage_xml({"geno_lewm/foo.py": {1: 1, 2: 0, 3: 0, 4: 0}}),
        encoding="utf-8",
    )
    diff.write_text(_diff({"geno_lewm/foo.py": [1, 2, 3, 4]}), encoding="utf-8")

    code = coverage_gate.main(
        ["--coverage-xml", str(xml), "--diff-file", str(diff), "--threshold", "0.9"]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "FAIL" in captured.out
    assert "below" in captured.err


def test_main_fails_for_package_relative_cobertura_filename(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Canonical pytest-cov paths must intersect repository-relative diff paths."""
    xml = tmp_path / "coverage.xml"
    diff = tmp_path / "diff.patch"
    xml.write_text(
        _coverage_xml(
            {"encoder/carbon.py": {100: 1, 101: 0}},
            source=str(coverage_gate.REPO_ROOT / "geno_lewm"),
        ),
        encoding="utf-8",
    )
    diff.write_text(
        _diff({"geno_lewm/encoder/carbon.py": [100, 101]}),
        encoding="utf-8",
    )

    code = coverage_gate.main(
        ["--coverage-xml", str(xml), "--diff-file", str(diff), "--threshold", "0.9"]
    )
    captured = capsys.readouterr()

    assert code == 1
    assert "geno_lewm/encoder/carbon.py" in captured.out
    assert "FAIL" in captured.out
    assert "no measurable changed lines" not in captured.out


def test_main_writes_output_json_on_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    xml = tmp_path / "coverage.xml"
    diff = tmp_path / "diff.patch"
    output = tmp_path / "reports" / "coverage-gate-report.json"
    xml.write_text(
        _coverage_xml({"geno_lewm/foo.py": {1: 1, 2: 0, 3: 0, 4: 0}}),
        encoding="utf-8",
    )
    diff.write_text(_diff({"geno_lewm/foo.py": [1, 2, 3, 4]}), encoding="utf-8")

    code = coverage_gate.main(
        [
            "--coverage-xml",
            str(xml),
            "--diff-file",
            str(diff),
            "--threshold",
            "0.9",
            "--output-json",
            str(output),
        ]
    )
    capsys.readouterr()

    assert code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["summary"]["failing_files"] == 1
    assert report["files"][0]["path"] == "geno_lewm/foo.py"
    assert report["files"][0]["status"] == "fail"


def test_main_missing_coverage_xml(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = coverage_gate.main(["--coverage-xml", str(tmp_path / "missing.xml")])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_main_skips_when_no_changes_in_prefix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Touching only files outside the prefix → exit 0, gate is no-op."""
    xml = tmp_path / "coverage.xml"
    diff = tmp_path / "diff.patch"
    xml.write_text(_coverage_xml({"geno_lewm/foo.py": {1: 1}}), encoding="utf-8")
    diff.write_text(_diff({"docs/index.md": [1, 2]}), encoding="utf-8")

    code = coverage_gate.main(["--coverage-xml", str(xml), "--diff-file", str(diff)])
    captured = capsys.readouterr()
    assert code == 0
    assert "no measurable changed lines" in captured.out


def test_main_threshold_boundary(tmp_path: Path) -> None:
    """Exactly meeting threshold passes; below fails."""
    xml = tmp_path / "coverage.xml"
    diff = tmp_path / "diff.patch"
    # 9 of 10 covered = 0.9 → passes at threshold 0.9, fails at 0.91.
    file_lines = dict.fromkeys(range(1, 10), 1)
    file_lines[10] = 0
    xml.write_text(_coverage_xml({"geno_lewm/foo.py": file_lines}), encoding="utf-8")
    diff.write_text(_diff({"geno_lewm/foo.py": list(range(1, 11))}), encoding="utf-8")

    assert (
        coverage_gate.main(
            ["--coverage-xml", str(xml), "--diff-file", str(diff), "--threshold", "0.9"]
        )
        == 0
    )
    assert (
        coverage_gate.main(
            ["--coverage-xml", str(xml), "--diff-file", str(diff), "--threshold", "0.91"]
        )
        == 1
    )


# ---------------------------------------------------------------------------
# run_git_diff smoke (real `git` invocation)
# ---------------------------------------------------------------------------


def test_run_git_diff_against_head_is_empty(tmp_path: Path) -> None:
    """Diffing HEAD against HEAD must be empty (sanity check the subprocess wrapper)."""
    text = coverage_gate.run_git_diff("HEAD", coverage_gate.REPO_ROOT)
    assert text == ""


def test_run_git_diff_invalid_base_returns_exit_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-existent ref should be reported with exit code 2."""
    xml = tmp_path / "coverage.xml"
    xml.write_text(_coverage_xml({"geno_lewm/foo.py": {1: 1}}), encoding="utf-8")
    code = coverage_gate.main(["--coverage-xml", str(xml), "--base", "definitely-not-a-real-ref"])
    captured = capsys.readouterr()
    assert code == 2
    assert "git diff" in captured.err
