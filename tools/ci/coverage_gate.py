# SPDX-License-Identifier: Apache-2.0
"""Changed-files coverage gate (RFC-0015 §3.7).

Compute branch-coverage on the lines added or modified in the current
PR (vs. ``origin/main``) and fail if any touched Python file under
``geno_lewm/`` falls below ``--threshold`` (default 90 %).

Why changed-files and not project-wide? RFC-0015 §4.2: a global ratchet
punishes new code that lands in a file whose siblings happen to be
under-covered. Changed-files coverage is fair and equally tight.

Usage::

    # Typical CI invocation, after a `pytest --cov=geno_lewm` run that
    # wrote ``coverage.xml`` in cobertura format.
    python -m tools.ci.coverage_gate \\
        --coverage-xml coverage.xml \\
        --base origin/main \\
        --threshold 0.9 \\
        --output-json coverage-gate-report.json

Inputs:

- ``--coverage-xml`` — Cobertura XML written by ``pytest --cov-report=xml``.
- ``--base`` — git ref to diff against (default ``origin/main``).
- ``--diff-file`` — read a unified diff from a file instead of running git
  (used by tests; lets the gate stay reproducible without a real repo).
- ``--threshold`` — minimum ratio of covered changed lines per file.
- ``--prefix`` — only files whose path starts with this prefix are gated
  (default ``geno_lewm/``).
- ``--output-json`` — optional deterministic machine-readable report for
  CI artifacts and coverage ratchet reviews.

Exit codes:

- ``0`` — every gated file at or above threshold (or no changed lines in
  the gated prefix at all).
- ``1`` — at least one file below threshold.
- ``2`` — invalid inputs (e.g., missing coverage XML).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_THRESHOLD: float = 0.90
DEFAULT_BASE: str = "origin/main"
DEFAULT_PREFIX: str = "geno_lewm/"


@dataclass(frozen=True)
class FileResult:
    """Coverage outcome for one changed Python file."""

    path: str
    total_changed: int
    covered: int

    @property
    def ratio(self) -> float:
        """Fraction of *tracked* changed lines that were hit."""
        if self.total_changed == 0:
            return 1.0
        return self.covered / self.total_changed


def parse_coverage_xml(
    path: Path,
) -> tuple[dict[str, frozenset[int]], dict[str, frozenset[int]]]:
    """Parse a Cobertura XML; return ``(tracked, covered)`` line sets.

    ``tracked`` is every line the coverage tool considered executable
    code (the universe over which we measure). ``covered`` is the
    subset whose ``hits`` attribute is positive. Coverage XML keys are
    repo-relative POSIX paths (``geno_lewm/foo.py``).
    """

    tree = ET.parse(path)
    root = tree.getroot()
    tracked: dict[str, set[int]] = {}
    covered: dict[str, set[int]] = {}
    for cls in root.iter("class"):
        fn = cls.get("filename")
        if fn is None:
            continue
        norm = _normalize_path(fn)
        t_set = tracked.setdefault(norm, set())
        c_set = covered.setdefault(norm, set())
        for line in cls.iter("line"):
            number = int(line.get("number", "0"))
            hits = int(line.get("hits", "0"))
            if number <= 0:
                continue
            t_set.add(number)
            if hits > 0:
                c_set.add(number)
    return (
        {k: frozenset(v) for k, v in tracked.items()},
        {k: frozenset(v) for k, v in covered.items()},
    )


def _normalize_path(p: str) -> str:
    """Strip drive letters and normalize to POSIX-style relative paths."""
    norm = p.replace("\\", "/").lstrip("./")
    # Strip leading repo-absolute prefix if any tool wrote one in.
    repo_str = str(REPO_ROOT).replace("\\", "/") + "/"
    norm = norm.removeprefix(repo_str)
    return norm


_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def parse_added_lines(diff_text: str) -> dict[str, frozenset[int]]:
    """Parse a ``git diff --unified=0`` and return added lines per file.

    Only ``+`` lines (in the *new* side of the diff) are returned. Pure
    deletions contribute nothing because there is nothing to cover.
    Renames are followed via the ``+++ b/<path>`` header.
    """

    added: dict[str, set[int]] = {}
    current: str | None = None
    cur_line: int = 0
    in_file = False
    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            target = target.removeprefix("b/")
            current = None if target == "/dev/null" else _normalize_path(target)
            in_file = current is not None
            continue
        if not in_file:
            continue
        if raw.startswith("@@"):
            m = _HUNK_RE.match(raw)
            if m is None or current is None:
                continue
            cur_line = int(m.group(1))
            continue
        if current is None:
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            added.setdefault(current, set()).add(cur_line)
            cur_line += 1
        elif raw.startswith("-"):
            # deletion: no line advance on the new side.
            continue
        else:
            # context lines (only present when unified != 0) — advance.
            cur_line += 1
    return {k: frozenset(v) for k, v in added.items()}


def run_git_diff(base: str, repo_root: Path) -> str:
    """Run ``git diff --unified=0 --no-color <base>...HEAD`` and return stdout."""
    cmd = ["git", "diff", "--unified=0", "--no-color", f"{base}...HEAD"]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def compute_results(
    *,
    changed: Mapping[str, frozenset[int]],
    tracked: Mapping[str, frozenset[int]],
    covered: Mapping[str, frozenset[int]],
    prefix: str,
) -> list[FileResult]:
    """Intersect changed lines with the tracked-code universe per file.

    Files outside ``prefix``, non-``.py`` files, and files whose changed
    lines are entirely non-code (blank lines, comments, docstrings) are
    omitted from the report — they cannot legitimately be measured.
    """
    out: list[FileResult] = []
    for path, lines in sorted(changed.items()):
        if not path.startswith(prefix):
            continue
        if not path.endswith(".py"):
            continue
        tracked_lines = tracked.get(path, frozenset())
        covered_lines = covered.get(path, frozenset())
        relevant = lines & tracked_lines
        if not relevant:
            continue
        hit = relevant & covered_lines
        out.append(
            FileResult(
                path=path,
                total_changed=len(relevant),
                covered=len(hit),
            )
        )
    return out


def format_report(results: Iterable[FileResult], threshold: float) -> str:
    """Human-readable summary table; intentionally ASCII-only."""
    rows = list(results)
    if not rows:
        return "coverage_gate: no measurable changed lines in gated prefix.\n"
    header = f"{'File':<60} {'Hit':>5} {'Lines':>6} {'%':>6}  Status"
    out = [header, "-" * len(header)]
    for r in rows:
        status = "PASS" if r.ratio >= threshold else "FAIL"
        out.append(
            f"{r.path:<60} {r.covered:>5} {r.total_changed:>6} {r.ratio * 100:>5.1f}%  {status}"
        )
    return "\n".join(out) + "\n"


def build_json_report(
    results: Iterable[FileResult],
    *,
    threshold: float,
    base: str,
    prefix: str,
    coverage_xml: Path,
    diff_file: Path | None,
) -> dict[str, object]:
    """Return a deterministic machine-readable changed-files coverage report."""
    rows = list(results)
    failing = [r for r in rows if r.ratio < threshold]
    total_changed = sum(r.total_changed for r in rows)
    total_covered = sum(r.covered for r in rows)
    return {
        "generated_by": "tools.ci.coverage_gate",
        "base": base,
        "prefix": prefix,
        "threshold": threshold,
        "coverage_xml": str(coverage_xml),
        "diff_source": str(diff_file) if diff_file is not None else "git",
        "passed": not failing,
        "summary": {
            "measured_files": len(rows),
            "total_changed_lines": total_changed,
            "total_covered_lines": total_covered,
            "overall_changed_line_ratio": (
                (total_covered / total_changed) if total_changed else None
            ),
            "minimum_file_ratio": min((r.ratio for r in rows), default=None),
            "failing_files": len(failing),
        },
        "files": [
            {
                "path": r.path,
                "changed_lines": r.total_changed,
                "covered_lines": r.covered,
                "coverage_ratio": r.ratio,
                "status": "pass" if r.ratio >= threshold else "fail",
            }
            for r in rows
        ],
    }


def write_json_report(report: Mapping[str, object], output: Path) -> None:
    """Write a stable JSON report, creating parent directories if needed."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="coverage_gate",
        description="Changed-files coverage gate (RFC-0015 §3.7).",
    )
    parser.add_argument(
        "--coverage-xml",
        type=Path,
        default=Path("coverage.xml"),
        help="Cobertura XML written by pytest-cov (default: coverage.xml)",
    )
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help=f"git ref to diff against (default: {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--diff-file",
        type=Path,
        default=None,
        help="read diff from file instead of `git diff` (mainly for tests)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"minimum covered-ratio per file (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"only gate files whose path starts with this (default: {DEFAULT_PREFIX})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="write a deterministic machine-readable report to this path",
    )
    args = parser.parse_args(argv)

    if not args.coverage_xml.is_file():
        print(
            f"coverage_gate: coverage XML not found at {args.coverage_xml}",
            file=sys.stderr,
        )
        return 2

    tracked, covered = parse_coverage_xml(args.coverage_xml)

    if args.diff_file is not None:
        diff_text = args.diff_file.read_text(encoding="utf-8")
    else:
        try:
            diff_text = run_git_diff(args.base, REPO_ROOT)
        except subprocess.CalledProcessError as exc:
            print(
                f"coverage_gate: `git diff` failed against {args.base}: {exc.stderr}",
                file=sys.stderr,
            )
            return 2

    changed = parse_added_lines(diff_text)
    results = compute_results(
        changed=changed,
        tracked=tracked,
        covered=covered,
        prefix=args.prefix,
    )
    sys.stdout.write(format_report(results, args.threshold))
    if args.output_json is not None:
        write_json_report(
            build_json_report(
                results,
                threshold=args.threshold,
                base=args.base,
                prefix=args.prefix,
                coverage_xml=args.coverage_xml,
                diff_file=args.diff_file,
            ),
            args.output_json,
        )

    failing = [r for r in results if r.ratio < args.threshold]
    if failing:
        print(
            f"coverage_gate: {len(failing)} file(s) below {args.threshold * 100:.0f}% threshold",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
