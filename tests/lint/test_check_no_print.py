"""Unit tests for ``tools.lint.check_no_print``."""

from __future__ import annotations

from pathlib import Path

from tools.lint import check_no_print as linter


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def test_print_in_package_root_is_flagged(tmp_path: Path) -> None:
    f = _write(tmp_path / "geno_lewm", "core.py", "def f():\n    print('x')\n")
    # Simulate the directory being a logical sub-tree of geno_lewm.
    v = linter.check_file(f)
    # Our allowlist check is on the real geno_lewm path, so tmp_path
    # files are not allowlisted. Confirm the violation is emitted.
    assert len(v) == 1
    assert "bare print" in v[0].message


def test_no_print_returns_clean(tmp_path: Path) -> None:
    f = _write(tmp_path, "ok.py", "def f():\n    pass\n")
    assert linter.check_file(f) == []


def test_print_in_cli_dir_is_allowed() -> None:
    # The CLI verify file uses print() legitimately.
    cli_file = linter.PACKAGE_DIR / "cli" / "verify.py"
    if cli_file.is_file():
        # Allowlisted → no violations even though the file uses print().
        assert linter.check_file(cli_file) == []


def test_logger_calls_are_not_flagged(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "core.py",
        "def f(lg):\n    lg.info('event')\n",
    )
    assert linter.check_file(f) == []


def test_real_package_passes() -> None:
    # The shipped package should not raise the linter today.
    assert linter.main([str(linter.PACKAGE_DIR)]) == 0


def test_main_returns_one_on_violation(tmp_path: Path) -> None:
    _write(tmp_path, "p.py", "print('hi')\n")
    assert linter.main([str(tmp_path)]) == 1
