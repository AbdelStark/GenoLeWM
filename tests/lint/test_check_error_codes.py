"""Unit tests for ``tools.lint.check_error_codes``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.lint import check_error_codes as linter


def write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_registry_discovery_finds_known_classes() -> None:
    registered = linter.discover_registered_classes()
    # Spot-check across families. The full inventory lives in test_errors.py.
    expected = {
        "InvalidEditError",
        "UnsupportedEditError",
        "WindowMismatchError",
        "CacheCorruptError",
        "NaNLossError",
        "EvalRegressionError",
        "ManifestHashMismatchError",
        "InvariantViolation",
    }
    assert expected.issubset(registered), expected - registered
    # The root class is intentionally absent from the registry: raising it
    # directly is a smell — leaf classes only (RFC-0012).
    assert "GenoLeWMError" not in registered


def test_bare_reraise_is_allowed(tmp_path: Path) -> None:
    f = write(
        tmp_path,
        "ok_bare.py",
        "def f():\n    try:\n        1/0\n    except Exception:\n        raise\n",
    )
    assert linter.check_file(f, linter.discover_registered_classes()) == []


def test_registered_raise_passes(tmp_path: Path) -> None:
    f = write(
        tmp_path,
        "ok_registered.py",
        "from geno_lewm.errors import InvalidEditError\n"
        "def f():\n"
        "    raise InvalidEditError('bad')\n",
    )
    assert linter.check_file(f, linter.discover_registered_classes()) == []


def test_builtin_raise_is_flagged(tmp_path: Path) -> None:
    f = write(
        tmp_path,
        "bad_builtin.py",
        "def f():\n    raise ValueError('nope')\n",
    )
    v = linter.check_file(f, linter.discover_registered_classes())
    assert len(v) == 1
    assert v[0].check == "raise_geno_lewm_error"
    assert "ValueError" in v[0].message


def test_unregistered_class_is_flagged(tmp_path: Path) -> None:
    f = write(
        tmp_path,
        "bad_unregistered.py",
        "def f():\n    raise SomeGhostError('?')\n",
    )
    v = linter.check_file(f, linter.discover_registered_classes())
    assert len(v) == 1
    # The name ends in 'Error' so it lands in the "looks like a stdlib /
    # subclass but not registered" bucket.
    assert v[0].check == "raise_geno_lewm_error"


def test_unregistered_non_error_class_is_flagged_as_registry_violation(tmp_path: Path) -> None:
    f = write(
        tmp_path,
        "bad_unknown.py",
        "def f():\n    raise Bogus('?')\n",
    )
    v = linter.check_file(f, linter.discover_registered_classes())
    assert len(v) == 1
    assert v[0].check == "registered_error_code"


def test_raise_from_clause_still_inspects_class(tmp_path: Path) -> None:
    f = write(
        tmp_path,
        "bad_from.py",
        "def f(e):\n    raise RuntimeError('x') from e\n",
    )
    v = linter.check_file(f, linter.discover_registered_classes())
    assert len(v) == 1
    assert v[0].check == "raise_geno_lewm_error"


def test_attribute_call_uses_attr_name(tmp_path: Path) -> None:
    # ``raise errors.InvalidEditError(...)`` is the alias-style raise.
    f = write(
        tmp_path,
        "ok_attr.py",
        "from geno_lewm import errors\ndef f():\n    raise errors.InvalidEditError('bad')\n",
    )
    assert linter.check_file(f, linter.discover_registered_classes()) == []


def test_main_returns_zero_for_clean_package() -> None:
    # Running the linter against the real package should pass at this
    # point in history — no production raises exist yet.
    assert linter.main([str(linter.PACKAGE_DIR)]) == 0


def test_main_returns_one_with_violations(tmp_path: Path) -> None:
    write(tmp_path, "bad.py", "raise ValueError('x')\n")
    rc = linter.main([str(tmp_path)])
    assert rc == 1


def test_violation_format_includes_path_line_col(tmp_path: Path) -> None:
    bad = write(tmp_path, "fmt.py", "\n\nraise ValueError('x')\n")
    [v] = linter.check_file(bad, linter.discover_registered_classes())
    formatted = v.format(tmp_path)
    assert formatted.startswith("fmt.py:3:")
    assert "error:" in formatted
    assert "[raise_geno_lewm_error]" in formatted


def test_syntax_error_reported_as_violation(tmp_path: Path) -> None:
    f = write(tmp_path, "broken.py", "def f(:\n")
    v = linter.check_file(f, linter.discover_registered_classes())
    assert len(v) == 1
    assert "could not parse" in v[0].message


def test_errors_module_itself_is_skipped() -> None:
    # The internal ``raise InvariantViolation(...)`` inside errors.py
    # would otherwise create a self-referential lint loop.
    rc = linter.main([str(linter.ERRORS_MODULE)])
    assert rc == 0


@pytest.mark.parametrize(
    "raise_line",
    [
        "raise ValueError('x')",
        "raise TypeError('x')",
        "raise RuntimeError('x')",
        "raise NotImplementedError",
        "raise KeyError('x')",
        "raise OSError('x')",
    ],
)
def test_common_builtins_are_all_flagged(tmp_path: Path, raise_line: str) -> None:
    f = write(tmp_path, "b.py", f"def f():\n    {raise_line}\n")
    v = linter.check_file(f, linter.discover_registered_classes())
    assert len(v) == 1
    assert v[0].check == "raise_geno_lewm_error"
