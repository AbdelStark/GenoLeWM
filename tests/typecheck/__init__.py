# SPDX-License-Identifier: Apache-2.0
"""Static-typing regression tests (testing contract).

This package holds tests that confirm the public type surface stays
intentional — e.g., asserting that downstream callers can rely on the
documented `__all__`, that the `py.typed` marker ships, and that the
public-API snapshot still resolves. The big mypy gate runs separately
in CI (`make types`); the tests here cover the smaller, runtime-visible
type contracts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import geno_lewm


def test_py_typed_marker_is_present() -> None:
    """The ``py.typed`` marker must ship with the package."""
    pkg_dir = Path(geno_lewm.__file__).resolve().parent
    assert (pkg_dir / "py.typed").is_file(), "py.typed missing from wheel layout"


def test_public_all_is_a_sorted_tuple_or_list() -> None:
    """``geno_lewm.__all__`` must be ordered and contain no duplicates."""
    names = list(geno_lewm.__all__)
    assert names == sorted(names), "geno_lewm.__all__ must be alphabetically sorted"
    assert len(names) == len(set(names)), "geno_lewm.__all__ contains duplicates"


def test_every_public_name_resolves() -> None:
    """Every entry in ``__all__`` must be importable from the package."""
    for name in geno_lewm.__all__:
        assert hasattr(geno_lewm, name), f"geno_lewm.{name} declared but missing"


@pytest.mark.parametrize(
    "name",
    [
        "EditSpec",
        "Manifest",
        "Receipt",
        "ReceiptOutput",
        "DtypeConfig",
        "PoolingConfig",
    ],
)
def test_named_dataclasses_have_typed_init(name: str) -> None:
    """Each public dataclass must expose a typed ``__init__`` for downstream IDEs."""
    cls = getattr(geno_lewm, name)
    init = cls.__init__
    # Frozen dataclasses synthesise __init__ with the field type hints,
    # so the function should carry an ``__annotations__`` dict with at
    # least one annotated field.
    assert getattr(init, "__annotations__", {}), f"{name}.__init__ is missing annotations"
