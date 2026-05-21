"""Tests for ``geno_lewm.api`` lifetime decorators."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from geno_lewm import api
from geno_lewm.errors import InputError


@pytest.fixture(autouse=True)
def _reset() -> Any:
    api._experimental_warned.clear()
    api._deprecated_warned.clear()
    yield
    api._experimental_warned.clear()
    api._deprecated_warned.clear()


# ---------------------------------------------------------------------------
# @experimental on functions.


def test_experimental_function_warns_once() -> None:
    @api.experimental
    def f(x: int) -> int:
        return x + 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert f(1) == 2
        assert f(2) == 3
        assert f(3) == 4
    # Only the first call emits a FutureWarning.
    matched = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert len(matched) == 1
    assert "experimental" in str(matched[0].message)


def test_experimental_metadata_preserved() -> None:
    @api.experimental
    def g(x: int) -> int:
        "g docstring"
        return x

    assert g.__name__ == "g"
    assert g.__doc__ == "g docstring"
    assert g.__wrapped__.__name__ == "g"  # functools.wraps populates this
    assert g.__geno_lewm_experimental__ is True


def test_experimental_with_reason_parameter() -> None:
    @api.experimental(reason="API under review")
    def h() -> None:
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h()
    [w] = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert "API under review" in str(w.message)


# ---------------------------------------------------------------------------
# @experimental on classes.


def test_experimental_class_warns_on_first_instantiation() -> None:
    @api.experimental
    class C:
        def __init__(self, x: int) -> None:
            self.x = x

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a = C(1)
        b = C(2)
    matched = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert len(matched) == 1
    assert a.x == 1 and b.x == 2
    assert C.__geno_lewm_experimental__ is True


# ---------------------------------------------------------------------------
# @deprecated on functions.


def test_deprecated_emits_once_per_call_site() -> None:
    @api.deprecated("use new_thing instead")
    def old() -> None:
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Two calls on the SAME line → one warning.
        # fmt: off
        old(); old()  # noqa: E702
        # fmt: on
    matched = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(matched) == 1
    assert "use new_thing instead" in str(matched[0].message)


def test_deprecated_emits_again_from_different_call_site() -> None:
    @api.deprecated("use new_thing instead")
    def old() -> int:
        return 1

    def call_from_helper() -> None:
        old()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old()
        call_from_helper()
    matched = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    # Two distinct call sites → two warnings.
    assert len(matched) == 2


def test_deprecated_preserves_function_metadata() -> None:
    @api.deprecated("see new_thing")
    def g(x: int) -> int:
        "g docstring"
        return x

    assert g.__name__ == "g"
    assert g.__doc__ == "g docstring"
    assert g.__wrapped__.__name__ == "g"
    assert g.__geno_lewm_deprecated__ is True


def test_deprecated_class_warns_per_instantiation_site() -> None:
    @api.deprecated("use NewC instead")
    class OldC:
        def __init__(self, x: int) -> None:
            self.x = x

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # fmt: off
        a = OldC(1); a2 = OldC(2)  # noqa: E702  — same line, one warning
        # fmt: on
        b = OldC(3)  # different line, another warning
    matched = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(matched) == 2
    assert a.x == 1 and a2.x == 2 and b.x == 3


def test_deprecated_reason_must_be_string() -> None:
    with pytest.raises(InputError):
        api.deprecated(reason=123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Mypy / IDE introspection — at runtime we can check inspect signatures.


def test_signature_visible_through_decorator() -> None:
    import inspect

    @api.experimental
    def f(x: int, y: int = 2) -> int:
        return x + y

    sig = inspect.signature(f)
    assert list(sig.parameters) == ["x", "y"]
    assert sig.parameters["y"].default == 2


def test_class_attributes_accessible_through_experimental() -> None:
    @api.experimental
    class K:
        CONSTANT = 7

        def method(self) -> int:
            return 1

    assert K.CONSTANT == 7
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        instance = K()
    assert instance.method() == 1


# ---------------------------------------------------------------------------
# Cross-decorator independence.


def test_experimental_and_deprecated_seen_sets_are_independent() -> None:
    @api.experimental
    def e() -> None:
        return None

    @api.deprecated("x")
    def d() -> None:
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        e()
        d()
    fut = [w for w in caught if issubclass(w.category, FutureWarning)]
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(fut) == 1
    assert len(dep) == 1
