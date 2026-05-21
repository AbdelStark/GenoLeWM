# SPDX-License-Identifier: Apache-2.0
"""Public-API lifetime decorators (RFC-0014 §3.3, §3.6).

Two decorators mark stability on the public surface:

- :func:`experimental` — wraps a function or class that may change
  without notice. Emits :class:`FutureWarning` **once per process per
  decorated object** on first call / instantiation.
- :func:`deprecated` — wraps a function or class scheduled for removal.
  Emits :class:`DeprecationWarning` **once per process per call site**
  (i.e. once per ``(file, line)`` of the calling code). Takes a
  free-text ``reason`` describing what replaces the deprecated API.

Both decorators preserve ``__name__``, ``__doc__``, ``__module__``,
``__qualname__``, ``__wrapped__``, and (for classes) the original
attribute surface, so mypy and IDE introspection are unaffected.
"""

from __future__ import annotations

import functools
import inspect
import sys
import threading
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, cast, overload

__all__ = ["deprecated", "experimental"]

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

# Module-level "seen" sets, guarded by a lock. ``experimental`` warns
# once per decorated object; ``deprecated`` warns once per call site
# (file + line of the caller).
_experimental_warned: set[int] = set()
_deprecated_warned: set[tuple[int, str, int]] = set()
_warned_lock = threading.Lock()


def _emit_once(seen_key: object, category: type[Warning], message: str, stacklevel: int) -> None:
    """Issue a warning iff the (decorated-object | call-site) hasn't
    been warned about yet in this process."""
    with _warned_lock:
        if seen_key in _experimental_warned or seen_key in _deprecated_warned:
            return
    # Decide which set to dirty.
    with _warned_lock:
        if category is FutureWarning:
            _experimental_warned.add(cast(int, seen_key))
        else:
            _deprecated_warned.add(cast("tuple[int, str, int]", seen_key))
    warnings.warn(message, category=category, stacklevel=stacklevel)


def _caller_site(depth: int = 2) -> tuple[str, int]:
    """Return ``(filename, lineno)`` of the caller ``depth`` frames up.

    ``depth=2`` points at the user of the decorated function: frame 0
    is :func:`_caller_site` itself, frame 1 is the wrapper, frame 2 is
    the call site.
    """
    frame = sys._getframe(depth)
    return frame.f_code.co_filename, frame.f_lineno


# ---------------------------------------------------------------------------
# @experimental


@overload
def experimental(obj: F) -> F: ...
@overload
def experimental(*, reason: str = "") -> Callable[[F], F]: ...


def experimental(obj: Any = None, *, reason: str = "") -> Any:
    """Mark ``obj`` (a function or class) as experimental.

    Usage::

        @experimental
        def f(...): ...

        @experimental(reason="API shape under review")
        class C: ...

    Emits :class:`FutureWarning` once per process when the decorated
    object is first invoked (function) or instantiated (class). Later
    calls are silent.
    """

    def _decorate(target: Any) -> Any:
        msg = f"{target.__module__}.{target.__qualname__} is experimental and may change without notice."
        if reason:
            msg += f" {reason}"

        if inspect.isclass(target):
            original_init = target.__init__
            sentinel = id(target)

            @functools.wraps(original_init)
            def __init__(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: N807
                _emit_once(sentinel, FutureWarning, msg, stacklevel=2)
                original_init(self, *args, **kwargs)

            setattr(target, "__init__", __init__)  # noqa: B010
            target.__geno_lewm_experimental__ = True
            return target

        sentinel_func = id(target)

        @functools.wraps(target)
        def _wrap(*args: Any, **kwargs: Any) -> Any:
            _emit_once(sentinel_func, FutureWarning, msg, stacklevel=2)
            return target(*args, **kwargs)

        _wrap.__geno_lewm_experimental__ = True  # type: ignore[attr-defined]
        return _wrap

    # Distinguish bare ``@experimental`` from parameterised ``@experimental(reason=…)``.
    if obj is not None and callable(obj):
        return _decorate(obj)
    return _decorate


# ---------------------------------------------------------------------------
# @deprecated


def deprecated(reason: str = "") -> Callable[[F], F]:
    """Mark ``obj`` as deprecated.

    Usage::

        @deprecated("use new_thing() instead; removed in v0.3")
        def old_thing(...): ...

    Emits :class:`DeprecationWarning` once per process **per call
    site** — that is, once per ``(filename, lineno)`` of the calling
    code. Multiple call sites all warn; the same site warns only once.

    ``reason`` is appended to the warning message; pass a short
    actionable string ("use X instead").
    """
    if not isinstance(reason, str):
        from geno_lewm.errors import InputError  # type: ignore[unreachable]

        raise InputError(
            "deprecated(reason) must be str",
            details={"type": type(reason).__name__},
        )

    def _decorate(target: F) -> F:
        base_msg = f"{target.__module__}.{target.__qualname__} is deprecated."
        if reason:
            base_msg += f" {reason}"

        is_class = inspect.isclass(target)
        if is_class:
            original_init: Callable[..., None] = target.__init__

            @functools.wraps(original_init)
            def __init__(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: N807
                file, line = _caller_site(depth=2)
                _emit_once(
                    (id(target), file, line),
                    DeprecationWarning,
                    base_msg,
                    stacklevel=3,
                )
                original_init(self, *args, **kwargs)

            setattr(target, "__init__", __init__)  # noqa: B010
            target.__geno_lewm_deprecated__ = True  # type: ignore[attr-defined]
            return target

        @functools.wraps(target)
        def _wrap(*args: Any, **kwargs: Any) -> Any:
            file, line = _caller_site(depth=2)
            _emit_once(
                (id(target), file, line),
                DeprecationWarning,
                base_msg,
                stacklevel=3,
            )
            return target(*args, **kwargs)

        _wrap.__geno_lewm_deprecated__ = True  # type: ignore[attr-defined]
        return cast(F, _wrap)

    return _decorate
