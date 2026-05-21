"""Resolve the public surface of ``geno_lewm`` and serialize it.

Defined by RFC-0014 §3.7. The output is a deterministic JSON document
committed at ``tests/api/public_surface.json``. The CI gate
``tests/api/test_public_surface.py`` diffs the committed snapshot
against a freshly-computed one:

- New entry → MINOR-or-MAJOR. Snapshot regeneration is the PR author's
  responsibility; CI warns but does not block.
- Removed entry → MAJOR required. CI **fails** unless the snapshot is
  explicitly updated.
- Changed signature → MAJOR required. CI **fails**.

"Public" means anything reachable from ``geno_lewm`` (or any
non-underscore submodule). Signatures are normalized so they are
stable across Python releases — enums in particular surface as
``enum(<MEMBER>=<value>, ...)`` instead of the synthesized
``__init__`` signature that varies between 3.10 / 3.11 / 3.12 / 3.13.
"""

from __future__ import annotations

import enum
import importlib
import inspect
import json
import pkgutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "geno_lewm"
SNAPSHOT_PATH = REPO_ROOT / "tests" / "api" / "public_surface.json"


def _iter_public_modules() -> Iterable[str]:
    """Yield import names of every non-underscore submodule."""
    root = importlib.import_module(PACKAGE)
    yield PACKAGE
    for info in pkgutil.walk_packages(root.__path__, prefix=PACKAGE + "."):
        # Skip dunder / single-underscore-prefixed names anywhere in the
        # dotted path (e.g. geno_lewm._redaction).
        parts = info.name.split(".")
        if any(p.startswith("_") for p in parts):
            continue
        yield info.name


def _enum_signature(obj: type[enum.Enum]) -> str:
    """Return a Python-version-stable signature for an ``enum.Enum`` subclass.

    Inspect's signature for an ``IntEnum`` reflects ``__init__`` of the
    underlying machinery, which changes shape between Python releases
    (``(*values)`` in 3.10, ``(value, names=None, ...)`` in 3.11+,
    ``(value, names=None, ..., boundary=None)`` in 3.12+, etc.). The
    members themselves are what users depend on; pin those.
    """
    # ``obj`` is the enum class; ``obj.__mro__`` is (Class, BaseEnum, ..., object).
    mro = obj.__mro__
    base = mro[1].__name__ if len(mro) > 1 else "Enum"  # e.g. "IntEnum"
    members = ", ".join(f"{m.name}={m.value!r}" for m in obj)
    return f"enum[{base}]({members})"


def _signature_for(obj: Any) -> str:
    """Return a stable string signature for ``obj``.

    For callables we use :func:`inspect.signature`; for classes we
    include the ``__init__`` signature. Enums get a hand-rolled,
    version-stable rendering. Anything else is recorded as its type
    name.
    """
    if inspect.isclass(obj) and issubclass(obj, enum.Enum):
        return _enum_signature(obj)

    try:
        if inspect.isclass(obj):
            try:
                sig = inspect.signature(obj)
                return f"class{sig!s}"
            except (TypeError, ValueError):
                return "class()"
        if callable(obj):
            sig = inspect.signature(obj)
            return f"callable{sig!s}"
    except (TypeError, ValueError):
        pass
    return f"value:{type(obj).__name__}"


def _public_attrs(module: Any) -> list[str]:
    """Return the names in ``module.__all__`` if set, else every name
    not beginning with an underscore."""
    declared = getattr(module, "__all__", None)
    if declared is not None:
        return [n for n in declared if not n.startswith("_")]
    return [n for n in dir(module) if not n.startswith("_")]


def compute_snapshot() -> dict[str, Any]:
    """Walk the package and return ``{symbol -> signature}`` ordered."""
    surface: dict[str, str] = {}

    for mod_name in sorted(set(_iter_public_modules())):
        try:
            module = importlib.import_module(mod_name)
        except Exception:  # pragma: no cover - import failure is itself a surface change
            surface[f"{mod_name}.__import_error__"] = "could not import"
            continue
        for name in sorted(_public_attrs(module)):
            obj = getattr(module, name, None)
            if obj is None:
                continue
            qualified = f"{mod_name}.{name}"
            surface[qualified] = _signature_for(obj)

    return {
        "package": PACKAGE,
        "schema": 1,
        "symbols": surface,
    }


def write_snapshot(path: Path = SNAPSHOT_PATH) -> Path:
    """Write the current snapshot to ``path`` and return the path."""
    snap = compute_snapshot()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def diff_snapshots(
    committed: dict[str, Any], current: dict[str, Any]
) -> tuple[list[str], list[str], list[tuple[str, str, str]]]:
    """Return ``(added, removed, changed)``.

    ``changed`` is a list of ``(symbol, committed_sig, current_sig)``.
    """
    c = committed.get("symbols", {})
    n = current.get("symbols", {})
    added = sorted(set(n) - set(c))
    removed = sorted(set(c) - set(n))
    changed = sorted((sym, c[sym], n[sym]) for sym in set(c) & set(n) if c[sym] != n[sym])
    return added, removed, changed


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Manage the GenoLeWM public-API snapshot.")
    parser.add_argument(
        "command",
        choices=["write", "show", "check"],
        help="write: regenerate the snapshot file; "
        "show: print the current snapshot to stdout; "
        "check: diff committed vs current and exit non-zero on removal/change.",
    )
    args = parser.parse_args(argv)

    if args.command == "write":
        path = write_snapshot()
        print(f"wrote {path.relative_to(REPO_ROOT)}")
        return 0

    if args.command == "show":
        print(json.dumps(compute_snapshot(), indent=2, sort_keys=True))
        return 0

    # check
    if not SNAPSHOT_PATH.is_file():
        print(f"snapshot missing: {SNAPSHOT_PATH}", flush=True)
        return 2
    committed = load_snapshot()
    current = compute_snapshot()
    added, removed, changed = diff_snapshots(committed, current)
    if added:
        print(f"[snapshot] {len(added)} new symbol(s) — review:")
        for sym in added:
            print(f"  + {sym}")
    if removed:
        print(f"[snapshot] {len(removed)} REMOVED symbol(s) — MAJOR required:")
        for sym in removed:
            print(f"  - {sym}")
    if changed:
        print(f"[snapshot] {len(changed)} CHANGED signature(s) — MAJOR required:")
        for sym, old, new in changed:
            print(f"  ~ {sym}\n      was: {old}\n      now: {new}")
    if removed or changed:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
