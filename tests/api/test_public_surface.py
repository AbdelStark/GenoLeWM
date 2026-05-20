"""Public-API snapshot test (RFC-0014 §3.7).

Diffs ``tests/api/public_surface.json`` against a freshly-computed
snapshot. Removals or signature changes fail; new symbols pass with a
warning so MINOR additions don't block PRs.

To regenerate after an intentional API change::

    python -m tools.api.snapshot write
"""

from __future__ import annotations

import warnings

import pytest

from tools.api import snapshot


@pytest.fixture(scope="module")
def committed() -> dict[str, object]:
    if not snapshot.SNAPSHOT_PATH.is_file():
        pytest.fail(
            f"public_surface.json missing at {snapshot.SNAPSHOT_PATH}; "
            f"regenerate with `python -m tools.api.snapshot write`."
        )
    return snapshot.load_snapshot()


@pytest.fixture(scope="module")
def current() -> dict[str, object]:
    return snapshot.compute_snapshot()


def test_no_symbols_removed(committed: dict[str, object], current: dict[str, object]) -> None:
    _added, removed, _changed = snapshot.diff_snapshots(committed, current)
    if removed:
        pytest.fail(
            "public symbol(s) removed — MAJOR bump required:\n  "
            + "\n  ".join(f"- {sym}" for sym in removed)
            + "\nRegenerate snapshot only after agreeing the removal:\n  "
            "  python -m tools.api.snapshot write"
        )


def test_no_signatures_changed(committed: dict[str, object], current: dict[str, object]) -> None:
    _added, _removed, changed = snapshot.diff_snapshots(committed, current)
    if changed:
        lines = []
        for sym, old, new in changed:
            lines.append(f"~ {sym}\n      was: {old}\n      now: {new}")
        pytest.fail(
            "public signature(s) changed — MAJOR bump required:\n  "
            + "\n  ".join(lines)
            + "\nRegenerate snapshot only after agreeing the change:\n  "
            "  python -m tools.api.snapshot write"
        )


def test_new_symbols_are_only_warnings(
    committed: dict[str, object], current: dict[str, object]
) -> None:
    added, _removed, _changed = snapshot.diff_snapshots(committed, current)
    # Adding a symbol is OK — it's a MINOR change at worst. We just
    # surface it as a (test-collected) warning so reviewers see it.
    if added:
        warnings.warn(
            "Public surface gained "
            f"{len(added)} symbol(s); regenerate the snapshot if intentional:\n  "
            + "\n  ".join(f"+ {sym}" for sym in added),
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Unit tests for the snapshot helpers themselves.


def test_snapshot_has_schema_field() -> None:
    snap = snapshot.compute_snapshot()
    assert snap["schema"] == 1
    assert snap["package"] == "geno_lewm"
    assert isinstance(snap["symbols"], dict)


def test_diff_handles_added_removed_changed() -> None:
    committed = {"symbols": {"a": "sig1", "b": "sigB"}}
    current = {"symbols": {"a": "sig1NEW", "c": "sigC"}}
    added, removed, changed = snapshot.diff_snapshots(committed, current)
    assert added == ["c"]
    assert removed == ["b"]
    assert changed == [("a", "sig1", "sig1NEW")]


def test_underscore_modules_skipped() -> None:
    snap = snapshot.compute_snapshot()
    # The _redaction module is intentionally private.
    for sym in snap["symbols"]:
        assert "._redaction" not in sym
        assert not any(part.startswith("_") for part in sym.split("."))


def test_committed_snapshot_lists_known_symbols() -> None:
    """Smoke check: the committed file mentions current-PR symbols.

    Reduces the risk that the snapshot drifts silently on main.
    """
    snap = snapshot.load_snapshot()
    syms = set(snap["symbols"])
    # Pick a handful that should always be there once a PR has landed.
    expected = {
        "geno_lewm.action.EditSpec",
        "geno_lewm.action.apply_edit",
        "geno_lewm.errors.GenoLeWMError",
        "geno_lewm.api.experimental",
        "geno_lewm.api.deprecated",
    }
    missing = expected - syms
    assert not missing, missing
