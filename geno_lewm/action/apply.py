# SPDX-License-Identifier: Apache-2.0
"""Pure-Python ``apply_edit`` / ``apply_edits`` helpers.

These functions produce the post-edit window string used to encode the
training target ``s_{t+1}`` (RFC-0003 §3.7). They are pure, importable
without torch, and load-bearing for both training and eval.

The multi-edit form applies edits **right-to-left** by descending
``rel_pos`` (INV-ARCH-4) so each edit's relative position stays valid
through the sequence of mutations. Overlapping edits raise
:class:`geno_lewm.errors.OverlappingEditsError`. Reference-bases
mismatch raises :class:`geno_lewm.errors.WindowMismatchError`. Edits
whose locus is outside the window raise
:class:`geno_lewm.errors.OutOfWindowError`.

After applying edits the post-edit window length may change (indels).
:func:`apply_edit` and :func:`apply_edits` accept an optional
``preserve_length=True`` argument that truncates / pads on the side
*opposite* the edit to preserve maximum context around the locus.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from geno_lewm.action.spec import EditType, RelEdit
from geno_lewm.errors import (
    OutOfWindowError,
    OverlappingEditsError,
    WindowMismatchError,
)

__all__ = ["apply_edit", "apply_edits"]


_PAD_BASE = "N"  # used when preserve_length=True and we need to pad


def apply_edit(window: str, edit: RelEdit, *, preserve_length: bool = False) -> str:
    """Return ``window`` with ``edit`` applied.

    ``window`` is the pre-edit base string (uppercase ACGTN). The
    function does not validate window contents beyond what the edit
    locus requires; that is the caller's responsibility.

    The reference bases at the edit locus must match ``edit.ref_bases``
    case-insensitively — otherwise :class:`WindowMismatchError` is
    raised with the locus context attached.

    Pass ``preserve_length=True`` to truncate / pad the result back to
    the original window length on the side opposite the edit. The
    default leaves the indel length change intact (length-preserving
    is the trainer's responsibility for ``s_{t+1}`` encoding).
    """
    original_len = len(window)
    end = edit.rel_pos + len(edit.ref_bases)
    if edit.rel_pos < 0 or end > original_len:
        raise OutOfWindowError(
            "edit locus is outside the window",
            details={
                "rel_pos": edit.rel_pos,
                "ref_len": len(edit.ref_bases),
                "window_len": original_len,
            },
        )

    observed = window[edit.rel_pos : end]
    if observed.upper() != edit.ref_bases.upper():
        raise WindowMismatchError(
            "window bases do not match edit.ref_bases at locus",
            details={
                "rel_pos": edit.rel_pos,
                "expected_ref": edit.ref_bases,
                "observed_ref": observed,
            },
            remediation="re-fetch the window, or correct the EditSpec.ref",
        )

    edited = window[: edit.rel_pos] + edit.alt_bases + window[end:]

    if not preserve_length:
        return edited

    return _truncate_or_pad(edited, original_len, edit_locus=edit.rel_pos)


def apply_edits(
    window: str,
    edits: Sequence[RelEdit],
    *,
    preserve_length: bool = False,
) -> str:
    """Apply a sequence of edits to ``window``.

    The edits are sorted by descending ``rel_pos`` and applied in that
    order (INV-ARCH-4). Edits must not overlap in genomic coordinates;
    overlap raises :class:`OverlappingEditsError`.

    Equivalent inputs (same set of edits in any caller-supplied order)
    produce equivalent outputs — the function is order-invariant after
    the internal sort, which is the property the training pipeline
    relies on.

    The ``preserve_length`` flag truncates / pads back to the input
    window length using the position of the **first** (left-most)
    edit as the reference locus, so the side opposite the edit cluster
    is the one trimmed.
    """
    if not edits:
        return window

    _assert_disjoint(edits)

    # Apply right-to-left. With preserve_length=False on the inner
    # calls so we only truncate once at the end (intermediate lengths
    # change with indels, which is fine).
    ordered = sorted(edits, key=lambda e: e.rel_pos, reverse=True)
    out = window
    for edit in ordered:
        out = apply_edit(out, edit, preserve_length=False)

    if not preserve_length:
        return out

    leftmost = min(e.rel_pos for e in edits)
    return _truncate_or_pad(out, len(window), edit_locus=leftmost)


# ---------------------------------------------------------------------------
# Helpers


def _assert_disjoint(edits: Sequence[RelEdit]) -> None:
    """Raise :class:`OverlappingEditsError` if any two edits overlap."""
    intervals = sorted(
        ((e.rel_pos, e.rel_pos + len(e.ref_bases), idx) for idx, e in enumerate(edits)),
        key=lambda t: t[0],
    )
    for (s1, e1, i1), (s2, e2, i2) in itertools.pairwise(intervals):
        if s2 < e1:
            raise OverlappingEditsError(
                "edits overlap in genomic coordinates",
                details={
                    "first_index": i1,
                    "first_interval": [s1, e1],
                    "second_index": i2,
                    "second_interval": [s2, e2],
                },
                remediation="decompose the haplotype upstream so edits do not share bases",
            )


def _truncate_or_pad(window: str, target_len: int, *, edit_locus: int) -> str:
    """Bring ``window`` back to ``target_len`` opposite the edit locus.

    For insertions / MNVs / SNVs the trimming side is decided by which
    side has more context to spare. Specifically:

    - If the edit is in the left half (``edit_locus < target_len // 2``)
      we trim / pad on the RIGHT (the far side).
    - Otherwise we trim / pad on the LEFT.

    Padding uses ``N`` per the encoder convention.
    """
    current = len(window)
    if current == target_len:
        return window

    trim_right = edit_locus < target_len // 2

    if current > target_len:
        excess = current - target_len
        if trim_right:
            return window[:target_len]
        return window[excess:]

    needed = target_len - current
    if trim_right:
        return window + _PAD_BASE * needed
    return _PAD_BASE * needed + window


# RelEdit kept in the imports for re-export checks; suppress unused import
# warning when ruff scans this module in isolation.
_ = (EditType, RelEdit)
