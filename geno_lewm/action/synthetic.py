# SPDX-License-Identifier: Apache-2.0
"""Synthetic edit samplers (RFC-0003 §3.8).

These samplers produce :class:`RelEdit` objects keyed to an existing
window string. Used by the training data pipeline (RFC-0006 §3.4) to
ensure uniform action-space coverage when natural variants are sparse
in a given region.

All samplers are deterministic with respect to a seeded
:class:`random.Random` instance (passed in as ``rng``), so training
runs are reproducible end-to-end (RFC-0005 §3.6).

A minimum distance from each window edge is enforced (``edge_margin``,
default 64 bp). This guarantees the pooling step has enough context on
both sides of the edit, matching the encoder's pooling assumptions
(RFC-0002 §3.4).
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from geno_lewm.action.spec import V1_MAX_LEN, EditType, RelEdit
from geno_lewm.errors import InputError

__all__ = [
    "DEFAULT_EDGE_MARGIN",
    "indel",
    "mnv",
    "uniform_snv",
]

DEFAULT_EDGE_MARGIN: int = 64

_BASES: tuple[str, ...] = ("A", "C", "G", "T")
_OTHER_BASE: dict[str, tuple[str, str, str]] = {
    "A": ("C", "G", "T"),
    "C": ("A", "G", "T"),
    "G": ("A", "C", "T"),
    "T": ("A", "C", "G"),
}


def _validate_window(window: str, edge_margin: int) -> None:
    if edge_margin < 0:
        raise InputError(
            "edge_margin must be non-negative",
            details={"edge_margin": edge_margin},
        )
    if 2 * edge_margin >= len(window):
        raise InputError(
            "edge_margin leaves no interior region",
            details={"window_len": len(window), "edge_margin": edge_margin},
            remediation="reduce edge_margin or use a longer window",
        )


def _pick_position(rng: random.Random, window_len: int, edge_margin: int) -> int:
    """Return a uniformly random base-pair position inside the
    interior of the window."""
    return rng.randint(edge_margin, window_len - edge_margin - 1)


def _draw_indel_length(
    rng: random.Random,
    length_dist: Mapping[int, float] | Sequence[float] | None,
) -> int:
    """Draw an indel length from the configurable distribution.

    The default distribution is a truncated geometric over [1, 16] with
    ``p = 0.5`` (matches the RFC text).
    """
    if length_dist is None:
        # Truncated geometric on [1, V1_MAX_LEN], normalised.
        weights = [0.5**k for k in range(1, V1_MAX_LEN + 1)]
        total = sum(weights)
        probs = [w / total for w in weights]
        r = rng.random()
        cum = 0.0
        for k, p in enumerate(probs, start=1):
            cum += p
            if r <= cum:
                return k
        return V1_MAX_LEN  # numerical safety

    if isinstance(length_dist, Mapping):
        lengths = sorted(length_dist.keys())
        weights = [length_dist[k] for k in lengths]
    else:
        # Sequence[float] indexed by (length - 1).
        lengths = list(range(1, len(length_dist) + 1))
        weights = list(length_dist)

    if not lengths:
        raise InputError(
            "length_dist must be non-empty",
            details={"length_dist": length_dist},
        )
    if any(w < 0 for w in weights):
        raise InputError(
            "length_dist weights must be non-negative",
            details={"weights": weights},
        )
    if any(k < 1 or k > V1_MAX_LEN for k in lengths):
        raise InputError(
            "length_dist keys must be in [1, V1_MAX_LEN]",
            details={"lengths": lengths, "v1_max_len": V1_MAX_LEN},
        )
    total = sum(weights)
    if total <= 0:
        raise InputError("length_dist weights sum to zero", details={"weights": weights})
    probs = [w / total for w in weights]
    r = rng.random()
    cum = 0.0
    for k, p in zip(lengths, probs, strict=True):
        cum += p
        if r <= cum:
            return k
    return lengths[-1]


def _rand_bases(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(_BASES) for _ in range(n))


def uniform_snv(
    window: str,
    n: int,
    *,
    rng: random.Random,
    edge_margin: int = DEFAULT_EDGE_MARGIN,
) -> list[RelEdit]:
    """Sample ``n`` uniform SNVs anchored inside ``window``.

    Each SNV's ``alt`` is uniformly drawn from the three non-reference
    bases at the chosen position, so the contract "alt is always
    non-reference" is enforced by construction.

    Returns edits in the order they were sampled. The list may contain
    duplicates by position — the caller (data pipeline) is responsible
    for deduplication if it needs disjoint edits.
    """
    _validate_window(window, edge_margin)
    if n < 0:
        raise InputError("n must be non-negative", details={"n": n})

    out: list[RelEdit] = []
    for _ in range(n):
        pos = _pick_position(rng, len(window), edge_margin)
        ref = window[pos]
        if ref not in _OTHER_BASE:
            # Window contains 'N' or other non-ACGT at this position; resample.
            # Simple bounded retry; if window is mostly N's the caller
            # should not be using a synthetic sampler.
            for _retry in range(10):
                pos = _pick_position(rng, len(window), edge_margin)
                ref = window[pos]
                if ref in _OTHER_BASE:
                    break
            else:  # pragma: no cover - defensive
                raise InputError(
                    "could not find an ACGT position in the window's interior",
                    details={"window_len": len(window), "edge_margin": edge_margin},
                )
        alt = rng.choice(_OTHER_BASE[ref])
        out.append(RelEdit(rel_pos=pos, edit_type=EditType.SNV, ref_bases=ref, alt_bases=alt))
    return out


def indel(
    window: str,
    n: int,
    *,
    rng: random.Random,
    length_dist: Mapping[int, float] | Sequence[float] | None = None,
    type_mix: tuple[float, float] = (0.5, 0.5),
    edge_margin: int = DEFAULT_EDGE_MARGIN,
) -> list[RelEdit]:
    """Sample ``n`` indels (INS or DEL).

    ``length_dist`` is the *event* length (number of bases inserted or
    deleted, exclusive of the VCF anchor base). Default is a truncated
    geometric over ``[1, V1_MAX_LEN-1]``.

    ``type_mix`` is ``(p_ins, p_del)``. Default 50/50.
    """
    _validate_window(window, edge_margin)
    if n < 0:
        raise InputError("n must be non-negative", details={"n": n})
    if any(p < 0 for p in type_mix) or sum(type_mix) <= 0:
        raise InputError(
            "type_mix must contain non-negative probs that sum > 0",
            details={"type_mix": list(type_mix)},
        )

    p_ins = type_mix[0] / sum(type_mix)

    out: list[RelEdit] = []
    for _ in range(n):
        pos = _pick_position(rng, len(window), edge_margin)
        ref_anchor = window[pos]
        if ref_anchor not in _OTHER_BASE:
            continue  # skip non-ACGT anchor
        # Event length in [1, V1_MAX_LEN-1] so total ref or alt length ≤ V1_MAX_LEN.
        # We respect the caller's distribution but clip to V1_MAX_LEN-1.
        ev_len = min(_draw_indel_length(rng, length_dist), V1_MAX_LEN - 1)

        if rng.random() < p_ins:
            # Insertion: ref = anchor, alt = anchor + ev_len random bases.
            inserted = _rand_bases(rng, ev_len)
            out.append(
                RelEdit(
                    rel_pos=pos,
                    edit_type=EditType.INS,
                    ref_bases=ref_anchor,
                    alt_bases=ref_anchor + inserted,
                )
            )
        else:
            # Deletion: ref = anchor + ev_len following bases, alt = anchor.
            end = pos + 1 + ev_len
            if end > len(window) - edge_margin:
                # Cannot fit deletion without crossing right margin; retry as INS.
                inserted = _rand_bases(rng, ev_len)
                out.append(
                    RelEdit(
                        rel_pos=pos,
                        edit_type=EditType.INS,
                        ref_bases=ref_anchor,
                        alt_bases=ref_anchor + inserted,
                    )
                )
                continue
            ref_seg = window[pos:end]
            # Skip when ref segment contains N's (cannot construct valid RelEdit).
            if any(c not in _OTHER_BASE for c in ref_seg):
                continue
            out.append(
                RelEdit(
                    rel_pos=pos,
                    edit_type=EditType.DEL,
                    ref_bases=ref_seg,
                    alt_bases=ref_anchor,
                )
            )
    return out


def mnv(
    window: str,
    n: int,
    *,
    rng: random.Random,
    length_dist: Mapping[int, float] | Sequence[float] | None = None,
    edge_margin: int = DEFAULT_EDGE_MARGIN,
) -> list[RelEdit]:
    """Sample ``n`` MNVs (length-preserving multi-base substitutions).

    Length is drawn from ``length_dist`` (default uniform over [2, 8]
    per RFC text). The alt is guaranteed different from ref at every
    base (otherwise constructing a RelEdit with that ref/alt would be
    rejected by EditSpec validation).
    """
    _validate_window(window, edge_margin)
    if n < 0:
        raise InputError("n must be non-negative", details={"n": n})

    if length_dist is None:
        length_dist = dict.fromkeys(range(2, 9), 1.0)  # uniform on [2, 8]

    out: list[RelEdit] = []
    for _ in range(n):
        pos = _pick_position(rng, len(window), edge_margin)
        length = max(2, min(_draw_indel_length(rng, length_dist), V1_MAX_LEN))
        end = pos + length
        if end > len(window) - edge_margin:
            continue
        ref_seg = window[pos:end]
        if any(c not in _OTHER_BASE for c in ref_seg):
            continue
        # Build alt by perturbing every base to a non-self draw.
        alt_chars = [rng.choice(_OTHER_BASE[c]) for c in ref_seg]
        alt_seg = "".join(alt_chars)
        if alt_seg == ref_seg:
            continue  # extremely unlikely; skip
        out.append(
            RelEdit(
                rel_pos=pos,
                edit_type=EditType.MNV,
                ref_bases=ref_seg,
                alt_bases=alt_seg,
            )
        )
    return out
