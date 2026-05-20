"""Canonical edit types — ``EditSpec``, ``EditType``, ``RelEdit``.

Defined by RFC-0003 §3.1–§3.3 and ``docs/spec/03-data-model.md#editspec``.

Every downstream subsystem (encoder, predictor, planner, scorer)
consumes these types. They are intentionally tiny dataclasses with
hard validation — no ``Optional`` fields, no string-typed kinds, no
post-construction mutation.

Validation maps to the typed error hierarchy from RFC-0012:

- malformed input (bad bases, ``ref == alt``, ``pos < 1``, etc.) →
  :class:`geno_lewm.errors.InvalidEditError`.
- ``len(ref) > 16`` or ``len(alt) > 16`` (i.e. structural variant) →
  :class:`geno_lewm.errors.UnsupportedEditError` whose ``details``
  payload includes ``edit_type=EditType.SV``.
- ``rel_pos`` outside the window during :meth:`EditSpec.relative_to` →
  :class:`geno_lewm.errors.OutOfWindowError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from geno_lewm.errors import InvalidEditError, OutOfWindowError, UnsupportedEditError

__all__ = ["EditType", "EditSpec", "RelEdit", "V1_MAX_LEN"]


#: Maximum ``ref`` / ``alt`` length in v1. Edits longer than this are
#: structural variants and are routed through the SV adapter in v2
#: (RFC-0003 §3.5).
V1_MAX_LEN: int = 16


class EditType(IntEnum):
    """The six v1 edit categories (RFC-0003 §3.2).

    Members are deterministic functions of ``(len(ref), len(alt))`` —
    callers do not pass this value; it is computed during construction.
    """

    SNV = 0
    INS = 1
    DEL = 2
    MNV = 3
    INDEL = 4
    SV = 5


_VALID_BASES: frozenset[str] = frozenset("ACGT")


def _derive_edit_type(ref: str, alt: str) -> EditType:
    """Derive :class:`EditType` from ``(ref, alt)`` lengths."""
    lr, la = len(ref), len(alt)
    if lr > V1_MAX_LEN or la > V1_MAX_LEN:
        return EditType.SV
    if lr == 1 and la == 1:
        return EditType.SNV
    if lr == 1 and la > 1:
        return EditType.INS
    if lr > 1 and la == 1:
        return EditType.DEL
    if lr == la:  # both > 1
        return EditType.MNV
    return EditType.INDEL  # both > 1 and unequal


def _validate_bases(name: str, value: str) -> None:
    if not value:
        raise InvalidEditError(
            f"{name} must be non-empty",
            details={"field": name, "value": value},
            remediation="provide at least one base",
        )
    if value != value.upper():
        raise InvalidEditError(
            f"{name} must be uppercase ACGT",
            details={"field": name, "value": value},
            remediation="uppercase the bases (e.g. ref.upper())",
        )
    bad = set(value) - _VALID_BASES
    if bad:
        raise InvalidEditError(
            f"{name} contains non-ACGT character(s)",
            details={"field": name, "value": value, "bad_chars": sorted(bad)},
            remediation="bases must be one of A, C, G, T (no N, no IUPAC, no lowercase)",
        )


@dataclass(frozen=True, slots=True)
class EditSpec:
    """A canonical, frozen genomic edit (RFC-0003 §3.1).

    Construct with absolute VCF-style coordinates; the derived
    :attr:`edit_type` is filled in by ``__post_init__``.

    ``pos`` is 1-based per VCF convention; both ``ref`` and ``alt`` are
    explicit base strings (no ``<DEL>`` / ``<INS>`` symbolic alleles —
    they're deferred to v2).
    """

    chrom: str
    pos: int
    ref: str
    alt: str
    edit_type: EditType = EditType.SNV  # placeholder; overwritten below

    def __post_init__(self) -> None:
        if not isinstance(self.chrom, str) or not self.chrom:
            raise InvalidEditError(
                "chrom must be a non-empty string",
                details={"chrom": self.chrom},
            )
        if not isinstance(self.pos, int) or isinstance(self.pos, bool):
            raise InvalidEditError(
                "pos must be int",
                details={"pos": self.pos, "type": type(self.pos).__name__},
            )
        if self.pos < 1:
            raise InvalidEditError(
                "pos must be >= 1 (VCF 1-based convention)",
                details={"pos": self.pos},
            )

        _validate_bases("ref", self.ref)
        _validate_bases("alt", self.alt)

        if self.ref == self.alt:
            raise InvalidEditError(
                "ref and alt must differ",
                details={"ref": self.ref, "alt": self.alt},
            )

        derived = _derive_edit_type(self.ref, self.alt)
        if derived is EditType.SV:
            raise UnsupportedEditError(
                "edit length exceeds V1_MAX_LEN; structural variants are v2 (RFC-0003 §3.5)",
                details={
                    "ref_len": len(self.ref),
                    "alt_len": len(self.alt),
                    "v1_max_len": V1_MAX_LEN,
                    "edit_type": int(EditType.SV),
                },
                remediation="decompose the SV upstream or wait for the v2 SV adapter",
            )

        # frozen + slots ⇒ object.__setattr__ is the only way to fill
        # the derived field at construction time.
        object.__setattr__(self, "edit_type", derived)

    def relative_to(self, window_start_bp: int, window_end_bp: int) -> RelEdit:
        """Return the window-relative form (RFC-0003 §3.3).

        ``window_start_bp`` and ``window_end_bp`` are 0-based inclusive
        coordinates on the same chromosome as :attr:`chrom`. The
        predictor sees only the relative offset; absolute coordinates
        never enter the model.
        """
        if window_end_bp < window_start_bp:
            raise InvalidEditError(
                "window_end_bp must be >= window_start_bp",
                details={"start": window_start_bp, "end": window_end_bp},
            )
        rel_pos = self.pos - 1 - window_start_bp  # convert 1-based VCF → 0-based offset
        if rel_pos < 0 or rel_pos + len(self.ref) > (window_end_bp - window_start_bp + 1):
            raise OutOfWindowError(
                "edit falls outside the window",
                details={
                    "pos": self.pos,
                    "ref_len": len(self.ref),
                    "window_start_bp": window_start_bp,
                    "window_end_bp": window_end_bp,
                    "rel_pos": rel_pos,
                },
                remediation="re-center the encoder window over the edit, or skip the edit",
            )
        return RelEdit(
            rel_pos=rel_pos,
            edit_type=self.edit_type,
            ref_bases=self.ref,
            alt_bases=self.alt,
        )


@dataclass(frozen=True, slots=True)
class RelEdit:
    """Window-relative form consumed by the action encoder."""

    rel_pos: int
    edit_type: EditType
    ref_bases: str
    alt_bases: str

    def __post_init__(self) -> None:
        if not isinstance(self.rel_pos, int) or isinstance(self.rel_pos, bool):
            raise InvalidEditError(
                "rel_pos must be int",
                details={"rel_pos": self.rel_pos, "type": type(self.rel_pos).__name__},
            )
        if self.rel_pos < 0:
            raise InvalidEditError(
                "rel_pos must be >= 0",
                details={"rel_pos": self.rel_pos},
            )
        # ``edit_type`` arrives as an IntEnum from EditSpec.relative_to;
        # allow plain ints too for downstream serializers.
        if not isinstance(self.edit_type, EditType):
            try:
                object.__setattr__(self, "edit_type", EditType(int(self.edit_type)))
            except (ValueError, TypeError) as exc:
                raise InvalidEditError(
                    "edit_type must be an EditType member",
                    details={"edit_type": self.edit_type},
                ) from exc
        _validate_bases("ref_bases", self.ref_bases)
        _validate_bases("alt_bases", self.alt_bases)
