# SPDX-License-Identifier: Apache-2.0
"""Canonical, content-addressed genomic variant identities.

The v0.3 data contract uses one identity spelling everywhere: GRCh38,
canonical chromosome names without a ``chr`` prefix, 1-based coordinates,
and uppercase explicit A/C/G/T alleles.  Human-friendly input spellings are
normalized at construction time; serialized keys are validated strictly by
``CanonicalVariant.from_key`` so artifacts cannot drift silently.

This module removes redundant context from an allele pair, but it deliberately
does not perform reference-aware left alignment across repeats.  Dataset
preparation must perform and receipt-bind that upstream transformation before
constructing these identities; the reference sequence is intentionally not an
implicit dependency of this pure value object.
"""

from __future__ import annotations

from dataclasses import dataclass

from geno_lewm.errors import InputError
from geno_lewm.provenance import canonical_json_sha256

__all__ = [
    "CANONICAL_CHROMOSOMES",
    "SUPPORTED_ASSEMBLIES",
    "CanonicalVariant",
    "canonicalize_assembly",
    "canonicalize_chromosome",
]


SUPPORTED_ASSEMBLIES: frozenset[str] = frozenset({"GRCh38"})
"""Reference assemblies accepted by the v0.3 identity contract."""

CANONICAL_CHROMOSOMES: tuple[str, ...] = (
    *map(str, range(1, 23)),
    "X",
    "Y",
    "MT",
)
"""Canonical primary chromosome spellings, in genomic sort order."""

_VALID_BASES = frozenset("ACGT")


def canonicalize_assembly(value: str) -> str:
    """Return the canonical assembly spelling accepted by v0.3."""
    if not isinstance(value, str) or not value.strip():
        raise InputError(
            "assembly must be a non-empty string",
            details={"assembly": value, "type": type(value).__name__},
        )
    normalized = value.strip()
    if normalized.casefold() == "grch38":
        return "GRCh38"
    raise InputError(
        "assembly is not supported by the v0.3 identity contract",
        details={"assembly": value, "supported": sorted(SUPPORTED_ASSEMBLIES)},
    )


def canonicalize_chromosome(value: str) -> str:
    """Return a primary chromosome without a ``chr`` prefix."""
    if not isinstance(value, str) or not value.strip():
        raise InputError(
            "chromosome must be a non-empty string",
            details={"chrom": value, "type": type(value).__name__},
        )
    normalized = value.strip()
    if normalized[:3].casefold() == "chr":
        normalized = normalized[3:]
    upper = normalized.upper()
    if upper == "M":
        upper = "MT"
    if upper.isdecimal():
        upper = str(int(upper))
    if upper not in CANONICAL_CHROMOSOMES:
        raise InputError(
            "chromosome must identify a canonical primary chromosome",
            details={"chrom": value, "supported": list(CANONICAL_CHROMOSOMES)},
        )
    return upper


def _canonicalize_allele(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputError(
            f"{name} must be a non-empty allele string",
            details={"field": name, "value": value, "type": type(value).__name__},
        )
    normalized = value.upper()
    invalid = set(normalized) - _VALID_BASES
    if invalid:
        raise InputError(
            f"{name} must contain only explicit A/C/G/T bases",
            details={"field": name, "value": value, "invalid": sorted(invalid)},
        )
    return normalized


def _trim_shared_context(pos: int, ref: str, alt: str) -> tuple[int, str, str]:
    """Remove redundant suffix then prefix bases while preserving VCF anchors."""
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref = ref[:-1]
        alt = alt[:-1]
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref = ref[1:]
        alt = alt[1:]
        pos += 1
    return pos, ref, alt


@dataclass(frozen=True, slots=True)
class CanonicalVariant:
    """One parsimonious assembly-qualified variant with a stable key and digest.

    Shared allele context is trimmed deterministically. Repeat-aware left
    alignment requires a reference sequence and remains an upstream,
    receipt-bound dataset-preparation responsibility.
    """

    assembly: str
    chrom: str
    pos: int
    ref: str
    alt: str

    def __post_init__(self) -> None:
        if not isinstance(self.pos, int) or isinstance(self.pos, bool) or self.pos < 1:
            raise InputError(
                "pos must be a positive 1-based integer",
                details={"pos": self.pos, "type": type(self.pos).__name__},
            )
        assembly = canonicalize_assembly(self.assembly)
        chrom = canonicalize_chromosome(self.chrom)
        ref = _canonicalize_allele("ref", self.ref)
        alt = _canonicalize_allele("alt", self.alt)
        if ref == alt:
            raise InputError(
                "ref and alt must differ",
                details={"ref": ref, "alt": alt},
            )
        pos, ref, alt = _trim_shared_context(self.pos, ref, alt)
        object.__setattr__(self, "assembly", assembly)
        object.__setattr__(self, "chrom", chrom)
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "alt", alt)

    @classmethod
    def from_key(cls, value: str) -> CanonicalVariant:
        """Parse an already-canonical variant key, rejecting spelling drift."""
        if not isinstance(value, str):
            raise InputError(
                "value must be a canonical variant key",
                details={"value": value, "type": type(value).__name__},
            )
        parts = value.split(":")
        if len(parts) != 5:
            raise InputError(
                "value must be a canonical variant key",
                details={"value": value, "expected": "assembly:chrom:pos:ref:alt"},
            )
        assembly, chrom, pos_text, ref, alt = parts
        try:
            pos = int(pos_text)
            variant = cls(assembly=assembly, chrom=chrom, pos=pos, ref=ref, alt=alt)
        except (InputError, ValueError) as exc:
            raise InputError(
                "value must be a canonical variant key",
                details={"value": value, "expected": "assembly:chrom:pos:ref:alt"},
            ) from exc
        if variant.key != value:
            raise InputError(
                "value must be a canonical variant key",
                details={"value": value, "canonical": variant.key},
            )
        return variant

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-native identity payload."""
        return {
            "assembly": self.assembly,
            "chrom": self.chrom,
            "pos": self.pos,
            "ref": self.ref,
            "alt": self.alt,
        }

    @property
    def key(self) -> str:
        """Return the strict ``assembly:chrom:pos:ref:alt`` key."""
        return f"{self.assembly}:{self.chrom}:{self.pos}:{self.ref}:{self.alt}"

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical JSON identity."""
        return canonical_json_sha256(self.to_dict())
