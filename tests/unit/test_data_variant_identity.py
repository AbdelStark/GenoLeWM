"""Behavior tests for canonical variant identities."""

from __future__ import annotations

import pytest

from geno_lewm.data.variant_identity import CanonicalVariant
from geno_lewm.errors import InputError


def test_variant_identity_canonicalizes_equivalent_grch38_snv_inputs() -> None:
    variant = CanonicalVariant(
        assembly="grch38",
        chrom="chr22",
        pos=42,
        ref="a",
        alt="g",
    )

    assert variant.to_dict() == {
        "assembly": "GRCh38",
        "chrom": "22",
        "pos": 42,
        "ref": "A",
        "alt": "G",
    }
    assert variant.key == "GRCh38:22:42:A:G"
    assert (
        variant.digest == "sha256:f5c41a0eea52b2bc6ed20f6a5b8c70aecf3662d7480a7de45c724273cba52c38"
    )


@pytest.mark.parametrize(
    ("padded_args", "parsimonious_args", "expected_key"),
    [
        ((100, "CAT", "CGT"), (101, "A", "G"), "GRCh38:22:101:A:G"),
        ((100, "CAT", "CGAT"), (100, "C", "CG"), "GRCh38:22:100:C:CG"),
        ((100, "CAT", "CT"), (100, "CA", "C"), "GRCh38:22:100:CA:C"),
    ],
)
def test_padded_variant_representations_collapse_to_one_parsimonious_identity(
    padded_args: tuple[int, str, str],
    parsimonious_args: tuple[int, str, str],
    expected_key: str,
) -> None:
    padded = CanonicalVariant("GRCh38", "chr22", *padded_args)
    parsimonious = CanonicalVariant("GRCh38", "22", *parsimonious_args)

    assert padded == parsimonious
    assert padded.key == parsimonious.key == expected_key
    assert padded.digest == parsimonious.digest


@pytest.mark.parametrize(
    "drifted_key",
    [
        "GRCh38:chr22:42:A:G",
        "grch38:22:42:A:G",
        "GRCh38:22:42:a:g",
        "GRCh38|22|42|A|G",
        "GRCh38:22:42:A:G:extra",
        "GRCh38:22:100:CAT:CGT",
    ],
)
def test_variant_key_parser_rejects_noncanonical_or_drifted_keys(drifted_key: str) -> None:
    canonical = CanonicalVariant.from_key("GRCh38:22:42:A:G")

    assert canonical == CanonicalVariant("GRCh38", "22", 42, "A", "G")
    with pytest.raises(InputError, match="canonical variant key"):
        CanonicalVariant.from_key(drifted_key)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"assembly": "hg38"}, "assembly is not supported"),
        ({"chrom": "GL000220.1"}, "canonical primary chromosome"),
        ({"pos": 0}, "positive 1-based integer"),
        ({"pos": True}, "positive 1-based integer"),
        ({"ref": "N"}, "explicit A/C/G/T"),
        ({"alt": "<DEL>"}, "explicit A/C/G/T"),
        ({"alt": "A"}, "ref and alt must differ"),
    ],
)
def test_variant_identity_rejects_ambiguous_or_invalid_coordinates_and_alleles(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "assembly": "GRCh38",
        "chrom": "22",
        "pos": 42,
        "ref": "A",
        "alt": "G",
    }
    values.update(kwargs)

    with pytest.raises(InputError, match=message):
        CanonicalVariant(**values)  # type: ignore[arg-type]


def test_variant_identity_rejects_empty_fields_and_invalid_key_types() -> None:
    with pytest.raises(InputError, match="assembly must be a non-empty string"):
        CanonicalVariant("", "22", 1, "A", "G")
    with pytest.raises(InputError, match="chromosome must be a non-empty string"):
        CanonicalVariant("GRCh38", "", 1, "A", "G")
    with pytest.raises(InputError, match="non-empty allele string"):
        CanonicalVariant("GRCh38", "22", 1, "", "G")
    with pytest.raises(InputError, match="canonical variant key"):
        CanonicalVariant.from_key(42)  # type: ignore[arg-type]
    with pytest.raises(InputError, match="canonical variant key"):
        CanonicalVariant.from_key("GRCh38:22:not-an-int:A:G")

    assert CanonicalVariant("GRCh38", "chrM", 1, "a", "g").chrom == "MT"
