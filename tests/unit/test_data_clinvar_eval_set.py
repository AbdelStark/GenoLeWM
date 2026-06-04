"""Unit tests for the ClinVar eval-set (labels JSONL + VCF) builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from tools.data.clinvar_eval_set import build_clinvar_eval_set

_VCF = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
21\t100\t11\tA\tT\t.\t.\tCLNSIG=Pathogenic
21\t200\t12\tC\tG\t.\t.\tCLNSIG=Benign
21\t300\t13\tG\tA\t.\t.\tCLNSIG=Uncertain_significance
22\t400\t14\tT\tC\t.\t.\tCLNSIG=Likely_pathogenic
21\t100\t15\tA\tT\t.\t.\tCLNSIG=Pathogenic
21\t500\t16\tA\tG\t.\t.\tCLNSIG=Pathogenic
21\t500\t17\tA\tG\t.\t.\tCLNSIG=Benign
"""


def _write_vcf(tmp_path: Path) -> Path:
    path = tmp_path / "clinvar.vcf"
    path.write_text(_VCF, encoding="utf-8")
    return path


def test_builds_deduped_labels_and_matching_vcf(tmp_path: Path) -> None:
    vcf = _write_vcf(tmp_path)
    labels_out = tmp_path / "labels.jsonl"
    vcf_out = tmp_path / "variants.vcf"

    summary = build_clinvar_eval_set(
        input_vcf=vcf, labels_out=labels_out, vcf_out=vcf_out, chrom="21"
    )

    # chr21: 100(P, deduped), 200(B) kept; 300 is VUS; 500 conflicts (P vs B); 22 filtered out.
    assert summary["variants"] == 2
    assert summary["positives"] == 1
    assert summary["negatives"] == 1
    assert summary["dropped_conflicting"] == 1
    assert summary["labelled_rows_seen"] == 5

    label_rows = [json.loads(line) for line in labels_out.read_text().splitlines()]
    assert label_rows == [
        {"chrom": "21", "pos": 100, "ref": "A", "alt": "T", "clinical_significance": "P"},
        {"chrom": "21", "pos": 200, "ref": "C", "alt": "G", "clinical_significance": "B"},
    ]

    # The emitted VCF carries exactly the kept variant keys (scores cover labels).
    data_rows = [
        line.split("\t")[:5]
        for line in vcf_out.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert data_rows == [
        ["21", "100", ".", "A", "T"],
        ["21", "200", ".", "C", "G"],
    ]


def test_no_chrom_filter_keeps_all_contigs(tmp_path: Path) -> None:
    vcf = _write_vcf(tmp_path)
    summary = build_clinvar_eval_set(
        input_vcf=vcf,
        labels_out=tmp_path / "labels.jsonl",
        vcf_out=tmp_path / "variants.vcf",
    )
    # Adds chr22 400 (LP, positive) to the chr21 keepers.
    assert summary["variants"] == 3
    assert summary["positives"] == 2
    assert summary["negatives"] == 1


def test_fasta_filter_keeps_scoreable_variants(tmp_path: Path) -> None:
    vcf = _write_vcf(tmp_path)
    # chr21 contig where pos100 is 'A' (matches 21:100 A>T) and pos200 is 'C'
    # (matches 21:200 C>G): both kept variants stay scoreable.
    seq = list("A" * 600)
    seq[199] = "C"
    fasta = tmp_path / "ref.fa"
    fasta.write_text(">21\n" + "".join(seq) + "\n", encoding="utf-8")

    summary = build_clinvar_eval_set(
        input_vcf=vcf,
        labels_out=tmp_path / "labels.jsonl",
        vcf_out=tmp_path / "variants.vcf",
        chrom="21",
        fasta=fasta,
    )
    assert summary["variants"] == 2
    assert summary["dropped_unscoreable"] == 0


def test_fasta_filter_drops_ref_mismatch(tmp_path: Path) -> None:
    vcf = _write_vcf(tmp_path)
    # All-'A' contig: 21:100 (ref A) matches, 21:200 (ref C) does NOT -> dropped.
    fasta = tmp_path / "ref.fa"
    fasta.write_text(">21\n" + ("A" * 600) + "\n", encoding="utf-8")

    summary = build_clinvar_eval_set(
        input_vcf=vcf,
        labels_out=tmp_path / "labels.jsonl",
        vcf_out=tmp_path / "variants.vcf",
        chrom="21",
        fasta=fasta,
    )
    assert summary["variants"] == 1
    assert summary["dropped_unscoreable"] == 1
    label_rows = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert [r["pos"] for r in label_rows] == [100]


def test_max_variants_caps_size_and_keeps_both_classes(tmp_path: Path) -> None:
    header = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    rows = []
    rid = 0
    for i in range(8):  # 8 pathogenic
        rid += 1
        rows.append(f"21\t{1000 + i}\t{rid}\tA\tT\t.\t.\tCLNSIG=Pathogenic")
    for i in range(8):  # 8 benign
        rid += 1
        rows.append(f"21\t{2000 + i}\t{rid}\tC\tG\t.\t.\tCLNSIG=Benign")
    vcf = tmp_path / "many.vcf"
    vcf.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")

    summary = build_clinvar_eval_set(
        input_vcf=vcf,
        labels_out=tmp_path / "labels.jsonl",
        vcf_out=tmp_path / "variants.vcf",
        chrom="21",
        max_variants=6,
        subset_seed=3,
    )
    assert summary["variants"] == 6
    assert summary["subsampled_from"] == 16
    assert summary["positives"] >= 1
    assert summary["negatives"] >= 1
    assert summary["positives"] + summary["negatives"] == 6
    # Deterministic for a fixed seed.
    again = build_clinvar_eval_set(
        input_vcf=vcf,
        labels_out=tmp_path / "labels2.jsonl",
        vcf_out=tmp_path / "variants2.jsonl",
        chrom="21",
        max_variants=6,
        subset_seed=3,
    )
    assert (tmp_path / "labels.jsonl").read_text() == (tmp_path / "labels2.jsonl").read_text()
    assert again["variants"] == 6


def test_raises_when_no_labelled_variants_match(tmp_path: Path) -> None:
    vcf = _write_vcf(tmp_path)
    with pytest.raises(InputError, match="no labelled ClinVar variants"):
        build_clinvar_eval_set(
            input_vcf=vcf,
            labels_out=tmp_path / "labels.jsonl",
            vcf_out=tmp_path / "variants.vcf",
            chrom="7",
        )
