"""Tests for v0.2 benchmark input generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.release import v02_benchmark_inputs

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def test_v02_benchmark_inputs_write_split_vcfs_labels_and_report(tmp_path: Path) -> None:
    clinvar = tmp_path / "clinvar.parquet"
    traitgym = tmp_path / "traitgym.parquet"
    source_vcf = tmp_path / "clinvar.vcf"
    brca_scores = tmp_path / "brca2_scores.csv"
    brca_mapped = tmp_path / "brca2_mapped.json"
    report = tmp_path / "benchmark_inputs" / "v02_benchmark_inputs_report.json"

    _write_parquet(
        clinvar,
        [
            _clinvar_row("1", 10, "A", "C", "P", "GENE1", "1"),
            _clinvar_row("1", 11, "G", "T", "B", "GENE1", "2"),
            _clinvar_row("1", 20, "A", "G", "LP", "GENE2", "3"),
            _clinvar_row("1", 21, "C", "A", "LB", "GENE2", "4"),
        ],
    )
    source_vcf.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
                "1\t10\t.\tA\tC\t.\tPASS\tMC=SO:0001583|missense_variant",
                "1\t11\t.\tG\tT\t.\tPASS\tMC=SO:0001819|synonymous_variant",
                "1\t20\t.\tA\tG\t.\tPASS\tMC=SO:0001627|intron_variant",
                "1\t21\t.\tC\tA\t.\tPASS\tMC=SO:0001624|3_prime_UTR_variant",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_parquet(
        traitgym,
        [
            {
                "chrom": "2",
                "pos": 30,
                "ref": "A",
                "alt": "G",
                "label": True,
                "consequence": "PLS",
                "OMIM": "100",
            },
            {
                "chrom": "2",
                "pos": 31,
                "ref": "C",
                "alt": "T",
                "label": False,
                "consequence": "5_prime_UTR_variant",
                "OMIM": "101",
            },
        ],
    )
    brca_scores.write_text(
        "accession,hgvs_nt,hgvs_splice,hgvs_pro,score\n"
        "urn:mavedb:test#1,NA,NA,NA,-0.5\n",
        encoding="utf-8",
    )
    brca_mapped.write_text(
        json.dumps(
            [
                {
                    "current": True,
                    "variantUrn": "urn:mavedb:test#1",
                    "postMapped": {
                        "location": {
                            "end": 40,
                            "sequenceReference": {"label": "NC_000013.11"},
                        },
                        "state": {"sequence": "T"},
                        "extensions": [
                            {"name": "vrs_ref_allele_seq", "value": "A"},
                        ],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = v02_benchmark_inputs.write_v02_benchmark_inputs(
        artifact_root=tmp_path,
        clinvar_parquet=clinvar,
        clinvar_source_vcf=source_vcf,
        traitgym_parquet=traitgym,
        brca2_scores_csv=brca_scores,
        brca2_mapped_variants_json=brca_mapped,
        output_report=report,
        max_clinvar_per_split=2,
    )

    assert payload["ok"] is True
    assert payload["generated_by"] == "tools.release.v02_benchmark_inputs"
    assert payload["splits"]["clinvar_coding"]["rows"] == 2
    assert payload["splits"]["clinvar_noncoding"]["rows"] == 2
    assert payload["splits"]["brca2_saturation"]["rows"] == 1
    assert payload["splits"]["traitgym_mendelian"]["rows"] == 2
    assert (tmp_path / "benchmark_inputs" / "clinvar_coding.vcf").read_text(
        encoding="utf-8"
    ).startswith("##fileformat=VCFv4.2")
    labels = [
        json.loads(line)
        for line in (tmp_path / "eval" / "traitgym_mendelian.labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["functional_score"] for row in labels} == {0.0, 1.0}


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _clinvar_row(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    clinical_significance: str,
    gene_symbol: str,
    clinvar_id: str,
) -> dict[str, object]:
    return {
        "chrom": chrom,
        "pos": pos,
        "ref": ref,
        "alt": alt,
        "clinical_significance": clinical_significance,
        "gene_symbol": gene_symbol,
        "clinvar_id": clinvar_id,
    }
