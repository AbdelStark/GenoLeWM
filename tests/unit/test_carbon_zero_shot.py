"""Tests for Carbon zero-shot baseline score artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm._artifact_sources import (
    CARBON_ZERO_SHOT_GENERATED_BY,
    CARBON_ZERO_SHOT_SCHEMA_VERSION,
)
from geno_lewm.carbon_zero_shot import (
    CARBON_ZERO_SHOT_SCORE_FIELD,
    write_carbon_zero_shot_scores,
)
from geno_lewm.errors import InputError


def test_write_carbon_zero_shot_scores_records_ref_minus_alt_logp(
    tmp_path: Path,
) -> None:
    vcf, fasta = _write_variant_inputs(tmp_path)
    output = tmp_path / "carbon_zero_shot_scores.jsonl"
    metadata = tmp_path / "carbon_zero_shot_summary.json"
    cache = tmp_path / "carbon_zero_shot_logp_cache.jsonl"

    summary = write_carbon_zero_shot_scores(
        vcf_path=vcf,
        fasta_path=fasta,
        output_scores=output,
        scorer=_count_a_logp,
        carbon_model="HuggingFaceBio/Carbon-500M",
        carbon_revision="main",
        window_bp=4096,
        logp_cache_jsonl=cache,
        metadata_output=metadata,
        generated_at="2026-06-01T00:00:00Z",
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == CARBON_ZERO_SHOT_SCHEMA_VERSION
    assert row["generated_by"] == CARBON_ZERO_SHOT_GENERATED_BY
    assert row["chrom"] == "1"
    assert row["pos"] == 1
    assert row["ref"] == "A"
    assert row["alt"] == "T"
    assert row["carbon_alt_minus_ref_log_likelihood"] == -1.0
    assert row[CARBON_ZERO_SHOT_SCORE_FIELD] == 1.0
    assert len(row["reference_window_sha256"]) == 64
    assert len(row["alternate_window_sha256"]) == 64
    assert summary.records == 1
    assert summary.score_field == CARBON_ZERO_SHOT_SCORE_FIELD
    assert summary.new_logp_evaluations == 2
    assert summary.logp_cache_entries == 2
    assert json.loads(metadata.read_text(encoding="utf-8"))["records"] == 1
    cache_rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    assert len(cache_rows) == 2
    assert {row["generated_by"] for row in cache_rows} == {CARBON_ZERO_SHOT_GENERATED_BY}
    assert {row["schema_version"] for row in cache_rows} == {CARBON_ZERO_SHOT_SCHEMA_VERSION}
    assert {row["carbon_model"] for row in cache_rows} == {"HuggingFaceBio/Carbon-500M"}
    assert {row["carbon_revision"] for row in cache_rows} == {"main"}


def test_write_carbon_zero_shot_scores_reuses_logp_cache(
    tmp_path: Path,
) -> None:
    vcf, fasta = _write_variant_inputs(tmp_path, duplicate=True)
    output = tmp_path / "scores.jsonl"
    cache = tmp_path / "cache.jsonl"
    calls: list[str] = []

    def scorer(sequence: str) -> float:
        calls.append(sequence)
        return _count_a_logp(sequence)

    first = write_carbon_zero_shot_scores(
        vcf_path=vcf,
        fasta_path=fasta,
        output_scores=output,
        scorer=scorer,
        carbon_model="Carbon",
        carbon_revision="main",
        window_bp=4096,
        logp_cache_jsonl=cache,
    )
    second = write_carbon_zero_shot_scores(
        vcf_path=vcf,
        fasta_path=fasta,
        output_scores=output,
        scorer=scorer,
        carbon_model="Carbon",
        carbon_revision="main",
        window_bp=4096,
        logp_cache_jsonl=cache,
    )

    assert first.records == 2
    assert first.new_logp_evaluations == 2
    assert second.new_logp_evaluations == 0
    assert len(calls) == 2


def test_write_carbon_zero_shot_scores_ignores_cache_from_other_carbon_model(
    tmp_path: Path,
) -> None:
    vcf, fasta = _write_variant_inputs(tmp_path)
    output = tmp_path / "scores.jsonl"
    cache = tmp_path / "cache.jsonl"
    calls: list[str] = []

    def stale_scorer(_: str) -> float:
        return 999.0

    first = write_carbon_zero_shot_scores(
        vcf_path=vcf,
        fasta_path=fasta,
        output_scores=output,
        scorer=stale_scorer,
        carbon_model="Carbon-A",
        carbon_revision="main",
        window_bp=4096,
        logp_cache_jsonl=cache,
    )

    def scorer(sequence: str) -> float:
        calls.append(sequence)
        return _count_a_logp(sequence)

    second = write_carbon_zero_shot_scores(
        vcf_path=vcf,
        fasta_path=fasta,
        output_scores=output,
        scorer=scorer,
        carbon_model="Carbon-B",
        carbon_revision="main",
        window_bp=4096,
        logp_cache_jsonl=cache,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    cache_rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    assert first.new_logp_evaluations == 2
    assert second.new_logp_evaluations == 2
    assert len(calls) == 2
    assert rows[0]["carbon_zero_shot_score"] == 1.0
    assert {row["carbon_model"] for row in cache_rows} == {"Carbon-B"}


def test_write_carbon_zero_shot_scores_rejects_invalid_logp_cache_json(
    tmp_path: Path,
) -> None:
    vcf, fasta = _write_variant_inputs(tmp_path)
    output = tmp_path / "scores.jsonl"
    cache = tmp_path / "cache.jsonl"
    cache.write_text("{not-json\n", encoding="utf-8")

    with pytest.raises(InputError, match="logp cache row JSON is invalid"):
        write_carbon_zero_shot_scores(
            vcf_path=vcf,
            fasta_path=fasta,
            output_scores=output,
            scorer=_count_a_logp,
            carbon_model="Carbon",
            carbon_revision="main",
            window_bp=4096,
            logp_cache_jsonl=cache,
        )


def test_write_carbon_zero_shot_scores_rejects_duplicate_compatible_cache_keys(
    tmp_path: Path,
) -> None:
    vcf, fasta = _write_variant_inputs(tmp_path)
    output = tmp_path / "scores.jsonl"
    cache = tmp_path / "cache.jsonl"
    key = "a" * 64
    rows = [
        _cache_row(sequence_sha256=key, log_likelihood=1.0),
        _cache_row(sequence_sha256=key, log_likelihood=2.0),
    ]
    cache.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(InputError, match="sequence_sha256 values must be unique"):
        write_carbon_zero_shot_scores(
            vcf_path=vcf,
            fasta_path=fasta,
            output_scores=output,
            scorer=_count_a_logp,
            carbon_model="Carbon",
            carbon_revision="main",
            window_bp=4096,
            logp_cache_jsonl=cache,
        )


def _write_variant_inputs(tmp_path: Path, *, duplicate: bool = False) -> tuple[Path, Path]:
    fasta = tmp_path / "ref.fa"
    fasta.write_text(">1\nACGTACGT\n", encoding="utf-8")
    vcf = tmp_path / "variants.vcf"
    rows = [
        "##fileformat=VCFv4.3",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
        "1\t1\t.\tA\tT\t.\tPASS\t.",
    ]
    if duplicate:
        rows.append("1\t1\t.\tA\tT\t.\tPASS\t.")
    vcf.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return vcf, fasta


def _count_a_logp(sequence: str) -> float:
    return float(sequence.count("A"))


def _cache_row(*, sequence_sha256: str, log_likelihood: float) -> dict[str, object]:
    return {
        "schema_version": CARBON_ZERO_SHOT_SCHEMA_VERSION,
        "generated_by": CARBON_ZERO_SHOT_GENERATED_BY,
        "carbon_model": "Carbon",
        "carbon_revision": "main",
        "sequence_sha256": sequence_sha256,
        "log_likelihood": log_likelihood,
    }
