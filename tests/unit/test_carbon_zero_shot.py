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


def test_log_likelihood_resolves_small_deltas_under_bf16_logits() -> None:
    """A one-base edit's likelihood shift must survive a bf16 forward.

    Regression test for a silent numerical failure. A window log-likelihood sums
    ~10^3 token terms and lands near -5000, while logP(alt) - logP(ref) for a
    single SNV is of order 1. Accumulated in bf16 (8-bit mantissa, spacing 16 at
    that magnitude) the difference quantizes to the grid and almost always
    cancels to exactly zero -- scoring 12,993 real SNVs this way produced 90.7%
    exact zeros over 18 distinct values, all multiples of 16. The scorer must
    therefore accumulate in fp32 even when the model computes in bf16.
    """
    torch = pytest.importorskip("torch")
    from geno_lewm.carbon_zero_shot import _autoregressive_log_likelihood

    generator = torch.Generator().manual_seed(20260715)
    n_tokens, vocab = 1024, 64
    base = torch.randn(1, n_tokens, vocab, generator=generator) * 4.0
    ids = torch.randint(0, vocab, (1, n_tokens), generator=generator)
    mask = torch.ones(1, n_tokens, dtype=torch.long)

    # An edited window differs from the reference at exactly one token.
    edited = base.clone()
    edited[0, n_tokens // 2, :] += torch.randn(vocab, generator=generator) * 2.0

    def score(logits: object) -> float:
        return _autoregressive_log_likelihood(
            torch=torch, logits=logits, input_ids=ids, attention_mask=mask
        )

    reference = score(base.to(torch.float32))
    fp32_delta = score(edited.to(torch.float32)) - reference
    # The planted edit must actually move the likelihood, or the test proves nothing.
    assert abs(fp32_delta) > 1e-3

    bf16_delta = score(edited.to(torch.bfloat16)) - score(base.to(torch.bfloat16))
    # bf16 logits must still yield the fp32 answer: the cast happens inside.
    assert bf16_delta == pytest.approx(fp32_delta, abs=0.5)
    assert bf16_delta != 0.0

    # And the magnitudes must not be quantized onto the coarse bf16 grid.
    assert (
        score(base.to(torch.bfloat16))
        != pytest.approx(round(score(base.to(torch.bfloat16)) / 16.0) * 16.0, abs=1e-9)
        or abs(score(base.to(torch.bfloat16))) < 1e-9
    )


def test_to_float32_passes_through_objects_without_a_float_method() -> None:
    """Stub tensors lacking ``float`` must survive the fp32 cast unchanged.

    This module resolves torch entry points through ``getattr`` so it can run
    against stubs, so the fp32 cast has to tolerate the same. Without the guard
    a stub-backed caller would raise on a missing attribute instead of scoring.
    """
    from geno_lewm.carbon_zero_shot import _to_float32

    class _StubTensor:
        """A tensor-like object that never learned ``.float()``."""

    stub = _StubTensor()
    assert _to_float32(stub) is stub
