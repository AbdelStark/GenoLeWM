"""Tests for cache-backed rollout-state example generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.encoder import POOL_CENTERED_MEAN, WindowCacheRecord, write_shard
from geno_lewm.errors import InputError
from geno_lewm.provenance import sha256_file
from tools.release import rollout_state_examples, rollout_state_rows


def test_rollout_state_examples_generate_jsonl_and_report(tmp_path: Path) -> None:
    cache_dir, records = _write_cache(tmp_path / "cache")
    spec = tmp_path / "eval" / "rollout_state_example_specs.jsonl"
    output = tmp_path / "eval" / "rollout_state_examples.jsonl"
    report = tmp_path / "eval" / "rollout_state_examples_report.json"
    spec.parent.mkdir(parents=True)
    spec.write_text(json.dumps(_spec_row(records)) + "\n", encoding="utf-8")

    payload = rollout_state_examples.write_rollout_state_example_artifacts(
        spec_jsonl=spec,
        cache_dir=cache_dir,
        artifact_root=tmp_path,
        output_jsonl=output,
        output_report=report,
        command=("python", "-m", "tools.release.rollout_state_examples"),
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["schema_version"] == "1.2.0"
    assert row["cache_schema_version"] == "3.0.0"
    assert row["cached_state_value_contract"] == "raw_pooled_v1"
    assert row["materialized_state_contract"] == "legacy_raw_v1"
    assert row["generated_by"] == "tools.release.rollout_state_examples"
    assert row["id"] == "phased-k2-a"
    assert row["normalize"] is False
    assert row["source_state"] == [0.0, 1.0]
    assert row["target_state"] == [1.0, 0.0]
    assert row["candidates"][0]["state"] == [1.0, 0.0]
    assert payload["generated_by"] == "tools.release.rollout_state_examples"
    assert payload["rows"] == 1
    assert payload["splits"] == ["rollout_phased_haplotypes"]
    assert payload["horizons"] == [2]
    assert payload["normalization_views"] == [False]
    assert payload["unique_cache_state_keys"] == 3
    assert payload["inputs"]["spec_jsonl"]["path"] == "eval/rollout_state_example_specs.jsonl"
    assert payload["inputs"]["cache_dir"]["path"] == "cache"
    assert payload["outputs"]["examples_jsonl"]["sha256"] == sha256_file(output)
    assert len(rollout_state_rows.load_rollout_state_examples(output)) == 1


def test_rollout_state_examples_apply_explicit_normalized_view(tmp_path: Path) -> None:
    cache_dir, records = _write_cache(tmp_path / "cache")
    spec = tmp_path / "rollout_state_example_specs.jsonl"
    spec.write_text(json.dumps(_spec_row(records, normalize=True)) + "\n", encoding="utf-8")

    specs = rollout_state_examples.load_rollout_state_example_specs(spec)
    rows = rollout_state_examples.generate_rollout_state_examples(specs, cache_dir=cache_dir)

    assert specs[0].normalize is True
    assert rows[0]["normalize"] is True
    candidate = rows[0]["candidates"][1]
    assert isinstance(candidate, dict)
    assert candidate["state"] == pytest.approx([0.242535625, 0.9701425])


def test_rollout_state_examples_reject_target_candidate_key_drift(tmp_path: Path) -> None:
    cache_dir, records = _write_cache(tmp_path / "cache")
    spec = tmp_path / "rollout_state_example_specs.jsonl"
    row = _spec_row(records)
    candidates = row["candidates"]
    assert isinstance(candidates, list)
    target = candidates[0]
    assert isinstance(target, dict)
    target["state_key"] = _key(records[2])
    spec.write_text(json.dumps(row) + "\n", encoding="utf-8")
    specs = rollout_state_examples.load_rollout_state_example_specs(spec)

    with pytest.raises(InputError, match="target candidate key must match target_state_key"):
        rollout_state_examples.generate_rollout_state_examples(specs, cache_dir=cache_dir)


def test_rollout_state_examples_reject_missing_cache_embedding(tmp_path: Path) -> None:
    cache_dir, records = _write_cache(tmp_path / "cache")
    spec = tmp_path / "rollout_state_example_specs.jsonl"
    row = _spec_row(records)
    row["source_state_key"] = {
        **_key(records[0]),
        "window_hash": "ff" * 32,
    }
    spec.write_text(json.dumps(row) + "\n", encoding="utf-8")
    specs = rollout_state_examples.load_rollout_state_example_specs(spec)

    with pytest.raises(InputError, match="missing cache embedding"):
        rollout_state_examples.generate_rollout_state_examples(specs, cache_dir=cache_dir)


def test_rollout_state_examples_reject_state_representation_drift(tmp_path: Path) -> None:
    cache_dir, records = _write_cache(tmp_path / "cache")
    spec = tmp_path / "rollout_state_example_specs.jsonl"
    row = _spec_row(records)
    target_key = row["target_state_key"]
    assert isinstance(target_key, dict)
    target_key["state_layer"] = -2
    candidates = row["candidates"]
    assert isinstance(candidates, list)
    target_candidate = candidates[0]
    assert isinstance(target_candidate, dict)
    target_candidate["state_key"] = dict(target_key)
    spec.write_text(json.dumps(row) + "\n", encoding="utf-8")
    specs = rollout_state_examples.load_rollout_state_example_specs(spec)

    with pytest.raises(InputError, match="must share one state representation"):
        rollout_state_examples.generate_rollout_state_examples(specs, cache_dir=cache_dir)


def test_rollout_state_examples_reject_cache_outside_artifact_root(tmp_path: Path) -> None:
    cache_dir, records = _write_cache(tmp_path / "cache")
    artifact_root = tmp_path / "artifact"
    spec = artifact_root / "eval" / "rollout_state_example_specs.jsonl"
    spec.parent.mkdir(parents=True)
    spec.write_text(json.dumps(_spec_row(records)) + "\n", encoding="utf-8")
    output = artifact_root / "eval" / "rollout_state_examples.jsonl"
    report = artifact_root / "eval" / "rollout_state_examples_report.json"

    with pytest.raises(InputError, match="must stay under artifact_root"):
        rollout_state_examples.write_rollout_state_example_artifacts(
            spec_jsonl=spec,
            cache_dir=cache_dir,
            artifact_root=artifact_root,
            output_jsonl=output,
            output_report=report,
        )


def _write_cache(cache_dir: Path) -> tuple[Path, tuple[WindowCacheRecord, ...]]:
    records = (
        _record(1, embedding=(0.0, 1.0)),
        _record(2, embedding=(1.0, 0.0)),
        _record(3, embedding=(0.2, 0.8)),
    )
    write_shard(cache_dir, encoder_id="carbon", contig="1", stride_block=0, records=records)
    return cache_dir, records


def _record(seed: int, *, embedding: tuple[float, ...]) -> WindowCacheRecord:
    return WindowCacheRecord(
        chrom="1",
        start_bp=seed * 10,
        end_bp=(seed * 10) + 12_288,
        window_hash=_hash(seed),
        encoder_hash=_hash(100),
        state_layer=-1,
        pool_type=POOL_CENTERED_MEAN,
        pool_radius=256,
        center_token=0,
        dtype="fp16",
        embedding=embedding,
        untargeted=False,
        created_at=seed,
    )


def _spec_row(
    records: tuple[WindowCacheRecord, ...],
    *,
    normalize: bool | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "1.2.0",
        "generated_by": "tools.release.rollout_state_example_specs",
        "cache_schema_version": "3.0.0",
        "cached_state_value_contract": "raw_pooled_v1",
        "id": "phased-k2-a",
        "split": "rollout_phased_haplotypes",
        "normalize": bool(normalize),
        "source_state_key": _key(records[0]),
        "target_state_key": _key(records[1]),
        "target_candidate_id": "target",
        "edits": [
            {"rel_pos": 3, "edit_type": 0, "ref_bases": "A", "alt_bases": "C"},
            {"rel_pos": 7, "edit_type": 0, "ref_bases": "G", "alt_bases": "T"},
        ],
        "candidates": [
            {"id": "target", "state_key": _key(records[1])},
            {"id": "source-like", "state_key": _key(records[2])},
        ],
    }
    return row


def _key(record: WindowCacheRecord) -> dict[str, object]:
    return {
        "window_hash": record.window_hash.hex(),
        "encoder_hash": record.encoder_hash.hex(),
        "state_layer": record.state_layer,
        "pool_type": record.pool_type,
        "pool_radius": record.pool_radius,
        "center_token": record.center_token,
        "dtype": record.dtype,
    }


def _hash(seed: int) -> bytes:
    return bytes([seed % 256]) * 32
