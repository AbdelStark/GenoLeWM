"""Contract tests for finite, resumable window-cache builds."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar

import pytest

import geno_lewm.encoder.cache_build as cache_build_module
from geno_lewm.encoder import (
    POOL_CENTERED_MEAN,
    POOL_GLOBAL_MEAN,
    inspect_cache_shard,
    shard_path_for,
)
from geno_lewm.encoder.cache_build import build_window_cache as _build_window_cache
from geno_lewm.errors import CacheCorruptError, CacheKeyAlreadyIndexedError, InputError
from geno_lewm.observability import get_logger, shutdown_run
from geno_lewm.provenance import canonical_json_sha256

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="corrected cache I/O is intentionally fail-closed without POSIX dirfd primitives",
)


class FakeRawEncoder:
    encoder_hash = bytes.fromhex("ab" * 32)
    state_layer = 20
    pool_type = POOL_CENTERED_MEAN
    pool_radius = 8
    dtype = "bf16"
    normalize = False
    device = "cpu"

    def __init__(self) -> None:
        self.encoded_batches: list[tuple[tuple[str, ...], tuple[int | None, ...]]] = []

    def pooling_identity(
        self,
        window: str,
        edit_locus: int | None,
    ) -> tuple[str, int, int | None]:
        if edit_locus is None:
            return POOL_GLOBAL_MEAN, 0, None
        return POOL_CENTERED_MEAN, self.pool_radius, 1 + edit_locus // 6

    def encode_batch(
        self,
        windows: list[str],
        edit_loci: list[int | None],
    ) -> tuple[tuple[float, ...], ...]:
        self.encoded_batches.append((tuple(windows), tuple(edit_loci)))
        return tuple(
            (float(len(window)), float(-1 if locus is None else locus))
            for window, locus in zip(windows, edit_loci, strict=True)
        )


class FailIfEncoded(FakeRawEncoder):
    def encode_batch(
        self,
        windows: list[str],
        edit_loci: list[int | None],
    ) -> tuple[tuple[float, ...], ...]:
        raise AssertionError("resume must verify completed shards without encoding")


class InterruptAfterOneShard(FakeRawEncoder):
    def encode_batch(
        self,
        windows: list[str],
        edit_loci: list[int | None],
    ) -> tuple[tuple[float, ...], ...]:
        if self.encoded_batches:
            raise RuntimeError("simulated interruption")
        return super().encode_batch(windows, edit_loci)


class InterruptBeforeAnyShard(FakeRawEncoder):
    def encode_batch(
        self,
        windows: list[str],
        edit_loci: list[int | None],
    ) -> tuple[tuple[float, ...], ...]:
        raise RuntimeError("simulated interruption before first shard")


def build_window_cache(**kwargs: object):  # type: ignore[no-untyped-def]
    """Supply the explicit fixture runtime identity required by the public API."""
    kwargs.setdefault("hardware", "fixture CPU")
    kwargs.setdefault(
        "resolved_config",
        {"encoder": {"revision": "fixture", "state_contract_version": "l2_normalized_v2"}},
    )
    encoder = kwargs["encoder"]
    digest = "sha256:" + encoder.encoder_hash.hex()  # type: ignore[attr-defined]
    kwargs.setdefault(
        "encoder_runtime_identity",
        {
            "state_contract_version": "l2_normalized_v2",
            "expected": digest,
            "observed": digest,
        },
    )
    return _build_window_cache(**kwargs)  # type: ignore[arg-type]


def _write_requests(path: Path) -> None:
    rows = (
        {
            "request_id": "same-window-center-a",
            "chrom": "22",
            "start_bp": 100,
            "end_bp": 112,
            "window": "ACGTACGTACGT",
            "edit_locus": 0,
        },
        {
            "request_id": "same-window-center-b",
            "chrom": "22",
            "start_bp": 100,
            "end_bp": 112,
            "window": "ACGTACGTACGT",
            "edit_locus": 6,
        },
        {
            "request_id": "duplicate-center-a",
            "chrom": "22",
            "start_bp": 100,
            "end_bp": 112,
            "window": "ACGTACGTACGT",
            "edit_locus": 0,
        },
    )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_request_rows(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_window_cache_keeps_distinct_centers_and_deduplicates_exact_keys(
    tmp_path: Path,
) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_requests(requests)
    encoder = FakeRawEncoder()

    report = build_window_cache(
        requests_jsonl=requests,
        cache_dir=tmp_path / "cache",
        evidence_dir=tmp_path / "evidence",
        encoder=encoder,
        encoder_id="HuggingFaceBio/Carbon-500M",
        batch_size=2,
        rows_per_shard=1,
        created_at_ns=1_750_000_000_000_000_000,
    )

    payload = report.to_dict()
    assert payload["requests"] == {
        "duplicate_rows": 1,
        "input_rows": 3,
        "sha256": payload["requests"]["sha256"],
        "size_bytes": requests.stat().st_size,
        "unique_cache_keys": 2,
    }
    assert payload["build"]["encoded_rows"] == 2
    assert payload["build"]["resumed_rows"] == 0
    assert payload["build"]["completed_shards"] == 2
    assert len(encoder.encoded_batches) == 2
    assert payload["cache_contract"]["normalized_states_persisted"] is False
    assert payload["configuration"]["hardware"] == {
        "description": "fixture CPU",
        "encoder_device": "cpu",
    }
    assert "inside encoder.encode_batch calls" in payload["throughput"]["measurement_scope"]
    assert "sha256" not in payload["cache_artifacts"]["index"]
    assert payload["claim_boundary"]["ten_percent_corpus_completed"] is False
    inspections = [
        inspect_cache_shard(tmp_path / "cache", shard["path"])
        for shard in payload["cache_artifacts"]["shards"]
    ]
    records = [record for inspection in inspections for record in inspection.records]
    assert {record.center_token for record in records} == {1, 2}
    assert {record.created_at for record in records} == {1_750_000_000_000_000_000}
    assert (tmp_path / "evidence" / "cache_build_plan.json").is_file()
    assert (tmp_path / "evidence" / "cache_build_state.json").is_file()
    assert (tmp_path / "evidence" / "cache_build_report.json").is_file()
    assert (tmp_path / "evidence" / "SHA256SUMS").is_file()


def test_completed_build_is_fully_verified_without_reencoding(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_requests(requests)
    kwargs = {
        "requests_jsonl": requests,
        "cache_dir": tmp_path / "cache",
        "evidence_dir": tmp_path / "evidence",
        "encoder_id": "HuggingFaceBio/Carbon-500M",
        "batch_size": 2,
        "rows_per_shard": 1,
        "created_at_ns": 1_750_000_000_000_000_000,
    }
    first = build_window_cache(encoder=FakeRawEncoder(), **kwargs)
    checksums_before = first.checksums_path.read_bytes()

    resumed = build_window_cache(encoder=FailIfEncoded(), **kwargs)

    assert resumed.to_dict() == first.to_dict()
    assert resumed.checksums_path.read_bytes() == checksums_before


def test_resume_verifies_completed_shard_before_encoding_missing_work(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_requests(requests)
    kwargs = {
        "requests_jsonl": requests,
        "cache_dir": tmp_path / "cache",
        "evidence_dir": tmp_path / "evidence",
        "encoder_id": "HuggingFaceBio/Carbon-500M",
        "batch_size": 2,
        "rows_per_shard": 1,
        "created_at_ns": 1_750_000_000_000_000_000,
    }
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build_window_cache(encoder=InterruptAfterOneShard(), **kwargs)
    partial_state = json.loads(
        (tmp_path / "evidence" / "cache_build_state.json").read_text(encoding="utf-8")
    )
    assert len(partial_state["completed_shards"]) == 1

    resumed_encoder = FakeRawEncoder()
    report = build_window_cache(encoder=resumed_encoder, **kwargs).to_dict()

    assert len(resumed_encoder.encoded_batches) == 1
    assert report["build"]["encoded_rows"] == 1
    assert report["build"]["resumed_rows"] == 1
    assert report["throughput"]["measured_encoded_rows"] == 2


def test_resume_rejects_tampered_shard_before_encoder_work(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_requests(requests)
    kwargs = {
        "requests_jsonl": requests,
        "cache_dir": tmp_path / "cache",
        "evidence_dir": tmp_path / "evidence",
        "encoder_id": "HuggingFaceBio/Carbon-500M",
        "batch_size": 2,
        "rows_per_shard": 1,
        "created_at_ns": 1_750_000_000_000_000_000,
    }
    report = build_window_cache(encoder=FakeRawEncoder(), **kwargs).to_dict()
    first_shard = tmp_path / "cache" / report["cache_artifacts"]["shards"][0]["path"]
    first_shard.write_bytes(first_shard.read_bytes()[:32])

    with pytest.raises(CacheCorruptError, match="cache shard"):
        build_window_cache(encoder=FailIfEncoded(), **kwargs)


def test_cache_build_rejects_normalized_encoder(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_requests(requests)
    encoder = FakeRawEncoder()
    encoder.normalize = True

    with pytest.raises(InputError, match="normalize=false"):
        build_window_cache(
            requests_jsonl=requests,
            cache_dir=tmp_path / "cache",
            evidence_dir=tmp_path / "evidence",
            encoder=encoder,
            encoder_id="HuggingFaceBio/Carbon-500M",
            batch_size=2,
            rows_per_shard=1,
            created_at_ns=1_750_000_000_000_000_000,
        )


def test_cache_build_emits_registered_structured_progress(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_requests(requests)
    log_dir = tmp_path / "logs"
    logger = get_logger("cache-build", run_id="cache-proof-test", log_dir=log_dir)

    build_window_cache(
        requests_jsonl=requests,
        cache_dir=tmp_path / "cache",
        evidence_dir=tmp_path / "evidence",
        encoder=FakeRawEncoder(),
        encoder_id="HuggingFaceBio/Carbon-500M",
        batch_size=2,
        rows_per_shard=1,
        created_at_ns=1_750_000_000_000_000_000,
        logger=logger,
    )
    shutdown_run("cache-proof-test", log_dir)

    records = [
        json.loads(line)
        for line in (log_dir / "cache-proof-test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    events = [record["event"] for record in records]
    assert events[0] == "data.cache.build.start"
    assert events[-1] == "data.cache.build.end"
    assert events.count("data.cache.build.progress") == 2
    assert events.count("data.shard.write") == 2
    progress = next(record for record in records if record["event"] == "data.cache.build.progress")
    assert progress["data"]["total_shards"] == 2
    assert progress["data"]["status"] == "encoded"


def test_cache_build_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        '{"request_id":"a","request_id":"b","chrom":"22","start_bp":0,'
        '"end_bp":6,"window":"ACGTAC","edit_locus":0}\n',
        encoding="utf-8",
    )

    with pytest.raises(InputError, match="duplicate JSON key"):
        build_window_cache(
            requests_jsonl=requests,
            cache_dir=tmp_path / "cache",
            evidence_dir=tmp_path / "evidence",
            encoder=FakeRawEncoder(),
            encoder_id="HuggingFaceBio/Carbon-500M",
            batch_size=1,
            rows_per_shard=1,
            created_at_ns=1_750_000_000_000_000_000,
        )


def test_resume_rejects_state_digest_drift_before_encoder_work(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_requests(requests)
    kwargs = {
        "requests_jsonl": requests,
        "cache_dir": tmp_path / "cache",
        "evidence_dir": tmp_path / "evidence",
        "encoder_id": "HuggingFaceBio/Carbon-500M",
        "batch_size": 2,
        "rows_per_shard": 1,
        "created_at_ns": 1_750_000_000_000_000_000,
    }
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build_window_cache(encoder=InterruptAfterOneShard(), **kwargs)
    state_path = tmp_path / "evidence" / "cache_build_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["completed_shards"][0]["sha256"] = "sha256:" + "0" * 64
    state_path.chmod(0o644)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(CacheCorruptError, match="durable build state"):
        build_window_cache(encoder=FailIfEncoded(), **kwargs)


def test_resume_rejects_impossible_state_origin_before_encoder_work(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_requests(requests)
    kwargs = {
        "requests_jsonl": requests,
        "cache_dir": tmp_path / "cache",
        "evidence_dir": tmp_path / "evidence",
        "encoder_id": "HuggingFaceBio/Carbon-500M",
        "batch_size": 2,
        "rows_per_shard": 1,
        "created_at_ns": 1_750_000_000_000_000_000,
    }
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build_window_cache(encoder=InterruptAfterOneShard(), **kwargs)
    state_path = tmp_path / "evidence" / "cache_build_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["completed_shards"][0]["origin"] = "adopted_after_interruption"
    state_path.chmod(0o644)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(CacheCorruptError, match="origin and measurements"):
        build_window_cache(encoder=FailIfEncoded(), **kwargs)


def test_two_fresh_builds_produce_identical_cache_and_plan_bytes(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_requests(requests)
    reports = [
        build_window_cache(
            requests_jsonl=requests,
            cache_dir=tmp_path / run_name / "cache",
            evidence_dir=tmp_path / run_name / "evidence",
            encoder=FakeRawEncoder(),
            encoder_id="HuggingFaceBio/Carbon-500M",
            batch_size=2,
            rows_per_shard=1,
            created_at_ns=1_750_000_000_000_000_000,
        ).to_dict()
        for run_name in ("a", "b")
    ]

    assert (tmp_path / "a/evidence/cache_build_plan.json").read_bytes() == (
        tmp_path / "b/evidence/cache_build_plan.json"
    ).read_bytes()
    assert reports[0]["cache_artifacts"] == reports[1]["cache_artifacts"]
    for first, second in zip(
        reports[0]["cache_artifacts"]["shards"],
        reports[1]["cache_artifacts"]["shards"],
        strict=True,
    ):
        assert (tmp_path / "a/cache" / first["path"]).read_bytes() == (
            tmp_path / "b/cache" / second["path"]
        ).read_bytes()
    assert (tmp_path / "a/cache/embeddings/index.sqlite").read_bytes() == (
        tmp_path / "b/cache/embeddings/index.sqlite"
    ).read_bytes()


def test_resume_adopts_crash_gap_shard_without_reencoding(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_requests(requests)
    evidence = tmp_path / "evidence"
    kwargs = {
        "requests_jsonl": requests,
        "cache_dir": tmp_path / "cache",
        "evidence_dir": evidence,
        "encoder_id": "HuggingFaceBio/Carbon-500M",
        "batch_size": 2,
        "rows_per_shard": 1,
        "created_at_ns": 1_750_000_000_000_000_000,
    }
    build_window_cache(encoder=FakeRawEncoder(), **kwargs)
    (evidence / "SHA256SUMS").unlink()
    (evidence / "cache_build_report.json").unlink()
    state_path = evidence / "cache_build_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    removed = state["completed_shards"].pop()
    state_path.chmod(0o644)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_window_cache(encoder=FailIfEncoded(), **kwargs).to_dict()
    repaired_state = json.loads(state_path.read_text(encoding="utf-8"))

    assert report["build"]["encoded_rows"] == 0
    assert report["build"]["resumed_rows"] == 2
    repaired = {entry["plan_shard_id"]: entry for entry in repaired_state["completed_shards"]}
    assert repaired[removed["plan_shard_id"]]["origin"] == "adopted_after_interruption"
    assert repaired[removed["plan_shard_id"]]["sha256"] == removed["sha256"]


def test_resume_rejects_missing_shard_named_complete_before_encoder_work(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_requests(requests)
    kwargs = {
        "requests_jsonl": requests,
        "cache_dir": tmp_path / "cache",
        "evidence_dir": tmp_path / "evidence",
        "encoder_id": "HuggingFaceBio/Carbon-500M",
        "batch_size": 2,
        "rows_per_shard": 1,
        "created_at_ns": 1_750_000_000_000_000_000,
    }
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build_window_cache(encoder=InterruptAfterOneShard(), **kwargs)
    state = json.loads((tmp_path / "evidence/cache_build_state.json").read_text(encoding="utf-8"))
    completed_path = tmp_path / "cache" / state["completed_shards"][0]["path"]
    completed_path.unlink()

    with pytest.raises(CacheCorruptError, match="completed shard that is missing"):
        build_window_cache(encoder=FailIfEncoded(), **kwargs)


def test_existing_evidence_input_is_never_replaced(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_requests(requests)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    collision = evidence / "cache_build_requests.jsonl"
    collision.write_bytes(b"do-not-replace\n")

    with pytest.raises(InputError, match="does not match this build"):
        build_window_cache(
            requests_jsonl=requests,
            cache_dir=tmp_path / "cache",
            evidence_dir=evidence,
            encoder=FailIfEncoded(),
            encoder_id="HuggingFaceBio/Carbon-500M",
            batch_size=1,
            rows_per_shard=1,
            created_at_ns=1_750_000_000_000_000_000,
        )

    assert collision.read_bytes() == b"do-not-replace\n"
    # The exact immutable plan is installed before staging is attempted. The
    # conflicting request copy remains untouched and no state/report is born.
    assert (evidence / "cache_build_plan.json").is_file()
    assert not (evidence / "cache_build_state.json").exists()


def test_plan_only_resume_rederives_pooling_identity_before_encoder_work(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_requests(requests)
    evidence = tmp_path / "evidence"
    kwargs = {
        "requests_jsonl": requests,
        "cache_dir": tmp_path / "cache",
        "evidence_dir": evidence,
        "encoder_id": "HuggingFaceBio/Carbon-500M",
        "batch_size": 2,
        "rows_per_shard": 1,
        "created_at_ns": 1_750_000_000_000_000_000,
    }
    with pytest.raises(RuntimeError, match="before first shard"):
        build_window_cache(encoder=InterruptBeforeAnyShard(), **kwargs)

    # Simulate a crash after the immutable plan was published but before the
    # first state file. Rewrite one center and its dependent shard ID so the
    # plan remains internally self-consistent but disagrees with the encoder.
    (evidence / "cache_build_state.json").unlink()
    plan_path = evidence / "cache_build_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    first_shard = plan["shards"][0]
    first_shard["rows"][0]["key"]["center_token"] = 99
    first_shard["shard_id"] = canonical_json_sha256(
        {
            "path": first_shard["path"],
            "keys": [row["key"] for row in first_shard["rows"]],
        }
    )
    plan_path.chmod(0o644)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="exact plan rederived from immutable inputs"):
        build_window_cache(encoder=FailIfEncoded(), **kwargs)


def test_plan_only_resume_rejects_self_consistent_repartition_from_immutable_inputs(
    tmp_path: Path,
) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_requests(requests)
    evidence = tmp_path / "evidence"
    kwargs = {
        "requests_jsonl": requests,
        "cache_dir": tmp_path / "cache",
        "evidence_dir": evidence,
        "encoder_id": "HuggingFaceBio/Carbon-500M",
        "batch_size": 2,
        "rows_per_shard": 1,
        "created_at_ns": 1_750_000_000_000_000_000,
    }
    with pytest.raises(RuntimeError, match="before first shard"):
        build_window_cache(encoder=InterruptBeforeAnyShard(), **kwargs)
    (evidence / "cache_build_state.json").unlink()
    plan_path = evidence / "cache_build_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    first, second = plan["shards"]
    first["rows"], second["rows"] = second["rows"], first["rows"]
    for shard in (first, second):
        shard["shard_id"] = canonical_json_sha256(
            {
                "path": shard["path"],
                "keys": [row["key"] for row in shard["rows"]],
            }
        )
    plan_path.chmod(0o644)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="exact plan rederived from immutable inputs"):
        build_window_cache(encoder=FailIfEncoded(), **kwargs)


def test_overlapping_request_namespace_reuses_unique_key_and_survives_index_growth(
    tmp_path: Path,
) -> None:
    first_requests = tmp_path / "first.jsonl"
    second_requests = tmp_path / "second.jsonl"
    growth_requests = tmp_path / "growth.jsonl"
    common = {
        "chrom": "22",
        "start_bp": 100,
        "end_bp": 112,
        "window": "ACGTACGTACGT",
        "edit_locus": 0,
    }
    _write_request_rows(first_requests, ({"request_id": "first", **common},))
    _write_request_rows(second_requests, ({"request_id": "overlap", **common},))
    _write_request_rows(
        growth_requests,
        (
            {
                "request_id": "growth",
                "chrom": "22",
                "start_bp": 200,
                "end_bp": 212,
                "window": "TTTTTTTTTTTT",
                "edit_locus": 0,
            },
        ),
    )
    cache = tmp_path / "cache"
    first = build_window_cache(
        requests_jsonl=first_requests,
        cache_dir=cache,
        evidence_dir=tmp_path / "first-evidence",
        encoder=FakeRawEncoder(),
        encoder_id="HuggingFaceBio/Carbon-500M",
        batch_size=1,
        rows_per_shard=4,
        created_at_ns=1_750_000_000_000_000_000,
    ).to_dict()
    second_report = build_window_cache(
        requests_jsonl=second_requests,
        cache_dir=cache,
        evidence_dir=tmp_path / "second-evidence",
        encoder=FailIfEncoded(),
        encoder_id="HuggingFaceBio/Carbon-500M",
        batch_size=1,
        rows_per_shard=4,
        created_at_ns=1_750_000_000_000_000_001,
    )
    second = second_report.to_dict()
    second_checksums = second_report.checksums_path.read_bytes()

    assert first["configuration"]["cache_namespace"] != second["configuration"]["cache_namespace"]
    assert second["build"]["encoded_rows"] == 0
    assert second["build"]["resumed_rows"] == 0
    assert second["build"]["reused_rows"] == 1
    assert (
        second["cache_artifacts"]["shards"][0]["path"]
        == first["cache_artifacts"]["shards"][0]["path"]
    )

    build_window_cache(
        requests_jsonl=growth_requests,
        cache_dir=cache,
        evidence_dir=tmp_path / "growth-evidence",
        encoder=FakeRawEncoder(),
        encoder_id="HuggingFaceBio/Carbon-500M",
        batch_size=1,
        rows_per_shard=4,
        created_at_ns=1_750_000_000_000_000_002,
    )
    verified = build_window_cache(
        requests_jsonl=second_requests,
        cache_dir=cache,
        evidence_dir=tmp_path / "second-evidence",
        encoder=FailIfEncoded(),
        encoder_id="HuggingFaceBio/Carbon-500M",
        batch_size=1,
        rows_per_shard=4,
        created_at_ns=1_750_000_000_000_000_001,
    )

    assert verified.to_dict() == second
    assert verified.checksums_path.read_bytes() == second_checksums


def test_partial_overlap_encodes_only_misses_in_the_new_namespace(tmp_path: Path) -> None:
    first_requests = tmp_path / "first.jsonl"
    mixed_requests = tmp_path / "mixed.jsonl"
    shared = {
        "request_id": "shared-first",
        "chrom": "22",
        "start_bp": 100,
        "end_bp": 112,
        "window": "ACGTACGTACGT",
        "edit_locus": 0,
    }
    novel = {
        "request_id": "novel",
        "chrom": "22",
        "start_bp": 200,
        "end_bp": 212,
        "window": "TTTTTTTTTTTT",
        "edit_locus": 0,
    }
    _write_request_rows(first_requests, (shared,))
    _write_request_rows(
        mixed_requests,
        ({**shared, "request_id": "shared-second"}, novel),
    )
    cache = tmp_path / "cache"
    build_window_cache(
        requests_jsonl=first_requests,
        cache_dir=cache,
        evidence_dir=tmp_path / "first-evidence",
        encoder=FakeRawEncoder(),
        encoder_id="HuggingFaceBio/Carbon-500M",
        batch_size=2,
        rows_per_shard=4,
        created_at_ns=1_750_000_000_000_000_000,
    )
    mixed_encoder = FakeRawEncoder()
    mixed = build_window_cache(
        requests_jsonl=mixed_requests,
        cache_dir=cache,
        evidence_dir=tmp_path / "mixed-evidence",
        encoder=mixed_encoder,
        encoder_id="HuggingFaceBio/Carbon-500M",
        batch_size=2,
        rows_per_shard=4,
        created_at_ns=1_750_000_000_000_000_001,
    ).to_dict()

    assert mixed_encoder.encoded_batches == [(("TTTTTTTTTTTT",), (0,))]
    assert mixed["build"]["encoded_rows"] == 1
    assert mixed["build"]["resumed_rows"] == 0
    assert mixed["build"]["reused_rows"] == 1
    assert mixed["build"]["resolved_unique_rows"] == 2
    assert len(mixed["cache_artifacts"]["shards"]) == 2
    state = json.loads(
        (tmp_path / "mixed-evidence/cache_build_state.json").read_text(encoding="utf-8")
    )
    assert state["completed_shards"][0]["encoded_rows"] == 1
    assert len(state["completed_shards"][0]["row_keys"]) == 1

    resumed = build_window_cache(
        requests_jsonl=mixed_requests,
        cache_dir=cache,
        evidence_dir=tmp_path / "mixed-evidence",
        encoder=FailIfEncoded(),
        encoder_id="HuggingFaceBio/Carbon-500M",
        batch_size=2,
        rows_per_shard=4,
        created_at_ns=1_750_000_000_000_000_001,
    )
    assert resumed.to_dict() == mixed


@pytest.mark.parametrize("drift", ["batch_size", "hardware", "resolved_config", "device"])
def test_resume_rejects_execution_identity_drift_before_encoder_work(
    tmp_path: Path,
    drift: str,
) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_requests(requests)
    kwargs: dict[str, object] = {
        "requests_jsonl": requests,
        "cache_dir": tmp_path / "cache",
        "evidence_dir": tmp_path / "evidence",
        "encoder_id": "HuggingFaceBio/Carbon-500M",
        "batch_size": 2,
        "rows_per_shard": 1,
        "created_at_ns": 1_750_000_000_000_000_000,
        "hardware": "fixture CPU",
        "resolved_config": {"seed": 17},
    }
    build_window_cache(encoder=FakeRawEncoder(), **kwargs)
    plan_before = (tmp_path / "evidence/cache_build_plan.json").read_bytes()
    encoder = FailIfEncoded()
    if drift == "batch_size":
        kwargs["batch_size"] = 1
    elif drift == "hardware":
        kwargs["hardware"] = "different CPU"
    elif drift == "resolved_config":
        kwargs["resolved_config"] = {"seed": 18}
    else:
        encoder.device = "cuda"

    with pytest.raises(InputError, match="exact plan rederived from immutable inputs"):
        build_window_cache(encoder=encoder, **kwargs)

    assert (tmp_path / "evidence/cache_build_plan.json").read_bytes() == plan_before


def test_plan_validation_precedes_input_artifact_staging(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    config = tmp_path / "source-config.yaml"
    _write_requests(requests)
    config.write_text("seed: 17\n", encoding="utf-8")
    kwargs = {
        "requests_jsonl": requests,
        "cache_dir": tmp_path / "cache",
        "evidence_dir": tmp_path / "evidence",
        "encoder_id": "HuggingFaceBio/Carbon-500M",
        "batch_size": 2,
        "rows_per_shard": 1,
        "created_at_ns": 1_750_000_000_000_000_000,
        "input_artifacts": {"source-config.yaml": config},
    }
    build_window_cache(encoder=FakeRawEncoder(), **kwargs)
    staged = tmp_path / "evidence/inputs/source-config.yaml"
    staged_before = staged.read_bytes()
    config.write_text("seed: 18\n", encoding="utf-8")

    with pytest.raises(InputError, match="exact plan rederived from immutable inputs"):
        build_window_cache(encoder=FailIfEncoded(), **kwargs)

    assert staged.read_bytes() == staged_before


def test_unexpected_evidence_file_is_rejected_instead_of_dynamically_checksummed(
    tmp_path: Path,
) -> None:
    requests = tmp_path / "requests.jsonl"
    evidence = tmp_path / "evidence"
    _write_requests(requests)
    evidence.mkdir()
    (evidence / "mutable.log").write_text("not evidence\n", encoding="utf-8")

    with pytest.raises(CacheCorruptError, match="unexpected artifact"):
        build_window_cache(
            requests_jsonl=requests,
            cache_dir=tmp_path / "cache",
            evidence_dir=evidence,
            encoder=FailIfEncoded(),
            encoder_id="HuggingFaceBio/Carbon-500M",
            batch_size=1,
            rows_per_shard=1,
            created_at_ns=1_750_000_000_000_000_000,
        )

    assert not (evidence / "SHA256SUMS").exists()
    assert not (evidence / "cache_build_plan.json").exists()


def test_completed_bundle_rejects_posthoc_unclosed_artifact(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    evidence = tmp_path / "evidence"
    _write_requests(requests)
    kwargs = {
        "requests_jsonl": requests,
        "cache_dir": tmp_path / "cache",
        "evidence_dir": evidence,
        "encoder_id": "HuggingFaceBio/Carbon-500M",
        "batch_size": 2,
        "rows_per_shard": 1,
        "created_at_ns": 1_750_000_000_000_000_000,
    }
    build_window_cache(encoder=FakeRawEncoder(), **kwargs)
    (evidence / "late-report.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(CacheCorruptError, match="unexpected artifact"):
        build_window_cache(encoder=FailIfEncoded(), **kwargs)


def test_builder_does_not_retain_decoded_embeddings_for_all_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = tmp_path / "requests.jsonl"
    rows = tuple(
        {
            "request_id": f"row-{index}",
            "chrom": "22",
            "start_bp": index * 12,
            "end_bp": index * 12 + 12,
            "window": format(index, "012b").translate(str.maketrans("01", "AC")),
            "edit_locus": 0,
        }
        for index in range(12)
    )
    _write_request_rows(requests, rows)
    real_inspect = cache_build_module.inspect_cache_shard

    class TrackedInspection:
        live: ClassVar[int] = 0
        peak: ClassVar[int] = 0

        def __init__(self, base: object) -> None:
            type(self).live += 1
            type(self).peak = max(type(self).peak, type(self).live)
            self.path = base.path  # type: ignore[attr-defined]
            self.records = base.records  # type: ignore[attr-defined]
            self.sha256 = base.sha256  # type: ignore[attr-defined]
            self.size_bytes = base.size_bytes  # type: ignore[attr-defined]

        def __del__(self) -> None:
            type(self).live -= 1

    def tracked_inspect(cache_dir: Path, shard_path: str) -> TrackedInspection:
        return TrackedInspection(real_inspect(cache_dir, shard_path))

    monkeypatch.setattr(cache_build_module, "inspect_cache_shard", tracked_inspect)
    build_window_cache(
        requests_jsonl=requests,
        cache_dir=tmp_path / "cache",
        evidence_dir=tmp_path / "evidence",
        encoder=FakeRawEncoder(),
        encoder_id="HuggingFaceBio/Carbon-500M",
        batch_size=1,
        rows_per_shard=1,
        created_at_ns=1_750_000_000_000_000_000,
    )

    assert TrackedInspection.peak <= 2


def test_unrelated_write_shard_corruption_is_never_reclassified_as_a_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_request_rows(
        requests,
        (
            {
                "request_id": "row",
                "chrom": "22",
                "start_bp": 0,
                "end_bp": 12,
                "window": "ACGTACGTACGT",
                "edit_locus": 0,
            },
        ),
    )

    def corrupt_planned_path(
        cache_dir: Path,
        *,
        encoder_id: str,
        contig: str,
        stride_block: int,
        records: object,
    ) -> Path:
        first = records[0]  # type: ignore[index]
        path = shard_path_for(
            cache_dir,
            encoder_id=encoder_id,
            state_layer=first.state_layer,
            pool_type=first.pool_type,
            pool_radius=first.pool_radius,
            contig=contig,
            stride_block=stride_block,
            encoder_hash=first.encoder_hash,
            dtype=first.dtype,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"corrupt planned path")
        raise CacheCorruptError("synthetic planned-path corruption")

    monkeypatch.setattr(cache_build_module, "write_shard", corrupt_planned_path)
    evidence = tmp_path / "evidence"
    with pytest.raises(CacheCorruptError, match="planned-path corruption"):
        build_window_cache(
            requests_jsonl=requests,
            cache_dir=tmp_path / "cache",
            evidence_dir=evidence,
            encoder=FakeRawEncoder(),
            encoder_id="HuggingFaceBio/Carbon-500M",
            batch_size=1,
            rows_per_shard=1,
            created_at_ns=1_750_000_000_000_000_000,
        )

    assert not (evidence / "cache_build_report.json").exists()
    assert not (evidence / "SHA256SUMS").exists()


def test_precise_equivalent_key_publication_race_is_verified_and_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_request_rows(
        requests,
        (
            {
                "request_id": "row",
                "chrom": "22",
                "start_bp": 0,
                "end_bp": 12,
                "window": "ACGTACGTACGT",
                "edit_locus": 0,
            },
        ),
    )
    real_write = cache_build_module.write_shard

    def publish_concurrent_winner(
        cache_dir: Path,
        *,
        encoder_id: str,
        contig: str,
        stride_block: int,
        records: object,
    ) -> Path:
        real_write(
            cache_dir,
            encoder_id="independent-equivalent-winner",
            contig=contig,
            stride_block=stride_block,
            records=records,  # type: ignore[arg-type]
        )
        return real_write(
            cache_dir,
            encoder_id=encoder_id,
            contig=contig,
            stride_block=stride_block,
            records=records,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(cache_build_module, "write_shard", publish_concurrent_winner)
    kwargs = {
        "requests_jsonl": requests,
        "cache_dir": tmp_path / "cache",
        "evidence_dir": tmp_path / "evidence",
        "encoder_id": "HuggingFaceBio/Carbon-500M",
        "batch_size": 1,
        "rows_per_shard": 1,
        "created_at_ns": 1_750_000_000_000_000_000,
    }

    first = build_window_cache(encoder=FakeRawEncoder(), **kwargs)
    replay = build_window_cache(encoder=FailIfEncoded(), **kwargs)
    state = json.loads((tmp_path / "evidence/cache_build_state.json").read_text(encoding="utf-8"))

    assert first.to_dict()["build"] == {
        "planned_shards": 1,
        "completed_shards": 1,
        "encoded_shards": 0,
        "encoded_rows": 0,
        "resumed_rows": 0,
        "reused_rows": 1,
        "resolved_unique_rows": 1,
    }
    assert state["completed_shards"] == []
    assert replay.to_dict() == first.to_dict()
    assert replay.checksums_path.read_bytes() == first.checksums_path.read_bytes()


def test_key_race_with_unowned_planned_path_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_request_rows(
        requests,
        (
            {
                "request_id": "row",
                "chrom": "22",
                "start_bp": 0,
                "end_bp": 12,
                "window": "ACGTACGTACGT",
                "edit_locus": 0,
            },
        ),
    )

    def leave_unowned_path(
        cache_dir: Path,
        *,
        encoder_id: str,
        contig: str,
        stride_block: int,
        records: object,
    ) -> Path:
        first = records[0]  # type: ignore[index]
        path = shard_path_for(
            cache_dir,
            encoder_id=encoder_id,
            state_layer=first.state_layer,
            pool_type=first.pool_type,
            pool_radius=first.pool_radius,
            contig=contig,
            stride_block=stride_block,
            encoder_hash=first.encoder_hash,
            dtype=first.dtype,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unowned")
        raise CacheKeyAlreadyIndexedError("synthetic key race")

    monkeypatch.setattr(cache_build_module, "write_shard", leave_unowned_path)
    with pytest.raises(CacheCorruptError, match="unowned path"):
        build_window_cache(
            requests_jsonl=requests,
            cache_dir=tmp_path / "cache",
            evidence_dir=tmp_path / "evidence",
            encoder=FakeRawEncoder(),
            encoder_id="HuggingFaceBio/Carbon-500M",
            batch_size=1,
            rows_per_shard=1,
            created_at_ns=1_750_000_000_000_000_000,
        )


@pytest.mark.parametrize(
    "drift",
    ["created_at_ns", "rows_per_shard", "batch_size", "hardware", "resolved_config", "runtime"],
)
def test_same_requests_with_distinct_immutable_plans_get_distinct_namespaces(
    tmp_path: Path,
    drift: str,
) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_request_rows(
        requests,
        (
            {
                "request_id": "row",
                "chrom": "22",
                "start_bp": 0,
                "end_bp": 12,
                "window": "ACGTACGTACGT",
                "edit_locus": 0,
            },
        ),
    )
    common: dict[str, object] = {
        "requests_jsonl": requests,
        "cache_dir": tmp_path / "cache",
        "encoder_id": "HuggingFaceBio/Carbon-500M",
        "batch_size": 2,
        "rows_per_shard": 2,
        "created_at_ns": 1_750_000_000_000_000_000,
        "hardware": "fixture CPU",
        "resolved_config": {"seed": 17},
    }
    first = build_window_cache(
        evidence_dir=tmp_path / "first-evidence",
        encoder=FakeRawEncoder(),
        **common,
    ).to_dict()
    second_args = dict(common)
    if drift == "created_at_ns":
        second_args["created_at_ns"] = 1_750_000_000_000_000_001
    elif drift == "rows_per_shard":
        second_args["rows_per_shard"] = 1
    elif drift == "batch_size":
        second_args["batch_size"] = 1
    elif drift == "hardware":
        second_args["hardware"] = "different fixture CPU"
    elif drift == "resolved_config":
        second_args["resolved_config"] = {"seed": 18}
    else:
        digest = "sha256:" + FakeRawEncoder.encoder_hash.hex()
        second_args["encoder_runtime_identity"] = {
            "state_contract_version": "different-contract-version",
            "expected": digest,
            "observed": digest,
        }

    second = build_window_cache(
        evidence_dir=tmp_path / "second-evidence",
        encoder=FailIfEncoded(),
        **second_args,
    ).to_dict()
    second_state = json.loads(
        (tmp_path / "second-evidence/cache_build_state.json").read_text(encoding="utf-8")
    )

    assert first["configuration"]["cache_namespace"] != second["configuration"]["cache_namespace"]
    assert first["plan"]["sha256"] != second["plan"]["sha256"]
    assert second["build"]["encoded_rows"] == 0
    assert second["build"]["resumed_rows"] == 0
    assert second["build"]["reused_rows"] == 1
    assert second_state["completed_shards"] == []


@pytest.mark.parametrize(
    "artifacts",
    [
        {"unsafe\nname.yaml": b"x"},
        {"Config.yaml": b"first", "config.yaml": b"second"},
        {"sha256sums": b"reserved portable alias"},
    ],
)
def test_input_artifact_names_are_portable_and_case_unique(
    tmp_path: Path,
    artifacts: dict[str, bytes],
) -> None:
    requests = tmp_path / "requests.jsonl"
    _write_requests(requests)

    with pytest.raises(InputError, match="safe unique basename"):
        build_window_cache(
            requests_jsonl=requests,
            cache_dir=tmp_path / "cache",
            evidence_dir=tmp_path / "evidence",
            encoder=FailIfEncoded(),
            encoder_id="HuggingFaceBio/Carbon-500M",
            batch_size=1,
            rows_per_shard=1,
            created_at_ns=1_750_000_000_000_000_000,
            input_artifacts=artifacts,
        )

    assert not (tmp_path / "evidence/cache_build_plan.json").exists()
