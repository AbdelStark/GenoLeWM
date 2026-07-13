"""Contract tests for finite, resumable window-cache builds."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from geno_lewm.encoder import POOL_CENTERED_MEAN, POOL_GLOBAL_MEAN, inspect_cache_shard
from geno_lewm.encoder.cache_build import build_window_cache
from geno_lewm.errors import CacheCorruptError, InputError
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
    repaired = {entry["shard_id"]: entry for entry in repaired_state["completed_shards"]}
    assert repaired[removed["shard_id"]]["origin"] == "adopted_after_interruption"
    assert repaired[removed["shard_id"]]["sha256"] == removed["sha256"]


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
    assert not (evidence / "cache_build_plan.json").exists()


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

    with pytest.raises(InputError, match="pooling identity"):
        build_window_cache(encoder=FailIfEncoded(), **kwargs)
