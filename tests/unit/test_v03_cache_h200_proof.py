# SPDX-License-Identifier: Apache-2.0
"""Contracts for the exact-trace H200 cache interruption proof."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from itertools import pairwise
from pathlib import Path
from typing import Any

import jsonschema
import pytest

import tools.research.v03_cache_h200_proof as proof_module
from geno_lewm.encoder.cache import reindex_cache
from geno_lewm.encoder.cache_build import build_window_cache
from geno_lewm.errors import InputError
from geno_lewm.provenance import sha256_bytes
from tools.research.v03_cache_h200_proof import (
    DEFAULT_SCHEMA_PATH,
    GENERATED_BY,
    HARDWARE_GENERATED_BY,
    HARDWARE_SCHEMA_VERSION,
    ProofExpectations,
    author_proof_bundle,
    capture_partial_bundle,
    finalize_interruption,
    retire_cache_runtime_lock,
    validate_runtime_identity,
    verify_existing_bundle,
)

RUNTIME_IDENTITY = Path("configs/data_v03/carbon-500m-l2-runtime-identity.json")
TRACE_ARTIFACT_PATH = (
    "training-traces/v0.3/"
    "geno-lewm-v03-training-trace-48b5bf71397f-712d612d85ea-"
    "job-6a55f38e85d9643ce16d29e7-r1/success"
)
TRACE_REVISION = "da0d86cde7bf88de2015ab7c516f356e9ae89469"
pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="H200 interruption proof and cache publication locking are POSIX-only",
)


def test_committed_runtime_identity_is_the_exact_corrected_carbon_contract() -> None:
    payload = validate_runtime_identity(RUNTIME_IDENTITY)

    assert payload == {
        "model_id": "/carbon",
        "revision": "5d31d59b3c845b288a13aedb1358934196852eec",
        "runtime_hash": "sha256:add3c1a663a35fb92fbd3fd935b067da1aed8aeb143ea01f7d92c2cd3ed2aa5e",
        "schema_version": "1.0.0",
        "state_contract_version": "l2_normalized_v2",
    }
    assert json.loads(RUNTIME_IDENTITY.read_text(encoding="utf-8")) == payload


def test_proof_schema_is_draft_2020_12_and_closed() -> None:
    schema = json.loads(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    for field in (
        "producer",
        "trace",
        "runtime",
        "interruption",
        "resume",
        "cache",
        "hardware",
        "claim_boundary",
    ):
        assert schema["properties"][field]["additionalProperties"] is False
    claim = schema["properties"]["claim_boundary"]
    assert claim["properties"]["hf_job_terminal_status_attested"] == {"const": False}
    assert claim["properties"]["interrupted_in_flight_encode_batch_work_counted"] == {
        "const": False
    }
    resume = schema["properties"]["resume"]
    assert resume["properties"]["durable_completed_shard_encode_batch_calls"] == {"const": 928}
    assert resume["properties"]["completed_shards_reencoded"] == {"const": 0}


def test_trace_namespace_accepts_the_exact_public_path_and_rejects_candidates() -> None:
    assert proof_module._validate_trace_publication_arguments(
        "abdelstark/geno-lewm-data",
        TRACE_REVISION,
        TRACE_ARTIFACT_PATH,
    ) == (
        "abdelstark/geno-lewm-data",
        TRACE_REVISION,
        TRACE_ARTIFACT_PATH,
    )
    with pytest.raises(InputError, match="training-trace namespace"):
        proof_module._validate_trace_publication_arguments(
            "abdelstark/geno-lewm-data",
            TRACE_REVISION,
            "candidates/v0.3/fixture/success",
        )
    with pytest.raises(InputError, match="exact corrected public training trace"):
        proof_module._validate_trace_publication_arguments(
            "abdelstark/geno-lewm-data",
            "0" * 40,
            TRACE_ARTIFACT_PATH,
        )


class _FakeRawEncoder:
    encoder_hash = bytes.fromhex("add3c1a663a35fb92fbd3fd935b067da1aed8aeb143ea01f7d92c2cd3ed2aa5e")
    state_layer = 20
    pool_type = "centered_mean"
    pool_radius = 8
    dtype = "bf16"
    normalize = False
    device = "cuda"

    def pooling_identity(self, window: str, edit_locus: int | None) -> tuple[str, int, int]:
        del window
        assert edit_locus is not None
        return "centered_mean", 8, 1 + edit_locus // 6

    def encode_batch(
        self,
        windows: list[str],
        edit_loci: list[int | None],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            (float(len(window)), float(-1 if locus is None else locus))
            for window, locus in zip(windows, edit_loci, strict=True)
        )


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _identity(path: str, body: bytes) -> dict[str, object]:
    return {"path": path, "sha256": sha256_bytes(body), "size_bytes": len(body)}


def _write_json(path: Path, payload: object) -> None:
    if path.exists():
        path.chmod(0o600)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))
    path.chmod(0o400)


def _close_directory(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    if checksum_path.exists():
        checksum_path.chmod(0o600)
        checksum_path.unlink()
    names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    checksum_path.write_text(
        "".join(f"{sha256_bytes((root / name).read_bytes())[7:]}  {name}\n" for name in names),
        encoding="ascii",
    )
    checksum_path.chmod(0o400)


def _request_bytes() -> bytes:
    windows = ("A" * 12, "C" * 12, "G" * 12, "T" * 12, "ACGT" * 3)
    rows = [
        {
            "request_id": f"request-{index}",
            "chrom": "22",
            "start_bp": index * 100,
            "end_bp": index * 100 + len(window),
            "window": window,
            "edit_locus": 0,
        }
        for index, window in enumerate(windows)
    ]
    rows.append({**rows[0], "request_id": "request-duplicate"})
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows
    )


def _make_bundle(root: Path) -> tuple[Path, ProofExpectations, Path, Path, Path]:
    bundle = root.resolve() / "bundle"
    for name in ("cache", "evidence", "trace", "proof"):
        (bundle / name).mkdir(parents=True, exist_ok=True)
    requests = _request_bytes()
    config = Path("configs/data_v03/train-carbon-500m-snv-l2-epoch-r1.yaml").read_bytes()
    expectations = ProofExpectations(
        request_rows=6,
        request_sha256=sha256_bytes(requests),
        request_size_bytes=len(requests),
        unique_cache_keys=5,
        duplicate_rows=1,
        batch_size=2,
        rows_per_shard=2,
        shard_row_counts=(2, 2, 1),
        durable_completed_shard_encode_batch_calls=3,
        training_config_sha256=sha256_bytes(config),
        created_at_ns=1_783_965_600_000_000_000,
    )
    trace_schema = _json_bytes({"$schema": "https://json-schema.org/draft/2020-12/schema"})
    trace_report = _json_bytes(
        {
            "trace": {
                "request_rows": 6,
                "requests": _identity("cache_build_requests.jsonl", requests),
            },
            "training": {
                "batch_size": 2,
                "state_contract_version": "l2_normalized_v2",
                "encoder_revision": "5d31d59b3c845b288a13aedb1358934196852eec",
                "pool_type": "centered_mean",
                "pool_radius": 8,
                "normalize": True,
            },
        }
    )
    for name, body in {
        "cache_build_requests.jsonl": requests,
        "training_config.yaml": config,
        "training_trace.schema.json": trace_schema,
        "training_trace_report.json": trace_report,
    }.items():
        (bundle / "trace" / name).write_bytes(body)
    _close_directory(bundle / "trace")

    runtime = RUNTIME_IDENTITY.read_bytes()
    build_window_cache(
        requests_jsonl=requests,
        cache_dir=bundle / "cache",
        evidence_dir=bundle / "evidence",
        encoder=_FakeRawEncoder(),
        encoder_id="/carbon",
        batch_size=2,
        rows_per_shard=2,
        created_at_ns=1_783_965_600_000_000_000,
        hardware=("NVIDIA H200; 147849216000 bytes; CUDA 12.8; driver 570.0; single GPU"),
        resolved_config=proof_module._resolved_config_from_yaml(config),
        encoder_runtime_identity=json.loads(runtime),
        input_artifacts={
            "encoder_config.yaml": config,
            "encoder_runtime_identity_source.json": runtime,
        },
    )
    attempt_log = root / "attempt.jsonl"
    attempt_log.write_text(
        json.dumps({"event": "data.cache.build.start", "data": {}}) + "\n",
        encoding="utf-8",
    )
    resume_log = root / "resume.jsonl"
    hardware = root / "hardware.json"
    return bundle, expectations, attempt_log, resume_log, hardware


def _convert_to_resumed_completion(bundle: Path, resume_log: Path) -> tuple[dict[str, Any], int]:
    state_path = bundle / "evidence" / "cache_build_state.json"
    report_path = bundle / "evidence" / "cache_build_report.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    partial_entries = state["completed_shards"][:2]
    partial_rows = sum(entry["row_count"] for entry in partial_entries)
    partial_state = {**state, "completed_shards": partial_entries, "completion": None}
    attempt_dir = bundle / "proof" / "attempt1"
    attempt_dir.mkdir(parents=True)
    plan_body = (bundle / "evidence" / "cache_build_plan.json").read_bytes()
    state_body = _json_bytes(partial_state)
    attempt_log_body = (bundle.parent / "attempt.jsonl").read_bytes()
    (attempt_dir / "cache_build_plan.json").write_bytes(plan_body)
    (attempt_dir / "cache_build_state.json").write_bytes(state_body)
    (attempt_dir / "cache-build.jsonl").write_bytes(attempt_log_body)
    partial_cache = bundle.parent / "partial-cache-snapshot"
    for entry in partial_entries:
        source = bundle / "cache" / entry["path"]
        destination = partial_cache / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    reindex_cache(partial_cache)
    archived_cache = attempt_dir / "cache"
    snapshot_names = [
        "embeddings/index.sqlite",
        *(entry["path"] for entry in partial_entries),
    ]
    for relative in snapshot_names:
        destination = archived_cache / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(partial_cache / relative, destination)
    cache_artifacts = [
        _identity(f"cache/{relative}", (archived_cache / relative).read_bytes())
        for relative in sorted(snapshot_names)
    ]
    cache_artifacts.insert(
        0,
        _identity("cache/embeddings/.publish.lock", b""),
    )
    capture = {
        "schema_version": "geno-lewm.v03-cache-h200-interruption-capture.v1",
        "generated_by": GENERATED_BY,
        "process": {
            "pid": 1234,
            "observed_state": "stopped",
            "argv": ["geno-lewm-cache-windows"],
            "comm": "python",
            "child_pids": [],
            "termination_sequence": "SIGTERM_while_stopped_then_SIGCONT",
            "expected_shell_exit_code": 143,
        },
        "partial": {
            "completed_shards": 2,
            "completed_rows": partial_rows,
            "durable_completed_shard_encode_batch_calls": sum(
                entry["encode_batch_calls"] for entry in partial_entries
            ),
            "plan": _identity("cache_build_plan.json", plan_body),
            "state": _identity("cache_build_state.json", state_body),
            "log": _identity("cache-build.jsonl", attempt_log_body),
            "cache_artifacts": cache_artifacts,
        },
    }
    _write_json(attempt_dir / "capture.json", capture)
    termination = {
        "schema_version": "geno-lewm.v03-cache-h200-interruption-termination.v1",
        "generated_by": GENERATED_BY,
        "process": {
            "shell_exit_code": 143,
            "wait_status_scope": "conventional_shell_status_consistent_with_SIGTERM",
            "kernel_waitpid_signal_attested": False,
            "termination_sequence": "SIGTERM_while_stopped_then_SIGCONT",
        },
        "post_termination": {
            "state_plan_log_and_cache_bytes_unchanged": True,
            "plan": capture["partial"]["plan"],
            "state": capture["partial"]["state"],
            "log": capture["partial"]["log"],
            "cache_artifacts": capture["partial"]["cache_artifacts"],
        },
    }
    _write_json(attempt_dir / "termination.json", termination)

    encoded_rows = 5 - partial_rows
    state["completion"] = {
        "encoded_rows": encoded_rows,
        "encoded_shards": 1,
        "resumed_rows": partial_rows,
        "reused_rows": 0,
        "resolved_unique_rows": 5,
        "planned_shards": 3,
        "invocation_elapsed_seconds": 1.0,
        "run_id": "fixture-resume",
    }
    _write_json(state_path, state)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["run_id"] = "fixture-resume"
    report["throughput"]["invocation_elapsed_seconds"] = 1.0
    report["build"].update(
        {
            "encoded_rows": encoded_rows,
            "encoded_shards": 1,
            "resumed_rows": partial_rows,
            "reused_rows": 0,
            "resolved_unique_rows": 5,
        }
    )
    report["evidence_artifacts"]["state"] = _identity(
        "cache_build_state.json", state_path.read_bytes()
    )
    _write_json(report_path, report)
    _close_directory(bundle / "evidence")
    resume_log.write_text(
        json.dumps(
            {
                "event": "data.cache.build.end",
                "data": {
                    "completed_shards": 3,
                    "encoded_rows": encoded_rows,
                    "resumed_rows": partial_rows,
                    "evidence_report": "cache_build_report.json",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return state, partial_rows


def _write_hardware(path: Path, *, source_commit: str, container_image: str) -> None:
    _write_json(
        path,
        {
            "schema_version": HARDWARE_SCHEMA_VERSION,
            "generated_by": HARDWARE_GENERATED_BY,
            "source_commit_sha": source_commit,
            "container_image": container_image,
            "nvidia_smi_query_raw": "0, NVIDIA H200, 141000, 9.0, 570.0",
            "device": {
                "type": "cuda",
                "index": 0,
                "name": "NVIDIA H200",
                "total_memory_bytes": 147_849_216_000,
                "compute_capability": "9.0",
            },
            "runtime": {
                "python_version": "3.11",
                "torch_version": "2.8",
                "cuda_version": "12.8",
                "driver_version": "570.0",
            },
        },
    )


def _permissive_schema(path: Path) -> Path:
    path.write_text(
        json.dumps({"$schema": "https://json-schema.org/draft/2020-12/schema"}),
        encoding="utf-8",
    )
    return path


def test_authored_bundle_replays_and_tampering_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, expectations, _attempt_log, resume_log, hardware = _make_bundle(tmp_path)
    _convert_to_resumed_completion(bundle, resume_log)
    assert (bundle / "cache" / "embeddings" / ".publish.lock").is_file()
    retire_cache_runtime_lock(bundle_dir=bundle, expectations=expectations)
    assert not (bundle / "cache" / "embeddings" / ".publish.lock").exists()
    source_commit = "a" * 40
    container = "example.invalid/geno-lewm@sha256:" + "b" * 64
    _write_hardware(hardware, source_commit=source_commit, container_image=container)
    monkeypatch.setattr(
        proof_module, "DEFAULT_SCHEMA_PATH", _permissive_schema(tmp_path / "proof.schema.json")
    )
    monkeypatch.setattr(proof_module, "_verify_producer_invocation", lambda **_kwargs: None)

    report = author_proof_bundle(
        bundle_dir=bundle,
        trace_repository="abdelstark/geno-lewm-data",
        trace_revision=TRACE_REVISION,
        trace_artifact_path=TRACE_ARTIFACT_PATH,
        runtime_identity_path=RUNTIME_IDENTITY,
        source_commit=source_commit,
        container_image=container,
        hardware_json=hardware,
        resume_log=resume_log,
        expectations=expectations,
        verify_remote=False,
    )

    assert verify_existing_bundle(bundle_dir=bundle, expectations=expectations) == report
    runtime_copy = bundle / "proof" / "encoder_runtime_identity.json"
    runtime_copy.chmod(0o600)
    runtime_copy.write_text("{}\n", encoding="utf-8")
    with pytest.raises(InputError, match="digest mismatch"):
        verify_existing_bundle(bundle_dir=bundle, expectations=expectations)


def _make_partial_bundle(
    tmp_path: Path,
) -> tuple[Path, ProofExpectations, Path, tuple[str, bytes]]:
    bundle, expectations, attempt_log, _resume, _hardware = _make_bundle(tmp_path)
    state_path = bundle / "evidence" / "cache_build_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    partial_entries = state["completed_shards"][:2]
    retained = {entry["path"] for entry in partial_entries}
    gap: tuple[str, bytes] | None = None
    for entry in state["completed_shards"]:
        path = bundle / "cache" / entry["path"]
        if entry["path"] not in retained:
            gap = (entry["path"], path.read_bytes())
            path.unlink()
    assert gap is not None
    reindex_cache(bundle / "cache")
    _write_json(state_path, {**state, "completed_shards": partial_entries, "completion": None})
    for name in ("cache_build_report.json", "SHA256SUMS"):
        path = bundle / "evidence" / name
        path.chmod(0o600)
        path.unlink()
    return bundle, expectations, attempt_log, gap


def _fake_stopped_process(_pid: int) -> dict[str, object]:
    return {
        "pid": 1234,
        "observed_state": "stopped",
        "argv": ["geno-lewm-cache-windows"],
        "comm": "python",
        "child_pids": [],
    }


def test_partial_capture_rejects_crash_gap_and_premature_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, expectations, attempt_log, (gap_name, gap_body) = _make_partial_bundle(tmp_path)
    monkeypatch.setattr(proof_module, "_capture_stopped_process", _fake_stopped_process)
    gap_path = bundle / "cache" / gap_name
    gap_path.parent.mkdir(parents=True, exist_ok=True)
    gap_path.write_bytes(gap_body)

    with pytest.raises(InputError, match=r"outside durable state|unknown file"):
        capture_partial_bundle(
            bundle_dir=bundle,
            attempt_log=attempt_log,
            stopped_pid=1234,
            expectations=expectations,
        )
    gap_path.unlink()
    (bundle / "evidence" / "SHA256SUMS").write_text("premature\n", encoding="utf-8")
    with pytest.raises(InputError, match=r"inventory|already complete"):
        capture_partial_bundle(
            bundle_dir=bundle,
            attempt_log=attempt_log,
            stopped_pid=1234,
            expectations=expectations,
        )


def test_interruption_finalization_rejects_rc_and_post_term_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, expectations, attempt_log, _gap = _make_partial_bundle(tmp_path)
    monkeypatch.setattr(proof_module, "_capture_stopped_process", _fake_stopped_process)
    capture_partial_bundle(
        bundle_dir=bundle,
        attempt_log=attempt_log,
        stopped_pid=1234,
        expectations=expectations,
    )

    with pytest.raises(InputError, match="status 143"):
        finalize_interruption(
            bundle_dir=bundle,
            attempt_log=attempt_log,
            attempt_exit_code=1,
            expectations=expectations,
        )
    state = bundle / "evidence" / "cache_build_state.json"
    state.chmod(0o600)
    state.write_bytes(state.read_bytes() + b" ")
    with pytest.raises(InputError, match="state changed"):
        finalize_interruption(
            bundle_dir=bundle,
            attempt_log=attempt_log,
            attempt_exit_code=143,
            expectations=expectations,
        )


@pytest.mark.parametrize(
    "drift",
    ["plan", "reused_rows", "count", "throughput", "extra_evidence", "hardware_receipt"],
)
def test_author_rejects_plan_state_reuse_and_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    bundle, expectations, _attempt_log, resume_log, hardware = _make_bundle(tmp_path)
    state, _partial_rows = _convert_to_resumed_completion(bundle, resume_log)
    retire_cache_runtime_lock(bundle_dir=bundle, expectations=expectations)
    source_commit = "a" * 40
    container = "example.invalid/geno-lewm@sha256:" + "b" * 64
    _write_hardware(hardware, source_commit=source_commit, container_image=container)
    monkeypatch.setattr(
        proof_module, "DEFAULT_SCHEMA_PATH", _permissive_schema(tmp_path / "proof.schema.json")
    )
    monkeypatch.setattr(proof_module, "_verify_producer_invocation", lambda **_kwargs: None)
    if drift == "plan":
        partial_plan = bundle / "proof" / "attempt1" / "cache_build_plan.json"
        payload = json.loads(partial_plan.read_text(encoding="utf-8"))
        payload["created_at_ns"] += 1
        _write_json(partial_plan, payload)
    elif drift == "reused_rows":
        state["completion"]["reused_rows"] = 1
        state["completion"]["encoded_rows"] -= 1
        _write_json(bundle / "evidence" / "cache_build_state.json", state)
        _close_directory(bundle / "evidence")
    elif drift == "count":
        report_path = bundle / "evidence" / "cache_build_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["build"]["completed_shards"] = 2
        _write_json(report_path, report)
        _close_directory(bundle / "evidence")
    elif drift == "throughput":
        report_path = bundle / "evidence" / "cache_build_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["throughput"]["measured_encoded_rows"] += 1
        _write_json(report_path, report)
        _close_directory(bundle / "evidence")
    elif drift == "extra_evidence":
        (bundle / "evidence" / "unexpected.json").write_text("{}\n", encoding="utf-8")
        _close_directory(bundle / "evidence")
    else:
        receipt = json.loads(hardware.read_text(encoding="utf-8"))
        receipt["runtime"]["driver_version"] = "571.0"
        receipt["nvidia_smi_query_raw"] = receipt["nvidia_smi_query_raw"].replace("570.0", "571.0")
        _write_json(hardware, receipt)

    with pytest.raises(InputError):
        author_proof_bundle(
            bundle_dir=bundle,
            trace_repository="abdelstark/geno-lewm-data",
            trace_revision=TRACE_REVISION,
            trace_artifact_path=TRACE_ARTIFACT_PATH,
            runtime_identity_path=RUNTIME_IDENTITY,
            source_commit=source_commit,
            container_image=container,
            hardware_json=hardware,
            resume_log=resume_log,
            expectations=expectations,
            verify_remote=False,
        )


def test_recomputed_outer_checksums_cannot_hide_runtime_identity_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, expectations, _attempt_log, resume_log, hardware = _make_bundle(tmp_path)
    _convert_to_resumed_completion(bundle, resume_log)
    retirement = retire_cache_runtime_lock(bundle_dir=bundle, expectations=expectations)
    assert retirement["retired_path"] == "cache/embeddings/.publish.lock"
    assert not (bundle / "cache" / "embeddings" / ".publish.lock").exists()
    source_commit = "a" * 40
    container = "example.invalid/geno-lewm@sha256:" + "b" * 64
    _write_hardware(hardware, source_commit=source_commit, container_image=container)
    monkeypatch.setattr(
        proof_module, "DEFAULT_SCHEMA_PATH", _permissive_schema(tmp_path / "proof.schema.json")
    )
    monkeypatch.setattr(proof_module, "_verify_producer_invocation", lambda **_kwargs: None)
    author_proof_bundle(
        bundle_dir=bundle,
        trace_repository="abdelstark/geno-lewm-data",
        trace_revision=TRACE_REVISION,
        trace_artifact_path=TRACE_ARTIFACT_PATH,
        runtime_identity_path=RUNTIME_IDENTITY,
        source_commit=source_commit,
        container_image=container,
        hardware_json=hardware,
        resume_log=resume_log,
        expectations=expectations,
        verify_remote=False,
    )
    runtime_copy = bundle / "proof" / "encoder_runtime_identity.json"
    runtime_copy.chmod(0o600)
    runtime_copy.write_text("{}\n", encoding="utf-8")
    outer = bundle / "SHA256SUMS"
    outer.chmod(0o600)
    outer.unlink()
    _close_directory(bundle)

    with pytest.raises(InputError, match="runtime identity"):
        verify_existing_bundle(bundle_dir=bundle, expectations=expectations)


@pytest.mark.parametrize("drift", ["alias", "record", "identity", "shard_order"])
def test_exact_trace_rederivation_rejects_arbitrary_plan_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    bundle, expectations, _attempt, _resume, _hardware = _make_bundle(tmp_path)
    plan = proof_module._validate_plan(
        bundle / "evidence" / "cache_build_plan.json",
        expectations=expectations,
    )
    payload = plan.payload
    shards = payload["shards"]
    assert isinstance(shards, list)
    first = shards[0]
    assert isinstance(first, dict)
    rows = first["rows"]
    assert isinstance(rows, list)
    row = rows[0]
    assert isinstance(row, dict)
    if drift == "alias":
        row["request_ids"] = ["forged-request"]
    elif drift == "record":
        record = row["record"]
        assert isinstance(record, dict)
        record["start_bp"] += 1
    elif drift == "identity":
        payload["plan_identity"] = "sha256:" + "0" * 64
    else:
        shards.reverse()

    with pytest.raises(InputError, match=r"exact trace|identity|rows or aliases"):
        proof_module._validate_plan_against_trace(
            plan,
            request_body=(bundle / "trace" / "cache_build_requests.jsonl").read_bytes(),
            expectations=expectations,
        )


def test_coordinated_report_and_sqlite_offset_drift_fails_decoded_order(
    tmp_path: Path,
) -> None:
    bundle, expectations, _attempt, resume_log, _hardware = _make_bundle(tmp_path)
    _convert_to_resumed_completion(bundle, resume_log)
    index_path = bundle / "cache" / "embeddings" / "index.sqlite"
    connection = sqlite3.connect(index_path)
    try:
        rows = connection.execute(
            "SELECT window_hash, shard_path, row_offset FROM window_index "
            "ORDER BY shard_path, row_offset"
        ).fetchall()
        first, second = next((left, right) for left, right in pairwise(rows) if left[1] == right[1])
        connection.execute(
            "UPDATE window_index SET row_offset = ? WHERE window_hash = ?",
            (second[2], first[0]),
        )
        connection.execute(
            "UPDATE window_index SET row_offset = ? WHERE window_hash = ?",
            (first[2], second[0]),
        )
        connection.commit()
    finally:
        connection.close()
    report_path = bundle / "evidence" / "cache_build_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_shard = next(
        shard for shard in report["cache_artifacts"]["shards"] if shard["path"] == first[1]
    )
    report_shard["request_rows"][first[2]]["row_offset"] = second[2]
    report_shard["request_rows"][second[2]]["row_offset"] = first[2]
    _write_json(report_path, report)
    _close_directory(bundle / "evidence")

    with pytest.raises(InputError, match=r"decoded|logical"):
        retire_cache_runtime_lock(bundle_dir=bundle, expectations=expectations)


def test_recomputed_checksums_cannot_substitute_a_permissive_proof_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, expectations, _attempt, resume_log, hardware = _make_bundle(tmp_path)
    _convert_to_resumed_completion(bundle, resume_log)
    retire_cache_runtime_lock(bundle_dir=bundle, expectations=expectations)
    source_commit = "a" * 40
    container = "example.invalid/geno-lewm@sha256:" + "b" * 64
    _write_hardware(hardware, source_commit=source_commit, container_image=container)
    committed_schema = _permissive_schema(tmp_path / "committed.schema.json")
    monkeypatch.setattr(proof_module, "DEFAULT_SCHEMA_PATH", committed_schema)
    monkeypatch.setattr(proof_module, "_verify_producer_invocation", lambda **_kwargs: None)
    author_proof_bundle(
        bundle_dir=bundle,
        trace_repository="abdelstark/geno-lewm-data",
        trace_revision=TRACE_REVISION,
        trace_artifact_path=TRACE_ARTIFACT_PATH,
        runtime_identity_path=RUNTIME_IDENTITY,
        source_commit=source_commit,
        container_image=container,
        hardware_json=hardware,
        resume_log=resume_log,
        expectations=expectations,
        verify_remote=False,
    )
    bundled_schema = bundle / "proof" / "cache-h200-proof.schema.json"
    bundled_schema.chmod(0o600)
    bundled_schema.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "description": "substituted permissive schema",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    outer = bundle / "SHA256SUMS"
    outer.chmod(0o600)
    outer.unlink()
    _close_directory(bundle)

    with pytest.raises(InputError, match="differs from the committed schema"):
        verify_existing_bundle(bundle_dir=bundle, expectations=expectations)


def test_recomputed_checksums_cannot_hide_inconsistent_hardware_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, expectations, _attempt, resume_log, hardware = _make_bundle(tmp_path)
    _convert_to_resumed_completion(bundle, resume_log)
    retire_cache_runtime_lock(bundle_dir=bundle, expectations=expectations)
    source_commit = "a" * 40
    container = "example.invalid/geno-lewm@sha256:" + "b" * 64
    _write_hardware(hardware, source_commit=source_commit, container_image=container)
    monkeypatch.setattr(
        proof_module,
        "DEFAULT_SCHEMA_PATH",
        _permissive_schema(tmp_path / "proof.schema.json"),
    )
    monkeypatch.setattr(proof_module, "_verify_producer_invocation", lambda **_kwargs: None)
    author_proof_bundle(
        bundle_dir=bundle,
        trace_repository="abdelstark/geno-lewm-data",
        trace_revision=TRACE_REVISION,
        trace_artifact_path=TRACE_ARTIFACT_PATH,
        runtime_identity_path=RUNTIME_IDENTITY,
        source_commit=source_commit,
        container_image=container,
        hardware_json=hardware,
        resume_log=resume_log,
        expectations=expectations,
        verify_remote=False,
    )
    receipt_path = bundle / "proof" / "hardware.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["nvidia_smi_query_raw"] = receipt["nvidia_smi_query_raw"].replace(
        "NVIDIA H200", "NVIDIA H200 forged"
    )
    receipt_path.chmod(0o600)
    receipt_path.write_bytes(_json_bytes(receipt))
    outer = bundle / "SHA256SUMS"
    outer.chmod(0o600)
    outer.unlink()
    _close_directory(bundle)

    with pytest.raises(InputError, match="hardware receipt"):
        verify_existing_bundle(bundle_dir=bundle, expectations=expectations)


@pytest.mark.parametrize("drift", ["extra_capture_field", "boolean_pid", "extra_post_field"])
def test_recomputed_checksums_cannot_open_interruption_receipt_schemas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    bundle, expectations, _attempt, resume_log, hardware = _make_bundle(tmp_path)
    _convert_to_resumed_completion(bundle, resume_log)
    retire_cache_runtime_lock(bundle_dir=bundle, expectations=expectations)
    source_commit = "a" * 40
    container = "example.invalid/geno-lewm@sha256:" + "b" * 64
    _write_hardware(hardware, source_commit=source_commit, container_image=container)
    monkeypatch.setattr(
        proof_module,
        "DEFAULT_SCHEMA_PATH",
        _permissive_schema(tmp_path / "proof.schema.json"),
    )
    monkeypatch.setattr(proof_module, "_verify_producer_invocation", lambda **_kwargs: None)
    author_proof_bundle(
        bundle_dir=bundle,
        trace_repository="abdelstark/geno-lewm-data",
        trace_revision=TRACE_REVISION,
        trace_artifact_path=TRACE_ARTIFACT_PATH,
        runtime_identity_path=RUNTIME_IDENTITY,
        source_commit=source_commit,
        container_image=container,
        hardware_json=hardware,
        resume_log=resume_log,
        expectations=expectations,
        verify_remote=False,
    )
    if drift == "extra_post_field":
        receipt_path = bundle / "proof" / "attempt1" / "termination.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["post_termination"]["unexpected"] = False
    else:
        receipt_path = bundle / "proof" / "attempt1" / "capture.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if drift == "extra_capture_field":
            receipt["partial"]["unexpected"] = False
        else:
            receipt["process"]["pid"] = True
    receipt_path.chmod(0o600)
    receipt_path.write_bytes(_json_bytes(receipt))
    outer = bundle / "SHA256SUMS"
    outer.chmod(0o600)
    outer.unlink()
    _close_directory(bundle)

    with pytest.raises(InputError, match=r"capture|termination receipt"):
        verify_existing_bundle(bundle_dir=bundle, expectations=expectations)
