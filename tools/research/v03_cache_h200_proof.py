# SPDX-License-Identifier: Apache-2.0
"""Author and replay the exact-trace v0.3 H200 cache interruption proof.

The proof deliberately sits outside :mod:`geno_lewm.encoder.cache_build`.  It
does not add a testing hook to the cache implementation: the production CLI is
stopped by the job supervisor after durable shard state is observed, terminated
while stopped, and resumed with byte-identical arguments.  This module closes
the resulting cache, builder evidence, trace, interruption snapshot, logs, and
hardware receipt into one independently replayable directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import yaml

from geno_lewm.config import config_to_dict, load_config
from geno_lewm.encoder.cache import (
    WindowCacheKey,
    inspect_cache_shard,
    resolve_cache_provenances,
    shard_path_for,
)
from geno_lewm.encoder.cache_build import _key_from_payload
from geno_lewm.encoder.windowing import canonicalize_dna, window_sha256
from geno_lewm.errors import InputError, RuntimeSetupError
from geno_lewm.provenance import canonical_json_sha256, sha256_bytes

SCHEMA_VERSION: Final = "geno-lewm.v03-cache-h200-proof.v2"
GENERATED_BY: Final = "tools.research.v03_cache_h200_proof"
HARDWARE_SCHEMA_VERSION: Final = "geno-lewm.v03-h200-hardware.v2"
HARDWARE_GENERATED_BY: Final = "tools.jobs.v03_cache_h200_proof"

TRACE_REQUESTS_NAME: Final = "cache_build_requests.jsonl"
TRACE_CONFIG_NAME: Final = "training_config.yaml"
TRACE_REPORT_NAME: Final = "training_trace_report.json"
TRACE_SCHEMA_NAME: Final = "training_trace.schema.json"
CHECKSUMS_NAME: Final = "SHA256SUMS"
PLAN_NAME: Final = "cache_build_plan.json"
STATE_NAME: Final = "cache_build_state.json"
CACHE_REPORT_NAME: Final = "cache_build_report.json"
RUNTIME_COPY_NAME: Final = "proof/encoder_runtime_identity.json"
SCHEMA_COPY_NAME: Final = "proof/cache-h200-proof.schema.json"
HARDWARE_COPY_NAME: Final = "proof/hardware.json"
PROOF_REPORT_NAME: Final = "proof/cache-h200-proof.json"
ATTEMPT_PLAN_NAME: Final = "proof/attempt1/cache_build_plan.json"
ATTEMPT_STATE_NAME: Final = "proof/attempt1/cache_build_state.json"
ATTEMPT_LOG_NAME: Final = "proof/attempt1/cache-build.jsonl"
ATTEMPT_CAPTURE_NAME: Final = "proof/attempt1/capture.json"
ATTEMPT_TERMINATION_NAME: Final = "proof/attempt1/termination.json"
ATTEMPT_CACHE_DIR: Final = "proof/attempt1/cache"
ATTEMPT_INDEX_NAME: Final = "proof/attempt1/cache/embeddings/index.sqlite"
RESUME_LOG_NAME: Final = "proof/resume/cache-build.jsonl"

_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH: Final = _REPOSITORY_ROOT / "configs/data_v03/cache-h200-proof.schema.json"
DEFAULT_RUNTIME_IDENTITY_PATH: Final = (
    _REPOSITORY_ROOT / "configs/data_v03/carbon-500m-l2-runtime-identity.json"
)
_CANONICAL_ORIGIN: Final = "https://github.com/AbdelStark/GenoLeWM.git"
_CANONICAL_GITHUB_REPOSITORY: Final = "AbdelStark/GenoLeWM"
_CANONICAL_TRACE_REPOSITORY: Final = "abdelstark/geno-lewm-data"
_CANONICAL_TRACE_REVISION: Final = "da0d86cde7bf88de2015ab7c516f356e9ae89469"
_CANONICAL_TRACE_ARTIFACT_PATH: Final = (
    "training-traces/v0.3/"
    "geno-lewm-v03-training-trace-48b5bf71397f-712d612d85ea-"
    "job-6a55f38e85d9643ce16d29e7-r1/success"
)
_HUB_ENDPOINT: Final = "https://huggingface.co"
_GITHUB_API_ENDPOINT: Final = "https://api.github.com"
_COMMIT: Final = re.compile(r"[0-9a-f]{40}\Z")
_CONTAINER: Final = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
_SHA256: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_TRACE_PATH: Final = re.compile(
    r"training-traces/v0\.3/geno-lewm-v03-training-trace-[A-Za-z0-9._-]+/success\Z"
)

_RUNTIME_IDENTITY: Final[dict[str, object]] = {
    "model_id": "/carbon",
    "revision": "5d31d59b3c845b288a13aedb1358934196852eec",
    "runtime_hash": "sha256:a1fd1dd20756c7248b7f9ca95c59c821f0329530fd49c6fea253a8df9a6a6311",
    "schema_version": "1.0.0",
    "state_contract_version": "l2_normalized_v2",
}


@dataclass(frozen=True, slots=True)
class ProofExpectations:
    """Immutable cardinalities for one exact request trace."""

    request_rows: int
    request_sha256: str
    request_size_bytes: int
    unique_cache_keys: int
    duplicate_rows: int
    batch_size: int
    rows_per_shard: int
    shard_row_counts: tuple[int, ...]
    durable_completed_shard_encode_batch_calls: int
    training_config_sha256: str
    created_at_ns: int

    def __post_init__(self) -> None:
        if self.request_rows != self.unique_cache_keys + self.duplicate_rows:
            raise ValueError("request cardinalities must close")
        if sum(self.shard_row_counts) != self.unique_cache_keys:
            raise ValueError("shard cardinalities must close")
        expected_calls = sum(math.ceil(rows / self.batch_size) for rows in self.shard_row_counts)
        if expected_calls != self.durable_completed_shard_encode_batch_calls:
            raise ValueError("encode-batch call cardinality must close")


PRODUCTION_EXPECTATIONS: Final = ProofExpectations(
    request_rows=7_504,
    request_sha256="sha256:38757425a1aa7a0df89e303c339ace68c430848e4eceb1137d5a6448572bea7c",
    request_size_bytes=31_680_405,
    unique_cache_keys=7_421,
    duplicate_rows=83,
    batch_size=8,
    rows_per_shard=256,
    shard_row_counts=(*(256 for _ in range(28)), 253),
    durable_completed_shard_encode_batch_calls=928,
    training_config_sha256=(
        "sha256:1e4a2999316c89e2e0152301a581a0b5d333c704968184718e8c0facc9484806"
    ),
    created_at_ns=1_783_987_200_000_000_000,
)


@dataclass(frozen=True, slots=True)
class _PlanFacts:
    payload: Mapping[str, object]
    body: bytes
    shards_by_id: Mapping[str, Mapping[str, object]]
    keys: tuple[WindowCacheKey, ...]


@dataclass(frozen=True, slots=True)
class _StateFacts:
    payload: Mapping[str, object]
    body: bytes
    completed_by_id: Mapping[str, Mapping[str, object]]
    completed_rows: int
    encode_batch_calls: int


@dataclass(frozen=True, slots=True)
class _TraceRequest:
    request_id: str
    chrom: str
    start_bp: int
    end_bp: int
    window: str
    edit_locus: int


def validate_runtime_identity(path: Path) -> dict[str, object]:
    """Return the exact committed corrected-Carbon runtime identity."""
    payload = _json_object(
        _read_regular_bytes(path, label="cache proof runtime identity"),
        label="cache proof runtime identity",
    )
    if payload != _RUNTIME_IDENTITY:
        raise InputError("cache proof runtime identity does not match the exact contract")
    return payload


def author_hardware_receipt(
    *,
    output_path: Path,
    source_commit: str,
    container_image: str,
    nvidia_smi_query_raw: str,
    cuda_device_count: int,
    cuda_device_index: int,
    cuda_device_name: str,
    cuda_total_memory_bytes: int,
    cuda_compute_capability: str,
    python_version: str,
    torch_version: str,
    cuda_version: str,
    driver_version: str,
) -> dict[str, object]:
    """Write one closed receipt without conflating CUDA and NVML memory reports."""
    _raw_index, _raw_name, nvidia_smi_total_memory_mib, _raw_capability, _raw_driver = (
        _parse_nvidia_smi_row(nvidia_smi_query_raw)
    )
    payload: dict[str, object] = {
        "schema_version": HARDWARE_SCHEMA_VERSION,
        "generated_by": HARDWARE_GENERATED_BY,
        "source_commit_sha": _commit(source_commit, label="source commit"),
        "container_image": _container(container_image, label="container image"),
        "nvidia_smi_query_raw": nvidia_smi_query_raw,
        "device": {
            "type": "cuda",
            "count": cuda_device_count,
            "index": cuda_device_index,
            "name": cuda_device_name,
            "compute_capability": cuda_compute_capability,
        },
        "memory": {
            "cuda_total_memory_bytes": cuda_total_memory_bytes,
            "nvidia_smi_total_memory_mib": nvidia_smi_total_memory_mib,
        },
        "runtime": {
            "python_version": python_version,
            "torch_version": torch_version,
            "cuda_version": cuda_version,
            "driver_version": driver_version,
        },
    }
    receipt = _validate_hardware_receipt(
        payload,
        source_commit=source_commit,
        container_image=container_image,
    )
    _write_once(output_path, _json_bytes(receipt))
    return receipt


def validate_hardware_receipt(
    path: Path,
    *,
    source_commit: str,
    container_image: str,
) -> dict[str, object]:
    """Read back and validate one closed H200 hardware receipt."""
    return _validate_hardware_receipt(
        _json_object(
            _read_regular_bytes(path, label="H200 hardware receipt"),
            label="H200 hardware receipt",
        ),
        source_commit=source_commit,
        container_image=container_image,
    )


def preflight_bundle(
    *,
    bundle_dir: Path,
    trace_dir: Path,
    trace_repository: str,
    trace_revision: str,
    trace_artifact_path: str,
    runtime_identity_path: Path,
    source_commit: str,
    container_image: str,
    expectations: ProofExpectations = PRODUCTION_EXPECTATIONS,
    verify_remote: bool = True,
) -> dict[str, object]:
    """Fail before Carbon work unless every immutable proof input is exact."""
    bundle = _require_physical_bundle(bundle_dir)
    if trace_dir.absolute() != bundle / "trace":
        raise InputError("cache proof trace must be staged at bundle/trace")
    _require_initial_bundle_inventory(bundle)
    runtime = validate_runtime_identity(runtime_identity_path)
    _validate_schema_file(DEFAULT_SCHEMA_PATH)
    trace = _validate_trace_bundle(trace_dir, expectations=expectations)
    repository, revision, artifact_path = _validate_trace_publication_arguments(
        trace_repository,
        trace_revision,
        trace_artifact_path,
    )
    if verify_remote:
        _verify_remote_trace_namespace(
            repository=repository,
            revision=revision,
            artifact_path=artifact_path,
            trace_dir=trace_dir,
        )
    _verify_producer_invocation(source_commit=source_commit, container_image=container_image)
    return {
        "ok": True,
        "source_commit": source_commit,
        "container_image": container_image,
        "trace": {
            "repository": repository,
            "revision": revision,
            "artifact_path": artifact_path,
            "requests": trace["requests"],
        },
        "runtime": runtime,
    }


def capture_partial_bundle(
    *,
    bundle_dir: Path,
    attempt_log: Path,
    stopped_pid: int,
    expectations: ProofExpectations = PRODUCTION_EXPECTATIONS,
) -> dict[str, object]:
    """Snapshot and validate the durable state while the first process is stopped."""
    bundle = _require_physical_bundle(bundle_dir)
    _reject_outer_closure(bundle)
    process_snapshot = _capture_stopped_process(stopped_pid)
    plan = _validate_plan(bundle / "evidence" / PLAN_NAME, expectations=expectations)
    state = _validate_state(
        bundle / "evidence" / STATE_NAME,
        plan=plan,
        expectations=expectations,
        require_completion=False,
    )
    partial_count = len(state.completed_by_id)
    if partial_count < 2 or partial_count >= len(expectations.shard_row_counts):
        raise InputError(
            "cache proof interruption must retain at least two but fewer than all shards",
            details={"completed_shards": partial_count},
        )
    _validate_partial_evidence(bundle, plan=plan, state=state, expectations=expectations)
    log_body, events = _read_log(attempt_log, label="attempt-1 cache log")
    if any(event.get("event") == "data.cache.build.end" for event in events):
        raise InputError("interrupted cache attempt must not contain a build.end event")
    if any((bundle / "proof").iterdir()):
        raise InputError("cache proof interruption snapshot directory is not empty")

    plan_body = plan.body
    state_body = state.body
    capture_payload: dict[str, object] = {
        "schema_version": "geno-lewm.v03-cache-h200-interruption-capture.v1",
        "generated_by": GENERATED_BY,
        "process": {
            **process_snapshot,
            "termination_sequence": "SIGTERM_while_stopped_then_SIGCONT",
            "expected_shell_exit_code": 143,
        },
        "partial": {
            "completed_shards": partial_count,
            "completed_rows": state.completed_rows,
            "durable_completed_shard_encode_batch_calls": state.encode_batch_calls,
            "plan": _identity(PLAN_NAME, plan_body),
            "state": _identity(STATE_NAME, state_body),
            "log": _identity("cache-build.jsonl", log_body),
            "cache_artifacts": _snapshot_cache_identities(
                bundle / "cache",
                completed=state.completed_by_id,
            ),
        },
    }
    snapshot_files = {
        PLAN_NAME: plan_body,
        STATE_NAME: state_body,
        "cache-build.jsonl": log_body,
        "capture.json": _json_bytes(capture_payload),
    }
    for relative in _partial_cache_snapshot_paths(state):
        snapshot_files[f"cache/{relative}"] = _read_regular_bytes(
            bundle / "cache" / relative,
            label=f"partial cache snapshot {relative}",
        )
    _install_directory_noreplace(
        bundle / "proof" / "attempt1",
        snapshot_files,
    )
    _validate_archived_partial_cache(
        bundle / ATTEMPT_CACHE_DIR,
        plan=plan,
        state=state,
    )
    if (
        _read_regular_bytes(bundle / "evidence" / STATE_NAME, label="live partial state")
        != state_body
    ):
        raise InputError("cache proof durable state changed across the stopped capture")
    return capture_payload


def finalize_interruption(
    *,
    bundle_dir: Path,
    attempt_log: Path,
    attempt_exit_code: int,
    expectations: ProofExpectations = PRODUCTION_EXPECTATIONS,
) -> dict[str, object]:
    """Prove the terminated attempt left its stopped durable snapshot unchanged."""
    bundle = _require_physical_bundle(bundle_dir)
    _reject_outer_closure(bundle)
    if attempt_exit_code != 143:
        raise InputError(
            "interrupted cache process did not return conventional SIGTERM status 143",
            details={"expected": 143, "observed": attempt_exit_code},
        )
    plan = _validate_plan(bundle / ATTEMPT_PLAN_NAME, expectations=expectations)
    state = _validate_state(
        bundle / ATTEMPT_STATE_NAME,
        plan=plan,
        expectations=expectations,
        require_completion=False,
    )
    capture = _validate_interruption_capture(
        bundle,
        partial_plan=plan,
        partial_state=state,
    )
    if _read_regular_bytes(bundle / "evidence" / PLAN_NAME, label="post-TERM plan") != plan.body:
        raise InputError("cache plan changed after the stopped process was terminated")
    if _read_regular_bytes(bundle / "evidence" / STATE_NAME, label="post-TERM state") != state.body:
        raise InputError("cache state changed after the stopped process was terminated")
    live_log = _read_regular_bytes(attempt_log, label="post-TERM attempt log")
    captured_log = _read_regular_bytes(bundle / ATTEMPT_LOG_NAME, label="captured attempt log")
    if live_log != captured_log:
        raise InputError("cache log changed after the stopped process was terminated")
    _validate_partial_evidence(
        bundle,
        plan=plan,
        state=state,
        expectations=expectations,
    )
    post_cache = _snapshot_cache_identities(
        bundle / "cache",
        completed=state.completed_by_id,
    )
    capture_partial = _mapping(capture.get("partial"), label="interruption capture partial")
    if capture_partial.get("cache_artifacts") != post_cache:
        raise InputError("cache shard or index bytes changed after SIGTERM/SIGCONT")
    receipt: dict[str, object] = {
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
            "plan": _identity(PLAN_NAME, plan.body),
            "state": _identity(STATE_NAME, state.body),
            "log": _identity("cache-build.jsonl", live_log),
            "cache_artifacts": post_cache,
        },
    }
    _write_once(bundle / ATTEMPT_TERMINATION_NAME, _json_bytes(receipt))
    return receipt


def author_proof_bundle(
    *,
    bundle_dir: Path,
    trace_repository: str,
    trace_revision: str,
    trace_artifact_path: str,
    runtime_identity_path: Path,
    source_commit: str,
    container_image: str,
    hardware_json: Path,
    resume_log: Path,
    expectations: ProofExpectations = PRODUCTION_EXPECTATIONS,
    verify_remote: bool = True,
) -> dict[str, object]:
    """Validate the resumed build and install the final outer checksum closure."""
    bundle = _require_physical_bundle(bundle_dir)
    _reject_outer_closure(bundle)
    repository, revision, artifact_path = _validate_trace_publication_arguments(
        trace_repository,
        trace_revision,
        trace_artifact_path,
    )
    trace = _validate_trace_bundle(bundle / "trace", expectations=expectations)
    if verify_remote:
        _verify_remote_trace_namespace(
            repository=repository,
            revision=revision,
            artifact_path=artifact_path,
            trace_dir=bundle / "trace",
        )
    runtime_body = _read_regular_bytes(runtime_identity_path, label="runtime identity source")
    validate_runtime_identity(runtime_identity_path)
    schema_body = _read_regular_bytes(DEFAULT_SCHEMA_PATH, label="cache H200 proof schema")
    schema = _validate_schema_bytes(schema_body)
    hardware_body = _read_regular_bytes(hardware_json, label="H200 hardware receipt")
    hardware = _validate_hardware_receipt(
        _json_object(hardware_body, label="H200 hardware receipt"),
        source_commit=source_commit,
        container_image=container_image,
    )
    resume_log_body, resume_events = _read_log(resume_log, label="resume cache log")
    _verify_producer_invocation(source_commit=source_commit, container_image=container_image)

    plan = _validate_plan(bundle / "evidence" / PLAN_NAME, expectations=expectations)
    _validate_plan_hardware_binding(plan, hardware=hardware)
    state = _validate_state(
        bundle / "evidence" / STATE_NAME,
        plan=plan,
        expectations=expectations,
        require_completion=True,
    )
    partial_plan = _validate_plan(bundle / ATTEMPT_PLAN_NAME, expectations=expectations)
    partial_state = _validate_state(
        bundle / ATTEMPT_STATE_NAME,
        plan=partial_plan,
        expectations=expectations,
        require_completion=False,
    )
    _validate_interruption_capture(bundle, partial_plan=partial_plan, partial_state=partial_state)
    termination = _validate_interruption_termination(
        bundle,
        partial_plan=partial_plan,
        partial_state=partial_state,
    )
    if partial_plan.body != plan.body:
        raise InputError("resumed cache plan bytes differ from the interrupted plan")
    for shard_id, partial_entry in partial_state.completed_by_id.items():
        if state.completed_by_id.get(shard_id) != partial_entry:
            raise InputError(
                "an interrupted shard state entry changed semantically during resume",
                details={"plan_shard_id": shard_id},
            )
    _validate_completed_evidence(
        bundle,
        plan=plan,
        state=state,
        partial=partial_state,
        expectations=expectations,
    )
    attempt_log_body, attempt_events = _read_log(
        bundle / ATTEMPT_LOG_NAME,
        label="captured attempt-1 cache log",
    )
    if any(event.get("event") == "data.cache.build.end" for event in attempt_events):
        raise InputError("interrupted cache attempt contains a build.end event")
    end_events = [event for event in resume_events if event.get("event") == "data.cache.build.end"]
    if len(end_events) != 1:
        raise InputError(
            "resumed cache invocation must contain exactly one build.end event",
            details={"observed": len(end_events)},
        )
    _validate_resume_end_event(end_events[0], state=state, partial=partial_state)

    proof_dir = bundle / "proof"
    observed_proof = _regular_inventory(proof_dir)
    expected_partial = {
        "attempt1/cache_build_plan.json",
        "attempt1/cache_build_state.json",
        "attempt1/cache-build.jsonl",
        "attempt1/capture.json",
        "attempt1/termination.json",
        *{
            f"attempt1/cache/{relative}"
            for relative in _partial_cache_snapshot_paths(partial_state)
        },
    }
    if observed_proof != expected_partial:
        raise InputError(
            "cache proof contains unexpected files before final authoring",
            details={"observed": sorted(observed_proof)},
        )
    _install_directory_noreplace(
        proof_dir / "resume",
        {"cache-build.jsonl": resume_log_body},
    )
    _write_once(bundle / RUNTIME_COPY_NAME, runtime_body)
    _write_once(bundle / SCHEMA_COPY_NAME, schema_body)
    _write_once(bundle / HARDWARE_COPY_NAME, hardware_body)
    report = _derive_report(
        bundle=bundle,
        repository=repository,
        revision=revision,
        artifact_path=artifact_path,
        source_commit=source_commit,
        container_image=container_image,
        interrupted_exit_code=cast(
            int,
            _mapping(termination["process"], label="termination process")["shell_exit_code"],
        ),
        trace=trace,
        plan=plan,
        state=state,
        partial=partial_state,
        hardware=hardware,
        attempt_log_body=attempt_log_body,
        resume_log_body=resume_log_body,
        expectations=expectations,
    )
    _validate_report(report, schema)
    _write_once(bundle / PROOF_REPORT_NAME, _json_bytes(report))
    _write_outer_checksums(bundle)
    verified = verify_existing_bundle(bundle_dir=bundle, expectations=expectations)
    if verified != report:
        raise InputError("cache proof changed across final read-only replay")
    return report


def retire_cache_runtime_lock(
    *,
    bundle_dir: Path,
    expectations: ProofExpectations = PRODUCTION_EXPECTATIONS,
) -> dict[str, object]:
    """Remove the empty cache publication lock after every builder has exited."""
    bundle = _require_physical_bundle(bundle_dir)
    _reject_outer_closure(bundle)
    _verify_checksum_closure(bundle / "evidence")
    plan = _validate_plan(bundle / "evidence" / PLAN_NAME, expectations=expectations)
    state = _validate_state(
        bundle / "evidence" / STATE_NAME,
        plan=plan,
        expectations=expectations,
        require_completion=True,
    )
    report = _json_object(
        _read_regular_bytes(bundle / "evidence" / CACHE_REPORT_NAME, label="cache report"),
        label="cache report",
    )
    _validate_cache_artifacts(
        bundle / "cache",
        plan=plan,
        completed=state.completed_by_id,
        require_all=True,
        report=report,
        runtime_lock_expected=True,
    )
    lock_path = bundle / "cache" / "embeddings" / ".publish.lock"
    parent_path = lock_path.parent
    parent_fd = os.open(parent_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptor = -1
    try:
        descriptor = os.open(
            lock_path.name,
            os.O_RDWR | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != 0:
            raise InputError("cache runtime publication lock is not an empty regular file")
        try:
            fcntl = importlib.import_module("fcntl")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InputError("cache runtime publication lock is still held by a builder") from exc
        rebound = os.stat(lock_path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (before.st_dev, before.st_ino) != (rebound.st_dev, rebound.st_ino):
            raise InputError("cache runtime publication lock binding changed")
        os.unlink(lock_path.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)
    if os.path.lexists(lock_path):
        raise InputError("cache runtime publication lock remained after retirement")
    return {
        "ok": True,
        "retired_path": "cache/embeddings/.publish.lock",
        "retirement_scope": "runtime coordination file excluded from immutable publication",
    }


def verify_existing_bundle(
    *,
    bundle_dir: Path,
    expectations: ProofExpectations = PRODUCTION_EXPECTATIONS,
) -> dict[str, object]:
    """Read-only replay of one complete outer proof bundle."""
    bundle = _require_physical_bundle(bundle_dir)
    _verify_checksum_closure(bundle)
    _require_complete_top_level(bundle)
    runtime = validate_runtime_identity(bundle / RUNTIME_COPY_NAME)
    del runtime
    bundled_schema_body = _read_regular_bytes(
        bundle / SCHEMA_COPY_NAME,
        label="published cache H200 proof schema",
    )
    committed_schema_body = _read_regular_bytes(
        DEFAULT_SCHEMA_PATH,
        label="committed cache H200 proof schema",
    )
    if bundled_schema_body != committed_schema_body:
        raise InputError("published cache H200 proof schema differs from the committed schema")
    schema = _validate_schema_bytes(bundled_schema_body)
    trace = _validate_trace_bundle(bundle / "trace", expectations=expectations)
    plan = _validate_plan(bundle / "evidence" / PLAN_NAME, expectations=expectations)
    state = _validate_state(
        bundle / "evidence" / STATE_NAME,
        plan=plan,
        expectations=expectations,
        require_completion=True,
    )
    partial_plan = _validate_plan(bundle / ATTEMPT_PLAN_NAME, expectations=expectations)
    partial_state = _validate_state(
        bundle / ATTEMPT_STATE_NAME,
        plan=partial_plan,
        expectations=expectations,
        require_completion=False,
    )
    _require_proof_inventory(bundle, partial_state=partial_state)
    if partial_plan.body != plan.body:
        raise InputError("published resumed plan differs from the interruption snapshot")
    _validate_interruption_capture(bundle, partial_plan=partial_plan, partial_state=partial_state)
    termination = _validate_interruption_termination(
        bundle,
        partial_plan=partial_plan,
        partial_state=partial_state,
    )
    for shard_id, partial_entry in partial_state.completed_by_id.items():
        if state.completed_by_id.get(shard_id) != partial_entry:
            raise InputError(
                "published interrupted shard entry is not semantically identical after resume"
            )
    _validate_completed_evidence(
        bundle,
        plan=plan,
        state=state,
        partial=partial_state,
        expectations=expectations,
    )
    attempt_log_body, attempt_events = _read_log(
        bundle / ATTEMPT_LOG_NAME,
        label="captured attempt-1 cache log",
    )
    resume_log_body, resume_events = _read_log(
        bundle / RESUME_LOG_NAME,
        label="captured resume cache log",
    )
    if any(event.get("event") == "data.cache.build.end" for event in attempt_events):
        raise InputError("published attempt-1 log contains a build.end event")
    end_events = [event for event in resume_events if event.get("event") == "data.cache.build.end"]
    if len(end_events) != 1:
        raise InputError("published resume log must contain exactly one build.end event")
    _validate_resume_end_event(end_events[0], state=state, partial=partial_state)
    hardware = _json_object(
        _read_regular_bytes(bundle / HARDWARE_COPY_NAME, label="published H200 receipt"),
        label="published H200 receipt",
    )
    report = _json_object(
        _read_regular_bytes(bundle / PROOF_REPORT_NAME, label="cache H200 proof report"),
        label="cache H200 proof report",
    )
    producer = _mapping(report.get("producer"), label="proof report producer")
    source_commit = _commit(producer.get("git_commit"), label="producer.git_commit")
    container_image = _container(
        producer.get("declared_container_image"),
        label="producer.declared_container_image",
    )
    hardware = _validate_hardware_receipt(
        hardware,
        source_commit=source_commit,
        container_image=container_image,
    )
    _validate_plan_hardware_binding(plan, hardware=hardware)
    trace_binding = _mapping(report.get("trace"), label="proof report trace")
    expected = _derive_report(
        bundle=bundle,
        repository=cast(str, trace_binding.get("repository")),
        revision=cast(str, trace_binding.get("revision")),
        artifact_path=cast(str, trace_binding.get("artifact_path")),
        source_commit=source_commit,
        container_image=container_image,
        interrupted_exit_code=cast(
            int,
            _mapping(termination["process"], label="termination process")["shell_exit_code"],
        ),
        trace=trace,
        plan=plan,
        state=state,
        partial=partial_state,
        hardware=hardware,
        attempt_log_body=attempt_log_body,
        resume_log_body=resume_log_body,
        expectations=expectations,
    )
    _validate_report(report, schema)
    if report != expected:
        raise InputError(
            "cache H200 proof report does not match deterministic replay",
            details={
                "expected_sha256": canonical_json_sha256(expected),
                "observed_sha256": canonical_json_sha256(report),
            },
        )
    return report


def _validate_trace_bundle(
    trace_dir: Path,
    *,
    expectations: ProofExpectations,
) -> dict[str, object]:
    expected_names = {
        TRACE_REQUESTS_NAME,
        TRACE_CONFIG_NAME,
        TRACE_REPORT_NAME,
        TRACE_SCHEMA_NAME,
        CHECKSUMS_NAME,
    }
    _verify_checksum_closure(trace_dir, expected_names=expected_names)
    requests = _read_regular_bytes(trace_dir / TRACE_REQUESTS_NAME, label="training trace requests")
    if _identity(TRACE_REQUESTS_NAME, requests) != {
        "path": TRACE_REQUESTS_NAME,
        "sha256": expectations.request_sha256,
        "size_bytes": expectations.request_size_bytes,
    }:
        raise InputError("training trace request bytes do not match the exact v0.3 contract")
    if requests.count(b"\n") != expectations.request_rows or not requests.endswith(b"\n"):
        raise InputError("training trace request row count does not match the exact contract")
    config = _read_regular_bytes(trace_dir / TRACE_CONFIG_NAME, label="training trace config")
    if sha256_bytes(config) != expectations.training_config_sha256:
        raise InputError("training trace config bytes do not match the exact v0.3 contract")
    schema = _json_object(
        _read_regular_bytes(trace_dir / TRACE_SCHEMA_NAME, label="training trace schema"),
        label="training trace schema",
    )
    _validator_type().check_schema(schema)
    report = _json_object(
        _read_regular_bytes(trace_dir / TRACE_REPORT_NAME, label="training trace report"),
        label="training trace report",
    )
    errors = sorted(
        _validator_type()(schema).iter_errors(report),
        key=lambda error: tuple(error.absolute_path),
    )
    if errors:
        raise InputError("training trace report does not satisfy its bundled schema")
    trace = _mapping(report.get("trace"), label="training trace report.trace")
    training = _mapping(report.get("training"), label="training trace report.training")
    request_identity = _mapping(trace.get("requests"), label="training trace requests identity")
    if (
        trace.get("request_rows") != expectations.request_rows
        or request_identity != _identity(TRACE_REQUESTS_NAME, requests)
        or training.get("batch_size") != expectations.batch_size
        or training.get("state_contract_version") != _RUNTIME_IDENTITY["state_contract_version"]
        or training.get("encoder_revision") != _RUNTIME_IDENTITY["revision"]
        or training.get("pool_type") != "centered_mean"
        or training.get("pool_radius") != 8
        or training.get("normalize") is not True
    ):
        raise InputError("training trace report does not bind the exact corrected cache contract")
    return {
        "requests": _identity(TRACE_REQUESTS_NAME, requests),
        "config": _identity(TRACE_CONFIG_NAME, config),
        "report": _identity(
            TRACE_REPORT_NAME,
            _read_regular_bytes(trace_dir / TRACE_REPORT_NAME, label="training trace report"),
        ),
        "checksums": _file_identity(trace_dir / CHECKSUMS_NAME, CHECKSUMS_NAME),
    }


def _validate_plan(path: Path, *, expectations: ProofExpectations) -> _PlanFacts:
    body = _read_regular_bytes(path, label="cache build plan")
    payload = _json_object(body, label="cache build plan")
    required = {
        "schema_version",
        "generated_by",
        "plan_identity",
        "requests",
        "encoder",
        "created_at_ns",
        "execution",
        "input_artifacts",
        "sharding",
        "shards",
        "claim_boundary",
    }
    if set(payload) != required:
        raise InputError("cache build plan has an invalid closed schema")
    requests = _mapping(payload["requests"], label="cache plan requests")
    if requests != {
        "sha256": expectations.request_sha256,
        "size_bytes": expectations.request_size_bytes,
        "input_rows": expectations.request_rows,
        "unique_cache_keys": expectations.unique_cache_keys,
        "duplicate_rows": expectations.duplicate_rows,
    }:
        raise InputError("cache build plan request cardinalities or bytes drifted")
    encoder = _mapping(payload["encoder"], label="cache plan encoder")
    execution = _mapping(payload["execution"], label="cache plan execution")
    hardware = _mapping(execution.get("hardware"), label="cache plan hardware")
    sharding = _mapping(payload["sharding"], label="cache plan sharding")
    runtime_identity = _artifact_identity(
        encoder.get("runtime_identity"), label="cache plan runtime identity"
    )
    resolved_config = _artifact_identity(
        execution.get("resolved_config"), label="cache plan resolved config"
    )
    input_artifacts = payload.get("input_artifacts")
    if not isinstance(input_artifacts, list):
        raise InputError("cache plan input_artifacts must be a list")
    for index, identity in enumerate(input_artifacts):
        _artifact_identity(identity, label=f"cache plan input artifact {index}")
    if (
        set(encoder)
        != {
            "id",
            "cache_namespace",
            "hash",
            "state_layer",
            "pool_type",
            "pool_radius",
            "dtype",
            "normalize",
            "runtime_identity",
        }
        or set(execution) != {"batch_size", "hardware", "resolved_config", "timing_scope"}
        or set(hardware) != {"description", "encoder_device"}
        or set(sharding) != {"rows_per_shard", "planned_shards", "ordering"}
    ):
        raise InputError("cache build plan nested schemas are not closed")
    del runtime_identity, resolved_config
    plan_identity = payload.get("plan_identity")
    expected_namespace = f"/carbon::plan::{plan_identity}"
    if (
        payload.get("schema_version") != "1.3.0"
        or payload.get("generated_by") != "geno_lewm.encoder.cache_build"
        or not isinstance(plan_identity, str)
        or _SHA256.fullmatch(plan_identity) is None
        or payload.get("created_at_ns") != expectations.created_at_ns
        or encoder.get("id") != "/carbon"
        or encoder.get("cache_namespace") != expected_namespace
        or encoder.get("hash") != _RUNTIME_IDENTITY["runtime_hash"]
        or encoder.get("state_layer") != 20
        or encoder.get("pool_type") != "centered_mean"
        or encoder.get("pool_radius") != 8
        or encoder.get("dtype") != "bf16"
        or encoder.get("normalize") is not False
        or execution.get("batch_size") != expectations.batch_size
        or hardware.get("encoder_device") != "cuda"
        or not isinstance(hardware.get("description"), str)
        or "H200" not in cast(str, hardware.get("description"))
        or sharding.get("rows_per_shard") != expectations.rows_per_shard
        or sharding.get("planned_shards") != len(expectations.shard_row_counts)
        or sharding.get("ordering") != "chrom,pool_type,pool_radius,cache_key"
        or execution.get("timing_scope")
        != (
            "wall time inside encoder.encode_batch calls only; excludes planning, "
            "record materialization, Parquet publication, indexing, and verification"
        )
        or payload.get("claim_boundary")
        != {
            "scope": "the exact finite cache_build_requests.jsonl artifact only",
            "ten_percent_corpus_completed": False,
            "twenty_four_hour_target_evaluated": False,
        }
    ):
        raise InputError("cache build plan is not the exact H200 corrected-Carbon contract")
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != len(expectations.shard_row_counts):
        raise InputError("cache build plan shard cardinality drifted")
    shards_by_id: dict[str, Mapping[str, object]] = {}
    keys: list[WindowCacheKey] = []
    request_ids: set[str] = set()
    observed_row_counts: list[int] = []
    observed_paths: list[str] = []
    observed_stride_blocks: list[int] = []
    for raw_shard in raw_shards:
        shard = _mapping(raw_shard, label="cache plan shard")
        if set(shard) != {"shard_id", "path", "contig", "stride_block", "rows"}:
            raise InputError("cache build plan shard has an invalid schema")
        shard_id = _text(shard.get("shard_id"), label="cache plan shard_id")
        relative = _safe_relative(shard.get("path"), label="cache plan shard path")
        stride_block = shard.get("stride_block")
        if isinstance(stride_block, bool) or not isinstance(stride_block, int) or stride_block < 0:
            raise InputError("cache build plan stride_block must be a non-negative integer")
        expected_path = shard_path_for(
            Path(),
            encoder_id=expected_namespace,
            state_layer=20,
            pool_type="centered_mean",
            pool_radius=8,
            contig="22",
            stride_block=stride_block,
            encoder_hash=bytes.fromhex(cast(str, _RUNTIME_IDENTITY["runtime_hash"])[7:]),
            dtype="bf16",
        ).as_posix()
        if not relative.startswith("embeddings/") or not relative.endswith(".parquet"):
            raise InputError("cache build plan shard path is outside the cache namespace")
        if shard_id in shards_by_id or shard.get("contig") != "22" or relative != expected_path:
            raise InputError("cache build plan shard identity or contig drifted")
        rows = shard.get("rows")
        if not isinstance(rows, list) or not rows:
            raise InputError("cache build plan shard rows must be non-empty")
        observed_row_counts.append(len(rows))
        observed_paths.append(relative)
        observed_stride_blocks.append(stride_block)
        for raw_row in rows:
            row = _mapping(raw_row, label="cache plan row")
            if set(row) != {"representative_request_id", "request_ids", "key", "record"}:
                raise InputError("cache build plan row has an invalid schema")
            key = _key_from_payload(row.get("key"))
            record = _mapping(row.get("record"), label="cache plan row record")
            ids = row.get("request_ids")
            if (
                key.pool_type != "centered_mean"
                or key.pool_radius != 8
                or key.encoder_hash.hex() != cast(str, _RUNTIME_IDENTITY["runtime_hash"])[7:]
                or key.state_layer != 20
                or key.dtype != "bf16"
                or key.center_token is None
                or record.get("chrom") != "22"
                or not isinstance(ids, list)
                or not ids
                or any(not isinstance(item, str) or not item for item in ids)
                or row.get("representative_request_id") not in ids
                or ids != sorted(cast(list[str], ids))
                or len(set(cast(list[str], ids))) != len(ids)
                or set(record) != {"chrom", "start_bp", "end_bp", "untargeted"}
                or not isinstance(record.get("start_bp"), int)
                or isinstance(record.get("start_bp"), bool)
                or not isinstance(record.get("end_bp"), int)
                or isinstance(record.get("end_bp"), bool)
                or cast(int, record.get("end_bp")) <= cast(int, record.get("start_bp"))
                or record.get("untargeted") is not False
            ):
                raise InputError("cache build plan row drifted from the chr22 centered contract")
            for request_id in ids:
                text = _text(request_id, label="cache plan request_id")
                if text in request_ids:
                    raise InputError("cache build plan contains a duplicate request_id")
                request_ids.add(text)
            keys.append(key)
        if shard_id != canonical_json_sha256(
            {"path": relative, "keys": [row["key"] for row in rows]}
        ):
            raise InputError("cache build plan shard identity is not deterministic")
        shards_by_id[shard_id] = shard
    if sorted(observed_stride_blocks) != list(range(len(expectations.shard_row_counts))):
        raise InputError("cache build plan stride blocks are not exact and contiguous")
    row_counts_by_stride = dict(zip(observed_stride_blocks, observed_row_counts, strict=True))
    if (
        tuple(row_counts_by_stride[index] for index in range(len(expectations.shard_row_counts)))
        != expectations.shard_row_counts
    ):
        raise InputError("cache build plan shard row distribution drifted")
    if len(set(observed_paths)) != len(observed_paths) or observed_paths != sorted(observed_paths):
        raise InputError("cache build plan shards are not in canonical path order")
    if len(keys) != expectations.unique_cache_keys or len(set(keys)) != len(keys):
        raise InputError("cache build plan logical keys are not exact and unique")
    if len(request_ids) != expectations.request_rows:
        raise InputError("cache build plan request IDs do not cover the exact trace")
    return _PlanFacts(
        payload=payload,
        body=body,
        shards_by_id=shards_by_id,
        keys=tuple(keys),
    )


def _validate_state(
    path: Path,
    *,
    plan: _PlanFacts,
    expectations: ProofExpectations,
    require_completion: bool,
) -> _StateFacts:
    body = _read_regular_bytes(path, label="cache build state")
    payload = _json_object(body, label="cache build state")
    if set(payload) != {
        "schema_version",
        "generated_by",
        "plan_sha256",
        "completed_shards",
        "completion",
    }:
        raise InputError("cache build state has an invalid closed schema")
    if (
        payload.get("schema_version") != "1.3.0"
        or payload.get("generated_by") != "geno_lewm.encoder.cache_build"
        or payload.get("plan_sha256") != sha256_bytes(plan.body)
    ):
        raise InputError("cache build state is not bound to the exact plan")
    raw_completed = payload.get("completed_shards")
    if not isinstance(raw_completed, list):
        raise InputError("cache build state completed_shards must be a list")
    completed: dict[str, Mapping[str, object]] = {}
    rows = 0
    calls = 0
    for raw_entry in raw_completed:
        entry = _mapping(raw_entry, label="cache build completed shard")
        if set(entry) != {
            "plan_shard_id",
            "execution_shard_id",
            "path",
            "row_keys",
            "sha256",
            "size_bytes",
            "row_count",
            "origin",
            "encoded_rows",
            "encode_batch_calls",
            "encode_batch_seconds",
        }:
            raise InputError("cache build completed shard has an invalid schema")
        shard_id = _text(entry.get("plan_shard_id"), label="state plan_shard_id")
        shard = plan.shards_by_id.get(shard_id)
        if shard is None or shard_id in completed:
            raise InputError("cache build state names an unknown or duplicate shard")
        plan_rows = cast(list[Mapping[str, object]], shard["rows"])
        expected_keys = [row["key"] for row in plan_rows]
        expected_execution_shard_id = canonical_json_sha256(
            {
                "plan_shard_id": shard_id,
                "path": shard["path"],
                "keys": expected_keys,
            }
        )
        row_count = len(plan_rows)
        batch_calls = math.ceil(row_count / expectations.batch_size)
        seconds = entry.get("encode_batch_seconds")
        if (
            entry.get("execution_shard_id") != expected_execution_shard_id
            or entry.get("path") != shard.get("path")
            or entry.get("row_keys") != expected_keys
            or entry.get("row_count") != row_count
            or entry.get("origin") != "encoded"
            or entry.get("encoded_rows") != row_count
            or entry.get("encode_batch_calls") != batch_calls
            or not isinstance(seconds, int | float)
            or isinstance(seconds, bool)
            or not math.isfinite(float(seconds))
            or float(seconds) < 0
            or not isinstance(entry.get("size_bytes"), int)
            or cast(int, entry.get("size_bytes")) <= 0
            or not isinstance(entry.get("sha256"), str)
            or _SHA256.fullmatch(cast(str, entry.get("sha256"))) is None
        ):
            raise InputError("cache build completed shard metadata drifted")
        completed[shard_id] = entry
        rows += row_count
        calls += batch_calls
    completion = payload.get("completion")
    if require_completion and not isinstance(completion, Mapping):
        raise InputError("completed cache proof is missing completion state")
    if not require_completion and completion is not None:
        raise InputError("interrupted cache state must have null completion")
    if require_completion:
        completion_payload = _mapping(completion, label="cache completion")
        if set(completion_payload) != {
            "encoded_rows",
            "encoded_shards",
            "resumed_rows",
            "reused_rows",
            "resolved_unique_rows",
            "planned_shards",
            "invocation_elapsed_seconds",
            "run_id",
        }:
            raise InputError("cache completion state has an invalid closed schema")
        elapsed = completion_payload.get("invocation_elapsed_seconds")
        run_id = completion_payload.get("run_id")
        if (
            not isinstance(elapsed, int | float)
            or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
            or (run_id is not None and (not isinstance(run_id, str) or not run_id))
        ):
            raise InputError("cache completion timing or run identity is invalid")
    return _StateFacts(
        payload=payload,
        body=body,
        completed_by_id=completed,
        completed_rows=rows,
        encode_batch_calls=calls,
    )


def _validate_partial_evidence(
    bundle: Path,
    *,
    plan: _PlanFacts,
    state: _StateFacts,
    expectations: ProofExpectations,
) -> None:
    evidence = bundle / "evidence"
    expected_evidence = {
        TRACE_REQUESTS_NAME,
        PLAN_NAME,
        STATE_NAME,
        "resolved_config.json",
        "encoder_runtime_identity.json",
        "inputs/encoder_config.yaml",
        "inputs/encoder_runtime_identity_source.json",
    }
    observed = _regular_inventory(evidence)
    if observed != expected_evidence:
        raise InputError(
            "interrupted cache evidence inventory is not exact",
            details={"observed": sorted(observed)},
        )
    for forbidden in (CACHE_REPORT_NAME, CHECKSUMS_NAME):
        if (evidence / forbidden).exists() or (evidence / forbidden).is_symlink():
            raise InputError("interrupted cache evidence is already complete")
    _validate_evidence_inputs(bundle, plan=plan, expectations=expectations)
    _validate_cache_artifacts(
        bundle / "cache",
        plan=plan,
        completed=state.completed_by_id,
        require_all=False,
    )


def _validate_completed_evidence(
    bundle: Path,
    *,
    plan: _PlanFacts,
    state: _StateFacts,
    partial: _StateFacts,
    expectations: ProofExpectations,
) -> None:
    expected_evidence = {
        TRACE_REQUESTS_NAME,
        PLAN_NAME,
        STATE_NAME,
        "resolved_config.json",
        "encoder_runtime_identity.json",
        "inputs/encoder_config.yaml",
        "inputs/encoder_runtime_identity_source.json",
        CACHE_REPORT_NAME,
        CHECKSUMS_NAME,
    }
    observed_evidence = _regular_inventory(bundle / "evidence")
    if observed_evidence != expected_evidence:
        raise InputError(
            "completed cache evidence inventory is not exact",
            details={
                "expected": sorted(expected_evidence),
                "observed": sorted(observed_evidence),
            },
        )
    _verify_checksum_closure(bundle / "evidence")
    _validate_evidence_inputs(bundle, plan=plan, expectations=expectations)
    if len(state.completed_by_id) != len(expectations.shard_row_counts):
        raise InputError("resumed cache did not complete every planned shard")
    if state.completed_rows != expectations.unique_cache_keys:
        raise InputError("resumed cache completed-row cardinality drifted")
    if state.encode_batch_calls != expectations.durable_completed_shard_encode_batch_calls:
        raise InputError("durable completed-shard encode_batch call cardinality drifted")
    completion = _mapping(state.payload.get("completion"), label="cache completion")
    partial_count = len(partial.completed_by_id)
    expected_completion = {
        "encoded_rows": expectations.unique_cache_keys - partial.completed_rows,
        "encoded_shards": len(expectations.shard_row_counts) - partial_count,
        "resumed_rows": partial.completed_rows,
        "reused_rows": 0,
        "resolved_unique_rows": expectations.unique_cache_keys,
        "planned_shards": len(expectations.shard_row_counts),
    }
    for field, expected in expected_completion.items():
        if completion.get(field) != expected:
            raise InputError(
                "resumed cache completion accounting drifted",
                details={"field": field, "expected": expected, "observed": completion.get(field)},
            )
    report = _json_object(
        _read_regular_bytes(bundle / "evidence" / CACHE_REPORT_NAME, label="cache build report"),
        label="cache build report",
    )
    if report.get("ok") is not True or report.get("requests") != plan.payload.get("requests"):
        raise InputError("cache build report is not bound to the exact plan")
    build = _mapping(report.get("build"), label="cache report build")
    for field, expected in {
        "planned_shards": len(expectations.shard_row_counts),
        "completed_shards": len(expectations.shard_row_counts),
        **expected_completion,
    }.items():
        if build.get(field) != expected:
            raise InputError("cache build report accounting differs from durable completion")
    _validate_cache_artifacts(
        bundle / "cache",
        plan=plan,
        completed=state.completed_by_id,
        require_all=True,
        report=report,
    )
    _validate_cache_report(
        bundle,
        report=report,
        plan=plan,
        state=state,
    )


def _validate_cache_report(
    bundle: Path,
    *,
    report: Mapping[str, object],
    plan: _PlanFacts,
    state: _StateFacts,
) -> None:
    completion = _mapping(state.payload.get("completion"), label="cache completion")
    requests = _mapping(plan.payload.get("requests"), label="cache plan requests")
    encoder = _mapping(plan.payload.get("encoder"), label="cache plan encoder")
    execution = _mapping(plan.payload.get("execution"), label="cache plan execution")
    hardware = _mapping(execution.get("hardware"), label="cache plan hardware")
    sharding = _mapping(plan.payload.get("sharding"), label="cache plan sharding")
    measured_rows = sum(
        cast(int, entry["encoded_rows"])
        for entry in state.completed_by_id.values()
        if entry.get("encode_batch_seconds") is not None
    )
    measured_seconds = sum(
        cast(float, entry["encode_batch_seconds"])
        for entry in state.completed_by_id.values()
        if entry.get("encode_batch_seconds") is not None
    )
    measured_rate = None if measured_seconds <= 0.0 else measured_rows / measured_seconds
    cache_artifacts = _mapping(report.get("cache_artifacts"), label="cache report artifacts")
    expected = {
        "schema_version": "1.3.0",
        "generated_by": "geno_lewm.encoder.cache_build",
        "ok": True,
        "run_id": completion["run_id"],
        "requests": dict(requests),
        "plan": _file_identity(bundle / "evidence" / PLAN_NAME, PLAN_NAME),
        "configuration": {
            "batch_size": execution["batch_size"],
            "rows_per_shard": sharding["rows_per_shard"],
            "created_at_ns": plan.payload["created_at_ns"],
            "cache_namespace": encoder["cache_namespace"],
            "hardware": dict(hardware),
            "resolved_config": dict(
                _mapping(execution["resolved_config"], label="cache plan resolved config")
            ),
            "encoder_runtime_identity": dict(
                _mapping(encoder["runtime_identity"], label="cache plan runtime identity")
            ),
        },
        "build": {
            "planned_shards": len(plan.shards_by_id),
            "completed_shards": len(plan.shards_by_id),
            "encoded_shards": completion["encoded_shards"],
            "encoded_rows": completion["encoded_rows"],
            "resumed_rows": completion["resumed_rows"],
            "reused_rows": completion["reused_rows"],
            "resolved_unique_rows": completion["resolved_unique_rows"],
        },
        "throughput": {
            "invocation_elapsed_seconds": round(
                max(cast(float, completion["invocation_elapsed_seconds"]), 0.0),
                6,
            ),
            "measured_encoded_rows": measured_rows,
            "measured_encoder_seconds": round(measured_seconds, 6),
            "measured_encoded_rows_per_second": measured_rate,
            "measurement_scope": (
                "wall time inside encoder.encode_batch calls for evidence-owned encoded rows; "
                "excludes planning, Python record materialization, Parquet publication, indexing, "
                "verification, and reused shared-cache rows"
            ),
            "measurement_hardware": dict(hardware),
            "measurement_batch_size": execution["batch_size"],
            "ten_percent_24h_target_evaluated": False,
        },
        "cache_contract": {
            "schema_version": "3.0.0",
            "storage_dtype": "fp32",
            "logical_dtype": encoder["dtype"],
            "normalized_states_persisted": False,
            "deduplication_key_includes_center_token": True,
        },
        "cache_artifacts": dict(cache_artifacts),
        "evidence_artifacts": {
            "requests": _file_identity(
                bundle / "evidence" / TRACE_REQUESTS_NAME,
                TRACE_REQUESTS_NAME,
            ),
            "plan": _file_identity(bundle / "evidence" / PLAN_NAME, PLAN_NAME),
            "state": _file_identity(bundle / "evidence" / STATE_NAME, STATE_NAME),
            "resolved_config": dict(
                _mapping(execution["resolved_config"], label="cache plan resolved config")
            ),
            "encoder_runtime_identity": dict(
                _mapping(encoder["runtime_identity"], label="cache plan runtime identity")
            ),
            "inputs": [
                dict(_mapping(item, label="cache plan input artifact"))
                for item in cast(list[object], plan.payload["input_artifacts"])
            ],
        },
        "progress_events": [
            "data.cache.build.start",
            "data.cache.build.progress",
            "data.shard.write",
            "data.cache.build.end",
        ],
        "claim_boundary": {
            "finite_request_artifact_completed": True,
            "ten_percent_corpus_completed": False,
            "twenty_four_hour_target_evaluated": False,
            "model_quality_evaluated": False,
            "statement": (
                "This report proves construction and byte-level verification of the exact finite "
                "request artifact only; it does not establish corpus coverage, the 24-hour target, "
                "model quality, or clinical validity."
            ),
        },
    }
    if report != expected:
        raise InputError(
            "cache build report does not match deterministic completion evidence",
            details={
                "expected_sha256": canonical_json_sha256(expected),
                "observed_sha256": canonical_json_sha256(report),
            },
        )


def _validate_evidence_inputs(
    bundle: Path,
    *,
    plan: _PlanFacts,
    expectations: ProofExpectations,
) -> None:
    evidence = bundle / "evidence"
    trace = bundle / "trace"
    request_body = _read_regular_bytes(
        evidence / TRACE_REQUESTS_NAME, label="cache evidence requests"
    )
    if request_body != _read_regular_bytes(
        trace / TRACE_REQUESTS_NAME,
        label="trace requests",
    ):
        raise InputError("cache evidence request copy differs from the exact trace")
    config_body = _read_regular_bytes(
        evidence / "inputs/encoder_config.yaml", label="cache encoder config"
    )
    if config_body != _read_regular_bytes(
        trace / TRACE_CONFIG_NAME,
        label="trace config",
    ):
        raise InputError("cache evidence config differs from the exact trace config")
    validate_runtime_identity(evidence / "inputs/encoder_runtime_identity_source.json")
    canonical_runtime = _json_object(
        _read_regular_bytes(
            evidence / "encoder_runtime_identity.json",
            label="canonical cache runtime identity",
        ),
        label="canonical cache runtime identity",
    )
    if canonical_runtime != _RUNTIME_IDENTITY:
        raise InputError("canonical cache runtime identity drifted")
    resolved_config = _json_object(
        _read_regular_bytes(evidence / "resolved_config.json", label="resolved cache config"),
        label="resolved cache config",
    )
    if resolved_config != _resolved_config_from_yaml(config_body):
        raise InputError("resolved cache config is not derived from the exact training config")
    if (
        sha256_bytes(
            _read_regular_bytes(evidence / "inputs/encoder_config.yaml", label="cache config")
        )
        != expectations.training_config_sha256
    ):
        raise InputError("cache config identity drifted")
    input_artifacts = plan.payload.get("input_artifacts")
    expected_inputs = [
        _file_identity(
            evidence / "inputs/encoder_config.yaml",
            "inputs/encoder_config.yaml",
        ),
        _file_identity(
            evidence / "inputs/encoder_runtime_identity_source.json",
            "inputs/encoder_runtime_identity_source.json",
        ),
    ]
    encoder = _mapping(plan.payload.get("encoder"), label="cache plan encoder")
    execution = _mapping(plan.payload.get("execution"), label="cache plan execution")
    if (
        not isinstance(input_artifacts, list)
        or input_artifacts != expected_inputs
        or encoder.get("runtime_identity")
        != _file_identity(
            evidence / "encoder_runtime_identity.json",
            "encoder_runtime_identity.json",
        )
        or execution.get("resolved_config")
        != _file_identity(evidence / "resolved_config.json", "resolved_config.json")
    ):
        raise InputError("cache plan input artifact inventory drifted")
    _validate_plan_against_trace(
        plan,
        request_body=request_body,
        expectations=expectations,
    )


def _resolved_config_from_yaml(body: bytes) -> dict[str, object]:
    try:
        text = body.decode("utf-8")
        raw = yaml.safe_load(text) or {}
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise InputError("exact cache training config is not valid UTF-8 YAML") from exc
    try:
        return cast(dict[str, object], config_to_dict(load_config(raw)))
    except Exception as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError("exact cache training config could not be resolved") from exc


def _validate_plan_against_trace(
    plan: _PlanFacts,
    *,
    request_body: bytes,
    expectations: ProofExpectations,
) -> None:
    requests = _parse_trace_requests(request_body)
    if len(requests) != expectations.request_rows:
        raise InputError("cache plan trace request count drifted")
    encoder_hash = bytes.fromhex(cast(str, _RUNTIME_IDENTITY["runtime_hash"])[7:])
    grouped: dict[WindowCacheKey, list[_TraceRequest]] = {}
    for request in requests:
        key = WindowCacheKey(
            window_hash=window_sha256(request.window),
            encoder_hash=encoder_hash,
            state_layer=20,
            pool_type="centered_mean",
            pool_radius=8,
            center_token=1 + request.edit_locus // 6,
            dtype="bf16",
        )
        grouped.setdefault(key, []).append(request)
    expected_rows: list[tuple[tuple[object, ...], dict[str, object]]] = []
    for key, aliases in grouped.items():
        ordered_aliases = sorted(aliases, key=_trace_request_sort_key)
        representative = ordered_aliases[0]
        row: dict[str, object] = {
            "representative_request_id": representative.request_id,
            "request_ids": sorted(alias.request_id for alias in aliases),
            "key": _key_payload(key),
            "record": {
                "chrom": representative.chrom,
                "start_bp": representative.start_bp,
                "end_bp": representative.end_bp,
                "untargeted": False,
            },
        }
        expected_rows.append((_planned_trace_row_sort_key(key, representative), row))
    expected_rows.sort(key=lambda item: item[0])
    logical_rows = [row for _, row in expected_rows]
    if len(logical_rows) != expectations.unique_cache_keys:
        raise InputError("trace-derived corrected Carbon cache key count drifted")

    payload = plan.payload
    encoder = dict(_mapping(payload["encoder"], label="cache plan encoder"))
    encoder.pop("cache_namespace")
    execution = dict(_mapping(payload["execution"], label="cache plan execution"))
    execution.pop("timing_scope")
    identity_payload = {
        "schema_version": payload["schema_version"],
        "requests": payload["requests"],
        "encoder": encoder,
        "created_at_ns": payload["created_at_ns"],
        "execution": execution,
        "input_artifacts": payload["input_artifacts"],
        "sharding": {"rows_per_shard": expectations.rows_per_shard},
        "logical_rows": logical_rows,
    }
    expected_plan_identity = canonical_json_sha256(identity_payload)
    if payload.get("plan_identity") != expected_plan_identity:
        raise InputError("cache build plan identity is not rederived from the exact trace")
    if (
        _mapping(payload["encoder"], label="cache plan encoder").get("cache_namespace")
        != f"/carbon::plan::{expected_plan_identity}"
    ):
        raise InputError("cache build plan namespace is not derived from its exact identity")

    raw_shards = cast(list[Mapping[str, object]], payload["shards"])
    expected_chunks = [
        logical_rows[offset : offset + expectations.rows_per_shard]
        for offset in range(0, len(logical_rows), expectations.rows_per_shard)
    ]
    if len(raw_shards) != len(expected_chunks):
        raise InputError("trace-derived cache plan shard count drifted")
    raw_paths = [
        _safe_relative(shard.get("path"), label="trace-derived cache plan shard path")
        for shard in raw_shards
    ]
    if len(set(raw_paths)) != len(raw_paths) or raw_paths != sorted(raw_paths):
        raise InputError("cache build plan shard order differs from the exact trace")
    shards_by_stride: dict[int, Mapping[str, object]] = {}
    for shard in raw_shards:
        stride_block = shard.get("stride_block")
        if (
            isinstance(stride_block, bool)
            or not isinstance(stride_block, int)
            or stride_block in shards_by_stride
        ):
            raise InputError("cache build plan stride blocks differ from the exact trace")
        shards_by_stride[stride_block] = shard
    if set(shards_by_stride) != set(range(len(expected_chunks))):
        raise InputError("cache build plan stride blocks differ from the exact trace")
    for stride_block, rows in enumerate(expected_chunks):
        if shards_by_stride[stride_block].get("rows") != rows:
            raise InputError("cache build plan rows or aliases differ from the exact trace")


def _parse_trace_requests(body: bytes) -> tuple[_TraceRequest, ...]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError("cache proof requests must be UTF-8 JSONL") from exc
    if not text or not text.endswith("\n"):
        raise InputError("cache proof requests must be non-empty and newline-terminated")
    expected_fields = {"request_id", "chrom", "start_bp", "end_bp", "window", "edit_locus"}
    requests: list[_TraceRequest] = []
    seen: set[str] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise InputError("cache proof requests contain a blank line")
        try:
            raw = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, InputError) as exc:
            raise InputError(
                "cache proof request line is invalid JSON",
                details={"line": line_no},
            ) from exc
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise InputError("cache proof request line has an invalid closed schema")
        request_id = _text(raw["request_id"], label=f"request line {line_no} id")
        if request_id in seen:
            raise InputError("cache proof request IDs are not unique")
        seen.add(request_id)
        chrom = _text(raw["chrom"], label=f"request line {line_no} chrom")
        start_bp = raw["start_bp"]
        end_bp = raw["end_bp"]
        edit_locus = raw["edit_locus"]
        window_raw = raw["window"]
        if (
            chrom != "22"
            or not isinstance(start_bp, int)
            or isinstance(start_bp, bool)
            or start_bp < 0
            or not isinstance(end_bp, int)
            or isinstance(end_bp, bool)
            or not isinstance(edit_locus, int)
            or isinstance(edit_locus, bool)
            or edit_locus < 0
            or not isinstance(window_raw, str)
        ):
            raise InputError("cache proof request differs from the targeted chr22 contract")
        window = canonicalize_dna(window_raw)
        if not window or end_bp - start_bp != len(window) or edit_locus >= len(window):
            raise InputError("cache proof request coordinates or edit locus are invalid")
        requests.append(
            _TraceRequest(
                request_id=request_id,
                chrom=chrom,
                start_bp=start_bp,
                end_bp=end_bp,
                window=window,
                edit_locus=edit_locus,
            )
        )
    return tuple(requests)


def _trace_request_sort_key(request: _TraceRequest) -> tuple[object, ...]:
    return (
        request.chrom,
        request.start_bp,
        request.end_bp,
        request.window,
        request.edit_locus,
        request.request_id,
    )


def _planned_trace_row_sort_key(
    key: WindowCacheKey,
    representative: _TraceRequest,
) -> tuple[object, ...]:
    return (
        key.window_hash.hex(),
        key.encoder_hash.hex(),
        key.state_layer,
        key.pool_type,
        key.pool_radius,
        -1 if key.center_token is None else key.center_token,
        key.dtype,
        *_trace_request_sort_key(representative),
    )


def _key_payload(key: WindowCacheKey) -> dict[str, object]:
    return {
        "window_hash": f"sha256:{key.window_hash.hex()}",
        "encoder_hash": f"sha256:{key.encoder_hash.hex()}",
        "state_layer": key.state_layer,
        "pool_type": key.pool_type,
        "pool_radius": key.pool_radius,
        "center_token": key.center_token,
        "dtype": key.dtype,
    }


def _validate_cache_artifacts(
    cache_dir: Path,
    *,
    plan: _PlanFacts,
    completed: Mapping[str, Mapping[str, object]],
    require_all: bool,
    report: Mapping[str, object] | None = None,
    runtime_lock_expected: bool | None = None,
) -> None:
    expected_paths = {cast(str, plan.shards_by_id[shard_id]["path"]) for shard_id in completed}
    if runtime_lock_expected is None:
        runtime_lock_expected = not require_all
    expected_inventory = {*expected_paths, "embeddings/index.sqlite"}
    if runtime_lock_expected:
        expected_inventory.add("embeddings/.publish.lock")
    observed = _regular_inventory(cache_dir)
    if observed != expected_inventory:
        raise InputError(
            "cache contains a plan-owned shard outside durable state or an unknown file",
            details={"expected": sorted(expected_inventory), "observed": sorted(observed)},
        )
    if any(".pending-publication" in path for path in observed):
        raise InputError("cache contains an unfinished publication marker")
    plan_key_by_payload: dict[str, WindowCacheKey] = {}
    expected_provenance: dict[str, tuple[str, int]] = {}
    expected_report_shards: list[dict[str, object]] = []
    created_at_ns = cast(int, plan.payload["created_at_ns"])
    for shard_id, entry in completed.items():
        relative = cast(str, entry["path"])
        plan_shard = plan.shards_by_id[shard_id]
        plan_rows = cast(list[Mapping[str, object]], plan_shard["rows"])
        inspection = inspect_cache_shard(cache_dir, relative)
        if (
            inspection.sha256 != entry.get("sha256")
            or inspection.size_bytes != entry.get("size_bytes")
            or len(inspection.records) != entry.get("row_count")
        ):
            raise InputError("cache Parquet bytes differ from durable shard state")
        expected_keys = [_key_from_payload(raw) for raw in cast(list[object], entry["row_keys"])]
        if len(plan_rows) != len(inspection.records) or len(expected_keys) != len(plan_rows):
            raise InputError("cache Parquet row cardinality differs from its immutable plan shard")
        request_rows: list[dict[str, object]] = []
        for offset, (record, row, expected_key) in enumerate(
            zip(inspection.records, plan_rows, expected_keys, strict=True)
        ):
            planned_record = _mapping(row.get("record"), label="cache plan row record")
            if (
                record.key != expected_key
                or record.key != _key_from_payload(row.get("key"))
                or record.chrom != planned_record.get("chrom")
                or record.start_bp != planned_record.get("start_bp")
                or record.end_bp != planned_record.get("end_bp")
                or record.untargeted != planned_record.get("untargeted")
                or record.created_at != created_at_ns
                or record.schema_version != "3.0.0"
                or record.dtype != expected_key.dtype
            ):
                raise InputError("cache Parquet logical row differs from the immutable plan")
            canonical = _canonical_key(expected_key)
            if canonical in plan_key_by_payload:
                raise InputError("cache Parquet inventory contains a duplicate logical key")
            plan_key_by_payload[canonical] = expected_key
            expected_provenance[canonical] = (relative, offset)
            request_rows.append(
                {
                    "key": row["key"],
                    "row_offset": offset,
                    "created_at_ns": created_at_ns,
                    "request_ids": row["request_ids"],
                }
            )
        expected_report_shards.append(
            {
                "path": relative,
                "sha256": inspection.sha256,
                "size_bytes": inspection.size_bytes,
                "row_count": len(inspection.records),
                "request_rows": request_rows,
            }
        )
    keys = tuple(plan_key_by_payload.values())
    provenances = resolve_cache_provenances(cache_dir, keys, policy="require_v3")
    if any(item is None for item in provenances):
        raise InputError("request-scoped cache index is missing a durable logical key")
    for key, raw_provenance in zip(keys, provenances, strict=True):
        assert raw_provenance is not None
        relative = raw_provenance.shard_path.absolute().relative_to(cache_dir).as_posix()
        if (
            raw_provenance.cache_schema_version != "3.0.0"
            or raw_provenance.physical_encoding != "fixed_size_list<float32>"
            or expected_provenance[_canonical_key(key)] != (relative, raw_provenance.row_offset)
        ):
            raise InputError("request-scoped cache index differs from decoded Parquet order")
    if not require_all:
        completed_keys = set(plan_key_by_payload)
        pending_keys = tuple(key for key in plan.keys if _canonical_key(key) not in completed_keys)
        pending_provenances = resolve_cache_provenances(
            cache_dir,
            pending_keys,
            policy="require_v3",
        )
        if any(item is not None for item in pending_provenances):
            raise InputError("partial cache index exposes a plan key outside durable state")
        return
    if report is None:
        raise AssertionError("completed cache validation requires its report")
    artifacts = _mapping(report.get("cache_artifacts"), label="cache report artifacts")
    expected_artifacts = {
        "index": {
            "path": "embeddings/index.sqlite",
            "schema_version": 4,
            "verified_logical_keys": len(keys),
            "identity_scope": (
                "request-scoped logical-key mappings; mutable shared index bytes are excluded"
            ),
        },
        "shards": sorted(expected_report_shards, key=lambda item: cast(str, item["path"])),
    }
    if artifacts != expected_artifacts:
        raise InputError("cache report artifacts do not match decoded Parquet and index evidence")


def _validate_interruption_capture(
    bundle: Path,
    *,
    partial_plan: _PlanFacts,
    partial_state: _StateFacts,
) -> dict[str, object]:
    capture_body = _read_regular_bytes(bundle / ATTEMPT_CAPTURE_NAME, label="interruption capture")
    capture = _json_object(capture_body, label="interruption capture")
    process = _mapping(capture.get("process"), label="interruption process")
    partial = _mapping(capture.get("partial"), label="interruption partial")
    log_body = _read_regular_bytes(bundle / ATTEMPT_LOG_NAME, label="interruption log")
    argv = process.get("argv")
    if (
        set(capture) != {"schema_version", "generated_by", "process", "partial"}
        or set(partial)
        != {
            "completed_shards",
            "completed_rows",
            "durable_completed_shard_encode_batch_calls",
            "plan",
            "state",
            "log",
            "cache_artifacts",
        }
        or set(process)
        != {
            "pid",
            "observed_state",
            "argv",
            "comm",
            "child_pids",
            "termination_sequence",
            "expected_shell_exit_code",
        }
        or (
            capture.get("schema_version") != "geno-lewm.v03-cache-h200-interruption-capture.v1"
            or capture.get("generated_by") != GENERATED_BY
            or process.get("observed_state") != "stopped"
            or process.get("termination_sequence") != "SIGTERM_while_stopped_then_SIGCONT"
            or process.get("expected_shell_exit_code") != 143
            or type(process.get("pid")) is not int
            or cast(int, process.get("pid")) <= 1
            or not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) or not item for item in argv)
            or not any(
                Path(cast(str, item)).name == "geno-lewm-cache-windows"
                for item in cast(list[object], argv)
            )
            or not isinstance(process.get("comm"), str)
            or not process.get("comm")
            or process.get("child_pids") != []
            or partial.get("completed_shards") != len(partial_state.completed_by_id)
            or partial.get("completed_rows") != partial_state.completed_rows
            or partial.get("durable_completed_shard_encode_batch_calls")
            != partial_state.encode_batch_calls
            or partial.get("plan") != _identity(PLAN_NAME, partial_plan.body)
            or partial.get("state") != _identity(STATE_NAME, partial_state.body)
            or partial.get("log") != _identity("cache-build.jsonl", log_body)
            or not isinstance(partial.get("cache_artifacts"), list)
        )
    ):
        raise InputError("interruption capture does not bind the stopped durable snapshot")
    captured_artifacts = partial.get("cache_artifacts")
    assert isinstance(captured_artifacts, list)
    expected_paths = {
        "cache/embeddings/.publish.lock",
        *(f"cache/{relative}" for relative in _partial_cache_snapshot_paths(partial_state)),
    }
    captured_by_path: dict[str, Mapping[str, object]] = {}
    for raw_identity in captured_artifacts:
        identity = _mapping(raw_identity, label="captured partial cache identity")
        path = _text(identity.get("path"), label="captured partial cache path")
        if set(identity) != {"path", "sha256", "size_bytes"} or path in captured_by_path:
            raise InputError("captured partial cache identity inventory is invalid")
        captured_by_path[path] = identity
    if set(captured_by_path) != expected_paths:
        raise InputError("captured partial cache identity inventory is not exact")
    if captured_by_path["cache/embeddings/.publish.lock"] != {
        "path": "cache/embeddings/.publish.lock",
        "sha256": sha256_bytes(b""),
        "size_bytes": 0,
    }:
        raise InputError("captured runtime publication lock is not the expected empty file")
    for relative in _partial_cache_snapshot_paths(partial_state):
        archived = _file_identity(bundle / ATTEMPT_CACHE_DIR / relative, f"cache/{relative}")
        if captured_by_path[f"cache/{relative}"] != archived:
            raise InputError("archived partial cache bytes differ from the stopped capture")
    _validate_archived_partial_cache(
        bundle / ATTEMPT_CACHE_DIR,
        plan=partial_plan,
        state=partial_state,
    )
    return capture


def _validate_interruption_termination(
    bundle: Path,
    *,
    partial_plan: _PlanFacts,
    partial_state: _StateFacts,
) -> dict[str, object]:
    capture = _validate_interruption_capture(
        bundle,
        partial_plan=partial_plan,
        partial_state=partial_state,
    )
    receipt = _json_object(
        _read_regular_bytes(bundle / ATTEMPT_TERMINATION_NAME, label="termination receipt"),
        label="termination receipt",
    )
    if set(receipt) != {"schema_version", "generated_by", "process", "post_termination"}:
        raise InputError("interruption termination receipt has an invalid closed schema")
    process = _mapping(receipt.get("process"), label="termination receipt process")
    post = _mapping(receipt.get("post_termination"), label="termination receipt post state")
    captured = _mapping(capture.get("partial"), label="interruption capture partial")
    if (
        set(post)
        != {
            "state_plan_log_and_cache_bytes_unchanged",
            "plan",
            "state",
            "log",
            "cache_artifacts",
        }
        or receipt.get("schema_version") != "geno-lewm.v03-cache-h200-interruption-termination.v1"
        or receipt.get("generated_by") != GENERATED_BY
        or process
        != {
            "shell_exit_code": 143,
            "wait_status_scope": "conventional_shell_status_consistent_with_SIGTERM",
            "kernel_waitpid_signal_attested": False,
            "termination_sequence": "SIGTERM_while_stopped_then_SIGCONT",
        }
        or post.get("state_plan_log_and_cache_bytes_unchanged") is not True
        or post.get("plan") != captured.get("plan")
        or post.get("state") != captured.get("state")
        or post.get("log") != captured.get("log")
        or post.get("cache_artifacts") != captured.get("cache_artifacts")
    ):
        raise InputError("termination receipt does not prove the stopped snapshot stayed unchanged")
    return receipt


def _snapshot_cache_identities(
    cache_dir: Path,
    *,
    completed: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    relative_paths = {"embeddings/.publish.lock", "embeddings/index.sqlite"}
    relative_paths.update(cast(str, entry["path"]) for entry in completed.values())
    return [
        _file_identity(cache_dir / relative, f"cache/{relative}")
        for relative in sorted(relative_paths)
    ]


def _partial_cache_snapshot_paths(state: _StateFacts) -> tuple[str, ...]:
    """Return the complete immutable partial-cache replay inventory."""
    paths = {"embeddings/index.sqlite"}
    paths.update(cast(str, entry["path"]) for entry in state.completed_by_id.values())
    return tuple(sorted(paths))


def _validate_archived_partial_cache(
    cache_dir: Path,
    *,
    plan: _PlanFacts,
    state: _StateFacts,
) -> None:
    _validate_cache_artifacts(
        cache_dir,
        plan=plan,
        completed=state.completed_by_id,
        require_all=False,
        runtime_lock_expected=False,
    )


def _validate_resume_end_event(
    event: Mapping[str, object],
    *,
    state: _StateFacts,
    partial: _StateFacts,
) -> None:
    completion = _mapping(state.payload.get("completion"), label="cache completion")
    data = _mapping(event.get("data"), label="cache build.end data")
    if (
        data.get("completed_shards") != completion.get("planned_shards")
        or data.get("encoded_rows") != completion.get("encoded_rows")
        or data.get("resumed_rows") != partial.completed_rows
        or data.get("evidence_report") != CACHE_REPORT_NAME
    ):
        raise InputError("resume build.end event differs from durable completion state")


def _derive_report(
    *,
    bundle: Path,
    repository: str,
    revision: str,
    artifact_path: str,
    source_commit: str,
    container_image: str,
    interrupted_exit_code: int,
    trace: Mapping[str, object],
    plan: _PlanFacts,
    state: _StateFacts,
    partial: _StateFacts,
    hardware: Mapping[str, object],
    attempt_log_body: bytes,
    resume_log_body: bytes,
    expectations: ProofExpectations,
) -> dict[str, object]:
    del plan, state
    return {
        "$schema": "./cache-h200-proof.schema.json",
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "ok": True,
        "producer": {
            "git_commit": source_commit,
            "origin": _CANONICAL_ORIGIN,
            "declared_container_image": container_image,
            "container_binding": "launcher_environment_declaration",
        },
        "trace": {
            "repository": repository,
            "revision": revision,
            "artifact_path": artifact_path,
            **dict(trace),
            "request_rows": expectations.request_rows,
            "unique_cache_keys": expectations.unique_cache_keys,
            "duplicate_rows": expectations.duplicate_rows,
        },
        "runtime": {
            "identity": _file_identity(bundle / RUNTIME_COPY_NAME, "encoder_runtime_identity.json"),
            **_RUNTIME_IDENTITY,
        },
        "interruption": {
            "signal_sequence": "SIGSTOP_barrier_then_SIGTERM_then_SIGCONT",
            "shell_exit_code": interrupted_exit_code,
            "partial_completed_shards": len(partial.completed_by_id),
            "partial_completed_rows": partial.completed_rows,
            "partial_durable_completed_shard_encode_batch_calls": partial.encode_batch_calls,
            "plan": _file_identity(bundle / ATTEMPT_PLAN_NAME, PLAN_NAME),
            "state": _file_identity(bundle / ATTEMPT_STATE_NAME, STATE_NAME),
            "capture": _file_identity(bundle / ATTEMPT_CAPTURE_NAME, "capture.json"),
            "termination": _file_identity(
                bundle / ATTEMPT_TERMINATION_NAME,
                "termination.json",
            ),
            "log": _identity("cache-build.jsonl", attempt_log_body),
        },
        "resume": {
            "identical_plan_bytes": True,
            "partial_entries_semantically_identical": True,
            "resumed_rows": partial.completed_rows,
            "encoded_rows": expectations.unique_cache_keys - partial.completed_rows,
            "reused_rows": 0,
            "completed_shards": len(expectations.shard_row_counts),
            "durable_completed_shard_encode_batch_calls": (
                expectations.durable_completed_shard_encode_batch_calls
            ),
            "completed_shards_reencoded": 0,
            "build_end_events": 1,
            "log": _identity("cache-build.jsonl", resume_log_body),
        },
        "cache": {
            "schema_version": "3.0.0",
            "rows_per_shard": expectations.rows_per_shard,
            "planned_shards": len(expectations.shard_row_counts),
            "unique_rows": expectations.unique_cache_keys,
            "evidence_checksums": _file_identity(
                bundle / "evidence" / CHECKSUMS_NAME,
                "evidence/SHA256SUMS",
            ),
            "report": _file_identity(
                bundle / "evidence" / CACHE_REPORT_NAME,
                f"evidence/{CACHE_REPORT_NAME}",
            ),
        },
        "hardware": {
            "receipt": _file_identity(bundle / HARDWARE_COPY_NAME, "hardware.json"),
            "device": dict(_mapping(hardware.get("device"), label="hardware device")),
            "memory": dict(_mapping(hardware.get("memory"), label="hardware memory")),
            "runtime": dict(_mapping(hardware.get("runtime"), label="hardware runtime")),
        },
        "claim_boundary": {
            "scope": "the exact 7,504-request production v0.3 training trace only",
            "finite_request_cache_completed": True,
            "interruption_resume_verified": True,
            "ten_percent_corpus_completed": False,
            "twenty_four_hour_target_evaluated": False,
            "throughput_gate_evaluated": False,
            "model_quality_evaluated": False,
            "runtime_container_attested": False,
            "hf_job_terminal_status_attested": False,
            "interrupted_in_flight_encode_batch_work_counted": False,
            "statement": (
                "This proof closes interruption and exact preservation without re-encoding of "
                "already completed shards while completing the exact production trace cache. "
                "The 928 calls count only durable completed shards and exclude interrupted "
                "in-flight work. It does not establish 10% corpus coverage, a 24-hour target, "
                "throughput, model quality, clinical validity, in-container image attestation, "
                "or terminal HF Job status."
            ),
        },
    }


def _validate_hardware_receipt(
    payload: Mapping[str, object],
    *,
    source_commit: str,
    container_image: str,
) -> dict[str, object]:
    source_commit = _commit(source_commit, label="source commit")
    container_image = _container(container_image, label="container image")
    if set(payload) != {
        "schema_version",
        "generated_by",
        "source_commit_sha",
        "container_image",
        "nvidia_smi_query_raw",
        "device",
        "memory",
        "runtime",
    }:
        raise InputError("H200 hardware receipt has an invalid closed schema")
    device = _mapping(payload.get("device"), label="hardware device")
    memory = _mapping(payload.get("memory"), label="hardware memory")
    runtime = _mapping(payload.get("runtime"), label="hardware runtime")
    if (
        set(device)
        != {
            "type",
            "count",
            "index",
            "name",
            "compute_capability",
        }
        or set(memory)
        != {
            "cuda_total_memory_bytes",
            "nvidia_smi_total_memory_mib",
        }
        or set(runtime)
        != {
            "python_version",
            "torch_version",
            "cuda_version",
            "driver_version",
        }
    ):
        raise InputError("H200 hardware receipt nested schema is not closed")
    name = device.get("name")
    cuda_total_memory_bytes = memory.get("cuda_total_memory_bytes")
    nvidia_smi_total_memory_mib = memory.get("nvidia_smi_total_memory_mib")
    compute_capability = device.get("compute_capability")
    raw_query = payload.get("nvidia_smi_query_raw")
    if not isinstance(raw_query, str):
        raise InputError("H200 hardware receipt has an empty nvidia-smi query")
    raw_index, raw_name, raw_memory_mib, raw_capability, raw_driver = _parse_nvidia_smi_row(
        raw_query
    )
    if (
        payload.get("schema_version") != HARDWARE_SCHEMA_VERSION
        or payload.get("generated_by") != HARDWARE_GENERATED_BY
        or payload.get("source_commit_sha") != source_commit
        or payload.get("container_image") != container_image
        or device.get("type") != "cuda"
        or type(device.get("count")) is not int
        or device.get("count") != 1
        or type(device.get("index")) is not int
        or device.get("index") != 0
        or not isinstance(name, str)
        or "H200" not in name.upper()
        or type(cuda_total_memory_bytes) is not int
        or cuda_total_memory_bytes <= 0
        or type(nvidia_smi_total_memory_mib) is not int
        or nvidia_smi_total_memory_mib <= 0
        or cuda_total_memory_bytes > nvidia_smi_total_memory_mib * 1024**2
        or compute_capability != "9.0"
        or raw_index != "0"
        or raw_name != name
        or raw_memory_mib != nvidia_smi_total_memory_mib
        or raw_capability != compute_capability
        or raw_driver != runtime.get("driver_version")
        or any(
            not isinstance(runtime.get(field), str) or not runtime.get(field) for field in runtime
        )
    ):
        raise InputError("hardware receipt does not attest one NVIDIA H200 CUDA runtime")
    return dict(payload)


def _parse_nvidia_smi_row(raw_query: str) -> tuple[str, str, int, str, str]:
    if not raw_query.strip():
        raise InputError("H200 hardware receipt has an empty nvidia-smi query")
    try:
        rows = list(csv.reader(StringIO(raw_query)))
    except csv.Error as exc:
        raise InputError("H200 hardware receipt nvidia-smi query is invalid CSV") from exc
    if len(rows) != 1 or len(rows[0]) != 5:
        raise InputError("H200 hardware receipt must contain one five-column nvidia-smi row")
    raw_index, raw_name, raw_memory_mib, raw_capability, raw_driver = (
        field.strip() for field in rows[0]
    )
    try:
        parsed_memory_mib = int(raw_memory_mib)
    except ValueError as exc:
        raise InputError("H200 hardware receipt nvidia-smi memory is not an integer") from exc
    return raw_index, raw_name, parsed_memory_mib, raw_capability, raw_driver


def _validate_plan_hardware_binding(
    plan: _PlanFacts,
    *,
    hardware: Mapping[str, object],
) -> None:
    device = _mapping(hardware.get("device"), label="hardware device")
    memory = _mapping(hardware.get("memory"), label="hardware memory")
    runtime = _mapping(hardware.get("runtime"), label="hardware runtime")
    expected_description = (
        f"{device['name']}; {memory['cuda_total_memory_bytes']} bytes; "
        f"CUDA {runtime['cuda_version']}; driver {runtime['driver_version']}; single GPU"
    )
    execution = _mapping(plan.payload.get("execution"), label="cache plan execution")
    observed = _mapping(execution.get("hardware"), label="cache plan hardware")
    if observed != {
        "description": expected_description,
        "encoder_device": "cuda",
    }:
        raise InputError("cache plan hardware differs from the measured H200 receipt")


def _validate_schema_file(path: Path) -> dict[str, object]:
    return _validate_schema_bytes(_read_regular_bytes(path, label="cache H200 proof schema"))


def _validate_schema_bytes(body: bytes) -> dict[str, object]:
    schema = _json_object(body, label="cache H200 proof schema")
    try:
        _validator_type().check_schema(schema)
    except Exception as exc:
        raise InputError("cache H200 proof schema is invalid") from exc
    return schema


def _validate_report(report: Mapping[str, object], schema: Mapping[str, object]) -> None:
    errors = sorted(
        _validator_type()(schema).iter_errors(report),
        key=lambda error: tuple(error.absolute_path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.absolute_path) or "$"
        raise InputError(
            "cache H200 proof report does not satisfy its bundled schema",
            details={"location": location, "error": first.message},
        )


def _validator_type() -> Any:
    try:
        jsonschema = importlib.import_module("jsonschema")
    except ImportError as exc:
        raise RuntimeSetupError(
            "cache H200 proof validation requires jsonschema",
            remediation="install geno-lewm[evidence]",
        ) from exc
    return jsonschema.Draft202012Validator


def _verify_checksum_closure(root: Path, *, expected_names: set[str] | None = None) -> None:
    observed = _regular_inventory(root)
    if CHECKSUMS_NAME not in observed:
        raise InputError("checksum-closed directory is missing SHA256SUMS")
    if expected_names is not None and observed != expected_names:
        raise InputError(
            "checksum-closed directory inventory drifted",
            details={"expected": sorted(expected_names), "observed": sorted(observed)},
        )
    body = _read_regular_bytes(root / CHECKSUMS_NAME, label=f"{root.name} SHA256SUMS")
    entries = _parse_checksums(body)
    expected_entries = observed - {CHECKSUMS_NAME}
    if set(entries) != expected_entries or list(entries) != sorted(entries):
        raise InputError("SHA256SUMS does not exactly and canonically close its directory")
    for relative, digest in entries.items():
        path = root / relative
        observed_digest = sha256_bytes(
            _read_regular_bytes(path, label=f"closed artifact {relative}")
        )
        if observed_digest != f"sha256:{digest}":
            raise InputError("checksum-closed artifact digest mismatch", details={"path": relative})


def _parse_checksums(body: bytes) -> dict[str, str]:
    try:
        text = body.decode("ascii")
    except UnicodeDecodeError as exc:
        raise InputError("SHA256SUMS must be ASCII") from exc
    entries: dict[str, str] = {}
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split("  ")
        if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
            raise InputError("SHA256SUMS contains a malformed line", details={"line": line_no})
        relative = _safe_relative(parts[1], label="SHA256SUMS path")
        if relative == CHECKSUMS_NAME or relative in entries:
            raise InputError("SHA256SUMS contains a reserved or duplicate path")
        entries[relative] = parts[0]
    return entries


def _write_outer_checksums(bundle: Path) -> None:
    inventory = _regular_inventory(bundle)
    if CHECKSUMS_NAME in inventory:
        raise InputError("outer cache proof SHA256SUMS already exists")
    body = "".join(
        f"{sha256_bytes(_read_regular_bytes(bundle / name, label=f'proof artifact {name}'))[7:]}  {name}\n"
        for name in sorted(inventory)
    ).encode("ascii")
    _write_once(bundle / CHECKSUMS_NAME, body)


def _regular_inventory(root: Path) -> set[str]:
    _require_directory(root, label=f"artifact directory {root}")
    files: set[str] = set()
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise InputError("artifact directory could not be scanned") from exc
        for entry in entries:
            path = Path(entry.path)
            metadata = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                raise InputError(
                    "artifact directory contains a symlink", details={"path": relative}
                )
            if stat.S_ISDIR(metadata.st_mode):
                stack.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                files.add(relative)
            else:
                raise InputError(
                    "artifact directory contains a non-regular object",
                    details={"path": relative},
                )
    return files


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        initial = path.lstat()
    except OSError as exc:
        raise InputError(f"{label} is missing", details={"path": str(path)}) from exc
    if not stat.S_ISREG(initial.st_mode):
        raise InputError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise InputError(f"{label} could not be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        rebound = path.lstat()
    finally:
        os.close(descriptor)
    if (
        _stable_identity(initial) != _stable_identity(before)
        or _stable_identity(before) != _stable_identity(after)
        or _stable_identity(after) != _stable_identity(rebound)
    ):
        raise InputError(f"{label} changed while it was being read")
    return b"".join(chunks)


def _stable_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _install_directory_noreplace(path: Path, files: Mapping[str, bytes]) -> None:
    if path.exists() or path.is_symlink():
        raise InputError("proof snapshot output already exists", details={"path": str(path)})
    _reject_symlink_ancestors(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        for relative, body in files.items():
            target = stage / _safe_relative(relative, label="proof snapshot path")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            target.chmod(0o400)
        stage.rename(path)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _write_once(path: Path, body: bytes) -> None:
    _reject_symlink_ancestors(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o400)
    except FileExistsError as exc:
        raise InputError("proof artifact already exists", details={"path": str(path)}) from exc
    try:
        view = memoryview(body)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_log(path: Path, *, label: str) -> tuple[bytes, tuple[Mapping[str, object], ...]]:
    body = _read_regular_bytes(path, label=label)
    if not body or not body.endswith(b"\n"):
        raise InputError(f"{label} must be non-empty newline-terminated JSONL")
    events: list[Mapping[str, object]] = []
    for line_no, line in enumerate(body.splitlines(), start=1):
        try:
            payload = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InputError(f"{label} contains invalid JSON", details={"line": line_no}) from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("event"), str):
            raise InputError(f"{label} contains a non-event row", details={"line": line_no})
        events.append(payload)
    return body, tuple(events)


def _capture_stopped_process(pid: int) -> dict[str, object]:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        raise InputError("stopped cache process PID is invalid")
    process_root = Path(f"/proc/{pid}")
    try:
        body = _read_regular_bytes(
            process_root / "status",
            label="stopped process status",
        ).decode("utf-8")
        argv_body = _read_regular_bytes(
            process_root / "cmdline",
            label="stopped process argv",
        )
        comm = (
            _read_regular_bytes(
                process_root / "comm",
                label="stopped process comm",
            )
            .decode("utf-8")
            .strip()
        )
        children_body = _read_regular_bytes(
            process_root / "task" / str(pid) / "children",
            label="stopped process children",
        ).decode("ascii")
    except (InputError, UnicodeDecodeError) as exc:
        raise InputError("stopped cache process could not be inspected through /proc") from exc
    states = [line for line in body.splitlines() if line.startswith("State:")]
    argv = [part.decode("utf-8") for part in argv_body.rstrip(b"\0").split(b"\0") if part]
    children = [int(value) for value in children_body.split()]
    if (
        len(states) != 1
        or states[0].split()[1] not in {"T", "t"}
        or not argv
        or not any(Path(value).name == "geno-lewm-cache-windows" for value in argv)
        or not comm
        or children
    ):
        raise InputError(
            "cache proof barrier did not stop the direct child-free cache process",
            details={"state": states, "argv": argv, "comm": comm, "child_pids": children},
        )
    return {
        "pid": pid,
        "observed_state": "stopped",
        "argv": argv,
        "comm": comm,
        "child_pids": children,
    }


def _require_physical_bundle(path: Path) -> Path:
    absolute = path.absolute()
    _require_directory(absolute, label="cache proof bundle")
    _reject_symlink_ancestors(absolute)
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise InputError("cache proof bundle cannot be resolved") from exc
    if resolved != absolute:
        raise InputError("cache proof bundle must use a physical non-symlink path")
    repository = _REPOSITORY_ROOT.resolve()
    try:
        absolute.relative_to(repository)
    except ValueError:
        return absolute
    raise InputError("cache proof bundle must remain outside the source checkout")


def _require_initial_bundle_inventory(bundle: Path) -> None:
    children = {path.name for path in bundle.iterdir()}
    if children != {"cache", "evidence", "trace", "proof"}:
        raise InputError("initial cache proof bundle top-level inventory is not exact")
    for name in ("cache", "evidence", "proof"):
        _require_directory(bundle / name, label=f"initial {name} directory")
        if any((bundle / name).iterdir()):
            raise InputError(f"initial cache proof {name} directory must be empty")


def _require_complete_top_level(bundle: Path) -> None:
    children = {path.name for path in bundle.iterdir()}
    if children != {"cache", "evidence", "trace", "proof", CHECKSUMS_NAME}:
        raise InputError("completed cache proof top-level inventory is not exact")


def _require_proof_inventory(bundle: Path, *, partial_state: _StateFacts) -> None:
    proof_inventory = _regular_inventory(bundle / "proof")
    expected = {
        "attempt1/cache_build_plan.json",
        "attempt1/cache_build_state.json",
        "attempt1/cache-build.jsonl",
        "attempt1/capture.json",
        "attempt1/termination.json",
        *{
            f"attempt1/cache/{relative}"
            for relative in _partial_cache_snapshot_paths(partial_state)
        },
        "resume/cache-build.jsonl",
        "encoder_runtime_identity.json",
        "cache-h200-proof.schema.json",
        "hardware.json",
        "cache-h200-proof.json",
    }
    if proof_inventory != expected:
        raise InputError(
            "completed cache proof proof/ inventory is not exact",
            details={"expected": sorted(expected), "observed": sorted(proof_inventory)},
        )


def _reject_outer_closure(bundle: Path) -> None:
    if (bundle / CHECKSUMS_NAME).exists() or (bundle / CHECKSUMS_NAME).is_symlink():
        raise InputError("cache proof outer bundle is already checksum-closed")


def _require_directory(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise InputError(f"{label} is missing", details={"path": str(path)}) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise InputError(f"{label} must be a non-symlink directory")


def _reject_symlink_ancestors(path: Path) -> None:
    for candidate in reversed((path, *path.parents)):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise InputError("proof output path contains an unsafe ancestor")


def _validate_trace_publication_arguments(
    repository: object,
    revision: object,
    artifact_path: object,
) -> tuple[str, str, str]:
    if repository != _CANONICAL_TRACE_REPOSITORY:
        raise InputError("trace repository must be the canonical public dataset")
    revision_text = _commit(revision, label="trace revision")
    if revision_text != _CANONICAL_TRACE_REVISION:
        raise InputError("trace revision is not the exact corrected public training trace")
    path = _text(artifact_path, label="trace artifact path")
    if (
        path != _CANONICAL_TRACE_ARTIFACT_PATH
        or _SAFE_TRACE_PATH.fullmatch(path) is None
        or _safe_relative(path, label="trace path") != path
    ):
        raise InputError("trace artifact path is not a successful v0.3 training-trace namespace")
    return _CANONICAL_TRACE_REPOSITORY, revision_text, path


def _verify_remote_trace_namespace(
    *,
    repository: str,
    revision: str,
    artifact_path: str,
    trace_dir: Path,
) -> None:
    try:
        hub = importlib.import_module("huggingface_hub")
        api = hub.HfApi(token=False, endpoint=_HUB_ENDPOINT)
        info = api.repo_info(
            repo_id=repository,
            repo_type="dataset",
            revision=revision,
            files_metadata=True,
        )
        if getattr(info, "sha", None) != revision:
            raise InputError("Hugging Face resolved a different trace revision")
        prefix = f"{artifact_path}/"
        siblings = getattr(info, "siblings", None)
        if not isinstance(siblings, list):
            raise InputError("Hugging Face trace metadata omitted file identities")
        remote = {
            name.removeprefix(prefix): sibling
            for sibling in siblings
            if isinstance((name := getattr(sibling, "rfilename", None)), str)
            and name.startswith(prefix)
        }
        inventory = _regular_inventory(trace_dir)
        if set(remote) != inventory:
            raise InputError("exact-revision remote trace inventory differs from local bytes")
        for relative, sibling in remote.items():
            body = _read_regular_bytes(trace_dir / relative, label=f"trace artifact {relative}")
            if getattr(sibling, "size", None) != len(body):
                raise InputError("remote trace artifact size differs from local bytes")
            lfs = getattr(sibling, "lfs", None)
            if lfs is not None:
                if getattr(lfs, "sha256", None) != hashlib.sha256(body).hexdigest():
                    raise InputError("remote trace LFS identity differs from local bytes")
            elif getattr(sibling, "blob_id", None) != _git_blob_sha1(body):
                raise InputError("remote trace Git blob identity differs from local bytes")
    except InputError:
        raise
    except Exception as exc:
        raise InputError("exact-revision remote trace could not be verified") from exc


def _verify_producer_invocation(*, source_commit: str, container_image: str) -> None:
    source_commit = _commit(source_commit, label="source commit")
    container_image = _container(container_image, label="container image")
    if os.environ.get("GENO_LEWM_CACHE_H200_PROOF_DECLARED_CONTAINER_IMAGE") != container_image:
        raise InputError("container image differs from the launcher declaration")
    if _git_output("rev-parse", "HEAD") != source_commit:
        raise InputError("cache proof source commit differs from the checkout")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise InputError("cache proof source checkout must be clean")
    if _git_output("remote", "get-url", "origin") != _CANONICAL_ORIGIN:
        raise InputError("cache proof source origin is not canonical")
    for relative in (
        "tools/research/v03_cache_h200_launch.py",
        "tools/research/v03_cache_h200_proof.py",
        "tools/research/verify_carbon_runtime_lock.py",
        "tools/jobs/v03_cache_h200_proof.sh",
        "configs/data_v03/cache-h200-proof.schema.json",
        "configs/data_v03/carbon-500m-l2-runtime-identity.json",
        "configs/data_v03/carbon-500m-runtime-content-lock.json",
        "configs/data_v03/carbon-500m-runtime-content-lock.schema.json",
        "geno_lewm/encoder/cache_build.py",
        "geno_lewm/cli/cache_windows.py",
    ):
        _git_output("cat-file", "-e", f"{source_commit}:{relative}")
    url = f"{_GITHUB_API_ENDPOINT}/repos/{_CANONICAL_GITHUB_REPOSITORY}/commits/{source_commit}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "GenoLeWM-v03-cache-H200-proof",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
            if response.status != 200 or response.geturl() != url:
                raise InputError("canonical GitHub source lookup did not resolve exactly")
    except InputError:
        raise
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise InputError("cache proof source commit is not publicly verifiable") from exc
    if not isinstance(payload, Mapping) or payload.get("sha") != source_commit:
        raise InputError("canonical GitHub source lookup returned a different commit")


def _git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(_REPOSITORY_ROOT), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InputError("cache proof Git state cannot be verified") from exc
    return completed.stdout.strip()


def _file_identity(path: Path, relative: str) -> dict[str, object]:
    return _identity(relative, _read_regular_bytes(path, label=f"proof artifact {relative}"))


def _identity(path: str, body: bytes) -> dict[str, object]:
    return {"path": path, "sha256": sha256_bytes(body), "size_bytes": len(body)}


def _artifact_identity(value: object, *, label: str) -> Mapping[str, object]:
    identity = _mapping(value, label=label)
    if set(identity) != {"path", "sha256", "size_bytes"}:
        raise InputError(f"{label} has an invalid closed schema")
    _safe_relative(identity.get("path"), label=f"{label}.path")
    digest = identity.get("sha256")
    size = identity.get("size_bytes")
    if (
        not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
    ):
        raise InputError(f"{label} has an invalid hash or size")
    return identity


def _json_object(body: bytes, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(body, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, InputError) as exc:
        raise InputError(f"{label} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise InputError(f"{label} must be a JSON object")
    return cast(dict[str, object], payload)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise InputError("duplicate JSON key is not allowed", details={"key": key})
        payload[key] = value
    return payload


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise InputError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputError(f"{label} must be non-empty text")
    return value


def _commit(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if _COMMIT.fullmatch(text) is None:
        raise InputError(f"{label} must be an exact lowercase 40-character commit")
    return text


def _container(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if _CONTAINER.fullmatch(text) is None:
        raise InputError(f"{label} must be digest-pinned")
    return text


def _safe_relative(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise InputError(f"{label} must be a normalized safe relative path")
    return text


def _canonical_key(key: object) -> str:
    if not hasattr(key, "window_hash"):
        raise InputError("cache key object is invalid")
    return json.dumps(
        {
            "window_hash": cast(Any, key).window_hash.hex(),
            "encoder_hash": cast(Any, key).encoder_hash.hex(),
            "state_layer": cast(Any, key).state_layer,
            "pool_type": cast(Any, key).pool_type,
            "pool_radius": cast(Any, key).pool_radius,
            "center_token": cast(Any, key).center_token,
            "dtype": cast(Any, key).dtype,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _git_blob_sha1(body: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(body)}\0".encode("ascii"))
    digest.update(body)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--bundle-dir", type=Path, required=True)
    preflight.add_argument("--trace-dir", type=Path, required=True)
    _add_publication_arguments(preflight)
    preflight.add_argument("--runtime-identity", type=Path, required=True)
    preflight.add_argument("--source-commit", required=True)
    preflight.add_argument("--container-image", required=True)

    capture = subparsers.add_parser("capture-partial")
    capture.add_argument("--bundle-dir", type=Path, required=True)
    capture.add_argument("--attempt-log", type=Path, required=True)
    capture.add_argument("--stopped-pid", type=int, required=True)

    finalize = subparsers.add_parser("finalize-interruption")
    finalize.add_argument("--bundle-dir", type=Path, required=True)
    finalize.add_argument("--attempt-log", type=Path, required=True)
    finalize.add_argument("--attempt-exit-code", type=int, required=True)

    author = subparsers.add_parser("author")
    author.add_argument("--bundle-dir", type=Path, required=True)
    _add_publication_arguments(author)
    author.add_argument("--runtime-identity", type=Path, required=True)
    author.add_argument("--source-commit", required=True)
    author.add_argument("--container-image", required=True)
    author.add_argument("--hardware-json", type=Path, required=True)
    author.add_argument("--resume-log", type=Path, required=True)

    validate_hardware = subparsers.add_parser("validate-hardware")
    validate_hardware.add_argument("--hardware-json", type=Path, required=True)
    validate_hardware.add_argument("--source-commit", required=True)
    validate_hardware.add_argument("--container-image", required=True)

    retire = subparsers.add_parser("retire-runtime-lock")
    retire.add_argument("--bundle-dir", type=Path, required=True)

    verify = subparsers.add_parser("verify-existing")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    return parser


def _add_publication_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trace-repository", required=True)
    parser.add_argument("--trace-revision", required=True)
    parser.add_argument("--trace-artifact-path", required=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight_bundle(
            bundle_dir=args.bundle_dir,
            trace_dir=args.trace_dir,
            trace_repository=args.trace_repository,
            trace_revision=args.trace_revision,
            trace_artifact_path=args.trace_artifact_path,
            runtime_identity_path=args.runtime_identity,
            source_commit=args.source_commit,
            container_image=args.container_image,
        )
    elif args.command == "capture-partial":
        result = capture_partial_bundle(
            bundle_dir=args.bundle_dir,
            attempt_log=args.attempt_log,
            stopped_pid=args.stopped_pid,
        )
    elif args.command == "finalize-interruption":
        result = finalize_interruption(
            bundle_dir=args.bundle_dir,
            attempt_log=args.attempt_log,
            attempt_exit_code=args.attempt_exit_code,
        )
    elif args.command == "retire-runtime-lock":
        result = retire_cache_runtime_lock(bundle_dir=args.bundle_dir)
    elif args.command == "validate-hardware":
        result = validate_hardware_receipt(
            args.hardware_json,
            source_commit=args.source_commit,
            container_image=args.container_image,
        )
    elif args.command == "author":
        result = author_proof_bundle(
            bundle_dir=args.bundle_dir,
            trace_repository=args.trace_repository,
            trace_revision=args.trace_revision,
            trace_artifact_path=args.trace_artifact_path,
            runtime_identity_path=args.runtime_identity,
            source_commit=args.source_commit,
            container_image=args.container_image,
            hardware_json=args.hardware_json,
            resume_log=args.resume_log,
        )
    else:
        result = verify_existing_bundle(bundle_dir=args.bundle_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI exercised by job contract.
    raise SystemExit(main())
