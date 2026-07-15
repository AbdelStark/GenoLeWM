# SPDX-License-Identifier: Apache-2.0
"""Author one deterministic v0.3 trainer epoch as cache-build requests."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast
from urllib import error as urllib_error, request as urllib_request

import yaml

from geno_lewm.config import GenoLeWMConfig, load_config
from geno_lewm.config._state_contract import L2_NORMALIZED_STATE_CONTRACT
from geno_lewm.data._membership_store_publish import _publish_directory_noreplace
from geno_lewm.errors import InputError, RuntimeSetupError
from geno_lewm.provenance import canonical_json_sha256, sha256_bytes
from geno_lewm.training._data_stream import PreparedTrainingStream
from geno_lewm.training.trainer import TrainerSeeds
from tools.release.v03_dataset_snapshot import verify_v03_dataset_snapshot

SCHEMA_VERSION: Final = "geno-lewm.v03-training-trace.v1"
GENERATED_BY: Final = "tools.data.v03_training_trace"
REQUESTS_NAME: Final = "cache_build_requests.jsonl"
TRAINING_CONFIG_NAME: Final = "training_config.yaml"
REPORT_NAME: Final = "training_trace_report.json"
OUTPUT_SCHEMA_NAME: Final = "training_trace.schema.json"
CHECKSUMS_NAME: Final = "SHA256SUMS"
_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH: Final = _REPOSITORY_ROOT / "configs/data_v03/training-trace.schema.json"
_SCHEMA_1_1: Final = "1.1.0"
_CANONICAL_ORIGIN: Final = "https://github.com/AbdelStark/GenoLeWM.git"
_GITHUB_API_ENDPOINT: Final = "https://api.github.com"
_GITHUB_REPOSITORY: Final = "AbdelStark/GenoLeWM"
_CANONICAL_DATASET_REPOSITORY: Final = "abdelstark/geno-lewm-data"
_HUB_ENDPOINT: Final = "https://huggingface.co"
_COMMIT: Final = re.compile(r"[0-9a-f]{40}\Z")
_CONTAINER_IMAGE: Final = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}\Z")
_DATASET_ARTIFACT_PATH: Final = re.compile(r"candidates/v0\.3/[A-Za-z0-9._/-]+/success\Z")
_MEMBERSHIP_BINDING_KEYS: Final = frozenset(
    {"membership_store", "report", "holdout_policy", "holdout_policy_identity"}
)
_MEMBERSHIP_STORE_KEYS: Final = frozenset(
    {"path", "artifact_id", "content_identity", "physical_identity", "rowset_sha256"}
)
_MEMBERSHIP_REPORT_KEYS: Final = frozenset({"path", "schema_path", "artifact_id", "schema_version"})
_MEMBERSHIP_POLICY_KEYS: Final = frozenset(
    {
        "schema_version",
        "membership_content_identity",
        "excluded_chromosomes",
        "selection",
        "lookup",
    }
)
_SHA256: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_V03_EXCLUDED_CHROMOSOMES: Final = ("20", "21")


def author_training_trace(
    *,
    stream: PreparedTrainingStream,
    dataset_manifest_path: Path,
    training_config_path: Path,
    output_dir: Path,
    producer_git_commit: str,
    container_image: str,
    dataset_repository: str,
    dataset_revision: str,
    dataset_artifact_path: str,
    _expected_dataset_checksums: bytes | None = None,
    _expected_training_config: bytes | None = None,
    _expected_schema: bytes | None = None,
) -> dict[str, object]:
    """Write a checksum-closed one-epoch request artifact from ``stream``."""
    if stream.schema_version != _SCHEMA_1_1:
        raise InputError(
            "v0.3 training trace requires dataset schema 1.1.0",
            details={"schema_version": stream.schema_version},
        )
    if stream.membership_identity is None:
        raise InputError("v0.3 training trace requires verified membership and split evidence")
    producer_git_commit = _require_commit(producer_git_commit)
    container_image = _require_container_image(container_image)
    dataset_repository = _require_dataset_repository(dataset_repository)
    dataset_revision = _require_dataset_revision(dataset_revision)
    dataset_artifact_path = _require_dataset_artifact_path(dataset_artifact_path)
    manifest_bytes = _read_regular_bytes(dataset_manifest_path, label="dataset manifest")
    dataset_checksums_bytes = _read_regular_bytes(
        dataset_manifest_path.parent / "SHA256SUMS",
        label="dataset SHA256SUMS",
    )
    _require_unchanged_bytes(
        "dataset SHA256SUMS",
        expected=_expected_dataset_checksums,
        observed=dataset_checksums_bytes,
    )
    manifest = _json_object(manifest_bytes, label="dataset manifest")
    if manifest.get("schema_version") != stream.schema_version:
        raise InputError("prepared stream schema does not match dataset manifest")
    if manifest.get("snapshot_id") != stream.dataset_snapshot_id:
        raise InputError("prepared stream snapshot does not match dataset manifest")
    _require_manifest_membership_binding(manifest, stream.membership_identity)

    config_bytes = _read_regular_bytes(training_config_path, label="training config")
    _require_unchanged_bytes(
        "training config",
        expected=_expected_training_config,
        observed=config_bytes,
    )
    config = _load_config_bytes(config_bytes)
    _require_v03_training_config(config)
    seeds = TrainerSeeds.from_base_seed(config.seed)
    if stream.seed != seeds.data:
        raise InputError(
            "prepared stream data seed does not match training config",
            details={"stream": stream.seed, "config": seeds.data},
        )
    schema_bytes = _read_regular_bytes(DEFAULT_SCHEMA_PATH, label="training trace schema")
    _require_unchanged_bytes(
        "training trace schema",
        expected=_expected_schema,
        observed=schema_bytes,
    )
    schema = _json_object(schema_bytes, label="training trace schema")
    _validate_schema(schema)

    requests_bytes, source_counts, fallback_counts = _request_artifact(stream)
    request_rows = sum(source_counts.values())
    if request_rows % config.data.batch_size:
        raise InputError(
            "v0.3 training trace must contain an integer number of training batches",
            details={"request_rows": request_rows, "batch_size": config.data.batch_size},
        )
    batches_per_epoch = request_rows // config.data.batch_size
    if config.training.max_steps != batches_per_epoch:
        raise InputError(
            "v0.3 training config max_steps must consume exactly one prepared epoch",
            details={
                "max_steps": config.training.max_steps,
                "batches_per_epoch": batches_per_epoch,
            },
        )
    request_identity = _identity(REQUESTS_NAME, requests_bytes)
    schema_identity = _identity(OUTPUT_SCHEMA_NAME, schema_bytes)
    report = {
        "$schema": f"./{OUTPUT_SCHEMA_NAME}",
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "producer": {
            "git_commit": producer_git_commit,
            "origin": _CANONICAL_ORIGIN,
            "source_publication": {
                "endpoint": _GITHUB_API_ENDPOINT,
                "method": "unauthenticated_exact_commit_lookup",
            },
            "declared_container_image": container_image,
            "container_binding": "launcher_environment_declaration",
        },
        "dataset": {
            "repository": dataset_repository,
            "revision": dataset_revision,
            "artifact_path": dataset_artifact_path,
            "publication_binding": {
                "endpoint": _HUB_ENDPOINT,
                "method": "exact_revision_namespace_content_address_identity",
                "checksums": _identity("SHA256SUMS", dataset_checksums_bytes),
            },
            "snapshot_id": stream.dataset_snapshot_id,
            "schema_version": stream.schema_version,
            "manifest": _identity(dataset_manifest_path.name, manifest_bytes),
            "membership_and_split_evidence": stream.membership_identity,
        },
        "training": {
            "config": _identity(TRAINING_CONFIG_NAME, config_bytes),
            "seed": config.seed,
            "data_seed": seeds.data,
            "epoch": 0,
            "batch_size": config.data.batch_size,
            "batches_per_epoch": batches_per_epoch,
            "max_steps": config.training.max_steps,
            "state_contract_version": config.encoder.state_contract_version,
            "encoder_revision": config.encoder.revision,
            "pool_type": config.encoder.pool_type,
            "pool_radius": config.encoder.pool_radius,
            "normalize": config.encoder.normalize,
            "mix": [{"source": entry.source, "count": entry.count} for entry in stream.mix],
            "fallback_sources": dict(sorted(stream.fallback_sources.items())),
        },
        "window_eligibility": {
            "filter_order": "before_epoch_rng_initialization",
            "input_windows": stream.input_window_count,
            "usable_windows": stream.usable_window_count,
            "excluded_windows": len(stream.exclusions),
            "exclusions": [item.to_dict() for item in stream.exclusions],
        },
        "trace": {
            "request_rows": request_rows,
            "rows_per_window": sum(entry.count for entry in stream.mix),
            "source_counts": source_counts,
            "fallback_counts": fallback_counts,
            "ordering": "trainer_epoch_order_then_source_mix_order",
            "requests": request_identity,
        },
        "artifacts": {"schema": schema_identity},
        "claim_boundary": {
            "scope": "one deterministic epoch of source-state cache requests",
            "cache_built": False,
            "throughput_measured": False,
            "runtime_container_attested": False,
        },
    }
    _validate_report(report, schema)
    report_bytes = _json_bytes(report)
    files = {
        REQUESTS_NAME: requests_bytes,
        TRAINING_CONFIG_NAME: config_bytes,
        OUTPUT_SCHEMA_NAME: schema_bytes,
        REPORT_NAME: report_bytes,
    }
    checksums = "".join(
        f"{_bare_sha256(body)}  {name}\n" for name, body in sorted(files.items())
    ).encode("utf-8")
    _publish_closed_directory(output_dir, {**files, CHECKSUMS_NAME: checksums})
    return cast(dict[str, object], json.loads(report_bytes))


def build_training_trace(
    *,
    dataset_dir: Path,
    training_config_path: Path,
    output_dir: Path,
    producer_git_commit: str,
    container_image: str,
    dataset_repository: str,
    dataset_revision: str,
    dataset_artifact_path: str,
) -> dict[str, object]:
    """Open the verified production stream and author its epoch-zero trace."""
    producer_git_commit = _require_commit(producer_git_commit)
    container_image = _require_container_image(container_image)
    _verify_producer_invocation(
        producer_git_commit=producer_git_commit,
        container_image=container_image,
    )
    dataset_checksums = _verify_dataset_publication_binding(
        dataset_dir=dataset_dir,
        repository=dataset_repository,
        revision=dataset_revision,
        artifact_path=dataset_artifact_path,
    )
    config_bytes = _read_regular_bytes(training_config_path, label="training config")
    schema_bytes = _read_regular_bytes(DEFAULT_SCHEMA_PATH, label="training trace schema")
    config = _load_config_bytes(config_bytes)
    output_dir = output_dir.absolute()
    _prepare_output_parent(output_dir)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.verified.",
        dir=output_dir.parent,
    ) as temporary:
        staged_evidence = Path(temporary) / "evidence"
        with PreparedTrainingStream.open(
            dataset_dir=dataset_dir,
            config=config,
            require_membership=True,
        ) as stream:
            report = author_training_trace(
                stream=stream,
                dataset_manifest_path=dataset_dir / "dataset_manifest.json",
                training_config_path=training_config_path,
                output_dir=staged_evidence,
                producer_git_commit=producer_git_commit,
                container_image=container_image,
                dataset_repository=dataset_repository,
                dataset_revision=dataset_revision,
                dataset_artifact_path=dataset_artifact_path,
                _expected_dataset_checksums=dataset_checksums,
                _expected_training_config=config_bytes,
                _expected_schema=schema_bytes,
            )
        postflight_checksums = _verify_dataset_publication_binding(
            dataset_dir=dataset_dir,
            repository=dataset_repository,
            revision=dataset_revision,
            artifact_path=dataset_artifact_path,
        )
        _require_unchanged_bytes(
            "dataset publication binding",
            expected=dataset_checksums,
            observed=postflight_checksums,
        )
        _require_current_file_bytes(
            training_config_path,
            expected=config_bytes,
            label="training config",
        )
        _require_current_file_bytes(
            DEFAULT_SCHEMA_PATH,
            expected=schema_bytes,
            label="training trace schema",
        )
        _verify_producer_invocation(
            producer_git_commit=producer_git_commit,
            container_image=container_image,
        )
        _promote_verified_evidence(staged_evidence, output_dir)
        return report


def verify_training_trace_evidence(
    *,
    stream: PreparedTrainingStream,
    dataset_manifest_path: Path,
    training_config_path: Path,
    evidence_dir: Path,
    producer_git_commit: str,
    container_image: str,
    dataset_repository: str,
    dataset_revision: str,
    dataset_artifact_path: str,
    _expected_dataset_checksums: bytes | None = None,
    _expected_training_config: bytes | None = None,
    _expected_schema: bytes | None = None,
) -> dict[str, object]:
    """Re-author one trace and require byte identity with an evidence directory."""
    with tempfile.TemporaryDirectory(
        prefix="geno-lewm-v03-trace-replay-",
        dir=Path(tempfile.gettempdir()).resolve(),
    ) as temporary:
        expected_dir = Path(temporary) / "evidence"
        report = author_training_trace(
            stream=stream,
            dataset_manifest_path=dataset_manifest_path,
            training_config_path=training_config_path,
            output_dir=expected_dir,
            producer_git_commit=producer_git_commit,
            container_image=container_image,
            dataset_repository=dataset_repository,
            dataset_revision=dataset_revision,
            dataset_artifact_path=dataset_artifact_path,
            _expected_dataset_checksums=_expected_dataset_checksums,
            _expected_training_config=_expected_training_config,
            _expected_schema=_expected_schema,
        )
        _require_exact_reauthoring(evidence_dir, expected_dir)
    return report


def verify_training_trace(
    *,
    dataset_dir: Path,
    training_config_path: Path,
    evidence_dir: Path,
    producer_git_commit: str,
    container_image: str,
    dataset_repository: str,
    dataset_revision: str,
    dataset_artifact_path: str,
) -> dict[str, object]:
    """Verify producer state, reopen the dataset, and replay exact trace bytes."""
    producer_git_commit = _require_commit(producer_git_commit)
    container_image = _require_container_image(container_image)
    _verify_producer_invocation(
        producer_git_commit=producer_git_commit,
        container_image=container_image,
    )
    dataset_checksums = _verify_dataset_publication_binding(
        dataset_dir=dataset_dir,
        repository=dataset_repository,
        revision=dataset_revision,
        artifact_path=dataset_artifact_path,
    )
    config_bytes = _read_regular_bytes(training_config_path, label="training config")
    schema_bytes = _read_regular_bytes(DEFAULT_SCHEMA_PATH, label="training trace schema")
    config = _load_config_bytes(config_bytes)
    with PreparedTrainingStream.open(
        dataset_dir=dataset_dir,
        config=config,
        require_membership=True,
    ) as stream:
        report = verify_training_trace_evidence(
            stream=stream,
            dataset_manifest_path=dataset_dir / "dataset_manifest.json",
            training_config_path=training_config_path,
            evidence_dir=evidence_dir,
            producer_git_commit=producer_git_commit,
            container_image=container_image,
            dataset_repository=dataset_repository,
            dataset_revision=dataset_revision,
            dataset_artifact_path=dataset_artifact_path,
            _expected_dataset_checksums=dataset_checksums,
            _expected_training_config=config_bytes,
            _expected_schema=schema_bytes,
        )
    postflight_checksums = _verify_dataset_publication_binding(
        dataset_dir=dataset_dir,
        repository=dataset_repository,
        revision=dataset_revision,
        artifact_path=dataset_artifact_path,
    )
    _require_unchanged_bytes(
        "dataset publication binding",
        expected=dataset_checksums,
        observed=postflight_checksums,
    )
    _require_current_file_bytes(
        training_config_path,
        expected=config_bytes,
        label="training config",
    )
    _require_current_file_bytes(
        DEFAULT_SCHEMA_PATH,
        expected=schema_bytes,
        label="training trace schema",
    )
    _verify_producer_invocation(
        producer_git_commit=producer_git_commit,
        container_image=container_image,
    )
    return report


def _request_artifact(
    stream: PreparedTrainingStream,
) -> tuple[bytes, dict[str, int], dict[str, int]]:
    slots = tuple(source for entry in stream.mix for source in (entry.source,) * entry.count)
    if not slots:
        raise InputError("training trace mix must produce at least one row per window")
    configured_sources = {entry.source for entry in stream.mix}
    configured_sources.update(stream.fallback_sources.values())
    source_counts: Counter[str] = Counter(dict.fromkeys(configured_sources, 0))
    fallback_counts: Counter[str] = Counter()
    body = bytearray()
    row_count = 0
    for index, item in enumerate(stream.iter_epoch(0)):
        window_index, slot_index = divmod(index, len(slots))
        if window_index >= stream.usable_window_count:
            raise InputError("prepared epoch produced more rows than its finite window contract")
        expected_window = stream.usable_windows[window_index]
        if item.source_window != expected_window:
            raise InputError(
                "prepared epoch order drifted from the finite usable-window contract",
                details={"row": index, "window": item.source_window.record_id},
            )
        requested_source = slots[slot_index]
        actual_source = item.training_tuple.edit_source
        if actual_source != requested_source:
            expected_fallback = stream.fallback_sources.get(requested_source)
            if expected_fallback != actual_source:
                raise InputError(
                    "prepared epoch emitted an undeclared source substitution",
                    details={
                        "row": index,
                        "requested_source": requested_source,
                        "actual_source": actual_source,
                    },
                )
            fallback_counts[f"{requested_source}->{actual_source}"] += 1
        source_counts[actual_source] += 1
        window = item.source_window
        if window.chrom is None:
            raise InputError(
                "v0.3 cache requests require placed source windows",
                details={"record_id": window.record_id},
            )
        edits = item.training_tuple.rel_edits
        if not edits:
            raise InputError("training trace item contains no edit locus")
        request = {
            "request_id": f"v03-train-e000000-r{index:08d}",
            "chrom": window.chrom,
            "start_bp": window.start_bp,
            "end_bp": window.end_bp,
            "window": window.sequence,
            "edit_locus": edits[0].rel_pos,
        }
        body.extend(json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        body.extend(b"\n")
        row_count += 1
    expected_rows = stream.usable_window_count * len(slots)
    if row_count != expected_rows:
        raise InputError(
            "prepared epoch row count does not match its finite window and mix contract",
            details={"expected": expected_rows, "observed": row_count},
        )
    return (
        bytes(body),
        dict(sorted(source_counts.items())),
        dict(sorted(fallback_counts.items())),
    )


def _load_config_bytes(body: bytes) -> GenoLeWMConfig:
    try:
        payload = yaml.safe_load(body)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise InputError("training config YAML is invalid") from exc
    if not isinstance(payload, Mapping):
        raise InputError("training config must decode to an object")
    return load_config(cast(Mapping[str, Any], payload))


def _require_v03_training_config(config: GenoLeWMConfig) -> None:
    if config.schema_version != "1.1.0":
        raise InputError("v0.3 training trace requires training config schema 1.1.0")
    if not config.deterministic:
        raise InputError("v0.3 training trace requires deterministic training")
    encoder = config.encoder
    if encoder.state_contract_version != L2_NORMALIZED_STATE_CONTRACT or not encoder.normalize:
        raise InputError("v0.3 training trace requires corrected l2_normalized_v2 states")
    if encoder.pool_type != "centered_mean":
        raise InputError("v0.3 training trace requires edit-centered source states")
    if encoder.trust_remote_code:
        raise InputError("v0.3 training trace requires the local pure-DNA tokenizer")
    if _COMMIT.fullmatch(encoder.revision) is None:
        raise InputError("v0.3 training trace requires an exact encoder revision")


def _require_unchanged_bytes(
    label: str,
    *,
    expected: bytes | None,
    observed: bytes,
) -> None:
    if expected is not None and observed != expected:
        raise InputError(
            f"{label} changed during training-trace construction",
            details={
                "expected": sha256_bytes(expected),
                "observed": sha256_bytes(observed),
            },
        )


def _require_current_file_bytes(path: Path, *, expected: bytes, label: str) -> None:
    observed = _read_regular_bytes(path, label=label)
    _require_unchanged_bytes(label, expected=expected, observed=observed)


def _prepare_output_parent(output_dir: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise InputError(
            "training trace output already exists",
            details={"path": str(output_dir)},
        )
    _reject_symlink_ancestors(output_dir.parent)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(output_dir.parent)


def _promote_verified_evidence(staged: Path, output_dir: Path) -> None:
    expected = _capture_evidence_directory(staged)
    _prepare_output_parent(output_dir)
    _publish_directory_noreplace(staged, output_dir)
    _verify_published_directory(output_dir, expected)


def _publish_closed_directory(output_dir: Path, files: Mapping[str, bytes]) -> None:
    output_dir = output_dir.absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise InputError(
            "training trace output already exists",
            details={"path": str(output_dir)},
        )
    _reject_symlink_ancestors(output_dir.parent)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(output_dir.parent)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for name, body in files.items():
            path = stage / name
            path.write_bytes(body)
            path.chmod(0o400)
        observed = {path.name for path in stage.iterdir() if path.is_file()}
        if observed != set(files):
            raise InputError("training trace staged file inventory is not closed")
        _publish_directory_noreplace(stage, output_dir)
        _verify_published_directory(output_dir, files)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _reject_symlink_ancestors(path: Path) -> None:
    for candidate in reversed((path, *path.parents)):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise InputError(
                "training trace output path must not contain symlink components",
                details={"path": str(candidate)},
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise InputError(
                "training trace output parent must contain directories only",
                details={"path": str(candidate)},
            )


def _validate_schema(schema: Mapping[str, object]) -> None:
    validator_type = _validator_type()
    try:
        validator_type.check_schema(schema)
    except Exception as exc:
        raise InputError("training trace report schema is invalid") from exc


def _validate_report(report: Mapping[str, object], schema: Mapping[str, object]) -> None:
    validator = _validator_type()(schema)
    errors = sorted(validator.iter_errors(report), key=lambda error: tuple(error.absolute_path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "$"
        raise InputError(
            "training trace report does not satisfy its bundled schema",
            details={"location": location, "error": first.message},
        )
    _validate_report_semantics(report)


def _validate_report_semantics(report: Mapping[str, object]) -> None:
    dataset = cast(Mapping[str, object], report["dataset"])
    membership = cast(Mapping[str, object], dataset["membership_and_split_evidence"])
    _validate_membership_identity(membership)

    training = cast(Mapping[str, object], report["training"])
    mix = cast(Sequence[Mapping[str, object]], training["mix"])
    rows_per_window = sum(cast(int, item["count"]) for item in mix)
    trace = cast(Mapping[str, object], report["trace"])
    if trace["rows_per_window"] != rows_per_window:
        raise InputError("training trace rows_per_window does not match the configured mix")

    eligibility = cast(Mapping[str, object], report["window_eligibility"])
    input_windows = cast(int, eligibility["input_windows"])
    usable_windows = cast(int, eligibility["usable_windows"])
    excluded_windows = cast(int, eligibility["excluded_windows"])
    exclusions = cast(Sequence[object], eligibility["exclusions"])
    if input_windows != usable_windows + excluded_windows:
        raise InputError("training trace window cardinalities are inconsistent")
    if excluded_windows != len(exclusions):
        raise InputError("training trace excluded_windows does not match exclusions")

    request_rows = cast(int, trace["request_rows"])
    batch_size = cast(int, training["batch_size"])
    batches_per_epoch = cast(int, training["batches_per_epoch"])
    if request_rows != batch_size * batches_per_epoch:
        raise InputError("training trace batch cardinalities do not cover request_rows exactly")
    if training["max_steps"] != batches_per_epoch:
        raise InputError("training trace max_steps does not consume exactly one epoch")
    if request_rows != usable_windows * rows_per_window:
        raise InputError("training trace request_rows does not match usable windows and mix")
    source_counts = cast(Mapping[str, int], trace["source_counts"])
    if sum(source_counts.values()) != request_rows:
        raise InputError("training trace source_counts do not sum to request_rows")
    fallback_counts = cast(Mapping[str, int], trace["fallback_counts"])
    if sum(fallback_counts.values()) > request_rows:
        raise InputError("training trace fallback_counts exceed request_rows")


def _require_manifest_membership_binding(
    manifest: Mapping[str, object],
    membership: Mapping[str, object],
) -> None:
    expected = {
        "membership_store": membership.get("membership_store"),
        "report": membership.get("report"),
    }
    if manifest.get("membership_and_split_evidence") != expected:
        raise InputError("prepared stream membership binding does not match the dataset manifest")


def _validate_membership_identity(membership: Mapping[str, object]) -> None:
    if frozenset(membership) != _MEMBERSHIP_BINDING_KEYS:
        raise InputError("training trace membership binding fields are not closed")
    store = cast(Mapping[str, object], membership["membership_store"])
    report = cast(Mapping[str, object], membership["report"])
    policy = cast(Mapping[str, object], membership["holdout_policy"])
    if frozenset(store) != _MEMBERSHIP_STORE_KEYS:
        raise InputError("training trace membership-store binding fields are not closed")
    if frozenset(report) != _MEMBERSHIP_REPORT_KEYS:
        raise InputError("training trace membership-report binding fields are not closed")
    if frozenset(policy) != _MEMBERSHIP_POLICY_KEYS:
        raise InputError("training trace membership holdout-policy fields are not closed")
    for field in ("content_identity", "physical_identity", "rowset_sha256"):
        if not isinstance((value := store[field]), str) or _SHA256.fullmatch(value) is None:
            raise InputError(
                "training trace membership-store identity is not a canonical SHA-256",
                details={"field": field},
            )
    if policy["membership_content_identity"] != store["content_identity"]:
        raise InputError("training trace holdout policy does not bind the membership-store content")
    if tuple(cast(Sequence[object], policy["excluded_chromosomes"])) != (_V03_EXCLUDED_CHROMOSOMES):
        raise InputError("training trace holdout policy does not bind the v0.3 split")
    if membership["holdout_policy_identity"] != canonical_json_sha256(dict(policy)):
        raise InputError("training trace holdout-policy identity is not canonical")


def _validator_type() -> Any:
    try:
        jsonschema = importlib.import_module("jsonschema")
    except ImportError as exc:
        raise RuntimeSetupError(
            "training trace authoring requires jsonschema",
            remediation="install geno-lewm[evidence]",
        ) from exc
    return jsonschema.Draft202012Validator


def _verify_published_directory(output_dir: Path, expected: Mapping[str, bytes]) -> None:
    observed: dict[str, bytes] = {}
    for path in output_dir.iterdir():
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise InputError("published training trace contains a non-regular file")
        if metadata.st_mode & stat.S_IWUSR:
            raise InputError(
                "published training trace files must not be owner-writable",
                details={"path": str(path)},
            )
        observed[path.name] = path.read_bytes()
    if observed != dict(expected):
        raise InputError("published training trace bytes do not match the verified staging bundle")
    schema = _json_object(observed[OUTPUT_SCHEMA_NAME], label="published training trace schema")
    report = _json_object(observed[REPORT_NAME], label="published training trace report")
    _validate_schema(schema)
    _validate_report(report, schema)


def _require_exact_reauthoring(evidence_dir: Path, expected_dir: Path) -> None:
    observed = _capture_evidence_directory(evidence_dir)
    expected = _capture_evidence_directory(expected_dir)
    if observed != expected:
        raise InputError(
            "training trace evidence does not match exact re-authoring",
            details={
                "evidence_dir": str(evidence_dir),
                "expected_files": sorted(expected),
                "observed_files": sorted(observed),
            },
        )
    schema = _json_object(observed[OUTPUT_SCHEMA_NAME], label="training trace schema")
    report = _json_object(observed[REPORT_NAME], label="training trace report")
    _validate_schema(schema)
    _validate_report(report, schema)


def _capture_evidence_directory(directory: Path) -> dict[str, bytes]:
    try:
        metadata = directory.lstat()
    except OSError as exc:
        raise InputError(
            "training trace evidence directory is missing",
            details={"path": str(directory)},
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise InputError("training trace evidence path must be a non-symlink directory")
    expected_names = {
        REQUESTS_NAME,
        TRAINING_CONFIG_NAME,
        REPORT_NAME,
        OUTPUT_SCHEMA_NAME,
        CHECKSUMS_NAME,
    }
    captured: dict[str, bytes] = {}
    for path in directory.iterdir():
        file_metadata = path.lstat()
        if not stat.S_ISREG(file_metadata.st_mode):
            raise InputError("training trace evidence contains a non-regular file")
        captured[path.name] = path.read_bytes()
    if set(captured) != expected_names:
        raise InputError(
            "training trace evidence file inventory is not closed",
            details={"expected": sorted(expected_names), "observed": sorted(captured)},
        )
    return captured


def _verify_producer_invocation(*, producer_git_commit: str, container_image: str) -> None:
    expected_container = os.environ.get("GENO_LEWM_TRAINING_TRACE_DECLARED_CONTAINER_IMAGE")
    if expected_container != container_image:
        raise InputError(
            "container image does not match the launcher declaration",
            details={"expected": expected_container, "observed": container_image},
        )
    observed_commit = _git_output("rev-parse", "HEAD")
    if observed_commit != producer_git_commit:
        raise InputError(
            "producer Git commit does not match the checked-out source",
            details={"expected": producer_git_commit, "observed": observed_commit},
        )
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise InputError("training trace producer checkout must be clean")
    origin = _git_output("remote", "get-url", "origin")
    if origin != _CANONICAL_ORIGIN:
        raise InputError(
            "training trace producer origin is not canonical",
            details={"origin": origin},
        )
    _verify_public_git_commit(producer_git_commit)
    for relative in (
        "geno_lewm/training/_data_stream.py",
        "geno_lewm/training/real.py",
        "tools/data/v03_training_trace.py",
        "configs/data_v03/training-trace.schema.json",
    ):
        _git_output("cat-file", "-e", f"{producer_git_commit}:{relative}")


def _verify_public_git_commit(commit: str) -> None:
    url = f"{_GITHUB_API_ENDPOINT}/repos/{_GITHUB_REPOSITORY}/commits/{commit}"
    request = urllib_request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "GenoLeWM-v03-training-trace",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib_request.urlopen(request, timeout=30) as response:
            if response.status != 200 or response.geturl() != url:
                raise InputError(
                    "canonical GitHub commit lookup did not resolve exactly",
                    details={"status": response.status, "resolved_url": response.geturl()},
                )
            payload = json.loads(response.read())
    except InputError:
        raise
    except (OSError, TimeoutError, urllib_error.URLError, json.JSONDecodeError) as exc:
        raise InputError(
            "producer Git commit is not publicly verifiable on canonical GitHub",
            details={"commit": commit, "error": str(exc)},
        ) from exc
    if not isinstance(payload, dict) or payload.get("sha") != commit:
        raise InputError(
            "canonical GitHub commit lookup returned a different source identity",
            details={
                "expected": commit,
                "observed": payload.get("sha") if isinstance(payload, dict) else None,
            },
        )


def _verify_dataset_publication_binding(
    *,
    dataset_dir: Path,
    repository: str,
    revision: str,
    artifact_path: str,
) -> bytes:
    """Bind a fully verified local snapshot to one exact remote namespace."""
    repository = _require_dataset_repository(repository)
    revision = _require_dataset_revision(revision)
    artifact_path = _require_dataset_artifact_path(artifact_path)
    verify_v03_dataset_snapshot(dataset_dir)
    expected_inventory = _regular_inventory(dataset_dir)
    checksums_path = dataset_dir / "SHA256SUMS"
    checksums_bytes = _read_regular_bytes(checksums_path, label="dataset SHA256SUMS")
    _verify_remote_dataset_namespace(
        repository=repository,
        revision=revision,
        artifact_path=artifact_path,
        local_root=dataset_dir,
        expected_inventory=expected_inventory,
        expected_checksums=checksums_bytes,
    )
    return checksums_bytes


def _verify_remote_dataset_namespace(
    *,
    repository: str,
    revision: str,
    artifact_path: str,
    local_root: Path,
    expected_inventory: set[str],
    expected_checksums: bytes,
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
        resolved_revision = getattr(info, "sha", None)
        if resolved_revision != revision:
            raise InputError(
                "Hugging Face resolved revision differs from dataset_revision",
                details={"expected": revision, "observed": resolved_revision},
            )
        prefix = f"{artifact_path}/"
        siblings = getattr(info, "siblings", None)
        if not isinstance(siblings, list):
            raise InputError("Hugging Face repository metadata omitted sibling identities")
        remote_files = {
            name.removeprefix(prefix): sibling
            for sibling in siblings
            if isinstance((name := getattr(sibling, "rfilename", None)), str)
            and name.startswith(prefix)
        }
        remote_inventory = set(remote_files)
        if remote_inventory != expected_inventory:
            raise InputError(
                "exact-revision dataset namespace inventory differs from the local snapshot",
                details={
                    "missing": sorted(expected_inventory - remote_inventory),
                    "unexpected": sorted(remote_inventory - expected_inventory),
                },
            )
        _verify_remote_artifact_content_identities(
            local_root=local_root,
            remote_files=remote_files,
            expected_checksums=expected_checksums,
        )
    except InputError:
        raise
    except Exception as exc:
        raise InputError(
            "cannot verify the exact-revision Hugging Face dataset namespace",
            details={
                "repository": repository,
                "revision": revision,
                "artifact_path": artifact_path,
                "error": str(exc),
            },
        ) from exc


def _verify_remote_artifact_content_identities(
    *,
    local_root: Path,
    remote_files: Mapping[str, object],
    expected_checksums: bytes,
) -> None:
    checksum_entries = _parse_checksum_entries(expected_checksums)
    for relative, sibling in remote_files.items():
        path = local_root / relative
        size = path.stat().st_size
        remote_size = getattr(sibling, "size", None)
        if remote_size != size:
            raise InputError(
                "exact-revision dataset artifact size differs from the local snapshot",
                details={"path": relative, "expected": size, "observed": remote_size},
            )
        lfs = getattr(sibling, "lfs", None)
        if lfs is not None:
            remote_sha256 = getattr(lfs, "sha256", None)
            local_sha256 = checksum_entries.get(relative)
            if remote_sha256 != local_sha256:
                raise InputError(
                    "exact-revision LFS identity differs from the verified checksum closure",
                    details={
                        "path": relative,
                        "expected": local_sha256,
                        "observed": remote_sha256,
                    },
                )
            continue
        remote_blob_id = getattr(sibling, "blob_id", None)
        local_blob_id = _git_blob_sha1(path)
        if remote_blob_id != local_blob_id:
            raise InputError(
                "exact-revision Git blob identity differs from the local snapshot",
                details={
                    "path": relative,
                    "expected": local_blob_id,
                    "observed": remote_blob_id,
                },
            )


def _parse_checksum_entries(body: bytes) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError("dataset SHA256SUMS is not UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
            raise InputError(
                "dataset SHA256SUMS entry is malformed",
                details={"line": line_number},
            )
        digest, relative = parts
        if relative in entries:
            raise InputError(
                "dataset SHA256SUMS contains a duplicate path",
                details={"path": relative},
            )
        entries[relative] = digest
    return entries


def _git_blob_sha1(path: Path) -> str:
    body = _read_regular_bytes(path, label=f"dataset artifact {path.name}")
    size = len(body)
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    digest.update(body)
    return digest.hexdigest()


def _regular_inventory(root: Path) -> set[str]:
    inventory: set[str] = set()
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise InputError(
                "verified dataset snapshot contains a non-regular artifact",
                details={"path": str(path)},
            )
        inventory.add(path.relative_to(root).as_posix())
    return inventory


def _git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(_REPOSITORY_ROOT), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InputError(
            "training trace producer Git state cannot be verified",
            details={"command": ["git", *arguments]},
        ) from exc
    return completed.stdout.strip()


def _require_commit(value: object) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise InputError("producer_git_commit must be an exact lowercase 40-hex commit")
    return value


def _require_container_image(value: object) -> str:
    if not isinstance(value, str) or _CONTAINER_IMAGE.fullmatch(value) is None:
        raise InputError("container_image must be digest-pinned")
    return value


def _require_dataset_repository(value: object) -> str:
    if value != _CANONICAL_DATASET_REPOSITORY:
        raise InputError(
            "dataset_repository must identify the canonical public dataset",
            details={"expected": _CANONICAL_DATASET_REPOSITORY, "observed": value},
        )
    return _CANONICAL_DATASET_REPOSITORY


def _require_dataset_revision(value: object) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise InputError("dataset_revision must be an exact lowercase 40-hex Hub commit")
    return value


def _require_dataset_artifact_path(value: object) -> str:
    if not isinstance(value, str) or _DATASET_ARTIFACT_PATH.fullmatch(value) is None:
        raise InputError("dataset_artifact_path must identify a successful v0.3 candidate")
    normalized = PurePosixPath(value)
    if normalized.is_absolute() or normalized.as_posix() != value:
        raise InputError("dataset_artifact_path must be a normalized relative Hub path")
    if any(part in {"", ".", ".."} for part in normalized.parts):
        raise InputError("dataset_artifact_path contains an unsafe path component")
    return value


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
        raise InputError(f"failed to read {label}", details={"path": str(path)}) from exc
    try:
        before = os.fstat(descriptor)
        if _stable_file_identity(initial) != _stable_file_identity(before):
            raise InputError(f"{label} changed before it could be read")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            body = handle.read()
        after = os.fstat(descriptor)
        observed = path.lstat()
    except OSError as exc:
        raise InputError(f"failed to read {label}", details={"path": str(path)}) from exc
    finally:
        os.close(descriptor)
    if _stable_file_identity(before) != _stable_file_identity(after) or _stable_file_identity(
        after
    ) != _stable_file_identity(observed):
        raise InputError(f"{label} changed while it was being read")
    return body


def _stable_file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _json_object(body: bytes, *, label: str) -> dict[str, object]:
    try:
        decoded = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"{label} JSON is invalid") from exc
    if not isinstance(decoded, dict):
        raise InputError(f"{label} must be a JSON object")
    return cast(dict[str, object], decoded)


def _identity(path: str, body: bytes) -> dict[str, object]:
    return {"path": path, "sha256": sha256_bytes(body), "size_bytes": len(body)}


def _bare_sha256(body: bytes) -> str:
    return sha256_bytes(body).removeprefix("sha256:")


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--producer-git-commit", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--dataset-repository", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-artifact-path", required=True)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="re-author the exact trace and byte-compare it with --output-dir",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kwargs = {
        "dataset_dir": args.dataset_dir,
        "training_config_path": args.training_config,
        "producer_git_commit": args.producer_git_commit,
        "container_image": args.container_image,
        "dataset_repository": args.dataset_repository,
        "dataset_revision": args.dataset_revision,
        "dataset_artifact_path": args.dataset_artifact_path,
    }
    if args.verify_existing:
        report = verify_training_trace(evidence_dir=args.output_dir, **kwargs)
    else:
        report = build_training_trace(output_dir=args.output_dir, **kwargs)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI contract tests.
    raise SystemExit(main())
