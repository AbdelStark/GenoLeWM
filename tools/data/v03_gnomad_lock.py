# SPDX-License-Identifier: Apache-2.0
"""Validate and resolve the generation-pinned gnomAD v0.3 source lock."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import importlib
import json
import math
import os
import re
import shlex
import stat
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from geno_lewm.data._v03_evidence_contract import (
    GNOMAD_REMOTE_POSTFLIGHT_SCHEMA_VERSION as REMOTE_POSTFLIGHT_SCHEMA_VERSION,
    GNOMAD_SOURCE_LOCK_SCHEMA_VERSION as LOCK_SCHEMA_VERSION,
    GNOMAD_STAGING_RECEIPT_SCHEMA_VERSION as STAGING_RECEIPT_SCHEMA_VERSION,
)
from tools.data._immutable_json import ImmutableJsonError, write_immutable_json

SELECTION_SCHEMA_VERSION = "geno-lewm.gnomad-stage-selection.v1"
METADATA_VERIFICATION_SCHEMA_VERSION = "geno-lewm.gnomad-gcs-metadata-verification.v1"
SOURCE_IDENTITY_SCHEMA_VERSION = "geno-lewm.gnomad-stream-identity.v1"
_HASH_CHUNK_SIZE = 1 << 20
_PARQUET_AUDIT_BATCH_ROWS = 131_072
_HF_UPLOAD_MAX_ATTEMPTS = 12
_HF_PARENT_CONFLICT_STATUS = 412
_HF_PARENT_CONFLICT_MESSAGE = "A commit has happened since. Please refresh and try again."
_AUTOSOMES = frozenset(str(chromosome) for chromosome in range(1, 23))


@dataclass(frozen=True, slots=True)
class SourceLockSnapshot:
    """One byte-captured source lock and its referenced checked schema."""

    lock_path: Path
    lock_bytes: bytes
    lock: Mapping[str, object]
    schema_path: Path
    schema_bytes: bytes
    schema: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _BinarySnapshot:
    stream: BinaryIO
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _CapturedJson:
    path: Path
    payload: bytes
    value: Mapping[str, object]
    sha256: str
    size_bytes: int


_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_CONTAINER_IMAGE = re.compile(r"[^@]+@sha256:[0-9a-f]{64}")
_SAFE_HF_NAMESPACE = re.compile(r"[A-Za-z0-9._/-]+")
_REMOTE_NAMESPACE_FILES = frozenset(
    {
        "data/gnomad/v4.1/variants.parquet",
        "evidence/gcs-metadata-verification.json",
        "evidence/gcs-object-metadata.json",
        "evidence/prepare-report.json",
        "evidence/receipt.json",
        "evidence/selection.json",
        "evidence/source-lock.json",
        "evidence/source-lock.schema.json",
        "evidence/source-stream-identity.json",
    }
)
_SOURCE_LOCK_REPO_PATH = "configs/data_v03/gnomad-v4.1-exomes-autosomes.source-lock.json"
_SOURCE_LOCK_SCHEMA_REPO_PATH = (
    "configs/data_v03/gnomad-v4.1-exomes-autosomes.source-lock.schema.json"
)


class SourceLockError(ValueError):
    """Raised when a v0.3 gnomAD source lock violates its closed contract."""


def main(argv: list[str] | None = None) -> int:
    """Validate one lock selection and write its resolved contract as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select", help="select one checked autosome entry")
    select_parser.add_argument("--lock-json", type=Path, required=True)
    select_parser.add_argument("--chromosome", required=True)
    select_parser.add_argument("--commit-sha", required=True)
    select_parser.add_argument("--container-image", required=True)
    select_parser.add_argument("--output-json", type=Path, required=True)
    metadata_parser = subparsers.add_parser(
        "verify-metadata", help="verify live GCS metadata against a selected lock entry"
    )
    metadata_parser.add_argument("--selection-json", type=Path, required=True)
    metadata_parser.add_argument("--metadata-json", type=Path, required=True)
    metadata_parser.add_argument("--output-json", type=Path, required=True)
    hash_parser = subparsers.add_parser(
        "hash-source", help="verify the downloaded bytes and record their streamed SHA-256"
    )
    hash_parser.add_argument("--selection-json", type=Path, required=True)
    hash_parser.add_argument("--input-vcf", type=Path, required=True)
    hash_parser.add_argument("--output-json", type=Path, required=True)
    receipt_parser = subparsers.add_parser(
        "author-receipt", help="reconcile source, transform, runtime, and output evidence"
    )
    receipt_parser.add_argument("--selection-json", type=Path, required=True)
    receipt_parser.add_argument("--metadata-verification-json", type=Path, required=True)
    receipt_parser.add_argument("--source-identity-json", type=Path, required=True)
    receipt_parser.add_argument("--prepare-report-json", type=Path, required=True)
    receipt_parser.add_argument("--input-vcf", type=Path, required=True)
    receipt_parser.add_argument("--dataset-root", type=Path, required=True)
    receipt_parser.add_argument("--output-parquet", type=Path, required=True)
    receipt_parser.add_argument("--output-json", type=Path, required=True)
    probe_parser = subparsers.add_parser(
        "probe-namespace", help="prove one immutable Hugging Face namespace is absent"
    )
    probe_parser.add_argument("--repo-id", required=True)
    probe_parser.add_argument("--repo-type", required=True)
    probe_parser.add_argument("--namespace", required=True)
    publish_parser = subparsers.add_parser(
        "publish", help="publish one immutable shard with stale-parent conflict retries"
    )
    publish_parser.add_argument("--repo-id", required=True)
    publish_parser.add_argument("--repo-type", required=True)
    publish_parser.add_argument("--namespace", required=True)
    publish_parser.add_argument("--publish-dir", type=Path, required=True)
    publish_parser.add_argument("--commit-message", required=True)
    postflight_parser = subparsers.add_parser(
        "remote-postflight",
        help="download and verify one staging namespace at an immutable Hub revision",
    )
    postflight_parser.add_argument("--repo-id", required=True)
    postflight_parser.add_argument("--revision", required=True)
    postflight_parser.add_argument("--namespace", required=True)
    postflight_parser.add_argument("--expected-source-commit", required=True)
    postflight_parser.add_argument("--expected-chromosome", required=True)
    postflight_parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "select":
            payload = select_source(
                args.lock_json,
                chromosome=args.chromosome,
                commit_sha=args.commit_sha,
                container_image=args.container_image,
            )
            _write_json(args.output_json, payload)
        elif args.command == "verify-metadata":
            payload = verify_gcs_metadata(args.selection_json, args.metadata_json)
            _write_json(args.output_json, payload)
        elif args.command == "hash-source":
            payload = hash_source(args.selection_json, args.input_vcf)
            _write_json(args.output_json, payload)
        elif args.command == "author-receipt":
            payload = author_receipt(
                selection_path=args.selection_json,
                metadata_verification_path=args.metadata_verification_json,
                source_identity_path=args.source_identity_json,
                prepare_report_path=args.prepare_report_json,
                input_vcf=args.input_vcf,
                dataset_root=args.dataset_root,
                output_parquet=args.output_parquet,
            )
            _write_json(args.output_json, payload)
        elif args.command == "probe-namespace":
            print(
                probe_hf_namespace_absent(
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    namespace=args.namespace,
                    token=_require_hf_token(),
                )
            )
        elif args.command == "publish":
            commit = publish_gnomad_folder(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                namespace=args.namespace,
                publish_dir=args.publish_dir,
                commit_message=args.commit_message,
                token=_require_hf_token(),
            )
            print(f"uploaded commit: {_require_str(getattr(commit, 'oid', None), 'commit.oid')}")
        elif args.command == "remote-postflight":
            payload = verify_remote_gnomad_namespace(
                repo_id=args.repo_id,
                revision=args.revision,
                namespace=args.namespace,
                expected_source_commit=args.expected_source_commit,
                expected_chromosome=args.expected_chromosome,
                token=os.environ.get("HF_TOKEN"),
            )
            _write_json(args.output_json, payload)
    except (OSError, json.JSONDecodeError, ImmutableJsonError, SourceLockError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def verify_remote_gnomad_namespace(
    *,
    repo_id: str,
    revision: str,
    namespace: str,
    expected_source_commit: str,
    expected_chromosome: str,
    token: str | None,
) -> dict[str, object]:
    """Verify one complete staging namespace downloaded at an immutable Hub commit."""
    _validate_remote_postflight_request(
        repo_id=repo_id,
        revision=revision,
        namespace=namespace,
        expected_source_commit=expected_source_commit,
        expected_chromosome=expected_chromosome,
    )
    trusted_lock_bytes = _read_git_blob_at_commit(
        expected_source_commit,
        path=_SOURCE_LOCK_REPO_PATH,
    )
    trusted_schema_bytes = _read_git_blob_at_commit(
        expected_source_commit,
        path=_SOURCE_LOCK_SCHEMA_REPO_PATH,
    )
    trusted_lock = _decode_json_mapping(trusted_lock_bytes, "trusted source lock")
    _validate_lock(trusted_lock)
    trusted_namespace = _locked_namespace(
        trusted_lock,
        lock_sha256=hashlib.sha256(trusted_lock_bytes).hexdigest(),
        chromosome=expected_chromosome,
        source_commit=expected_source_commit,
    )
    if namespace != trusted_namespace:
        raise SourceLockError(
            "requested namespace drifted from the trusted source lock: "
            f"expected {trusted_namespace!r}, observed {namespace!r}"
        )
    try:
        hub = importlib.import_module("huggingface_hub")
        api = hub.HfApi(token=token)
        repo_info = api.repo_info(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            files_metadata=False,
        )
        resolved_revision = _require_str(
            getattr(repo_info, "sha", None), "Hugging Face repository revision"
        )
        _require_equal(resolved_revision, revision, "Hugging Face resolved revision")
        repo_files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )
    except SourceLockError:
        raise
    except Exception as exc:
        raise SourceLockError(
            f"cannot inspect exact Hugging Face revision {revision}: {exc}"
        ) from exc

    remote_files = _remote_namespace_file_set(repo_files, namespace=namespace)
    if remote_files != _REMOTE_NAMESPACE_FILES:
        raise SourceLockError(
            "remote namespace file set drifted: "
            f"missing={sorted(_REMOTE_NAMESPACE_FILES - remote_files)}, "
            f"unexpected={sorted(remote_files - _REMOTE_NAMESPACE_FILES)}"
        )

    with tempfile.TemporaryDirectory(prefix="geno-lewm-gnomad-postflight-") as temporary:
        local_paths: dict[str, Path] = {}
        cache_directory = Path(temporary) / "hf-cache"
        for relative_path in sorted(_REMOTE_NAMESPACE_FILES):
            filename = f"{namespace}/{relative_path}"
            try:
                downloaded = hub.hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=revision,
                    filename=filename,
                    token=token,
                    local_dir=temporary,
                    cache_dir=cache_directory,
                    force_download=True,
                )
            except Exception as exc:
                raise SourceLockError(
                    f"cannot download {filename!r} at exact revision {revision}: {exc}"
                ) from exc
            local_path = Path(downloaded)
            if not local_path.is_file():
                raise SourceLockError(
                    f"exact-revision download is not a regular file: {relative_path}"
                )
            local_paths[relative_path] = local_path

        report = _audit_remote_gnomad_bundle(
            local_paths,
            repo_id=repo_id,
            revision=revision,
            namespace=namespace,
            expected_source_commit=expected_source_commit,
            expected_chromosome=expected_chromosome,
            trusted_lock_bytes=trusted_lock_bytes,
            trusted_schema_bytes=trusted_schema_bytes,
        )
    return report


def _read_git_blob_at_commit(commit_sha: str, *, path: str) -> bytes:
    """Read one trusted source artifact directly from an exact local Git commit."""
    try:
        result = subprocess.run(
            ["git", "cat-file", "blob", f"{commit_sha}:{path}"],
            check=False,
            capture_output=True,
            env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
        )
    except OSError as exc:
        raise SourceLockError(f"cannot read trusted source artifact from Git: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise SourceLockError(
            f"cannot read trusted source artifact {path!r} at {commit_sha}: {detail}"
        )
    return result.stdout


def _validate_remote_postflight_request(
    *,
    repo_id: str,
    revision: str,
    namespace: str,
    expected_source_commit: str,
    expected_chromosome: str,
) -> None:
    if repo_id.count("/") != 1 or any(not part for part in repo_id.split("/")):
        raise SourceLockError("Hugging Face repo_id must be a namespace/name pair")
    if _COMMIT_SHA.fullmatch(revision) is None:
        raise SourceLockError(
            "Hugging Face revision must be a full lowercase 40-character commit SHA"
        )
    if _COMMIT_SHA.fullmatch(expected_source_commit) is None:
        raise SourceLockError(
            "expected source commit must be a full lowercase 40-character Git SHA"
        )
    if expected_chromosome not in _AUTOSOMES:
        raise SourceLockError(
            f"expected chromosome must be one of 1..22, got {expected_chromosome!r}"
        )
    if (
        namespace != namespace.strip("/")
        or not namespace.startswith("staging/v0.3/")
        or _SAFE_HF_NAMESPACE.fullmatch(namespace) is None
    ):
        raise SourceLockError("Hugging Face namespace must be below staging/v0.3 without slashes")
    parts = namespace.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise SourceLockError("Hugging Face namespace contains an unsafe path component")


def _remote_namespace_file_set(repo_files: object, *, namespace: str) -> frozenset[str]:
    if not isinstance(repo_files, list) or any(not isinstance(path, str) for path in repo_files):
        raise SourceLockError("Hugging Face file listing must be an array of paths")
    prefix = f"{namespace}/"
    relative_paths = [path.removeprefix(prefix) for path in repo_files if path.startswith(prefix)]
    if len(relative_paths) != len(set(relative_paths)):
        raise SourceLockError("Hugging Face namespace file listing contains duplicate paths")
    return frozenset(relative_paths)


def _audit_remote_gnomad_bundle(
    paths: Mapping[str, Path],
    *,
    repo_id: str,
    revision: str,
    namespace: str,
    expected_source_commit: str,
    expected_chromosome: str,
    trusted_lock_bytes: bytes,
    trusted_schema_bytes: bytes,
) -> dict[str, object]:
    selection_path = paths["evidence/selection.json"]
    metadata_path = paths["evidence/gcs-object-metadata.json"]
    metadata_verification_path = paths["evidence/gcs-metadata-verification.json"]
    source_identity_path = paths["evidence/source-stream-identity.json"]
    prepare_report_path = paths["evidence/prepare-report.json"]
    receipt_path = paths["evidence/receipt.json"]
    source_lock_path = paths["evidence/source-lock.json"]
    source_lock_schema_path = paths["evidence/source-lock.schema.json"]
    parquet_path = paths["data/gnomad/v4.1/variants.parquet"]

    captures = {
        "evidence/source-lock.json": _capture_json(source_lock_path, "remote source lock"),
        "evidence/source-lock.schema.json": _capture_json(
            source_lock_schema_path, "remote source lock schema"
        ),
        "evidence/selection.json": _capture_json(selection_path, "remote selection"),
        "evidence/gcs-object-metadata.json": _capture_json(metadata_path, "remote GCS metadata"),
        "evidence/gcs-metadata-verification.json": _capture_json(
            metadata_verification_path, "remote metadata verification"
        ),
        "evidence/source-stream-identity.json": _capture_json(
            source_identity_path, "remote source identity"
        ),
        "evidence/prepare-report.json": _capture_json(prepare_report_path, "remote prepare report"),
        "evidence/receipt.json": _capture_json(receipt_path, "remote receipt"),
    }
    lock_capture = captures["evidence/source-lock.json"]
    schema_capture = captures["evidence/source-lock.schema.json"]
    lock = lock_capture.value
    schema = schema_capture.value
    _require_trusted_blob(
        lock_capture.payload,
        trusted_lock_bytes,
        "remote source lock bytes at source commit",
    )
    _require_trusted_blob(
        schema_capture.payload,
        trusted_schema_bytes,
        "remote source lock schema bytes at source commit",
    )
    _validate_lock(lock)
    _require_equal(
        schema.get("$schema"),
        "https://json-schema.org/draft/2020-12/schema",
        "remote source lock schema.$schema",
    )
    _require_equal(
        schema.get("additionalProperties"),
        False,
        "remote source lock schema.additionalProperties",
    )

    selection_capture = captures["evidence/selection.json"]
    selection = selection_capture.value
    _validate_remote_selection(
        selection,
        lock=lock,
        lock_sha256=lock_capture.sha256,
        schema_sha256=schema_capture.sha256,
        repo_id=repo_id,
        namespace=namespace,
        expected_source_commit=expected_source_commit,
        expected_chromosome=expected_chromosome,
    )

    metadata_verification = captures["evidence/gcs-metadata-verification.json"].value
    metadata_capture = captures["evidence/gcs-object-metadata.json"]
    recomputed_metadata = _verify_gcs_metadata_captures(selection_capture, metadata_capture)
    _require_equal(
        dict(metadata_verification),
        recomputed_metadata,
        "remote metadata verification",
    )

    source_identity = captures["evidence/source-stream-identity.json"].value
    _validate_remote_source_identity(
        source_identity,
        selection=selection,
        selection_sha256=selection_capture.sha256,
    )
    prepare_report = captures["evidence/prepare-report.json"].value
    with _private_binary_snapshot(parquet_path) as parquet_snapshot:
        counts, runtime, output_identity = _validate_remote_prepare_report(
            prepare_report,
            selection=selection,
            source_identity=source_identity,
            parquet_path=parquet_path,
            parquet_sha256=parquet_snapshot.sha256,
            parquet_size_bytes=parquet_snapshot.size_bytes,
        )

        source = _require_mapping(selection.get("source"), "selection.source")
        transform = _require_mapping(selection.get("transform"), "selection.transform")
        parquet_audit = _audit_gnomad_parquet_stream(
            parquet_snapshot.stream,
            chromosome=_require_str(source.get("chromosome"), "selection.source.chromosome"),
            expected_records=counts["records_written"],
            min_af=_require_number(transform.get("min_af"), "selection.transform.min_af"),
            max_allele_len=_require_int(
                transform.get("max_allele_len"), "selection.transform.max_allele_len"
            ),
        )
        receipt = captures["evidence/receipt.json"].value
        _validate_remote_receipt(
            receipt,
            selection=selection,
            source_identity=source_identity,
            prepare_report=prepare_report,
            evidence_captures={
                "selection": selection_capture,
                "metadata_verification": captures["evidence/gcs-metadata-verification.json"],
                "source_identity": captures["evidence/source-stream-identity.json"],
                "prepare_report": captures["evidence/prepare-report.json"],
            },
            counts=counts,
            runtime=runtime,
            output_identity=output_identity,
            parquet_audit=parquet_audit,
        )

        file_identities = {
            relative: (
                {
                    "sha256": parquet_snapshot.sha256,
                    "size_bytes": parquet_snapshot.size_bytes,
                }
                if path == parquet_path
                else {
                    "sha256": captures[relative].sha256,
                    "size_bytes": captures[relative].size_bytes,
                }
            )
            for relative, path in sorted(paths.items())
        }
    return {
        "schema_version": REMOTE_POSTFLIGHT_SCHEMA_VERSION,
        "ok": True,
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "namespace": namespace,
        "source_commit": expected_source_commit,
        "chromosome": expected_chromosome,
        "verified_files": sorted(paths),
        "file_identities": file_identities,
        "parquet_audit": parquet_audit,
        "checks": [
            "exact_hub_revision_resolved",
            "complete_namespace_file_set",
            "source_lock_and_schema_bound",
            "source_lock_and_schema_match_source_commit",
            "selection_rederived_from_source_lock",
            "metadata_verification_recomputed",
            "receipt_evidence_identities_recomputed",
            "parquet_sha256_and_size_recomputed",
            "parquet_full_scan_recomputed",
        ],
    }


def _validate_remote_selection(
    selection: Mapping[str, object],
    *,
    lock: Mapping[str, object],
    lock_sha256: str,
    schema_sha256: str,
    repo_id: str,
    namespace: str,
    expected_source_commit: str,
    expected_chromosome: str,
) -> None:
    _require_exact_keys(
        selection,
        {
            "schema_version",
            "source_lock",
            "dataset_id",
            "release",
            "reference_genome",
            "source",
            "transform",
            "execution",
            "publication",
            "claim_boundary",
        },
        "selection",
    )
    _require_equal(
        selection["schema_version"], SELECTION_SCHEMA_VERSION, "selection.schema_version"
    )
    for field in ("dataset_id", "release", "reference_genome", "claim_boundary"):
        _require_equal(selection[field], lock[field], f"selection.{field}")

    lock_binding = _require_mapping(selection["source_lock"], "selection.source_lock")
    _require_exact_keys(
        lock_binding,
        {"path", "sha256", "schema_version", "schema"},
        "selection.source_lock",
    )
    _require_equal(lock_binding["path"], _SOURCE_LOCK_REPO_PATH, "selection.source_lock.path")
    _require_equal(lock_binding["sha256"], lock_sha256, "selection.source_lock.sha256")
    _require_equal(
        lock_binding["schema_version"], LOCK_SCHEMA_VERSION, "selection.source_lock.schema_version"
    )
    schema_binding = _require_mapping(lock_binding["schema"], "selection.source_lock.schema")
    _require_exact_keys(
        schema_binding,
        {"path", "sha256", "draft"},
        "selection.source_lock.schema",
    )
    _require_equal(
        schema_binding["path"],
        _SOURCE_LOCK_SCHEMA_REPO_PATH,
        "selection.source_lock.schema.path",
    )
    _require_equal(schema_binding["sha256"], schema_sha256, "selection.source_lock.schema.sha256")
    _require_equal(
        schema_binding["draft"],
        "https://json-schema.org/draft/2020-12/schema",
        "selection.source_lock.schema.draft",
    )

    source = _require_mapping(selection["source"], "selection.source")
    source_config = _require_mapping(lock["source"], "source lock.source")
    selected = next(
        _require_mapping(entry, "source lock.objects[]")
        for entry in _require_list(lock["objects"], "source lock.objects")
        if _require_mapping(entry, "source lock.objects[]").get("chromosome") == expected_chromosome
    )
    object_name = _require_str(selected["object"], "source lock.objects[].object")
    generation = _require_str(selected["generation"], "source lock.objects[].generation")
    bucket = _require_str(source_config["bucket"], "source lock.source.bucket")
    encoded_object = urllib.parse.quote(object_name, safe="")
    expected_source = {
        "bucket": bucket,
        "chromosome": expected_chromosome,
        "split_role": selected["split_role"],
        "object": object_name,
        "generation": generation,
        "size_bytes": selected["size_bytes"],
        "md5_base64": selected["md5_base64"],
        "md5_hex": base64.b64decode(
            _require_str(selected["md5_base64"], "source lock.objects[].md5_base64"),
            validate=True,
        ).hex(),
        "metadata_url": (
            f"{source_config['metadata_api']}/b/{urllib.parse.quote(bucket, safe='')}/"
            f"o/{encoded_object}?generation={generation}"
        ),
        "media_url": (
            f"{source_config['media_api']}/b/{urllib.parse.quote(bucket, safe='')}/"
            f"o/{encoded_object}?alt=media&generation={generation}"
        ),
    }
    _require_exact_keys(source, set(expected_source), "selection.source")
    _require_equal(dict(source), expected_source, "selection.source")
    _require_equal(selection["transform"], lock["transform"], "selection.transform")

    job = _require_mapping(lock["job"], "source lock.job")
    execution = _require_mapping(selection["execution"], "selection.execution")
    expected_execution = {
        "commit_sha": expected_source_commit,
        "container_image": job["container_image"],
        "repository": job["repository"],
    }
    _require_exact_keys(execution, set(expected_execution), "selection.execution")
    _require_equal(dict(execution), expected_execution, "selection.execution")
    expected_namespace = _locked_namespace(
        lock,
        lock_sha256=lock_sha256,
        chromosome=expected_chromosome,
        source_commit=expected_source_commit,
    )
    _require_equal(namespace, expected_namespace, "requested namespace")
    publication = _require_mapping(selection["publication"], "selection.publication")
    expected_publication = {
        "repo": repo_id,
        "repo_type": "dataset",
        "namespace": expected_namespace,
    }
    _require_exact_keys(publication, set(expected_publication), "selection.publication")
    _require_equal(dict(publication), expected_publication, "selection.publication")
    _require_equal(job["upload_repo"], repo_id, "source lock.job.upload_repo")


def _locked_namespace(
    lock: Mapping[str, object],
    *,
    lock_sha256: str,
    chromosome: str,
    source_commit: str,
) -> str:
    job = _require_mapping(lock["job"], "source lock.job")
    selected = next(
        _require_mapping(entry, "source lock.objects[]")
        for entry in _require_list(lock["objects"], "source lock.objects")
        if _require_mapping(entry, "source lock.objects[]").get("chromosome") == chromosome
    )
    generation = _require_str(selected["generation"], "source lock.objects[].generation")
    return (
        f"{job['namespace_root']}/lock-{lock_sha256[:12]}/"
        f"chr{chromosome}-g{generation}-{source_commit[:12]}"
    )


def _validate_remote_source_identity(
    source_identity: Mapping[str, object],
    *,
    selection: Mapping[str, object],
    selection_sha256: str,
) -> None:
    _require_exact_keys(
        source_identity,
        {
            "schema_version",
            "ok",
            "selection_sha256",
            "path",
            "size_bytes",
            "md5_base64",
            "md5_hex",
            "sha256",
            "hash_method",
            "chunk_size_bytes",
        },
        "source identity",
    )
    _require_equal(
        source_identity["schema_version"],
        SOURCE_IDENTITY_SCHEMA_VERSION,
        "source identity.schema_version",
    )
    _require_equal(source_identity["ok"], True, "source identity.ok")
    _require_equal(
        source_identity["selection_sha256"], selection_sha256, "source identity.selection_sha256"
    )
    source = _require_mapping(selection["source"], "selection.source")
    for field in ("size_bytes", "md5_base64", "md5_hex"):
        _require_equal(source_identity[field], source[field], f"source identity.{field}")
    _require_sha256(source_identity["sha256"], "source identity.sha256")
    source_path = _require_str(source_identity["path"], "source identity.path")
    expected_basename = _require_str(source["object"], "selection.source.object").rsplit("/", 1)[-1]
    if not source_path.endswith(f"/{expected_basename}"):
        raise SourceLockError("source identity.path drifted from the locked source object name")
    _require_equal(
        source_identity["hash_method"],
        "single_pass_chunked_file_read",
        "source identity.hash_method",
    )
    _require_equal(
        source_identity["chunk_size_bytes"], _HASH_CHUNK_SIZE, "source identity.chunk_size_bytes"
    )


def _validate_remote_prepare_report(
    prepare_report: Mapping[str, object],
    *,
    selection: Mapping[str, object],
    source_identity: Mapping[str, object],
    parquet_path: Path,
    parquet_sha256: str,
    parquet_size_bytes: int,
) -> tuple[dict[str, int], Mapping[str, object], dict[str, object]]:
    _require_exact_keys(
        prepare_report,
        {
            "output_path",
            "input_path",
            "release",
            "records_read",
            "allele_records_seen",
            "records_written",
            "skipped_filter",
            "skipped_af",
            "skipped_allele",
            "input_sha256",
            "output_sha256",
            "input_size_bytes",
            "size_bytes",
            "elapsed_seconds",
            "already_exists",
            "command",
            "input_vcf",
            "output_parquet",
            "runtime",
        },
        "prepare report",
    )
    transform = _require_mapping(selection["transform"], "selection.transform")
    source_path = _require_str(source_identity["path"], "source identity.path")
    output_report = _require_mapping(
        prepare_report["output_parquet"], "prepare report.output_parquet"
    )
    _require_exact_keys(
        output_report, {"path", "sha256", "size_bytes"}, "prepare report.output_parquet"
    )
    output_path = _require_str(output_report["path"], "prepare report.output_parquet.path")
    expected_output_suffix = f"/data/gnomad/{selection['release']}/variants.parquet"
    if not output_path.endswith(expected_output_suffix):
        raise SourceLockError("prepare report.output_parquet.path drifted from the staging layout")
    output_identity = {
        "path": str(parquet_path),
        "sha256": parquet_sha256,
        "size_bytes": parquet_size_bytes,
    }
    _require_equal(
        _require_prefixed_sha256(output_report["sha256"], "prepare report.output_parquet.sha256"),
        output_identity["sha256"],
        "prepare report.output_parquet.sha256",
    )
    _require_equal(
        output_report["size_bytes"],
        output_identity["size_bytes"],
        "prepare report.output_parquet.size_bytes",
    )
    input_report = _require_mapping(prepare_report["input_vcf"], "prepare report.input_vcf")
    _require_exact_keys(input_report, {"path", "sha256", "size_bytes"}, "prepare report.input_vcf")
    _require_equal(input_report["path"], source_path, "prepare report.input_vcf.path")
    source_sha256 = _require_sha256(source_identity["sha256"], "source identity.sha256")
    _require_equal(
        _require_prefixed_sha256(input_report["sha256"], "prepare report.input_vcf.sha256"),
        source_sha256,
        "prepare report.input_vcf.sha256",
    )
    _require_equal(
        input_report["size_bytes"],
        source_identity["size_bytes"],
        "prepare report.input_vcf.size_bytes",
    )

    identity_fields = {
        "output_path": output_path,
        "input_path": source_path,
        "input_size_bytes": source_identity["size_bytes"],
        "size_bytes": output_identity["size_bytes"],
        "release": selection["release"],
        "already_exists": False,
    }
    for field, expected in identity_fields.items():
        _require_equal(prepare_report[field], expected, f"prepare report.{field}")
    _require_equal(
        _require_prefixed_sha256(prepare_report["input_sha256"], "prepare report.input_sha256"),
        source_sha256,
        "prepare report.input_sha256",
    )
    _require_equal(
        _require_prefixed_sha256(prepare_report["output_sha256"], "prepare report.output_sha256"),
        output_identity["sha256"],
        "prepare report.output_sha256",
    )
    elapsed_seconds = _require_number(
        prepare_report["elapsed_seconds"], "prepare report.elapsed_seconds"
    )
    if elapsed_seconds < 0:
        raise SourceLockError("prepare report.elapsed_seconds must be non-negative")

    command = _require_str(transform["command"], "selection.transform.command")
    dataset_root = output_path.removesuffix(f"/gnomad/{selection['release']}/variants.parquet")
    report_argv = [
        command,
        "--input-vcf",
        source_path,
        "--output",
        dataset_root,
        "--release",
        _require_str(selection["release"], "selection.release"),
        "--min-af",
        str(_require_number(transform["min_af"], "selection.transform.min_af")),
        "--max-allele-len",
        str(_require_int(transform["max_allele_len"], "selection.transform.max_allele_len")),
    ]
    _require_equal(prepare_report["command"], shlex.join(report_argv), "prepare report.command")

    count_fields = (
        "records_read",
        "allele_records_seen",
        "records_written",
        "skipped_filter",
        "skipped_af",
        "skipped_allele",
    )
    counts = {
        field: _require_int(prepare_report[field], f"prepare report.{field}")
        for field in count_fields
    }
    if any(value < 0 for value in counts.values()) or counts["records_written"] <= 0:
        raise SourceLockError("prepare report counts must be non-negative with records_written > 0")
    classified = (
        counts["records_written"]
        + counts["skipped_filter"]
        + counts["skipped_af"]
        + counts["skipped_allele"]
    )
    if classified != counts["allele_records_seen"]:
        raise SourceLockError(
            "prepare report allele counts do not reconcile: "
            f"classified={classified}, seen={counts['allele_records_seen']}"
        )

    runtime = _require_mapping(prepare_report["runtime"], "prepare report.runtime")
    _require_exact_keys(
        runtime,
        {"elapsed_seconds", "process_peak_rss_bytes", "peak_memory_note"},
        "prepare report.runtime",
    )
    runtime_elapsed = _require_number(
        runtime["elapsed_seconds"], "prepare report.runtime.elapsed_seconds"
    )
    if runtime_elapsed < 0:
        raise SourceLockError("prepare report.runtime.elapsed_seconds must be non-negative")
    if (
        _require_int(
            runtime["process_peak_rss_bytes"], "prepare report.runtime.process_peak_rss_bytes"
        )
        <= 0
    ):
        raise SourceLockError("prepare report.runtime.process_peak_rss_bytes must be positive")
    _require_str(runtime["peak_memory_note"], "prepare report.runtime.peak_memory_note")
    return counts, runtime, output_identity


def _validate_remote_receipt(
    receipt: Mapping[str, object],
    *,
    selection: Mapping[str, object],
    source_identity: Mapping[str, object],
    prepare_report: Mapping[str, object],
    evidence_captures: Mapping[str, _CapturedJson],
    counts: Mapping[str, int],
    runtime: Mapping[str, object],
    output_identity: Mapping[str, object],
    parquet_audit: Mapping[str, object],
) -> None:
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "created_at",
            "ok",
            "dataset_id",
            "release",
            "reference_genome",
            "source_lock",
            "source",
            "transform",
            "output",
            "execution",
            "publication",
            "evidence",
            "claim_boundary",
        },
        "receipt",
    )
    _require_equal(
        receipt["schema_version"], STAGING_RECEIPT_SCHEMA_VERSION, "receipt.schema_version"
    )
    _require_str(receipt["created_at"], "receipt.created_at")
    _require_equal(receipt["ok"], True, "receipt.ok")
    for field in (
        "dataset_id",
        "release",
        "reference_genome",
        "source_lock",
        "execution",
        "publication",
        "claim_boundary",
    ):
        _require_equal(receipt[field], selection[field], f"receipt.{field}")

    selection_source = _require_mapping(selection["source"], "selection.source")
    expected_receipt_source = {
        "chromosome": selection_source["chromosome"],
        "split_role": selection_source["split_role"],
        "bucket": selection_source["bucket"],
        "object": selection_source["object"],
        "generation": selection_source["generation"],
        "size_bytes": selection_source["size_bytes"],
        "upstream_md5_base64": selection_source["md5_base64"],
        "upstream_md5_hex": selection_source["md5_hex"],
        "streamed_sha256": source_identity["sha256"],
    }
    receipt_source = _require_mapping(receipt["source"], "receipt.source")
    _require_exact_keys(receipt_source, set(expected_receipt_source), "receipt.source")
    _require_equal(dict(receipt_source), expected_receipt_source, "receipt.source")

    selection_transform = _require_mapping(selection["transform"], "selection.transform")
    receipt_transform = _require_mapping(receipt["transform"], "receipt.transform")
    _require_exact_keys(
        receipt_transform, {"command", "argv", "filters", "runtime", "counts"}, "receipt.transform"
    )
    _require_equal(
        receipt_transform["command"], selection_transform["command"], "receipt.transform.command"
    )
    expected_filters = {
        "filter": selection_transform["filter"],
        "min_af": selection_transform["min_af"],
        "max_allele_len": selection_transform["max_allele_len"],
    }
    _require_equal(receipt_transform["filters"], expected_filters, "receipt.transform.filters")
    _require_equal(receipt_transform["runtime"], runtime, "receipt.transform.runtime")
    _require_equal(receipt_transform["counts"], dict(counts), "receipt.transform.counts")
    expected_argv = [
        "uv",
        "run",
        selection_transform["command"],
        "--quiet",
        "--no-banner",
        *shlex.split(_require_str(prepare_report["command"], "prepare report.command"))[1:],
    ]
    _require_equal(receipt_transform["argv"], expected_argv, "receipt.transform.argv")

    receipt_output = _require_mapping(receipt["output"], "receipt.output")
    _require_exact_keys(
        receipt_output, {"path", "sha256", "size_bytes", "parquet_audit"}, "receipt.output"
    )
    for field in ("sha256", "size_bytes"):
        _require_equal(receipt_output[field], output_identity[field], f"receipt.output.{field}")
    _require_equal(
        receipt_output["path"],
        _require_mapping(prepare_report["output_parquet"], "prepare report.output_parquet")["path"],
        "receipt.output.path",
    )
    _require_equal(
        receipt_output["parquet_audit"], dict(parquet_audit), "receipt.output.parquet_audit"
    )

    evidence = _require_mapping(receipt["evidence"], "receipt.evidence")
    expected_evidence = {
        "selection": (
            evidence_captures["selection"],
            "/publish/evidence/selection.json",
        ),
        "metadata_verification": (
            evidence_captures["metadata_verification"],
            "/publish/evidence/gcs-metadata-verification.json",
        ),
        "source_identity": (
            evidence_captures["source_identity"],
            "/publish/evidence/source-stream-identity.json",
        ),
        "prepare_report": (
            evidence_captures["prepare_report"],
            "/publish/evidence/prepare-report.json",
        ),
    }
    _require_exact_keys(evidence, set(expected_evidence), "receipt.evidence")
    for field, (capture, suffix) in expected_evidence.items():
        _validate_remote_file_binding(
            _require_mapping(evidence[field], f"receipt.evidence.{field}"),
            capture=capture,
            expected_path_suffix=suffix,
            field=f"receipt.evidence.{field}",
        )


def _validate_remote_file_binding(
    binding: Mapping[str, object],
    *,
    capture: _CapturedJson,
    expected_path_suffix: str,
    field: str,
) -> None:
    _require_exact_keys(binding, {"path", "sha256", "size_bytes"}, field)
    bound_path = _require_str(binding["path"], f"{field}.path")
    if not bound_path.endswith(expected_path_suffix):
        raise SourceLockError(f"{field}.path drifted from the staging layout")
    identity = {"sha256": capture.sha256, "size_bytes": capture.size_bytes}
    for identity_field in ("sha256", "size_bytes"):
        _require_equal(
            binding[identity_field], identity[identity_field], f"{field}.{identity_field}"
        )


def _require_trusted_blob(observed: bytes, expected: bytes, field: str) -> None:
    if observed != expected:
        raise SourceLockError(
            f"{field} drifted: expected sha256={hashlib.sha256(expected).hexdigest()}, "
            f"observed sha256={hashlib.sha256(observed).hexdigest()}"
        )


def verify_gcs_metadata(selection_path: Path, metadata_path: Path) -> dict[str, object]:
    """Verify generation, size, and upstream MD5 returned by the GCS JSON API."""
    return _verify_gcs_metadata_captures(
        _capture_json(selection_path, "selection"),
        _capture_json(metadata_path, "GCS metadata"),
    )


def _verify_gcs_metadata_captures(
    selection_capture: _CapturedJson,
    metadata_capture: _CapturedJson,
) -> dict[str, object]:
    selection = selection_capture.value
    metadata = metadata_capture.value
    _require_equal(
        selection.get("schema_version"), SELECTION_SCHEMA_VERSION, "selection.schema_version"
    )
    source = _require_mapping(selection.get("source"), "selection.source")

    expected = {
        "bucket": _require_str(source.get("bucket"), "selection.source.bucket"),
        "name": _require_str(source.get("object"), "selection.source.object"),
        "generation": _require_str(source.get("generation"), "selection.source.generation"),
        "size": str(_require_int(source.get("size_bytes"), "selection.source.size_bytes")),
        "md5Hash": _require_str(source.get("md5_base64"), "selection.source.md5_base64"),
    }
    for field, expected_value in expected.items():
        _require_equal(metadata.get(field), expected_value, f"GCS metadata.{field}")

    md5_base64 = expected["md5Hash"]
    return {
        "schema_version": METADATA_VERIFICATION_SCHEMA_VERSION,
        "ok": True,
        "selection_sha256": selection_capture.sha256,
        "bucket": expected["bucket"],
        "object": expected["name"],
        "generation": expected["generation"],
        "size_bytes": int(expected["size"]),
        "md5_base64": md5_base64,
        "md5_hex": base64.b64decode(md5_base64, validate=True).hex(),
        "verified_fields": ["bucket", "name", "generation", "size", "md5Hash"],
    }


def hash_source(selection_path: Path, input_vcf: Path) -> dict[str, object]:
    """Hash downloaded source bytes once and reject size or upstream-MD5 drift."""
    selection_bytes = selection_path.read_bytes()
    selection = _decode_json_mapping(selection_bytes, "selection")
    _require_equal(
        selection.get("schema_version"), SELECTION_SCHEMA_VERSION, "selection.schema_version"
    )
    source = _require_mapping(selection.get("source"), "selection.source")
    expected_size = _require_int(source.get("size_bytes"), "selection.source.size_bytes")
    expected_md5 = _require_str(source.get("md5_base64"), "selection.source.md5_base64")
    if not input_vcf.is_file():
        raise SourceLockError(f"downloaded source is not a regular file: {input_vcf}")

    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    observed_size = 0
    with input_vcf.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
            observed_size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    observed_md5 = base64.b64encode(md5.digest()).decode("ascii")
    _require_equal(observed_size, expected_size, "downloaded source size_bytes")
    _require_equal(observed_md5, expected_md5, "downloaded source MD5")

    return {
        "schema_version": SOURCE_IDENTITY_SCHEMA_VERSION,
        "ok": True,
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "path": str(input_vcf),
        "size_bytes": observed_size,
        "md5_base64": observed_md5,
        "md5_hex": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
        "hash_method": "single_pass_chunked_file_read",
        "chunk_size_bytes": _HASH_CHUNK_SIZE,
    }


def probe_hf_namespace_absent(
    *,
    repo_id: str,
    repo_type: str,
    namespace: str,
    token: str,
    timeout_seconds: float = 30.0,
) -> str:
    """Return the current Hub parent SHA only when ``namespace`` is absent."""
    if repo_type != "dataset":
        raise SourceLockError(f"unsupported locked repo type: {repo_type}")
    if repo_id.count("/") != 1:
        raise SourceLockError("Hugging Face repo_id must be a namespace/name pair")
    normalized_namespace = namespace.strip("/")
    if not normalized_namespace or normalized_namespace != namespace:
        raise SourceLockError("Hugging Face namespace must be non-empty without outer slashes")
    if not token:
        raise SourceLockError("HF_TOKEN must be non-empty")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise SourceLockError("Hugging Face probe timeout_seconds must be finite and positive")

    repo_path = urllib.parse.quote(repo_id, safe="/")
    namespace_path = urllib.parse.quote(normalized_namespace, safe="/")
    repo_url = f"https://huggingface.co/api/datasets/{repo_path}/revision/main"
    tree_url = (
        f"https://huggingface.co/api/datasets/{repo_path}/tree/main/{namespace_path}"
        "?recursive=false&expand=false"
    )
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with urllib.request.urlopen(
            urllib.request.Request(repo_url, headers=headers), timeout=timeout_seconds
        ) as response:
            repo_info: object = json.load(response)
    except urllib.error.HTTPError as exc:
        raise SourceLockError(f"cannot resolve remote parent commit: HTTP {exc.code}") from exc
    except (OSError, ValueError) as exc:
        raise SourceLockError(f"cannot resolve remote parent commit: {exc}") from exc

    repo_sha = _require_str(
        _require_mapping(repo_info, "Hugging Face repository metadata").get("sha"),
        "Hugging Face repository metadata.sha",
    )
    if _COMMIT_SHA.fullmatch(repo_sha) is None:
        raise SourceLockError("remote repository did not report a full lowercase parent commit")

    try:
        with urllib.request.urlopen(
            urllib.request.Request(tree_url, headers=headers), timeout=timeout_seconds
        ) as response:
            json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return repo_sha
        raise SourceLockError(f"cannot prove remote namespace absence: HTTP {exc.code}") from exc
    except (OSError, ValueError) as exc:
        raise SourceLockError(f"cannot prove remote namespace absence: {exc}") from exc
    raise SourceLockError(f"immutable namespace already exists: {namespace}")


def is_hf_parent_head_conflict(exc: BaseException) -> bool:
    """Recognize only the Hub's exact stale-parent precondition failure."""
    try:
        errors = importlib.import_module("huggingface_hub.errors")
    except ImportError:
        return False
    error_type = getattr(errors, "HfHubHTTPError", None)
    if not isinstance(error_type, type) or not isinstance(exc, error_type):
        return False
    response = getattr(exc, "response", None)
    return getattr(
        response, "status_code", None
    ) == _HF_PARENT_CONFLICT_STATUS and _HF_PARENT_CONFLICT_MESSAGE in str(exc)


def upload_folder_with_parent_retry(
    *,
    api: Any,
    repo_id: str,
    repo_type: str,
    namespace: str,
    publish_dir: Path,
    commit_message: str,
    prove_namespace_absent: Callable[[], str],
    max_attempts: int = _HF_UPLOAD_MAX_ATTEMPTS,
    sleep: Callable[[float], object] = time.sleep,
    is_parent_head_conflict: Callable[[BaseException], bool] = is_hf_parent_head_conflict,
) -> Any:
    """Upload one immutable path, retrying only a stale Hub parent precondition."""
    if max_attempts <= 0:
        raise SourceLockError("Hugging Face upload max_attempts must be positive")

    for attempt in range(1, max_attempts + 1):
        parent_commit = prove_namespace_absent()
        if _COMMIT_SHA.fullmatch(parent_commit) is None:
            raise SourceLockError("namespace proof did not return a full lowercase parent commit")
        try:
            return api.upload_folder(
                repo_id=repo_id,
                repo_type=repo_type,
                folder_path=publish_dir,
                path_in_repo=namespace,
                parent_commit=parent_commit,
                commit_message=commit_message,
            )
        except Exception as exc:
            if not is_parent_head_conflict(exc) or attempt == max_attempts:
                raise
            sleep(_hf_parent_conflict_backoff(namespace=namespace, conflict_number=attempt))

    raise AssertionError("bounded Hugging Face upload loop exhausted without returning or raising")


def publish_gnomad_folder(
    *,
    repo_id: str,
    repo_type: str,
    namespace: str,
    publish_dir: Path,
    commit_message: str,
    token: str,
) -> Any:
    """Publish a completed shard through the conflict-safe immutable upload path."""
    if not publish_dir.is_dir():
        raise SourceLockError(f"publish directory is not a directory: {publish_dir}")
    hub = importlib.import_module("huggingface_hub")
    api = hub.HfApi(token=token)
    return upload_folder_with_parent_retry(
        api=api,
        repo_id=repo_id,
        repo_type=repo_type,
        namespace=namespace,
        publish_dir=publish_dir,
        commit_message=commit_message,
        prove_namespace_absent=lambda: probe_hf_namespace_absent(
            repo_id=repo_id,
            repo_type=repo_type,
            namespace=namespace,
            token=token,
        ),
    )


def _hf_parent_conflict_backoff(*, namespace: str, conflict_number: int) -> float:
    """Return bounded exponential backoff with deterministic per-namespace jitter."""
    base_seconds = min(2.0 ** (conflict_number - 1), 8.0)
    digest = hashlib.sha256(f"{namespace}:{conflict_number}".encode()).digest()
    jitter_seconds = int.from_bytes(digest[:4], "big") / (2**32) * 0.5
    return base_seconds + jitter_seconds


def _require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SourceLockError("HF_TOKEN is required")
    return token


def audit_gnomad_parquet(
    path: Path,
    *,
    chromosome: str,
    expected_records: int,
    min_af: float,
    max_allele_len: int,
) -> dict[str, object]:
    """Independently scan a staged gnomAD Parquet shard and verify its row contract."""
    if not path.is_file():
        raise SourceLockError(f"Parquet audit input is not a regular file: {path}")
    with _private_binary_snapshot(path) as snapshot:
        return _audit_gnomad_parquet_stream(
            snapshot.stream,
            chromosome=chromosome,
            expected_records=expected_records,
            min_af=min_af,
            max_allele_len=max_allele_len,
        )


def _audit_gnomad_parquet_stream(
    stream: BinaryIO,
    *,
    chromosome: str,
    expected_records: int,
    min_af: float,
    max_allele_len: int,
) -> dict[str, object]:
    """Scan one already captured gnomAD Parquet byte sequence."""
    if chromosome not in _AUTOSOMES:
        raise SourceLockError(f"Parquet audit chromosome must be one of 1..22, got {chromosome!r}")
    if expected_records <= 0:
        raise SourceLockError("Parquet audit expected_records must be positive")
    if not math.isfinite(min_af):
        raise SourceLockError("Parquet audit min_af must be finite")
    if not 0.0 <= min_af <= 1.0:
        raise SourceLockError("Parquet audit min_af must be between 0 and 1")
    if max_allele_len <= 0:
        raise SourceLockError("Parquet audit max_allele_len must be positive")

    pa, pq = _require_pyarrow_for_audit()
    expected_schema = _expected_gnomad_schema(pa)
    stream.seek(0)
    parquet = pq.ParquetFile(stream)
    observed_schema = parquet.schema_arrow
    if not observed_schema.equals(expected_schema, check_metadata=True):
        raise SourceLockError(
            "Parquet schema drifted from the independent gnomAD schema 2.0.0 contract: "
            f"expected {expected_schema}, observed {observed_schema}"
        )

    metadata_row_count = int(parquet.metadata.num_rows)
    if metadata_row_count != expected_records:
        raise SourceLockError(
            "Parquet metadata/preparer row-count mismatch: "
            f"metadata={metadata_row_count}, preparer={expected_records}"
        )

    canonical_chromosome = chromosome
    stored_min_af = struct.unpack("!f", struct.pack("!f", min_af))[0]
    population_af_columns = (
        "af_afr",
        "af_ami",
        "af_amr",
        "af_asj",
        "af_eas",
        "af_fin",
        "af_mid",
        "af_nfe",
        "af_oth",
        "af_remaining",
        "af_sas",
    )
    required_v41_population_columns = (
        "af_afr",
        "af_amr",
        "af_asj",
        "af_eas",
        "af_fin",
        "af_mid",
        "af_nfe",
        "af_remaining",
        "af_sas",
    )
    population_af_non_null_counts = dict.fromkeys(population_af_columns, 0)
    scanned_row_count = 0
    position_min: int | None = None
    position_max: int | None = None
    for batch in parquet.iter_batches(batch_size=_PARQUET_AUDIT_BATCH_ROWS):
        columns = {
            name: batch.column(observed_schema.get_field_index(name)).to_pylist()
            for name in (
                "chrom",
                "pos",
                "ref",
                "alt",
                "af_global",
                *population_af_columns,
                "filter",
                "schema_version",
            )
        }
        for index, values in enumerate(
            zip(
                columns["chrom"],
                columns["pos"],
                columns["ref"],
                columns["alt"],
                columns["af_global"],
                columns["filter"],
                columns["schema_version"],
                strict=True,
            )
        ):
            chrom, pos, ref, alt, af_global, filter_value, schema_version = values
            row_number = scanned_row_count + 1
            if chrom != canonical_chromosome:
                raise SourceLockError(
                    f"Parquet row {row_number} chromosome drifted: "
                    f"expected {canonical_chromosome!r}, observed {chrom!r}"
                )
            if isinstance(pos, bool) or not isinstance(pos, int) or pos <= 0:
                raise SourceLockError(
                    f"Parquet row {row_number} position must be a positive integer"
                )
            if not _is_explicit_dna_allele(ref, max_allele_len=max_allele_len):
                raise SourceLockError(
                    f"Parquet row {row_number} REF must be explicit uppercase ACGT with "
                    f"length 1..{max_allele_len}"
                )
            if not _is_explicit_dna_allele(alt, max_allele_len=max_allele_len):
                raise SourceLockError(
                    f"Parquet row {row_number} ALT must be explicit uppercase ACGT with "
                    f"length 1..{max_allele_len}"
                )
            if ref == alt:
                raise SourceLockError(f"Parquet row {row_number} REF and ALT must differ")
            if isinstance(af_global, bool) or not isinstance(af_global, int | float):
                raise SourceLockError(f"Parquet row {row_number} af_global must be numeric")
            normalized_af = float(af_global)
            if not math.isfinite(normalized_af):
                raise SourceLockError(f"Parquet row {row_number} af_global must be finite")
            if not stored_min_af <= normalized_af <= 1.0:
                raise SourceLockError(
                    f"Parquet row {row_number} af_global must be within [{min_af}, 1.0]"
                )
            for column in population_af_columns:
                population_af = columns[column][index]
                if population_af is None:
                    continue
                if isinstance(population_af, bool) or not isinstance(population_af, int | float):
                    raise SourceLockError(
                        f"Parquet row {row_number} {column} must be numeric or null"
                    )
                normalized_population_af = float(population_af)
                if not math.isfinite(normalized_population_af):
                    raise SourceLockError(f"Parquet row {row_number} {column} must be finite")
                if not 0.0 <= normalized_population_af <= 1.0:
                    raise SourceLockError(
                        f"Parquet row {row_number} {column} must be within [0.0, 1.0]"
                    )
                population_af_non_null_counts[column] += 1
            if filter_value != "PASS":
                raise SourceLockError(
                    f"Parquet row {row_number} filter must be 'PASS', observed {filter_value!r}"
                )
            if schema_version != "2.0.0":
                raise SourceLockError(
                    f"Parquet row {row_number} schema_version must be '2.0.0', "
                    f"observed {schema_version!r}"
                )
            scanned_row_count += 1
            position_min = pos if position_min is None else min(position_min, pos)
            position_max = pos if position_max is None else max(position_max, pos)

    if scanned_row_count != metadata_row_count:
        raise SourceLockError(
            "Parquet full-scan/metadata row-count mismatch: "
            f"scanned={scanned_row_count}, metadata={metadata_row_count}"
        )
    if position_min is None or position_max is None:
        raise SourceLockError("Parquet audit found no rows")
    missing_v41_populations = [
        column
        for column in required_v41_population_columns
        if population_af_non_null_counts[column] == 0
    ]
    if missing_v41_populations:
        raise SourceLockError(
            "Parquet audit found no values for required gnomAD v4.1 population AF columns: "
            f"{missing_v41_populations}"
        )

    return {
        "audit_method": "pyarrow_metadata_and_full_iter_batches_scan_v1",
        "batch_size_rows": _PARQUET_AUDIT_BATCH_ROWS,
        "metadata_row_count": metadata_row_count,
        "scanned_row_count": scanned_row_count,
        "canonical_chromosome": canonical_chromosome,
        "position_min": position_min,
        "position_max": position_max,
        "schema_version": "2.0.0",
        "population_af_non_null_counts": population_af_non_null_counts,
        "locked_min_af": min_af,
        "stored_min_af_float32": stored_min_af,
        "schema": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in observed_schema
        ],
        "checks": [
            "exact_arrow_schema",
            "canonical_chromosome",
            "positive_position",
            "explicit_acgt_alleles",
            "distinct_ref_alt",
            "finite_global_af_in_locked_range",
            "finite_population_af_in_unit_interval_or_null",
            "pass_filter",
            "exact_schema_version",
            "v41_population_columns_nonempty",
            "metadata_scan_and_preparer_row_counts_equal",
        ],
    }


def author_receipt(
    *,
    selection_path: Path,
    metadata_verification_path: Path,
    source_identity_path: Path,
    prepare_report_path: Path,
    input_vcf: Path,
    dataset_root: Path,
    output_parquet: Path,
) -> dict[str, object]:
    """Reconcile all staging evidence and author the publishable receipt."""
    selection_bytes, selection = _read_json_mapping(selection_path, "selection")
    metadata_bytes, metadata = _read_json_mapping(
        metadata_verification_path, "metadata verification"
    )
    source_identity_bytes, source_identity = _read_json_mapping(
        source_identity_path, "source identity"
    )
    prepare_report_bytes, prepare_report = _read_json_mapping(prepare_report_path, "prepare report")
    _require_equal(
        selection.get("schema_version"), SELECTION_SCHEMA_VERSION, "selection.schema_version"
    )
    _require_equal(
        metadata.get("schema_version"),
        METADATA_VERIFICATION_SCHEMA_VERSION,
        "metadata verification.schema_version",
    )
    _require_equal(metadata.get("ok"), True, "metadata verification.ok")
    _require_equal(
        source_identity.get("schema_version"),
        SOURCE_IDENTITY_SCHEMA_VERSION,
        "source identity.schema_version",
    )
    _require_equal(source_identity.get("ok"), True, "source identity.ok")

    selection_sha256 = hashlib.sha256(selection_bytes).hexdigest()
    _require_equal(
        metadata.get("selection_sha256"),
        selection_sha256,
        "metadata verification.selection_sha256",
    )
    _require_equal(
        source_identity.get("selection_sha256"),
        selection_sha256,
        "source identity.selection_sha256",
    )
    source = _require_mapping(selection.get("source"), "selection.source")
    for field in ("generation", "size_bytes", "md5_base64"):
        _require_equal(metadata.get(field), source.get(field), f"metadata verification.{field}")
    for field in ("size_bytes", "md5_base64"):
        _require_equal(source_identity.get(field), source.get(field), f"source identity.{field}")

    source_sha256 = _require_sha256(source_identity.get("sha256"), "source identity.sha256")
    _require_equal(source_identity.get("path"), str(input_vcf), "source identity.path")
    if not input_vcf.is_file():
        raise SourceLockError(f"input VCF is not a regular file: {input_vcf}")

    input_report = _require_mapping(prepare_report.get("input_vcf"), "prepare report.input_vcf")
    _require_equal(input_report.get("path"), str(input_vcf), "prepare report.input_vcf.path")
    _require_equal(
        input_report.get("size_bytes"),
        source_identity.get("size_bytes"),
        "prepare report.input_vcf.size_bytes",
    )
    prepare_input_sha256 = _require_prefixed_sha256(
        input_report.get("sha256"), "prepare report.input_vcf.sha256"
    )
    _require_equal(
        prepare_input_sha256,
        source_sha256,
        "prepare report.input_vcf.sha256",
    )

    output_report = _require_mapping(
        prepare_report.get("output_parquet"), "prepare report.output_parquet"
    )
    if not output_parquet.is_file():
        raise SourceLockError(f"transform output is not a regular file: {output_parquet}")
    _require_equal(
        output_report.get("path"), str(output_parquet), "prepare report.output_parquet.path"
    )
    prepare_output_sha256 = _require_prefixed_sha256(
        output_report.get("sha256"), "prepare report.output_parquet.sha256"
    )

    transform = _require_mapping(selection.get("transform"), "selection.transform")
    command = _require_str(transform.get("command"), "selection.transform.command")
    release = _require_str(selection.get("release"), "selection.release")
    _require_equal(prepare_report.get("release"), release, "prepare report.release")
    min_af = _require_number(transform.get("min_af"), "selection.transform.min_af")
    max_allele_len = _require_int(
        transform.get("max_allele_len"), "selection.transform.max_allele_len"
    )
    report_argv = [
        command,
        "--input-vcf",
        str(input_vcf),
        "--output",
        str(dataset_root),
        "--release",
        release,
        "--min-af",
        str(min_af),
        "--max-allele-len",
        str(max_allele_len),
    ]
    _require_equal(prepare_report.get("command"), shlex.join(report_argv), "prepare report.command")
    _require_equal(prepare_report.get("already_exists"), False, "prepare report.already_exists")

    count_fields = (
        "records_read",
        "allele_records_seen",
        "records_written",
        "skipped_filter",
        "skipped_af",
        "skipped_allele",
    )
    counts = {
        field: _require_int(prepare_report.get(field), f"prepare report.{field}")
        for field in count_fields
    }
    if any(value < 0 for value in counts.values()):
        raise SourceLockError("prepare report counts must be non-negative")
    if counts["records_written"] <= 0:
        raise SourceLockError("prepare report.records_written must be positive")
    classified_alleles = (
        counts["records_written"]
        + counts["skipped_filter"]
        + counts["skipped_af"]
        + counts["skipped_allele"]
    )
    if classified_alleles != counts["allele_records_seen"]:
        raise SourceLockError(
            "prepare report allele counts do not reconcile: "
            f"classified={classified_alleles}, seen={counts['allele_records_seen']}"
        )
    with _private_binary_snapshot(output_parquet) as snapshot:
        output_identity = {
            "path": str(output_parquet),
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes,
        }
        _require_equal(
            prepare_output_sha256,
            snapshot.sha256,
            "prepare report.output_parquet.sha256",
        )
        _require_equal(
            output_report.get("size_bytes"),
            snapshot.size_bytes,
            "prepare report.output_parquet.size_bytes",
        )
        parquet_audit = _audit_gnomad_parquet_stream(
            snapshot.stream,
            chromosome=_require_str(source.get("chromosome"), "selection.source.chromosome"),
            expected_records=counts["records_written"],
            min_af=min_af,
            max_allele_len=max_allele_len,
        )

    runtime = _require_mapping(prepare_report.get("runtime"), "prepare report.runtime")
    elapsed_seconds = _require_number(
        runtime.get("elapsed_seconds"), "prepare report.runtime.elapsed_seconds"
    )
    if elapsed_seconds < 0:
        raise SourceLockError("prepare report.runtime.elapsed_seconds must be non-negative")
    peak_rss = _require_int(
        runtime.get("process_peak_rss_bytes"),
        "prepare report.runtime.process_peak_rss_bytes",
    )
    if peak_rss <= 0:
        raise SourceLockError("prepare report.runtime.process_peak_rss_bytes must be positive")
    peak_memory_note = _require_str(
        runtime.get("peak_memory_note"), "prepare report.runtime.peak_memory_note"
    )

    full_argv = ["uv", "run", command, "--quiet", "--no-banner", *report_argv[1:]]
    return {
        "schema_version": STAGING_RECEIPT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ok": True,
        "dataset_id": _require_str(selection.get("dataset_id"), "selection.dataset_id"),
        "release": release,
        "reference_genome": _require_str(
            selection.get("reference_genome"), "selection.reference_genome"
        ),
        "source_lock": dict(
            _require_mapping(selection.get("source_lock"), "selection.source_lock")
        ),
        "source": {
            "chromosome": _require_str(source.get("chromosome"), "selection.source.chromosome"),
            "split_role": _require_str(source.get("split_role"), "selection.source.split_role"),
            "bucket": _require_str(source.get("bucket"), "selection.source.bucket"),
            "object": _require_str(source.get("object"), "selection.source.object"),
            "generation": _require_str(source.get("generation"), "selection.source.generation"),
            "size_bytes": _require_int(source.get("size_bytes"), "selection.source.size_bytes"),
            "upstream_md5_base64": _require_str(
                source.get("md5_base64"), "selection.source.md5_base64"
            ),
            "upstream_md5_hex": _require_str(source.get("md5_hex"), "selection.source.md5_hex"),
            "streamed_sha256": source_sha256,
        },
        "transform": {
            "command": command,
            "argv": full_argv,
            "filters": {
                "filter": _require_str(transform.get("filter"), "selection.transform.filter"),
                "min_af": min_af,
                "max_allele_len": max_allele_len,
            },
            "runtime": {
                "elapsed_seconds": elapsed_seconds,
                "process_peak_rss_bytes": peak_rss,
                "peak_memory_note": peak_memory_note,
            },
            "counts": counts,
        },
        "output": {**output_identity, "parquet_audit": parquet_audit},
        "execution": dict(_require_mapping(selection.get("execution"), "selection.execution")),
        "publication": dict(
            _require_mapping(selection.get("publication"), "selection.publication")
        ),
        "evidence": {
            "selection": _bytes_identity(selection_path, selection_bytes),
            "metadata_verification": _bytes_identity(metadata_verification_path, metadata_bytes),
            "source_identity": _bytes_identity(source_identity_path, source_identity_bytes),
            "prepare_report": _bytes_identity(prepare_report_path, prepare_report_bytes),
        },
        "claim_boundary": _require_str(selection.get("claim_boundary"), "selection.claim_boundary"),
    }


def select_source(
    lock_path: Path,
    *,
    chromosome: str,
    commit_sha: str,
    container_image: str,
) -> dict[str, object]:
    """Return one verified lock entry plus deterministic runtime/publication fields."""
    return select_source_from_snapshot(
        capture_source_lock(lock_path),
        chromosome=chromosome,
        commit_sha=commit_sha,
        container_image=container_image,
    )


def capture_source_lock(lock_path: Path) -> SourceLockSnapshot:
    """Capture a source lock and its referenced schema exactly once."""
    lock_bytes = lock_path.read_bytes()
    lock = _decode_json_mapping(lock_bytes, "source lock")
    schema_reference = _require_str(lock["$schema"], "$schema")
    schema_path = lock_path.parent / schema_reference.removeprefix("./")
    schema_bytes = schema_path.read_bytes()
    schema = _decode_json_mapping(schema_bytes, "source lock schema")
    return SourceLockSnapshot(
        lock_path=lock_path,
        lock_bytes=lock_bytes,
        lock=lock,
        schema_path=schema_path,
        schema_bytes=schema_bytes,
        schema=schema,
    )


def select_source_from_snapshot(
    snapshot: SourceLockSnapshot,
    *,
    chromosome: str,
    commit_sha: str,
    container_image: str,
) -> dict[str, object]:
    """Select one source using only an already-captured lock snapshot."""
    lock = snapshot.lock
    schema = snapshot.schema
    _validate_lock(lock)
    _require_equal(
        schema.get("$schema"),
        "https://json-schema.org/draft/2020-12/schema",
        "source lock schema.$schema",
    )
    _require_equal(
        schema.get("additionalProperties"), False, "source lock schema.additionalProperties"
    )
    if chromosome not in _AUTOSOMES:
        raise SourceLockError(f"chromosome must be one of 1..22, got {chromosome!r}")
    if _COMMIT_SHA.fullmatch(commit_sha) is None:
        raise SourceLockError("commit_sha must be a full lowercase 40-character Git SHA")

    job = _require_mapping(lock["job"], "job")
    locked_container = _require_str(job["container_image"], "job.container_image")
    if container_image != locked_container:
        raise SourceLockError(
            f"container image drift: expected {locked_container!r}, observed {container_image!r}"
        )

    entries = _require_list(lock["objects"], "objects")
    selected = next(
        _require_mapping(entry, "objects[]")
        for entry in entries
        if _require_mapping(entry, "objects[]").get("chromosome") == chromosome
    )
    source_config = _require_mapping(lock["source"], "source")
    transform = _require_mapping(lock["transform"], "transform")
    bucket = _require_str(source_config["bucket"], "source.bucket")
    object_name = _require_str(selected["object"], "objects[].object")
    generation = _require_str(selected["generation"], "objects[].generation")
    encoded_object = urllib.parse.quote(object_name, safe="")
    metadata_api = _require_str(source_config["metadata_api"], "source.metadata_api")
    media_api = _require_str(source_config["media_api"], "source.media_api")
    lock_sha256 = hashlib.sha256(snapshot.lock_bytes).hexdigest()
    namespace_root = _require_str(job["namespace_root"], "job.namespace_root")
    md5_base64 = _require_str(selected["md5_base64"], "objects[].md5_base64")

    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "source_lock": {
            "path": snapshot.lock_path.as_posix(),
            "sha256": lock_sha256,
            "schema_version": LOCK_SCHEMA_VERSION,
            "schema": {
                "path": snapshot.schema_path.as_posix(),
                "sha256": hashlib.sha256(snapshot.schema_bytes).hexdigest(),
                "draft": "https://json-schema.org/draft/2020-12/schema",
            },
        },
        "dataset_id": _require_str(lock["dataset_id"], "dataset_id"),
        "release": _require_str(lock["release"], "release"),
        "reference_genome": _require_str(lock["reference_genome"], "reference_genome"),
        "source": {
            "bucket": bucket,
            "chromosome": chromosome,
            "split_role": _require_str(selected["split_role"], "objects[].split_role"),
            "object": object_name,
            "generation": generation,
            "size_bytes": _require_int(selected["size_bytes"], "objects[].size_bytes"),
            "md5_base64": md5_base64,
            "md5_hex": base64.b64decode(md5_base64, validate=True).hex(),
            "metadata_url": (
                f"{metadata_api}/b/{urllib.parse.quote(bucket, safe='')}/o/{encoded_object}"
                f"?generation={generation}"
            ),
            "media_url": (
                f"{media_api}/b/{urllib.parse.quote(bucket, safe='')}/o/{encoded_object}"
                f"?alt=media&generation={generation}"
            ),
        },
        "transform": dict(transform),
        "execution": {
            "commit_sha": commit_sha,
            "container_image": container_image,
            "repository": _require_str(job["repository"], "job.repository"),
        },
        "publication": {
            "repo": _require_str(job["upload_repo"], "job.upload_repo"),
            "repo_type": _require_str(job["upload_repo_type"], "job.upload_repo_type"),
            "namespace": (
                f"{namespace_root}/lock-{lock_sha256[:12]}/"
                f"chr{chromosome}-g{generation}-{commit_sha[:12]}"
            ),
        },
        "claim_boundary": _require_str(lock["claim_boundary"], "claim_boundary"),
    }


def _validate_lock(lock: Mapping[str, object]) -> None:
    expected_top_level = {
        "$schema",
        "schema_version",
        "dataset_id",
        "release",
        "reference_genome",
        "source",
        "transform",
        "job",
        "objects",
        "claim_boundary",
    }
    _require_exact_keys(lock, expected_top_level, "source lock")
    _require_equal(
        lock["$schema"], "./gnomad-v4.1-exomes-autosomes.source-lock.schema.json", "$schema"
    )
    _require_equal(lock["schema_version"], LOCK_SCHEMA_VERSION, "schema_version")
    _require_equal(lock["dataset_id"], "gnomad-v4.1-exomes-autosomes", "dataset_id")
    _require_equal(lock["release"], "v4.1", "release")
    _require_equal(lock["reference_genome"], "GRCh38", "reference_genome")

    source = _require_mapping(lock["source"], "source")
    _require_exact_keys(source, {"bucket", "metadata_api", "media_api"}, "source")
    _require_equal(source["bucket"], "gcp-public-data--gnomad", "source.bucket")
    _require_equal(
        source["metadata_api"],
        "https://storage.googleapis.com/storage/v1",
        "source.metadata_api",
    )
    _require_equal(
        source["media_api"],
        "https://storage.googleapis.com/download/storage/v1",
        "source.media_api",
    )

    transform = _require_mapping(lock["transform"], "transform")
    _require_exact_keys(
        transform,
        {"command", "filter", "min_af", "max_allele_len"},
        "transform",
    )
    _require_equal(transform["command"], "geno-lewm-prepare-gnomad", "transform.command")
    _require_equal(transform["filter"], "PASS", "transform.filter")
    min_af = _require_number(transform["min_af"], "transform.min_af")
    if not 0.0 <= min_af <= 1.0:
        raise SourceLockError("transform.min_af must be between 0 and 1")
    if _require_int(transform["max_allele_len"], "transform.max_allele_len") <= 0:
        raise SourceLockError("transform.max_allele_len must be positive")

    job = _require_mapping(lock["job"], "job")
    _require_exact_keys(
        job,
        {
            "repository",
            "container_image",
            "upload_repo",
            "upload_repo_type",
            "namespace_root",
        },
        "job",
    )
    _require_equal(
        job["repository"], "https://github.com/AbdelStark/GenoLeWM.git", "job.repository"
    )
    image = _require_str(job["container_image"], "job.container_image")
    if _CONTAINER_IMAGE.fullmatch(image) is None:
        raise SourceLockError("job.container_image must be pinned by sha256 digest")
    upload_repo = _require_str(job["upload_repo"], "job.upload_repo")
    if upload_repo.count("/") != 1:
        raise SourceLockError("job.upload_repo must be a namespace/name repository id")
    _require_equal(job["upload_repo_type"], "dataset", "job.upload_repo_type")
    namespace_root = _require_str(job["namespace_root"], "job.namespace_root")
    if not namespace_root.startswith("staging/v0.3/"):
        raise SourceLockError("job.namespace_root must be below staging/v0.3")

    objects = _require_list(lock["objects"], "objects")
    if len(objects) != 22:
        raise SourceLockError(f"objects must contain exactly 22 autosomes, got {len(objects)}")
    by_chromosome: dict[str, Mapping[str, object]] = {}
    for index, raw_entry in enumerate(objects):
        entry = _require_mapping(raw_entry, f"objects[{index}]")
        _require_exact_keys(
            entry,
            {"chromosome", "split_role", "object", "generation", "size_bytes", "md5_base64"},
            f"objects[{index}]",
        )
        chromosome = _require_str(entry["chromosome"], f"objects[{index}].chromosome")
        if chromosome in by_chromosome:
            raise SourceLockError(f"duplicate chromosome entry: {chromosome}")
        by_chromosome[chromosome] = entry
        expected_object = f"release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr{chromosome}.vcf.bgz"
        _require_equal(entry["object"], expected_object, f"objects[{index}].object")
        generation = _require_str(entry["generation"], f"objects[{index}].generation")
        if not generation.isdigit():
            raise SourceLockError(f"objects[{index}].generation must contain only digits")
        if _require_int(entry["size_bytes"], f"objects[{index}].size_bytes") <= 0:
            raise SourceLockError(f"objects[{index}].size_bytes must be positive")
        md5_base64 = _require_str(entry["md5_base64"], f"objects[{index}].md5_base64")
        try:
            decoded_md5 = base64.b64decode(md5_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise SourceLockError(f"objects[{index}].md5_base64 is invalid") from exc
        if len(decoded_md5) != 16:
            raise SourceLockError(f"objects[{index}].md5_base64 must encode 16 bytes")

    if set(by_chromosome) != _AUTOSOMES:
        missing = sorted(_AUTOSOMES - set(by_chromosome), key=int)
        unexpected = sorted(set(by_chromosome) - _AUTOSOMES)
        raise SourceLockError(
            f"objects must cover autosomes 1..22; missing={missing}, unexpected={unexpected}"
        )
    for chromosome, entry in by_chromosome.items():
        expected_role = (
            "validation" if chromosome == "20" else "evaluation" if chromosome == "21" else "train"
        )
        _require_equal(entry["split_role"], expected_role, f"objects[{chromosome}].split_role")

    claim_boundary = _require_str(lock["claim_boundary"], "claim_boundary")
    if len(claim_boundary) < 80 or "not evidence" not in claim_boundary:
        raise SourceLockError("claim_boundary must preserve the non-scientific staging scope")


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SourceLockError(f"{field} must be an object")
    return value


def _require_list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise SourceLockError(f"{field} must be an array")
    return value


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceLockError(f"{field} must be a non-empty string")
    return value


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourceLockError(f"{field} must be an integer")
    return value


def _require_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SourceLockError(f"{field} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise SourceLockError(f"{field} must be finite")
    return normalized


def _require_sha256(value: object, field: str) -> str:
    digest = _require_str(value, field)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise SourceLockError(f"{field} must be a lowercase SHA-256 hex digest")
    return digest


def _require_prefixed_sha256(value: object, field: str) -> str:
    digest = _require_str(value, field)
    match = re.fullmatch(r"sha256:([0-9a-f]{64})", digest)
    if match is None:
        raise SourceLockError(f"{field} must be a lowercase sha256:<hex> digest")
    return match.group(1)


def _require_pyarrow_for_audit() -> tuple[Any, Any]:
    try:
        pa = importlib.import_module("pyarrow")
        pq = importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise SourceLockError(
            "independent gnomAD Parquet audit requires pyarrow; install the train extra"
        ) from exc
    return pa, pq


def _expected_gnomad_schema(pa: Any) -> Any:
    return pa.schema(
        [
            ("chrom", pa.string()),
            ("pos", pa.int64()),
            ("ref", pa.string()),
            ("alt", pa.string()),
            ("af_global", pa.float32()),
            ("af_afr", pa.float32()),
            ("af_ami", pa.float32()),
            ("af_amr", pa.float32()),
            ("af_asj", pa.float32()),
            ("af_eas", pa.float32()),
            ("af_fin", pa.float32()),
            ("af_mid", pa.float32()),
            ("af_nfe", pa.float32()),
            ("af_oth", pa.float32()),
            ("af_remaining", pa.float32()),
            ("af_sas", pa.float32()),
            ("filter", pa.string()),
            ("schema_version", pa.string()),
        ]
    )


def _is_explicit_dna_allele(value: object, *, max_allele_len: int) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= max_allele_len
        and set(value) <= {"A", "C", "G", "T"}
    )


def _require_exact_keys(value: Mapping[str, object], expected: set[str], field: str) -> None:
    observed = set(value)
    if observed != expected:
        raise SourceLockError(
            f"{field} keys drifted: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _require_equal(observed: object, expected: object, field: str) -> None:
    if not _json_equal(observed, expected):
        raise SourceLockError(f"{field} drifted: expected {expected!r}, observed {observed!r}")


def _json_equal(observed: object, expected: object) -> bool:
    """Compare JSON-like values without Python's bool/int equality aliases."""
    if type(observed) is not type(expected):
        return False
    if isinstance(observed, Mapping):
        if not isinstance(expected, Mapping) or set(observed) != set(expected):
            return False
        return all(_json_equal(observed[key], expected[key]) for key in observed)
    if isinstance(observed, list):
        if not isinstance(expected, list) or len(observed) != len(expected):
            return False
        return all(
            _json_equal(observed_item, expected_item)
            for observed_item, expected_item in zip(observed, expected, strict=True)
        )
    return observed == expected


def _read_json_mapping(path: Path, field: str) -> tuple[bytes, Mapping[str, object]]:
    capture = _capture_json(path, field)
    return capture.payload, capture.value


def _capture_json(path: Path, field: str) -> _CapturedJson:
    payload = path.read_bytes()
    return _CapturedJson(
        path=path,
        payload=payload,
        value=_decode_json_mapping(payload, field),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _decode_json_mapping(payload: bytes, field: str) -> Mapping[str, object]:
    raw = json.loads(payload, object_pairs_hook=_reject_duplicate_pairs)
    return _require_mapping(raw, field)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise SourceLockError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


@contextmanager
def _private_binary_snapshot(path: Path) -> Iterator[_BinarySnapshot]:
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        before_open = path.lstat()
    except OSError as exc:
        raise SourceLockError(f"cannot inspect binary snapshot input {path}: {exc}") from exc
    if not stat.S_ISREG(before_open.st_mode):
        raise SourceLockError(
            f"binary snapshot input must be a regular file without following symlinks: {path}"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceLockError(
            f"binary snapshot input must be a regular file without following symlinks: {path}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before_open.st_dev
            or opened.st_ino != before_open.st_ino
        ):
            raise SourceLockError(
                f"binary snapshot input must be a stable regular file without following "
                f"symlinks: {path}"
            )
        source = os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise
    try:
        with tempfile.TemporaryFile(mode="w+b") as snapshot:
            with source:
                for chunk in iter(lambda: source.read(_HASH_CHUNK_SIZE), b""):
                    digest.update(chunk)
                    size_bytes += len(chunk)
                    snapshot.write(chunk)
            snapshot.seek(0)
            yield _BinarySnapshot(
                stream=snapshot,
                sha256=digest.hexdigest(),
                size_bytes=size_bytes,
            )
    finally:
        source.close()


def _bytes_identity(path: Path, payload: bytes) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    write_immutable_json(path, payload)


if __name__ == "__main__":  # pragma: no cover - exercised through ``main`` in tests
    raise SystemExit(main())
