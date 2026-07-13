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
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOCK_SCHEMA_VERSION = "geno-lewm.gnomad-source-lock.v1"
SELECTION_SCHEMA_VERSION = "geno-lewm.gnomad-stage-selection.v1"
METADATA_VERIFICATION_SCHEMA_VERSION = "geno-lewm.gnomad-gcs-metadata-verification.v1"
SOURCE_IDENTITY_SCHEMA_VERSION = "geno-lewm.gnomad-stream-identity.v1"
STAGING_RECEIPT_SCHEMA_VERSION = "geno-lewm.gnomad-staging-receipt.v1"
_HASH_CHUNK_SIZE = 1 << 20
_PARQUET_AUDIT_BATCH_ROWS = 131_072
_HF_UPLOAD_MAX_ATTEMPTS = 5
_HF_PARENT_CONFLICT_STATUS = 412
_HF_PARENT_CONFLICT_MESSAGE = "A commit has happened since. Please refresh and try again."
_AUTOSOMES = frozenset(str(chromosome) for chromosome in range(1, 23))
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_CONTAINER_IMAGE = re.compile(r"[^@]+@sha256:[0-9a-f]{64}")


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
    except (OSError, json.JSONDecodeError, SourceLockError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def verify_gcs_metadata(selection_path: Path, metadata_path: Path) -> dict[str, object]:
    """Verify generation, size, and upstream MD5 returned by the GCS JSON API."""
    selection_bytes = selection_path.read_bytes()
    selection_raw: object = json.loads(selection_bytes)
    metadata_raw: object = json.loads(metadata_path.read_bytes())
    selection = _require_mapping(selection_raw, "selection")
    metadata = _require_mapping(metadata_raw, "GCS metadata")
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
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
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
    selection_raw: object = json.loads(selection_bytes)
    selection = _require_mapping(selection_raw, "selection")
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
    if not path.is_file():
        raise SourceLockError(f"Parquet audit input is not a regular file: {path}")

    pa, pq = _require_pyarrow_for_audit()
    expected_schema = _expected_gnomad_schema(pa)
    parquet = pq.ParquetFile(path)
    observed_schema = parquet.schema_arrow
    if not observed_schema.equals(expected_schema, check_metadata=False):
        raise SourceLockError(
            "Parquet schema drifted from the independent gnomAD v1 contract: "
            f"expected {expected_schema}, observed {observed_schema}"
        )

    metadata_row_count = int(parquet.metadata.num_rows)
    if metadata_row_count != expected_records:
        raise SourceLockError(
            "Parquet metadata/preparer row-count mismatch: "
            f"metadata={metadata_row_count}, preparer={expected_records}"
        )

    canonical_chromosome = f"chr{chromosome}"
    stored_min_af = struct.unpack("!f", struct.pack("!f", min_af))[0]
    scanned_row_count = 0
    position_min: int | None = None
    position_max: int | None = None
    for batch in parquet.iter_batches(batch_size=_PARQUET_AUDIT_BATCH_ROWS):
        columns = {
            name: batch.column(observed_schema.get_field_index(name)).to_pylist()
            for name in ("chrom", "pos", "ref", "alt", "af_global", "filter", "schema_version")
        }
        for values in zip(
            columns["chrom"],
            columns["pos"],
            columns["ref"],
            columns["alt"],
            columns["af_global"],
            columns["filter"],
            columns["schema_version"],
            strict=True,
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
            if filter_value != "PASS":
                raise SourceLockError(
                    f"Parquet row {row_number} filter must be 'PASS', observed {filter_value!r}"
                )
            if schema_version != "1.0.0":
                raise SourceLockError(
                    f"Parquet row {row_number} schema_version must be '1.0.0', "
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

    return {
        "audit_method": "pyarrow_metadata_and_full_iter_batches_scan_v1",
        "batch_size_rows": _PARQUET_AUDIT_BATCH_ROWS,
        "metadata_row_count": metadata_row_count,
        "scanned_row_count": scanned_row_count,
        "canonical_chromosome": canonical_chromosome,
        "position_min": position_min,
        "position_max": position_max,
        "schema_version": "1.0.0",
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
            "pass_filter",
            "exact_schema_version",
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
    _require_equal(input_report.get("sha256"), source_sha256, "prepare report.input_vcf.sha256")

    output_report = _require_mapping(
        prepare_report.get("output_parquet"), "prepare report.output_parquet"
    )
    if not output_parquet.is_file():
        raise SourceLockError(f"transform output is not a regular file: {output_parquet}")
    output_identity = _file_identity(output_parquet)
    _require_equal(
        output_report.get("path"), str(output_parquet), "prepare report.output_parquet.path"
    )
    for field in ("sha256", "size_bytes"):
        _require_equal(
            output_report.get(field),
            output_identity[field],
            f"prepare report.output_parquet.{field}",
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
    parquet_audit = audit_gnomad_parquet(
        output_parquet,
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
    lock_bytes = lock_path.read_bytes()
    raw: object = json.loads(lock_bytes)
    lock = _require_mapping(raw, "source lock")
    _validate_lock(lock)
    schema_reference = _require_str(lock["$schema"], "$schema")
    schema_path = lock_path.parent / schema_reference.removeprefix("./")
    schema_bytes = schema_path.read_bytes()
    schema_raw: object = json.loads(schema_bytes)
    schema = _require_mapping(schema_raw, "source lock schema")
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
    lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
    namespace_root = _require_str(job["namespace_root"], "job.namespace_root")
    md5_base64 = _require_str(selected["md5_base64"], "objects[].md5_base64")

    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "source_lock": {
            "path": str(lock_path),
            "sha256": lock_sha256,
            "schema_version": LOCK_SCHEMA_VERSION,
            "schema": {
                "path": str(schema_path),
                "sha256": hashlib.sha256(schema_bytes).hexdigest(),
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
            ("af_nfe", pa.float32()),
            ("af_oth", pa.float32()),
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
    if observed != expected:
        raise SourceLockError(f"{field} drifted: expected {expected!r}, observed {observed!r}")


def _read_json_mapping(path: Path, field: str) -> tuple[bytes, Mapping[str, object]]:
    payload_bytes = path.read_bytes()
    raw: object = json.loads(payload_bytes)
    return payload_bytes, _require_mapping(raw, field)


def _file_identity(path: Path) -> dict[str, object]:
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
            sha256.update(chunk)
    return {"path": str(path), "sha256": sha256.hexdigest(), "size_bytes": path.stat().st_size}


def _bytes_identity(path: Path, payload: bytes) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":  # pragma: no cover - exercised through ``main`` in tests
    raise SystemExit(main())
