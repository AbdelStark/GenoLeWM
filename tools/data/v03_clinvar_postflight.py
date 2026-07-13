# SPDX-License-Identifier: Apache-2.0
"""Verify the corrected ClinVar staging bundle at one immutable Hub revision."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import BinaryIO

from tools.data._immutable_json import ImmutableJsonError, write_immutable_json

REMOTE_POSTFLIGHT_SCHEMA_VERSION = "geno-lewm.clinvar-remote-postflight.v1"
_HASH_CHUNK_SIZE = 1 << 20
_PARQUET_AUDIT_BATCH_ROWS = 131_072
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_PREFIXED_SHA256 = re.compile(r"sha256:([0-9a-f]{64})")
_MD5 = re.compile(r"[0-9a-f]{32}")
_CONTAINER_IMAGE = re.compile(r"[^@]+@sha256:[0-9a-f]{64}")
_SAFE_HF_NAMESPACE = re.compile(r"[A-Za-z0-9._/-]+")
_SOURCE_CONTRACT_PATHS = (
    "geno_lewm/cli/_prepare_report.py",
    "geno_lewm/cli/prepare_clinvar.py",
    "geno_lewm/data/_vcf.py",
    "geno_lewm/data/clinvar.py",
)


class ClinvarPostflightError(ValueError):
    """Raised when immutable ClinVar staging evidence does not reconcile."""


@dataclass(frozen=True, slots=True)
class _SourceContract:
    schema_version: str
    labelled_classes: tuple[str, ...]
    normalized_classes: tuple[str, ...]
    parquet_schema: tuple[tuple[str, str], ...]
    nullable_fields: tuple[str, ...]
    allele_alphabet: tuple[str, ...]
    cli_command: str
    max_allele_len: int
    output_path_template: str
    prepare_report_enrichments: tuple[str, ...]
    file_identity_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CapturedJson:
    payload: bytes
    value: Mapping[str, object]
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _BinarySnapshot:
    stream: BinaryIO
    sha256: str
    size_bytes: int


def main(argv: list[str] | None = None) -> int:
    """Verify one exact remote ClinVar namespace and write deterministic JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-release", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = verify_remote_clinvar_namespace(
            repo_id=args.repo_id,
            revision=args.revision,
            namespace=args.namespace,
            expected_source_commit=args.expected_source_commit,
            expected_release=args.expected_release,
            token=os.environ.get("HF_TOKEN"),
        )
        _write_json(args.output_json, report)
    except (
        OSError,
        json.JSONDecodeError,
        UnicodeError,
        ClinvarPostflightError,
        ImmutableJsonError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def verify_remote_clinvar_namespace(
    *,
    repo_id: str,
    revision: str,
    namespace: str,
    expected_source_commit: str,
    expected_release: str,
    token: str | None,
) -> dict[str, object]:
    """Download and independently verify one immutable ClinVar staging bundle."""
    _validate_request(
        repo_id=repo_id,
        revision=revision,
        namespace=namespace,
        expected_source_commit=expected_source_commit,
        expected_release=expected_release,
    )
    source_blobs = {
        path: _read_git_blob_at_commit(expected_source_commit, path=path)
        for path in _SOURCE_CONTRACT_PATHS
    }
    source_contract = _derive_source_contract(source_blobs)
    relative_files = _remote_files(expected_release)
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
    except ClinvarPostflightError:
        raise
    except Exception as exc:
        raise ClinvarPostflightError(
            f"cannot inspect exact Hugging Face revision {revision}: {exc}"
        ) from exc

    observed_files = _remote_namespace_file_set(repo_files, namespace=namespace)
    if observed_files != relative_files:
        raise ClinvarPostflightError(
            "remote ClinVar namespace file set drifted: "
            f"missing={sorted(relative_files - observed_files)}, "
            f"unexpected={sorted(observed_files - relative_files)}"
        )

    with tempfile.TemporaryDirectory(prefix="geno-lewm-clinvar-postflight-") as temporary:
        local_paths: dict[str, Path] = {}
        cache_directory = Path(temporary) / "hf-cache"
        for relative_path in sorted(relative_files):
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
                raise ClinvarPostflightError(
                    f"cannot download {filename!r} at exact revision {revision}: {exc}"
                ) from exc
            local_path = Path(downloaded)
            if not local_path.is_file():
                raise ClinvarPostflightError(
                    f"exact-revision download is not a regular file: {relative_path}"
                )
            local_paths[relative_path] = local_path

        return _audit_bundle(
            local_paths,
            repo_id=repo_id,
            revision=revision,
            namespace=namespace,
            source_commit=expected_source_commit,
            release=expected_release,
            source_blobs=source_blobs,
            source_contract=source_contract,
        )


def _validate_request(
    *,
    repo_id: str,
    revision: str,
    namespace: str,
    expected_source_commit: str,
    expected_release: str,
) -> None:
    if repo_id.count("/") != 1 or any(not part for part in repo_id.split("/")):
        raise ClinvarPostflightError("Hugging Face repo_id must be a namespace/name pair")
    if _COMMIT_SHA.fullmatch(revision) is None:
        raise ClinvarPostflightError(
            "Hugging Face revision must be a full lowercase 40-character commit SHA"
        )
    if _COMMIT_SHA.fullmatch(expected_source_commit) is None:
        raise ClinvarPostflightError(
            "expected source commit must be a full lowercase 40-character Git SHA"
        )
    try:
        parsed_release = date.fromisoformat(expected_release)
    except ValueError as exc:
        raise ClinvarPostflightError("expected release must be an ISO YYYY-MM-DD date") from exc
    if parsed_release.isoformat() != expected_release:
        raise ClinvarPostflightError("expected release must be a canonical ISO YYYY-MM-DD date")
    if (
        namespace != namespace.strip("/")
        or _SAFE_HF_NAMESPACE.fullmatch(namespace) is None
        or any(part in {"", ".", ".."} for part in namespace.split("/"))
    ):
        raise ClinvarPostflightError("Hugging Face namespace contains an unsafe path component")
    expected_namespace = _expected_namespace(expected_release, expected_source_commit)
    _require_equal(namespace, expected_namespace, "requested ClinVar namespace")


def _expected_namespace(release: str, source_commit: str) -> str:
    return f"staging/clinvar-{release}-archive-{source_commit[:12]}-r1"


def _remote_files(release: str) -> frozenset[str]:
    return frozenset(
        {
            f"clinvar/{release}/variants.parquet",
            "evidence/audit.json",
            "evidence/prepare_report.json",
            "evidence/runtime_report.json",
        }
    )


def _remote_namespace_file_set(repo_files: object, *, namespace: str) -> frozenset[str]:
    if not isinstance(repo_files, list) or any(not isinstance(path, str) for path in repo_files):
        raise ClinvarPostflightError("Hugging Face file listing must be an array of paths")
    prefix = f"{namespace}/"
    relative = [path.removeprefix(prefix) for path in repo_files if path.startswith(prefix)]
    if len(relative) != len(set(relative)):
        raise ClinvarPostflightError("Hugging Face namespace file listing contains duplicates")
    return frozenset(relative)


def _read_git_blob_at_commit(commit_sha: str, *, path: str) -> bytes:
    """Read a contract blob from one exact commit without replacement objects."""
    try:
        result = subprocess.run(
            ["git", "cat-file", "blob", f"{commit_sha}:{path}"],
            check=False,
            capture_output=True,
            env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
        )
    except OSError as exc:
        raise ClinvarPostflightError(
            f"cannot read trusted source artifact from Git: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ClinvarPostflightError(
            f"cannot read trusted source artifact {path!r} at {commit_sha}: {detail}"
        )
    return result.stdout


def _derive_source_contract(blobs: Mapping[str, bytes]) -> _SourceContract:
    clinvar_tree = _parse_source(blobs["geno_lewm/data/clinvar.py"], "ClinVar data module")
    vcf_tree = _parse_source(blobs["geno_lewm/data/_vcf.py"], "VCF data module")
    cli_tree = _parse_source(blobs["geno_lewm/cli/prepare_clinvar.py"], "ClinVar CLI module")
    report_tree = _parse_source(blobs["geno_lewm/cli/_prepare_report.py"], "prepare-report module")
    schema_version = _string_assignment(clinvar_tree, "CLINVAR_SCHEMA_VERSION")
    labelled_classes = _string_set_assignment(clinvar_tree, "CLINVAR_LABELLED_CLASSES")
    normalized_classes = _string_returns(clinvar_tree, "_clinical_significance")
    if not set(labelled_classes) < set(normalized_classes):
        raise ClinvarPostflightError(
            "trusted source label contract must contain labelled classes plus unlabelled classes"
        )
    parquet_schema = _parquet_schema_from_ast(clinvar_tree)
    variant_fields, nullable_fields = _variant_fields_from_ast(clinvar_tree)
    if tuple(name for name, _kind in parquet_schema) != variant_fields:
        raise ClinvarPostflightError(
            "trusted source ClinvarVariant fields drifted from its Parquet schema"
        )
    output_parts = _target_path_parts(clinvar_tree)
    if output_parts != ("clinvar", "{release}", "variants.parquet"):
        raise ClinvarPostflightError(
            f"trusted source output path contract drifted: observed {output_parts!r}"
        )
    allele_alphabet = _string_set_assignment(vcf_tree, "_ACGT")
    cli_command = _typer_command_name(cli_tree)
    max_allele_len = _function_default(cli_tree, "main", "max_allele_len")
    if max_allele_len <= 0:
        raise ClinvarPostflightError("trusted max_allele_len default must be positive")
    enrichments = _subscript_assignment_keys(report_tree, "augment_prepare_report", "enriched")
    expected_enrichments = ("command", "input_vcf", "output_parquet", "runtime")
    if enrichments != expected_enrichments:
        raise ClinvarPostflightError(
            "trusted prepare-report enrichment contract drifted: "
            f"expected {expected_enrichments!r}, observed {enrichments!r}"
        )
    file_identity_fields = _returned_dict_keys(report_tree, "_file_identity")
    expected_identity = ("path", "sha256", "size_bytes")
    if file_identity_fields != expected_identity:
        raise ClinvarPostflightError(
            "trusted prepare-report file identity contract drifted: "
            f"expected {expected_identity!r}, observed {file_identity_fields!r}"
        )
    return _SourceContract(
        schema_version=schema_version,
        labelled_classes=labelled_classes,
        normalized_classes=normalized_classes,
        parquet_schema=parquet_schema,
        nullable_fields=nullable_fields,
        allele_alphabet=allele_alphabet,
        cli_command=cli_command,
        max_allele_len=max_allele_len,
        output_path_template="clinvar/{release}/variants.parquet",
        prepare_report_enrichments=enrichments,
        file_identity_fields=file_identity_fields,
    )


def _parse_source(payload: bytes, field: str) -> ast.Module:
    try:
        return ast.parse(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, SyntaxError) as exc:
        raise ClinvarPostflightError(f"cannot parse trusted {field}: {exc}") from exc


def _string_assignment(tree: ast.Module, name: str) -> str:
    value = _assignment_value(tree, name)
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str) or not value.value:
        raise ClinvarPostflightError(f"trusted source {name} must be a non-empty string literal")
    return value.value


def _string_set_assignment(tree: ast.Module, name: str) -> tuple[str, ...]:
    value = _assignment_value(tree, name)
    if (
        not isinstance(value, ast.Call)
        or not isinstance(value.func, ast.Name)
        or value.func.id != "frozenset"
        or len(value.args) != 1
    ):
        raise ClinvarPostflightError(f"trusted source {name} must be a literal frozenset")
    literal = value.args[0]
    if isinstance(literal, ast.Set):
        values = tuple(
            sorted(
                element.value
                for element in literal.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )
        )
        literal_size = len(literal.elts)
    elif isinstance(literal, ast.Constant) and isinstance(literal.value, str):
        values = tuple(sorted(set(literal.value)))
        literal_size = len(set(literal.value))
    else:
        raise ClinvarPostflightError(
            f"trusted source {name} must contain a literal string or set of strings"
        )
    if len(values) != literal_size or not values:
        raise ClinvarPostflightError(f"trusted source {name} contains a non-string or no values")
    return values


def _assignment_value(tree: ast.Module, name: str) -> ast.expr:
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(matches) != 1:
        raise ClinvarPostflightError(
            f"trusted source must assign {name} exactly once, observed {len(matches)}"
        )
    return matches[0]


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ClinvarPostflightError(
            f"trusted source must define {name} exactly once, observed {len(matches)}"
        )
    return matches[0]


def _string_returns(tree: ast.Module, function_name: str) -> tuple[str, ...]:
    function = _function(tree, function_name)
    values = {
        node.value.value
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    if not values:
        raise ClinvarPostflightError(f"trusted {function_name} has no literal string outcomes")
    return tuple(sorted(values))


def _parquet_schema_from_ast(tree: ast.Module) -> tuple[tuple[str, str], ...]:
    function = _function(tree, "_parquet_schema")
    returns = [node for node in function.body if isinstance(node, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Call):
        raise ClinvarPostflightError("trusted _parquet_schema must directly return pa.schema(...)")
    call = returns[0].value
    if (
        not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "pa"
        or call.func.attr != "schema"
        or len(call.args) != 1
        or not isinstance(call.args[0], ast.List)
    ):
        raise ClinvarPostflightError("trusted _parquet_schema has an unsupported AST shape")
    fields: list[tuple[str, str]] = []
    for entry in call.args[0].elts:
        if not isinstance(entry, ast.Tuple) or len(entry.elts) != 2:
            raise ClinvarPostflightError("trusted Parquet schema entries must be two-tuples")
        raw_name, raw_type = entry.elts
        if not isinstance(raw_name, ast.Constant) or not isinstance(raw_name.value, str):
            raise ClinvarPostflightError("trusted Parquet schema field name must be a literal")
        if (
            not isinstance(raw_type, ast.Call)
            or raw_type.args
            or raw_type.keywords
            or not isinstance(raw_type.func, ast.Attribute)
            or not isinstance(raw_type.func.value, ast.Name)
            or raw_type.func.value.id != "pa"
            or raw_type.func.attr not in {"string", "int64"}
        ):
            raise ClinvarPostflightError(
                f"trusted Parquet type for {raw_name.value!r} is unsupported"
            )
        fields.append((raw_name.value, raw_type.func.attr))
    if len(fields) != len({name for name, _kind in fields}):
        raise ClinvarPostflightError("trusted Parquet schema contains duplicate fields")
    return tuple(fields)


def _variant_fields_from_ast(tree: ast.Module) -> tuple[tuple[str, ...], tuple[str, ...]]:
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ClinvarVariant"
    ]
    if len(classes) != 1:
        raise ClinvarPostflightError("trusted source must define ClinvarVariant exactly once")
    annotations = [node for node in classes[0].body if isinstance(node, ast.AnnAssign)]
    fields: list[str] = []
    nullable: list[str] = []
    for annotation in annotations:
        if not isinstance(annotation.target, ast.Name):
            raise ClinvarPostflightError("trusted ClinvarVariant field must be a simple name")
        fields.append(annotation.target.id)
        if any(
            isinstance(node, ast.Constant) and node.value is None
            for node in ast.walk(annotation.annotation)
        ):
            nullable.append(annotation.target.id)
    return tuple(fields), tuple(sorted(nullable))


def _target_path_parts(tree: ast.Module) -> tuple[str, ...]:
    function = _function(tree, "prepare_clinvar_shard")
    assignments = [
        node.value
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "target" for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ClinvarPostflightError("trusted prepare_clinvar_shard must assign target once")

    def flatten(node: ast.expr) -> list[ast.expr]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return [*flatten(node.left), *flatten(node.right)]
        return [node]

    raw_parts = flatten(assignments[0])
    if (
        not raw_parts
        or not isinstance(raw_parts[0], ast.Call)
        or not isinstance(raw_parts[0].func, ast.Name)
        or raw_parts[0].func.id != "Path"
    ):
        raise ClinvarPostflightError("trusted output target must begin with Path(output_dir)")
    parts: list[str] = []
    for part in raw_parts[1:]:
        if isinstance(part, ast.Name) and part.id == "release":
            parts.append("{release}")
        elif isinstance(part, ast.Constant) and isinstance(part.value, str):
            parts.append(part.value)
        else:
            raise ClinvarPostflightError("trusted output target contains a dynamic path component")
    return tuple(parts)


def _typer_command_name(tree: ast.Module) -> str:
    call = _assignment_value(tree, "app")
    if not isinstance(call, ast.Call):
        raise ClinvarPostflightError("trusted CLI app must be constructed by a call")
    names = [
        keyword.value
        for keyword in call.keywords
        if keyword.arg == "name"
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, str)
    ]
    if len(names) != 1:
        raise ClinvarPostflightError("trusted CLI app must have one literal name")
    return str(names[0].value)


def _function_default(tree: ast.Module, function_name: str, argument_name: str) -> int:
    function = _function(tree, function_name)
    arguments = [*function.args.posonlyargs, *function.args.args]
    defaults = [None] * (len(arguments) - len(function.args.defaults)) + list(
        function.args.defaults
    )
    matches = [
        default
        for argument, default in zip(arguments, defaults, strict=True)
        if argument.arg == argument_name
    ]
    if (
        len(matches) != 1
        or not isinstance(matches[0], ast.Constant)
        or isinstance(matches[0].value, bool)
        or not isinstance(matches[0].value, int)
    ):
        raise ClinvarPostflightError(
            f"trusted CLI {argument_name} default must be one integer literal"
        )
    return matches[0].value


def _subscript_assignment_keys(
    tree: ast.Module, function_name: str, mapping_name: str
) -> tuple[str, ...]:
    function = _function(tree, function_name)
    keys: list[str] = []
    for node in function.body:
        if not isinstance(node, ast.Assign):
            continue
        keys.extend(
            target.slice.value
            for target in node.targets
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == mapping_name
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            )
        )
    return tuple(keys)


def _returned_dict_keys(tree: ast.Module, function_name: str) -> tuple[str, ...]:
    function = _function(tree, function_name)
    returns = [node for node in function.body if isinstance(node, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Dict):
        raise ClinvarPostflightError(f"trusted {function_name} must directly return a dict")
    keys: list[str] = []
    for key in returns[0].value.keys:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise ClinvarPostflightError(f"trusted {function_name} contains a dynamic dict key")
        keys.append(key.value)
    return tuple(keys)


def _audit_bundle(
    paths: Mapping[str, Path],
    *,
    repo_id: str,
    revision: str,
    namespace: str,
    source_commit: str,
    release: str,
    source_blobs: Mapping[str, bytes],
    source_contract: _SourceContract,
) -> dict[str, object]:
    parquet_relative = f"clinvar/{release}/variants.parquet"
    parquet_path = paths[parquet_relative]
    audit_capture = _capture_json(paths["evidence/audit.json"], "remote audit")
    prepare_capture = _capture_json(paths["evidence/prepare_report.json"], "remote prepare report")
    runtime_capture = _capture_json(paths["evidence/runtime_report.json"], "remote runtime report")
    audit = audit_capture.value
    prepare = prepare_capture.value
    runtime = runtime_capture.value
    _validate_audit_header(
        audit,
        source_commit=source_commit,
        source_contract=source_contract,
    )
    audit_prepare = _require_mapping(audit["prepare_report"], "audit.prepare_report")
    _require_equal(dict(audit_prepare), dict(prepare), "audit.prepare_report")
    source_identity = _validate_source_and_prepare(
        audit,
        prepare=prepare,
        release=release,
        source_contract=source_contract,
    )
    _validate_runtime(audit, runtime=runtime, prepare=prepare, source_contract=source_contract)
    with _private_binary_snapshot(parquet_path) as parquet_snapshot:
        output_identity, expected_class_balance = _validate_output_evidence(
            audit,
            prepare=prepare,
            parquet_sha256=parquet_snapshot.sha256,
            parquet_size_bytes=parquet_snapshot.size_bytes,
            release=release,
            source_contract=source_contract,
        )
        parquet_audit = _audit_clinvar_parquet_stream(
            parquet_snapshot.stream,
            source_contract=source_contract,
            expected_records=_require_int(output_identity["records"], "output.records"),
            expected_class_balance=expected_class_balance,
        )
        captures: dict[str, _CapturedJson | _BinarySnapshot] = {
            parquet_relative: parquet_snapshot,
            "evidence/audit.json": audit_capture,
            "evidence/prepare_report.json": prepare_capture,
            "evidence/runtime_report.json": runtime_capture,
        }
        file_identities = {
            relative_path: {
                "sha256": capture.sha256,
                "size_bytes": capture.size_bytes,
            }
            for relative_path, capture in sorted(captures.items())
        }
    source_contract_files = {
        path: {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
        for path, payload in sorted(source_blobs.items())
    }
    claim_boundary = _require_str(audit["claim_boundary"], "audit.claim_boundary")
    return {
        "schema_version": REMOTE_POSTFLIGHT_SCHEMA_VERSION,
        "ok": True,
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "namespace": namespace,
        "source_commit": source_commit,
        "release": release,
        "verified_files": sorted(paths),
        "file_identities": file_identities,
        "trusted_source_contract": {
            "files": source_contract_files,
            "schema_version": source_contract.schema_version,
            "parquet_schema": [
                {"name": name, "type": kind} for name, kind in source_contract.parquet_schema
            ],
            "nullable_fields": list(source_contract.nullable_fields),
            "normalized_classes": list(source_contract.normalized_classes),
            "labelled_classes": list(source_contract.labelled_classes),
            "allele_alphabet": list(source_contract.allele_alphabet),
            "cli_command": source_contract.cli_command,
            "max_allele_len": source_contract.max_allele_len,
            "output_path_template": source_contract.output_path_template,
            "prepare_report_enrichments": list(source_contract.prepare_report_enrichments),
            "file_identity_fields": list(source_contract.file_identity_fields),
        },
        "source_identity": source_identity,
        "output_identity": output_identity,
        "parquet_audit": parquet_audit,
        "claim_boundary": claim_boundary,
        "checks": [
            "exact_hub_revision_resolved",
            "complete_namespace_file_set",
            "source_contract_loaded_from_exact_git_commit",
            "source_contract_derived_from_ast",
            "audit_prepare_runtime_reconciled",
            "source_release_sha256_and_size_reconciled",
            "parquet_sha256_and_size_recomputed",
            "parquet_schema_derived_from_source_commit",
            "parquet_full_scan_recomputed",
        ],
    }


def _validate_audit_header(
    audit: Mapping[str, object],
    *,
    source_commit: str,
    source_contract: _SourceContract,
) -> None:
    _require_exact_keys(
        audit,
        {
            "claim_boundary",
            "commit_sha",
            "container_image",
            "generated_at",
            "generated_by",
            "ok",
            "output",
            "prepare_report",
            "runtime",
            "schema_version",
            "source",
        },
        "audit",
    )
    _require_equal(audit["ok"], True, "audit.ok")
    _require_equal(audit["commit_sha"], source_commit, "audit.commit_sha")
    _require_equal(audit["schema_version"], source_contract.schema_version, "audit.schema_version")
    _require_equal(
        audit["generated_by"],
        "hf-job:clinvar-corrected-shard-audit",
        "audit.generated_by",
    )
    image = _require_str(audit["container_image"], "audit.container_image")
    if _CONTAINER_IMAGE.fullmatch(image) is None:
        raise ClinvarPostflightError("audit.container_image must be digest pinned")
    generated_at = _require_str(audit["generated_at"], "audit.generated_at")
    try:
        parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ClinvarPostflightError("audit.generated_at must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ClinvarPostflightError("audit.generated_at must include a timezone")
    claim = _require_str(audit["claim_boundary"], "audit.claim_boundary")
    required_boundaries = (
        "leakage-safe eval split",
        "label correctness",
        "clinical utility",
        "model quality",
    )
    if any(boundary not in claim for boundary in required_boundaries):
        raise ClinvarPostflightError("audit.claim_boundary lost a required scientific limitation")


def _validate_source_and_prepare(
    audit: Mapping[str, object],
    *,
    prepare: Mapping[str, object],
    release: str,
    source_contract: _SourceContract,
) -> dict[str, object]:
    _require_exact_keys(
        prepare,
        {
            "allele_records_seen",
            "already_exists",
            "command",
            "elapsed_seconds",
            "input_path",
            "input_sha256",
            "input_size_bytes",
            "input_vcf",
            "output_parquet",
            "output_path",
            "output_sha256",
            "records_read",
            "records_written",
            "release",
            "runtime",
            "size_bytes",
            "skipped_allele",
        },
        "prepare report",
    )
    source = _require_mapping(audit["source"], "audit.source")
    _require_exact_keys(source, {"md5", "release", "sha256", "size_bytes", "url"}, "source")
    _require_equal(source["release"], release, "audit.source.release")
    _require_equal(prepare["release"], release, "prepare report.release")
    sha256 = _require_prefixed_sha256(source["sha256"], "audit.source.sha256")
    md5 = _require_str(source["md5"], "audit.source.md5")
    if _MD5.fullmatch(md5) is None:
        raise ClinvarPostflightError("audit.source.md5 must be a lowercase MD5 hex digest")
    size_bytes = _require_positive_int(source["size_bytes"], "audit.source.size_bytes")
    expected_url = (
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/"
        f"{release[:4]}/clinvar_{release.replace('-', '')}.vcf.gz"
    )
    _require_equal(source["url"], expected_url, "audit.source.url")
    input_path = _require_str(prepare["input_path"], "prepare report.input_path")
    input_identity = _require_mapping(prepare["input_vcf"], "prepare report.input_vcf")
    _require_exact_keys(input_identity, {"path", "sha256", "size_bytes"}, "input_vcf")
    expected_input = {
        "path": input_path,
        "sha256": f"sha256:{sha256}",
        "size_bytes": size_bytes,
    }
    _require_equal(dict(input_identity), expected_input, "prepare report.input_vcf")
    _require_equal(prepare["input_sha256"], f"sha256:{sha256}", "prepare input_sha256")
    _require_equal(prepare["input_size_bytes"], size_bytes, "prepare input_size_bytes")
    _require_equal(prepare["already_exists"], False, "prepare report.already_exists")
    records_read = _require_positive_int(prepare["records_read"], "prepare records_read")
    allele_records_seen = _require_positive_int(
        prepare["allele_records_seen"], "prepare allele_records_seen"
    )
    records_written = _require_positive_int(prepare["records_written"], "prepare records_written")
    skipped_allele = _require_non_negative_int(prepare["skipped_allele"], "prepare skipped_allele")
    if records_written + skipped_allele != allele_records_seen:
        raise ClinvarPostflightError(
            "prepare allele counts do not reconcile: "
            f"written={records_written}, skipped={skipped_allele}, seen={allele_records_seen}"
        )
    if allele_records_seen < records_read:
        raise ClinvarPostflightError("prepare allele_records_seen cannot be below records_read")
    elapsed = _require_positive_number(prepare["elapsed_seconds"], "prepare elapsed_seconds")
    process_runtime = _require_mapping(prepare["runtime"], "prepare report.runtime")
    _require_exact_keys(
        process_runtime,
        {"elapsed_seconds", "peak_memory_note", "process_peak_rss_bytes"},
        "prepare report.runtime",
    )
    runtime_elapsed = _require_positive_number(
        process_runtime["elapsed_seconds"], "prepare runtime.elapsed_seconds"
    )
    if abs(elapsed - runtime_elapsed) > 1.0:
        raise ClinvarPostflightError(
            "prepare elapsed_seconds values differ by more than one second"
        )
    _require_positive_int(
        process_runtime["process_peak_rss_bytes"], "prepare runtime.process_peak_rss_bytes"
    )
    _require_str(process_runtime["peak_memory_note"], "prepare runtime.peak_memory_note")
    command = _require_str(prepare["command"], "prepare report.command")
    output_path = _require_str(prepare["output_path"], "prepare report.output_path")
    relative_output = source_contract.output_path_template.format(release=release)
    suffix = f"/{relative_output}"
    if not output_path.endswith(suffix):
        raise ClinvarPostflightError(
            f"prepare output path must end with trusted source path {suffix!r}"
        )
    output_root = output_path.removesuffix(suffix)
    expected_command = shlex.join(
        [
            source_contract.cli_command,
            "--input-vcf",
            input_path,
            "--output",
            output_root,
            "--release",
            release,
            "--max-allele-len",
            str(source_contract.max_allele_len),
        ]
    )
    _require_equal(command, expected_command, "prepare report.command")
    return {
        "url": expected_url,
        "release": release,
        "md5": md5,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "verification_scope": [
            "release_reconciled",
            "sha256_reconciled",
            "size_bytes_reconciled",
        ],
        "verification_limitation": (
            "The source archive is not included in the Hub namespace; its MD5 and URL are "
            "receipt fields, not bytes recomputed by this postflight."
        ),
    }


def _validate_runtime(
    audit: Mapping[str, object],
    *,
    runtime: Mapping[str, object],
    prepare: Mapping[str, object],
    source_contract: _SourceContract,
) -> None:
    _require_exact_keys(
        runtime,
        {"command", "peak_rss_bytes", "peak_rss_source", "returncode", "wall_time_seconds"},
        "runtime report",
    )
    audit_runtime = _require_mapping(audit["runtime"], "audit.runtime")
    _require_exact_keys(
        audit_runtime,
        {
            "command",
            "cpu_count",
            "flavor",
            "peak_rss_bytes",
            "peak_rss_source",
            "platform",
            "python",
            "ram_gb",
            "returncode",
            "wall_time_seconds",
        },
        "audit.runtime",
    )
    for field, value in runtime.items():
        _require_equal(audit_runtime[field], value, f"audit.runtime.{field}")
    _require_equal(runtime["returncode"], 0, "runtime report.returncode")
    _require_positive_int(runtime["peak_rss_bytes"], "runtime report.peak_rss_bytes")
    _require_positive_number(runtime["wall_time_seconds"], "runtime report.wall_time_seconds")
    _require_str(runtime["peak_rss_source"], "runtime report.peak_rss_source")
    _require_positive_int(audit_runtime["cpu_count"], "audit.runtime.cpu_count")
    _require_positive_int(audit_runtime["ram_gb"], "audit.runtime.ram_gb")
    for field in ("flavor", "platform", "python"):
        _require_str(audit_runtime[field], f"audit.runtime.{field}")
    raw_command = runtime["command"]
    if not isinstance(raw_command, list) or any(not isinstance(item, str) for item in raw_command):
        raise ClinvarPostflightError("runtime report.command must be an array of strings")
    prepare_tokens = shlex.split(_require_str(prepare["command"], "prepare report.command"))
    if not raw_command:
        raise ClinvarPostflightError("runtime report.command must not be empty")
    if Path(raw_command[0]).name != source_contract.cli_command:
        raise ClinvarPostflightError("runtime command executable drifted from trusted CLI command")
    expected_tail = ["--no-banner", *prepare_tokens[1:]]
    _require_equal(raw_command[1:], expected_tail, "runtime report.command arguments")


def _validate_output_evidence(
    audit: Mapping[str, object],
    *,
    prepare: Mapping[str, object],
    parquet_sha256: str,
    parquet_size_bytes: int,
    release: str,
    source_contract: _SourceContract,
) -> tuple[dict[str, object], dict[str, int]]:
    output = _require_mapping(audit["output"], "audit.output")
    _require_exact_keys(
        output,
        {"class_balance", "path", "records", "sha256", "size_bytes"},
        "audit.output",
    )
    relative_output = source_contract.output_path_template.format(release=release)
    _require_equal(output["path"], relative_output, "audit.output.path")
    digest = _require_prefixed_sha256(output["sha256"], "audit.output.sha256")
    size_bytes = _require_positive_int(output["size_bytes"], "audit.output.size_bytes")
    records = _require_positive_int(output["records"], "audit.output.records")
    output_path = _require_str(prepare["output_path"], "prepare output_path")
    expected_output_identity = {
        "path": output_path,
        "sha256": f"sha256:{digest}",
        "size_bytes": size_bytes,
    }
    nested_output = _require_mapping(prepare["output_parquet"], "prepare output_parquet")
    _require_exact_keys(nested_output, {"path", "sha256", "size_bytes"}, "output_parquet")
    _require_equal(dict(nested_output), expected_output_identity, "prepare output_parquet")
    _require_equal(prepare["output_sha256"], f"sha256:{digest}", "prepare output_sha256")
    _require_equal(prepare["size_bytes"], size_bytes, "prepare size_bytes")
    _require_equal(prepare["records_written"], records, "prepare records_written")
    _require_equal(parquet_sha256, digest, "Parquet SHA-256")
    _require_equal(parquet_size_bytes, size_bytes, "Parquet size_bytes")
    raw_balance = _require_mapping(output["class_balance"], "audit.output.class_balance")
    expected_labels = set(source_contract.normalized_classes)
    if not set(raw_balance) <= expected_labels or not raw_balance:
        raise ClinvarPostflightError("audit.output.class_balance contains unknown or no classes")
    class_balance = {
        label: _require_non_negative_int(value, f"class_balance.{label}")
        for label, value in sorted(raw_balance.items())
    }
    if sum(class_balance.values()) != records:
        raise ClinvarPostflightError("audit output class balance does not sum to records")
    return (
        {
            "path": relative_output,
            "sha256": digest,
            "size_bytes": size_bytes,
            "records": records,
            "class_balance": class_balance,
        },
        class_balance,
    )


def audit_clinvar_parquet(
    path: Path,
    *,
    source_contract: _SourceContract,
    expected_records: int,
    expected_class_balance: Mapping[str, int],
) -> dict[str, object]:
    """Full-scan one ClinVar shard against schema derived from source commit bytes."""
    with _private_binary_snapshot(path) as snapshot:
        return _audit_clinvar_parquet_stream(
            snapshot.stream,
            source_contract=source_contract,
            expected_records=expected_records,
            expected_class_balance=expected_class_balance,
        )


def _audit_clinvar_parquet_stream(
    stream: BinaryIO,
    *,
    source_contract: _SourceContract,
    expected_records: int,
    expected_class_balance: Mapping[str, int],
) -> dict[str, object]:
    """Full-scan one already captured ClinVar Parquet byte sequence."""
    try:
        pa = importlib.import_module("pyarrow")
        pq = importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise ClinvarPostflightError(
            "independent ClinVar Parquet audit requires pyarrow; install the dev or train extra"
        ) from exc
    type_factories = {"string": pa.string, "int64": pa.int64}
    expected_schema = pa.schema(
        [(name, type_factories[kind]()) for name, kind in source_contract.parquet_schema]
    )
    stream.seek(0)
    parquet = pq.ParquetFile(stream)
    observed_schema = parquet.schema_arrow
    if not observed_schema.equals(expected_schema, check_metadata=True):
        raise ClinvarPostflightError(
            f"ClinVar Parquet schema drifted: expected {expected_schema}, "
            f"observed {observed_schema}"
        )
    metadata_rows = int(parquet.metadata.num_rows)
    _require_equal(metadata_rows, expected_records, "Parquet metadata row count")
    columns = [name for name, _kind in source_contract.parquet_schema]
    nullable = set(source_contract.nullable_fields)
    alphabet = set(source_contract.allele_alphabet)
    class_balance: Counter[str] = Counter()
    chromosome_balance: Counter[str] = Counter()
    schema_versions: Counter[str] = Counter()
    null_counts: Counter[str] = Counter()
    scanned_rows = 0
    min_position: int | None = None
    max_position: int | None = None
    min_clinvar_id: int | None = None
    max_clinvar_id: int | None = None
    for batch in parquet.iter_batches(batch_size=_PARQUET_AUDIT_BATCH_ROWS, columns=columns):
        payload = batch.to_pydict()
        batch_rows = batch.num_rows
        scanned_rows += batch_rows
        for column in columns:
            null_counts[column] += int(
                batch.column(batch.schema.get_field_index(column)).null_count
            )
        for column in columns:
            if column not in nullable and null_counts[column] > 0:
                raise ClinvarPostflightError(
                    f"ClinVar Parquet required column {column!r} has nulls"
                )
        chrom_values = payload["chrom"]
        label_values = payload["clinical_significance"]
        version_values = payload["schema_version"]
        chromosome_balance.update(str(value) for value in chrom_values if value is not None)
        class_balance.update(str(value) for value in label_values if value is not None)
        schema_versions.update(str(value) for value in version_values if value is not None)
        if any(not isinstance(value, str) or not value for value in chrom_values):
            raise ClinvarPostflightError("ClinVar Parquet chromosome values must be non-empty")
        for column in ("ref", "alt"):
            for value in payload[column]:
                if (
                    not isinstance(value, str)
                    or not value
                    or len(value) > source_contract.max_allele_len
                    or not set(value) <= alphabet
                ):
                    raise ClinvarPostflightError(
                        f"ClinVar Parquet {column} violates the trusted allele contract"
                    )
        if any(ref == alt for ref, alt in zip(payload["ref"], payload["alt"], strict=True)):
            raise ClinvarPostflightError("ClinVar Parquet ref and alt must differ")
        for column in ("review_status",):
            if any(not isinstance(value, str) or not value for value in payload[column]):
                raise ClinvarPostflightError(f"ClinVar Parquet {column} values must be non-empty")
        for value in payload["gene_symbol"]:
            if value is not None and (not isinstance(value, str) or not value):
                raise ClinvarPostflightError(
                    "ClinVar Parquet gene_symbol must be null or a non-empty string"
                )
        positions = payload["pos"]
        ids = payload["clinvar_id"]
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in positions
        ):
            raise ClinvarPostflightError("ClinVar Parquet positions must be positive integers")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in ids
        ):
            raise ClinvarPostflightError(
                "ClinVar Parquet clinvar_id values must be positive integers"
            )
        batch_min_position = min(positions)
        batch_max_position = max(positions)
        batch_min_clinvar_id = min(ids)
        batch_max_clinvar_id = max(ids)
        min_position = (
            batch_min_position if min_position is None else min(min_position, batch_min_position)
        )
        max_position = (
            batch_max_position if max_position is None else max(max_position, batch_max_position)
        )
        min_clinvar_id = (
            batch_min_clinvar_id
            if min_clinvar_id is None
            else min(min_clinvar_id, batch_min_clinvar_id)
        )
        max_clinvar_id = (
            batch_max_clinvar_id
            if max_clinvar_id is None
            else max(max_clinvar_id, batch_max_clinvar_id)
        )
    _require_equal(scanned_rows, expected_records, "Parquet scanned row count")
    _require_equal(
        dict(sorted(class_balance.items())),
        dict(sorted(expected_class_balance.items())),
        "Parquet class balance",
    )
    unknown_classes = set(class_balance) - set(source_contract.normalized_classes)
    if unknown_classes:
        raise ClinvarPostflightError(
            f"ClinVar Parquet contains classes absent from source contract: {sorted(unknown_classes)}"
        )
    _require_equal(
        dict(schema_versions),
        {source_contract.schema_version: expected_records},
        "Parquet schema_version values",
    )
    if sum(chromosome_balance.values()) != expected_records:
        raise ClinvarPostflightError("Parquet chromosome counts do not sum to records")
    return {
        "metadata_row_count": metadata_rows,
        "scanned_row_count": scanned_rows,
        "class_balance": dict(sorted(class_balance.items())),
        "chromosome_balance": dict(sorted(chromosome_balance.items())),
        "schema_version_balance": dict(sorted(schema_versions.items())),
        "null_counts": dict(sorted(null_counts.items())),
        "position_range": {"min": min_position, "max": max_position},
        "clinvar_id_range": {"min": min_clinvar_id, "max": max_clinvar_id},
        "schema": [{"name": name, "type": kind} for name, kind in source_contract.parquet_schema],
    }


def _capture_json(path: Path, field: str) -> _CapturedJson:
    payload = path.read_bytes()
    return _CapturedJson(
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
            raise ClinvarPostflightError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


@contextmanager
def _private_binary_snapshot(path: Path) -> Iterator[_BinarySnapshot]:
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        before_open = path.lstat()
    except OSError as exc:
        raise ClinvarPostflightError(f"cannot inspect binary snapshot input {path}: {exc}") from exc
    if not stat.S_ISREG(before_open.st_mode):
        raise ClinvarPostflightError(
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
        raise ClinvarPostflightError(
            f"binary snapshot input must be a regular file without following symlinks: {path}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before_open.st_dev
            or opened.st_ino != before_open.st_ino
        ):
            raise ClinvarPostflightError(
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


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ClinvarPostflightError(f"{field} must be an object")
    return value


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ClinvarPostflightError(f"{field} must be a non-empty string")
    return value


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ClinvarPostflightError(f"{field} must be an integer")
    return value


def _require_non_negative_int(value: object, field: str) -> int:
    integer = _require_int(value, field)
    if integer < 0:
        raise ClinvarPostflightError(f"{field} must be non-negative")
    return integer


def _require_positive_int(value: object, field: str) -> int:
    integer = _require_int(value, field)
    if integer <= 0:
        raise ClinvarPostflightError(f"{field} must be positive")
    return integer


def _require_positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ClinvarPostflightError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ClinvarPostflightError(f"{field} must be finite and positive")
    return number


def _require_prefixed_sha256(value: object, field: str) -> str:
    text = _require_str(value, field)
    match = _PREFIXED_SHA256.fullmatch(text)
    if match is None:
        raise ClinvarPostflightError(f"{field} must be a lowercase sha256:<hex> digest")
    return match.group(1)


def _require_exact_keys(value: Mapping[str, object], expected: set[str], field: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ClinvarPostflightError(
            f"{field} keys drifted: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _require_equal(observed: object, expected: object, field: str) -> None:
    if not _json_equal(observed, expected):
        raise ClinvarPostflightError(
            f"{field} drifted: expected {expected!r}, observed {observed!r}"
        )


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


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    write_immutable_json(path, value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
