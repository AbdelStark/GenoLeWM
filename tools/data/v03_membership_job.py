# SPDX-License-Identifier: Apache-2.0
"""Author lineage-bound inputs for the hosted v0.3 membership build."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Final

from tools.data._immutable_json import ImmutableJsonError, write_immutable_json

LINEAGE_SCHEMA_VERSION: Final = "geno-lewm.v03-snapshot-lineage.v1"
BUILD_SPEC_SCHEMA_VERSION: Final = "geno-lewm.membership-build-spec.v1"
CANDIDATE_SNAPSHOT_ID: Final = "geno-lewm-data-v0.3.0-r1"
DATA_REPO: Final = "abdelstark/geno-lewm-data"
DATA_REPO_TYPE: Final = "dataset"
GNOMAD_REVISION: Final = "f3676763b3f7f71d0d0d098588e9bf377faa0c5c"
CLINVAR_REVISION: Final = "9e1a2b279681177a7ca00b30b9eb8048b511d1cb"

_COMMIT: Final = re.compile(r"[0-9a-f]{40}")
_CONTAINER: Final = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")
_SHA256: Final = re.compile(r"sha256:[0-9a-f]{64}")
_ARTIFACT_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_NAMESPACE: Final = re.compile(r"[A-Za-z0-9._/-]+")
_AUTOSOMES: Final = tuple(str(chromosome) for chromosome in range(1, 23))


class MembershipJobError(ValueError):
    """Raised when hosted membership-build inputs violate the closed contract."""


def main(argv: list[str] | None = None) -> int:
    """Run the hosted membership input-authoring contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser(
        "author-download-plan",
        help="validate one immutable lineage and author its exact 23-file download plan",
    )
    plan.add_argument("--lineage-json", type=Path, required=True)
    plan.add_argument("--expected-lineage-sha256", required=True)
    plan.add_argument("--expected-lineage-size-bytes", type=int, required=True)
    plan.add_argument("--output-json", type=Path, required=True)
    author = subparsers.add_parser(
        "author-spec",
        help="verify downloaded lineage sources and author one closed build spec",
    )
    author.add_argument("--lineage-json", type=Path, required=True)
    author.add_argument("--expected-lineage-sha256", required=True)
    author.add_argument("--expected-lineage-size-bytes", type=int, required=True)
    author.add_argument("--gnomad-download-root", type=Path, required=True)
    author.add_argument("--clinvar-download-root", type=Path, required=True)
    author.add_argument("--artifact-id", required=True)
    author.add_argument("--builder-git-commit", required=True)
    author.add_argument("--container-image", required=True)
    author.add_argument("--output-json", type=Path, required=True)
    author.add_argument("--identity-report-json", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "author-download-plan":
            payload = author_membership_download_plan(
                lineage_path=args.lineage_json,
                expected_lineage_sha256=args.expected_lineage_sha256,
                expected_lineage_size_bytes=args.expected_lineage_size_bytes,
            )
            write_immutable_json(args.output_json, payload)
        else:
            spec, report = author_membership_build_spec(
                lineage_path=args.lineage_json,
                expected_lineage_sha256=args.expected_lineage_sha256,
                expected_lineage_size_bytes=args.expected_lineage_size_bytes,
                gnomad_download_root=args.gnomad_download_root,
                clinvar_download_root=args.clinvar_download_root,
                artifact_id=args.artifact_id,
                builder_git_commit=args.builder_git_commit,
                container_image=args.container_image,
                spec_path=args.output_json,
            )
            write_immutable_json(args.output_json, spec)
            write_immutable_json(args.identity_report_json, report)
    except (ImmutableJsonError, MembershipJobError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def author_membership_download_plan(
    *,
    lineage_path: Path,
    expected_lineage_sha256: str,
    expected_lineage_size_bytes: int,
) -> dict[str, object]:
    """Return the exact 23 remote paths declared by one immutable lineage."""
    _require_fullmatch(_SHA256, expected_lineage_sha256, "expected lineage sha256")
    if expected_lineage_size_bytes <= 0:
        raise MembershipJobError("expected lineage size must be positive")
    lineage, lineage_identity = _capture_expected_lineage(
        lineage_path=lineage_path,
        expected_sha256=expected_lineage_sha256,
        expected_size_bytes=expected_lineage_size_bytes,
    )
    _require_equal(lineage.get("schema_version"), LINEAGE_SCHEMA_VERSION, "lineage schema")
    _require_equal(
        lineage.get("candidate_snapshot_id"), CANDIDATE_SNAPSHOT_ID, "candidate snapshot"
    )
    _require_equal(lineage.get("membership_status"), "not_created", "membership status")
    lineage_id = _require_text(lineage.get("lineage_id"), "lineage_id")
    _require_fullmatch(_SHA256, lineage_id, "lineage_id")

    gnomad = _require_mapping(lineage.get("gnomad"), "lineage gnomAD")
    _require_repository(gnomad, "gnomAD")
    shards = _ordered_gnomad_shards(gnomad)
    downloads: list[dict[str, object]] = []
    for chromosome, shard in zip(_AUTOSOMES, shards, strict=True):
        _require_equal(shard.get("revision"), GNOMAD_REVISION, "gnomAD revision")
        namespace = _require_namespace(shard.get("namespace"), "gnomAD namespace")
        output = _require_mapping(shard.get("output"), "gnomAD output")
        artifact_path = _require_artifact_path(
            output.get("artifact_path"), namespace=namespace, kind="gnomad"
        )
        downloads.append(
            _download_entry(
                kind="gnomad",
                chromosome=chromosome,
                revision=GNOMAD_REVISION,
                namespace=namespace,
                artifact_path=artifact_path,
                output=output,
            )
        )

    clinvar = _require_mapping(lineage.get("clinvar"), "lineage ClinVar")
    _require_repository(clinvar, "ClinVar")
    _require_equal(clinvar.get("revision"), CLINVAR_REVISION, "ClinVar revision")
    namespace = _require_namespace(clinvar.get("namespace"), "ClinVar namespace")
    output = _require_mapping(clinvar.get("output"), "ClinVar output")
    artifact_path = _require_artifact_path(
        output.get("artifact_path"), namespace=namespace, kind="clinvar"
    )
    downloads.append(
        _download_entry(
            kind="clinvar",
            chromosome=None,
            revision=CLINVAR_REVISION,
            namespace=namespace,
            artifact_path=artifact_path,
            output=output,
        )
    )
    return {
        "schema_version": "geno-lewm.membership-download-plan.v1",
        "generated_by": "tools.data.v03_membership_job",
        "candidate_snapshot_id": CANDIDATE_SNAPSHOT_ID,
        "lineage": {
            "lineage_id": lineage_id,
            "sha256": lineage_identity["sha256"],
            "size_bytes": lineage_identity["size_bytes"],
        },
        "downloads": downloads,
        "claim_boundary": (
            "This plan identifies exact source bytes for membership construction only. "
            "It does not create phased haplotypes, release a v0.3 snapshot, or establish "
            "dataset representativeness, model quality, benchmark performance, or clinical validity."
        ),
    }


def author_membership_build_spec(
    *,
    lineage_path: Path,
    expected_lineage_sha256: str,
    expected_lineage_size_bytes: int,
    gnomad_download_root: Path,
    clinvar_download_root: Path,
    artifact_id: str,
    builder_git_commit: str,
    container_image: str,
    spec_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Verify 23 downloaded artifacts and return their closed build contract."""
    _require_fullmatch(_SHA256, expected_lineage_sha256, "expected lineage sha256")
    if expected_lineage_size_bytes <= 0:
        raise MembershipJobError("expected lineage size must be positive")
    _require_fullmatch(_ARTIFACT_ID, artifact_id, "membership artifact_id")
    _require_fullmatch(_COMMIT, builder_git_commit, "builder git commit")
    _require_fullmatch(_CONTAINER, container_image, "builder container image")

    lineage, lineage_identity = _capture_expected_lineage(
        lineage_path=lineage_path,
        expected_sha256=expected_lineage_sha256,
        expected_size_bytes=expected_lineage_size_bytes,
    )
    _require_equal(lineage.get("schema_version"), LINEAGE_SCHEMA_VERSION, "lineage schema")
    _require_equal(
        lineage.get("candidate_snapshot_id"), CANDIDATE_SNAPSHOT_ID, "candidate snapshot"
    )
    _require_equal(lineage.get("membership_status"), "not_created", "membership status")
    lineage_id = _require_text(lineage.get("lineage_id"), "lineage_id")
    _require_fullmatch(_SHA256, lineage_id, "lineage_id")

    gnomad = _require_mapping(lineage.get("gnomad"), "lineage gnomAD")
    _require_repository(gnomad, "gnomAD")
    shards = _ordered_gnomad_shards(gnomad)

    sources: list[dict[str, object]] = []
    identities: list[dict[str, object]] = []
    for chromosome, shard in zip(_AUTOSOMES, shards, strict=True):
        _require_equal(shard.get("revision"), GNOMAD_REVISION, "gnomAD revision")
        namespace = _require_namespace(shard.get("namespace"), "gnomAD namespace")
        output = _require_mapping(shard.get("output"), "gnomAD output")
        artifact_path = _require_artifact_path(
            output.get("artifact_path"), namespace=namespace, kind="gnomad"
        )
        identity = _verify_declared_artifact(
            root=gnomad_download_root,
            artifact_path=artifact_path,
            output=output,
            label=f"gnomAD chromosome {chromosome}",
        )
        sources.append(
            {
                "kind": "gnomad",
                "chromosome": chromosome,
                "path": _relative_input_path(identity["local_path"], spec_path.parent),
            }
        )
        identities.append(
            {
                "kind": "gnomad",
                "chromosome": chromosome,
                "namespace": namespace,
                "revision": GNOMAD_REVISION,
                "artifact_path": artifact_path,
                "sha256": identity["sha256"],
                "size_bytes": identity["size_bytes"],
            }
        )

    clinvar = _require_mapping(lineage.get("clinvar"), "lineage ClinVar")
    _require_repository(clinvar, "ClinVar")
    _require_equal(clinvar.get("revision"), CLINVAR_REVISION, "ClinVar revision")
    clinvar_namespace = _require_namespace(clinvar.get("namespace"), "ClinVar namespace")
    clinvar_output = _require_mapping(clinvar.get("output"), "ClinVar output")
    clinvar_artifact_path = _require_artifact_path(
        clinvar_output.get("artifact_path"), namespace=clinvar_namespace, kind="clinvar"
    )
    clinvar_identity = _verify_declared_artifact(
        root=clinvar_download_root,
        artifact_path=clinvar_artifact_path,
        output=clinvar_output,
        label="ClinVar",
    )
    sources.append(
        {
            "kind": "clinvar",
            "path": _relative_input_path(clinvar_identity["local_path"], spec_path.parent),
        }
    )
    identities.append(
        {
            "kind": "clinvar",
            "namespace": clinvar_namespace,
            "revision": CLINVAR_REVISION,
            "artifact_path": clinvar_artifact_path,
            "sha256": clinvar_identity["sha256"],
            "size_bytes": clinvar_identity["size_bytes"],
        }
    )

    spec: dict[str, object] = {
        "$schema": "./membership-build-spec.schema.json",
        "schema_version": BUILD_SPEC_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "snapshot_lineage": _relative_input_path(lineage_path, spec_path.parent),
        "snapshot_lineage_sha256": expected_lineage_sha256,
        "builder": {
            "git_commit": builder_git_commit,
            "container_image": container_image,
        },
        "sources": sources,
    }
    report = {
        "schema_version": "geno-lewm.membership-source-download-identities.v1",
        "generated_by": "tools.data.v03_membership_job",
        "ok": True,
        "candidate_snapshot_id": CANDIDATE_SNAPSHOT_ID,
        "artifact_id": artifact_id,
        "source_count": len(identities),
        "lineage": {
            "lineage_id": lineage_id,
            "sha256": lineage_identity["sha256"],
            "size_bytes": lineage_identity["size_bytes"],
        },
        "repositories": {
            "gnomad": {
                "repo_id": DATA_REPO,
                "repo_type": DATA_REPO_TYPE,
                "revision": GNOMAD_REVISION,
            },
            "clinvar": {
                "repo_id": DATA_REPO,
                "repo_type": DATA_REPO_TYPE,
                "revision": CLINVAR_REVISION,
            },
        },
        "files": identities,
        "claim_boundary": (
            "This report verifies exact downloaded source bytes for membership construction. "
            "It does not create phased haplotypes, release a v0.3 snapshot, or establish "
            "dataset representativeness, model quality, benchmark performance, or clinical validity."
        ),
    }
    return spec, report


def _verify_declared_artifact(
    *,
    root: Path,
    artifact_path: str,
    output: Mapping[str, object],
    label: str,
) -> dict[str, object]:
    expected_sha256 = _require_text(output.get("sha256"), f"{label} sha256")
    _require_fullmatch(_SHA256, expected_sha256, f"{label} sha256")
    expected_size = output.get("size_bytes")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size <= 0:
        raise MembershipJobError(f"{label} size_bytes must be a positive integer")
    path = _confined_path(root, artifact_path)
    identity = _verify_file(
        path,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size,
        label=label,
    )
    return {**identity, "local_path": path}


def _capture_expected_lineage(
    *,
    lineage_path: Path,
    expected_sha256: str,
    expected_size_bytes: int,
) -> tuple[Mapping[str, object], dict[str, object]]:
    if not lineage_path.is_file() or lineage_path.is_symlink():
        raise MembershipJobError(
            f"snapshot lineage is missing or is not a regular file: {lineage_path}"
        )
    payload = lineage_path.read_bytes()
    observed_size = len(payload)
    if observed_size != expected_size_bytes:
        raise MembershipJobError(
            "snapshot lineage size mismatch: "
            f"expected {expected_size_bytes}, observed {observed_size}"
        )
    observed_sha256 = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if observed_sha256 != expected_sha256:
        raise MembershipJobError(
            "snapshot lineage sha256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    try:
        raw: object = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
        )
    except MembershipJobError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MembershipJobError(f"snapshot lineage is not valid JSON: {exc}") from exc
    return _require_mapping(raw, "snapshot lineage"), {
        "sha256": observed_sha256,
        "size_bytes": observed_size,
    }


def _ordered_gnomad_shards(
    gnomad: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    raw_shards = gnomad.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != 22:
        raise MembershipJobError("lineage gnomAD shards must contain exactly 22 entries")
    shards_by_chromosome: dict[str, Mapping[str, object]] = {}
    for index, raw_shard in enumerate(raw_shards):
        shard = _require_mapping(raw_shard, f"lineage gnomAD shard {index}")
        chromosome = _require_text(shard.get("chromosome"), "gnomAD chromosome")
        if chromosome in shards_by_chromosome:
            raise MembershipJobError(f"duplicate gnomAD chromosome: {chromosome}")
        shards_by_chromosome[chromosome] = shard
    try:
        observed = tuple(sorted(shards_by_chromosome, key=int))
    except ValueError as exc:
        raise MembershipJobError("lineage gnomAD chromosomes must be exactly 1..22") from exc
    if observed != _AUTOSOMES:
        raise MembershipJobError("lineage gnomAD chromosomes must be exactly 1..22")
    return tuple(shards_by_chromosome[chromosome] for chromosome in _AUTOSOMES)


def _download_entry(
    *,
    kind: str,
    chromosome: str | None,
    revision: str,
    namespace: str,
    artifact_path: str,
    output: Mapping[str, object],
) -> dict[str, object]:
    sha256 = _require_text(output.get("sha256"), f"{kind} sha256")
    _require_fullmatch(_SHA256, sha256, f"{kind} sha256")
    size_bytes = output.get("size_bytes")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
        raise MembershipJobError(f"{kind} size_bytes must be a positive integer")
    payload: dict[str, object] = {
        "kind": kind,
        "repo_id": DATA_REPO,
        "repo_type": DATA_REPO_TYPE,
        "revision": revision,
        "namespace": namespace,
        "artifact_path": artifact_path,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }
    if chromosome is not None:
        payload["chromosome"] = chromosome
    return payload


def _verify_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    label: str,
) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise MembershipJobError(f"{label} is missing or is not a regular file: {path}")
    observed_size = path.stat().st_size
    if observed_size != expected_size_bytes:
        raise MembershipJobError(
            f"{label} size mismatch: expected {expected_size_bytes}, observed {observed_size}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    observed_sha256 = f"sha256:{digest.hexdigest()}"
    if observed_sha256 != expected_sha256:
        raise MembershipJobError(
            f"{label} sha256 mismatch: expected {expected_sha256}, observed {observed_sha256}"
        )
    return {"sha256": observed_sha256, "size_bytes": observed_size}


def _require_repository(value: Mapping[str, object], label: str) -> None:
    _require_equal(value.get("repo"), DATA_REPO, f"{label} repo")
    _require_equal(value.get("repo_type"), DATA_REPO_TYPE, f"{label} repo_type")


def _require_artifact_path(value: object, *, namespace: str, kind: str) -> str:
    path = _require_text(value, f"{kind} artifact_path")
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != path:
        raise MembershipJobError(f"{kind} artifact_path must be a normalized relative path")
    suffix = (
        "data/gnomad/v4.1/variants.parquet"
        if kind == "gnomad"
        else "clinvar/2026-04-15/variants.parquet"
    )
    if path != f"{namespace}/{suffix}":
        raise MembershipJobError(f"{kind} artifact_path is not bound to its lineage namespace")
    return path


def _require_namespace(value: object, label: str) -> str:
    namespace = _require_text(value, label)
    relative = PurePosixPath(namespace)
    if (
        _NAMESPACE.fullmatch(namespace) is None
        or "//" in namespace
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != namespace
        or namespace.startswith("/")
        or namespace.endswith("/")
    ):
        raise MembershipJobError(f"{label} must be a normalized relative path")
    return namespace


def _confined_path(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise MembershipJobError("downloaded artifact path escapes its root") from exc
    return candidate


def _relative_input_path(path: object, root: Path) -> str:
    if not isinstance(path, Path):
        raise MembershipJobError("local source path is invalid")
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise MembershipJobError(
            "membership build inputs must be below the spec directory"
        ) from exc
    if not relative.parts:
        raise MembershipJobError("membership build input path must name a file")
    return PurePosixPath(*relative.parts).as_posix()


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise MembershipJobError(f"duplicate snapshot-lineage JSON key: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite_constant(value: str) -> object:
    raise MembershipJobError(f"non-finite snapshot-lineage JSON number: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise MembershipJobError(f"non-finite snapshot-lineage JSON number: {value}")
    return parsed


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MembershipJobError(f"{label} must be an object")
    return value


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MembershipJobError(f"{label} must be a non-empty string")
    return value


def _require_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise MembershipJobError(f"{label} mismatch: expected {expected!r}, observed {observed!r}")


def _require_fullmatch(pattern: re.Pattern[str], value: object, label: str) -> str:
    text = _require_text(value, label)
    if pattern.fullmatch(text) is None:
        raise MembershipJobError(f"{label} is not canonical")
    return text


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
