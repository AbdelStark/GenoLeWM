# SPDX-License-Identifier: Apache-2.0
"""Author and replay the terminal JobInfo receipt for the v0.3 H200 cache proof.

The cache proof is produced inside the Job and therefore cannot attest the
Job's terminal state.  This host-side receipt is authored only after an exact
Hub-revision download of that proof has passed its full read-only replay.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol, cast

from geno_lewm.errors import InputError, ResourceError, exit_code_for
from geno_lewm.provenance.hashing import canonical_json_bytes
from tools.research import v03_cache_h200_launch as launch

SCHEMA_VERSION: Final = "geno-lewm.v03-cache-h200-job-receipt.v1"
GENERATED_BY: Final = "tools.research.v03_cache_h200_job_receipt"
SCHEMA_NAME: Final = "cache-h200-job-receipt.schema.json"
RECEIPT_NAME: Final = "cache-h200-job-receipt.json"
CHECKSUMS_NAME: Final = "SHA256SUMS"
HUGGINGFACE_HUB_VERSION: Final = "1.8.0"
PROOF_REPOSITORY: Final = "abdelstark/geno-lewm-data"
PROOF_REPO_TYPE: Final = "dataset"

_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH: Final = _REPOSITORY_ROOT / "configs/data_v03" / SCHEMA_NAME
_CANONICAL_ORIGIN: Final = "https://github.com/AbdelStark/GenoLeWM.git"
_PROOF_PREFIX: Final = "candidates/v0.3/geno-lewm-data-v0.3.0-r1/cache-h200-proofs"
_COMMIT: Final = re.compile(r"[0-9a-f]{40}\Z")
_JOB_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_CHECKSUM_LINE: Final = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]*)\Z")


class HubReadClient(Protocol):
    """Minimal observable Hub boundary used by the receipt protocol."""

    def inspect_job(self, *, job_id: str, namespace: str) -> Any: ...

    def download_exact_namespace(
        self,
        *,
        repository: str,
        repo_type: str,
        revision: str,
        namespace: str,
        destination: Path,
        token: bool,
    ) -> Path: ...


ProofVerifier = Callable[[Path], dict[str, object]]


class HuggingFaceReadClient:
    """Pinned huggingface-hub adapter for JobInfo and exact-revision downloads."""

    def __init__(self, *, token: str | bool, hub_module: Any | None = None) -> None:
        try:
            hub = hub_module or importlib.import_module("huggingface_hub")
        except ImportError as exc:
            raise ResourceError("huggingface-hub==1.8.0 is required") from exc
        if getattr(hub, "__version__", None) != HUGGINGFACE_HUB_VERSION:
            raise ResourceError(
                "receipt tooling requires huggingface-hub==1.8.0; "
                "run it with `uv run --with huggingface-hub==1.8.0`"
            )
        try:
            self._api: Any = hub.HfApi(token=token)
            self._snapshot_download: Any = hub.snapshot_download
        except (AttributeError, TypeError) as exc:
            raise ResourceError("huggingface-hub lacks the required Jobs/download API") from exc

    def inspect_job(self, *, job_id: str, namespace: str) -> Any:
        try:
            return self._api.inspect_job(job_id=job_id, namespace=namespace)
        except Exception as exc:
            raise ResourceError(
                "terminal Hugging Face JobInfo could not be inspected",
                details={"job_id": job_id, "namespace": namespace},
            ) from exc

    def download_exact_namespace(
        self,
        *,
        repository: str,
        repo_type: str,
        revision: str,
        namespace: str,
        destination: Path,
        token: bool,
    ) -> Path:
        if repository != PROOF_REPOSITORY or repo_type != PROOF_REPO_TYPE:
            raise InputError("receipt downloads must use the canonical public proof dataset")
        revision = _exact_commit(revision, label="Hub download revision")
        namespace = _safe_namespace(namespace, label="Hub download namespace")
        _require_absent(destination, label="exact-revision download destination")
        try:
            info = self._api.repo_info(
                repo_id=repository,
                repo_type=repo_type,
                revision=revision,
                files_metadata=False,
                token=token,
            )
        except Exception as exc:
            raise ResourceError(
                "exact Hub revision could not be resolved",
                details={"repository": repository, "revision": revision},
            ) from exc
        if getattr(info, "sha", None) != revision:
            raise ResourceError(
                "Hugging Face resolved a different Hub revision",
                details={"repository": repository, "expected_revision": revision},
            )
        try:
            downloaded_root = Path(
                self._snapshot_download(
                    repo_id=repository,
                    repo_type=repo_type,
                    revision=revision,
                    allow_patterns=[f"{namespace}/**"],
                    local_dir=str(destination),
                    force_download=True,
                    token=token,
                )
            )
        except Exception as exc:
            raise ResourceError(
                "exact Hub namespace could not be downloaded",
                details={
                    "repository": repository,
                    "revision": revision,
                    "namespace": namespace,
                },
            ) from exc
        if downloaded_root.resolve() != destination.resolve():
            raise ResourceError("Hub download returned an unexpected local root")
        downloaded = downloaded_root / namespace
        _require_directory(downloaded, label="exact downloaded Hub namespace")
        return downloaded


def expected_proof_namespace(source_commit: str, run_attempt: int) -> str:
    """Return the one proof success namespace allowed by the launch contract."""
    commit = _exact_commit(source_commit, label="source commit")
    attempt = _positive_integer(run_attempt, label="run attempt")
    return f"{_PROOF_PREFIX}/geno-lewm-v03-cache-h200-proof-{commit[:12]}-r{attempt}/success"


def expected_receipt_namespace(proof_namespace: str) -> str:
    """Return the immutable terminal-receipt sibling of a proof success path."""
    path = _safe_namespace(proof_namespace, label="proof namespace")
    if not path.endswith("/success"):
        raise InputError("proof namespace must end in /success")
    return f"{path.removesuffix('/success')}/terminal-job-receipt"


def capture_terminal_receipt(
    *,
    output_dir: Path,
    proof_download_dir: Path,
    job_id: str,
    source_commit: str,
    run_attempt: int,
    proof_revision: str,
    proof_namespace: str,
    client: HubReadClient,
    proof_verifier: ProofVerifier | None = None,
) -> dict[str, object]:
    """Inspect COMPLETED JobInfo, replay the exact proof revision, and close a receipt."""
    commit = _exact_commit(source_commit, label="source commit")
    attempt = _positive_integer(run_attempt, label="run attempt")
    revision = _exact_commit(proof_revision, label="proof revision")
    job_id = _job_id(job_id)
    expected_proof = expected_proof_namespace(commit, attempt)
    if _safe_namespace(proof_namespace, label="proof namespace") != expected_proof:
        raise InputError("proof namespace differs from the exact source/run contract")
    _require_absent(output_dir, label="receipt output directory")
    _require_absent(proof_download_dir, label="proof download directory")

    job = client.inspect_job(job_id=job_id, namespace=launch.NAMESPACE)
    spec = launch.build_launch_spec(source_commit=commit, run_attempt=attempt)
    _validate_terminal_job(job, expected_job_id=job_id, expected=spec)
    downloaded_proof = client.download_exact_namespace(
        repository=PROOF_REPOSITORY,
        repo_type=PROOF_REPO_TYPE,
        revision=revision,
        namespace=expected_proof,
        destination=proof_download_dir,
        token=False,
    )
    verifier = proof_verifier or _default_proof_verifier
    proof_report = verifier(downloaded_proof)
    downloaded_report = _json_object(
        _read_regular_bytes(
            downloaded_proof / "proof" / "cache-h200-proof.json",
            label="downloaded cache proof report",
        ),
        label="downloaded cache proof report",
    )
    if downloaded_report != proof_report:
        raise InputError("downloaded cache proof report bytes differ from the replay result")
    proof_runtime_hash = _validate_proof_report(
        proof_report,
        source_commit=commit,
        expected=spec,
    )
    payload = _derive_receipt(
        job=job,
        expected=spec,
        source_commit=commit,
        run_attempt=attempt,
        proof_revision=revision,
        proof_namespace=expected_proof,
        proof_runtime_hash=proof_runtime_hash,
        proof_dir=downloaded_proof,
    )
    _write_bundle_once(output_dir, payload)
    replayed = verify_existing_receipt(output_dir)
    if replayed != payload:
        raise InputError("terminal JobInfo receipt changed across local replay")
    return payload


def verify_existing_receipt(bundle_dir: Path) -> dict[str, object]:
    """Read-only replay of a closed terminal JobInfo receipt bundle."""
    bundle = _require_directory(bundle_dir, label="terminal receipt bundle")
    expected_inventory = {SCHEMA_NAME, RECEIPT_NAME, CHECKSUMS_NAME}
    inventory = _regular_inventory(bundle)
    if inventory != expected_inventory:
        raise InputError(
            "terminal receipt bundle inventory is not exact",
            details={"expected": sorted(expected_inventory), "observed": sorted(inventory)},
        )
    _verify_checksums(bundle)
    bundled_schema = _read_regular_bytes(bundle / SCHEMA_NAME, label="bundled receipt schema")
    committed_schema = _read_regular_bytes(
        DEFAULT_SCHEMA_PATH,
        label="committed receipt schema",
    )
    if bundled_schema != committed_schema:
        raise InputError("bundled terminal receipt schema differs from the committed schema")
    schema = _schema_from_bytes(bundled_schema)
    receipt_body = _read_regular_bytes(bundle / RECEIPT_NAME, label="terminal receipt")
    payload = _json_object(receipt_body, label="terminal receipt")
    if receipt_body != canonical_json_bytes(payload):
        raise InputError("terminal receipt JSON is not canonical")
    _validate_payload(payload, schema=schema)
    _validate_receipt_semantics(payload)
    return payload


def verify_remote_receipt(
    *,
    receipt_revision: str,
    receipt_namespace: str,
    download_root: Path,
    client: HubReadClient,
    proof_verifier: ProofVerifier | None = None,
) -> dict[str, object]:
    """Exact-download a receipt, replay it, then re-download and replay its proof."""
    revision = _exact_commit(receipt_revision, label="receipt revision")
    namespace = _safe_namespace(receipt_namespace, label="receipt namespace")
    _require_absent(download_root, label="remote receipt download root")
    receipt_dir = client.download_exact_namespace(
        repository=PROOF_REPOSITORY,
        repo_type=PROOF_REPO_TYPE,
        revision=revision,
        namespace=namespace,
        destination=download_root / "receipt",
        token=False,
    )
    payload = verify_existing_receipt(receipt_dir)
    publication = _mapping(payload.get("publication"), label="receipt publication")
    if (
        publication.get("repository") != PROOF_REPOSITORY
        or publication.get("namespace") != namespace
    ):
        raise InputError("downloaded receipt namespace differs from its publication binding")

    proof = _mapping(payload.get("proof"), label="receipt proof")
    proof_revision = _exact_commit(proof.get("revision"), label="receipt proof revision")
    proof_namespace = _safe_namespace(proof.get("namespace"), label="receipt proof namespace")
    proof_dir = client.download_exact_namespace(
        repository=PROOF_REPOSITORY,
        repo_type=PROOF_REPO_TYPE,
        revision=proof_revision,
        namespace=proof_namespace,
        destination=download_root / "proof",
        token=False,
    )
    verifier = proof_verifier or _default_proof_verifier
    proof_report = verifier(proof_dir)
    downloaded_report_body = _read_regular_bytes(
        proof_dir / "proof" / "cache-h200-proof.json",
        label="remotely replayed cache proof report",
    )
    downloaded_report = _json_object(
        downloaded_report_body,
        label="remotely replayed cache proof report",
    )
    if downloaded_report != proof_report:
        raise InputError("remotely downloaded cache proof report bytes differ from replay")
    source = _mapping(payload.get("source"), label="receipt source")
    source_commit = _exact_commit(source.get("commit"), label="receipt source commit")
    run_attempt = _positive_integer(source.get("run_attempt"), label="receipt run attempt")
    expected = launch.build_launch_spec(source_commit=source_commit, run_attempt=run_attempt)
    proof_runtime_hash = _validate_proof_report(
        proof_report,
        source_commit=source_commit,
        expected=expected,
    )
    if proof.get("runtime_hash") != proof_runtime_hash:
        raise InputError("remotely replayed cache proof runtime hash differs from the receipt")
    if proof.get("checksums") != _file_identity(
        proof_dir / CHECKSUMS_NAME,
        CHECKSUMS_NAME,
    ) or proof.get("report") != _file_identity(
        proof_dir / "proof" / "cache-h200-proof.json",
        "proof/cache-h200-proof.json",
    ):
        raise InputError("remotely replayed cache proof identities differ from the receipt")
    return payload


def _derive_receipt(
    *,
    job: Any,
    expected: launch.LaunchSpec,
    source_commit: str,
    run_attempt: int,
    proof_revision: str,
    proof_namespace: str,
    proof_runtime_hash: str,
    proof_dir: Path,
) -> dict[str, object]:
    receipt_namespace = expected_receipt_namespace(proof_namespace)
    return {
        "$schema": f"./{SCHEMA_NAME}",
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "ok": True,
        "source": {
            "repository": _CANONICAL_ORIGIN,
            "commit": source_commit,
            "run_attempt": run_attempt,
        },
        "job": {
            "id": cast(str, job.id),
            "url": cast(str, job.url),
            "namespace": expected.namespace,
            "status": {"stage": "COMPLETED"},
            "image": expected.image,
            "space_id": None,
            "command": list(expected.command),
            "arguments": [],
            "environment": dict(expected.environment),
            "flavor": expected.flavor,
            "labels": dict(expected.labels),
            "secret_names": list(expected.secret_names),
            "volumes": [asdict(expected.volume)],
        },
        "trace": {
            "repository": launch.TRACE_REPOSITORY,
            "revision": launch.TRACE_REVISION,
            "artifact_path": launch.TRACE_ARTIFACT_PATH,
        },
        "proof": {
            "repository": PROOF_REPOSITORY,
            "revision": proof_revision,
            "namespace": proof_namespace,
            "checksums": _file_identity(proof_dir / CHECKSUMS_NAME, CHECKSUMS_NAME),
            "report": _file_identity(
                proof_dir / "proof" / "cache-h200-proof.json",
                "proof/cache-h200-proof.json",
            ),
            "runtime_hash": proof_runtime_hash,
            "exact_revision_download_replayed": True,
        },
        "publication": {
            "repository": PROOF_REPOSITORY,
            "namespace": receipt_namespace,
            "sibling_of_proof_namespace": True,
        },
        "claim_boundary": {
            "jobinfo_terminal_status_attested": True,
            "proof_exact_revision_replayed": True,
            "requested_submission_timeout": launch.TIMEOUT,
            "timeout_server_echo_attested": False,
            "statement": (
                "This host-side receipt binds the COMPLETED JobInfo fields and an exact-revision "
                "replay of the separately published cache proof. Hugging Face JobInfo does not "
                "echo the submitted timeout, so the requested eight-hour timeout is recorded "
                "without a server-echo attestation."
            ),
        },
    }


def _validate_terminal_job(
    job: Any,
    *,
    expected_job_id: str,
    expected: launch.LaunchSpec,
) -> None:
    try:
        launch.require_exact_job_contract(job, expected=expected)
    except RuntimeError as exc:
        raise InputError("terminal JobInfo differs from the accepted launch contract") from exc
    observed_id = getattr(job, "id", None)
    if observed_id != expected_job_id:
        raise InputError("terminal JobInfo id differs from the requested job id")
    expected_url = f"https://huggingface.co/jobs/{expected.namespace}/{expected_job_id}"
    if getattr(job, "url", None) != expected_url:
        raise InputError("terminal JobInfo URL differs from its namespace and id")
    status = getattr(job, "status", None)
    stage = getattr(status, "stage", None)
    stage_value = getattr(stage, "value", stage)
    if stage_value != "COMPLETED":
        raise InputError("terminal JobInfo status must be COMPLETED")


def _validate_proof_report(
    report: Mapping[str, object],
    *,
    source_commit: str,
    expected: launch.LaunchSpec,
) -> str:
    producer = _mapping(report.get("producer"), label="cache proof producer")
    trace = _mapping(report.get("trace"), label="cache proof trace")
    claim = _mapping(report.get("claim_boundary"), label="cache proof claim boundary")
    runtime = _mapping(report.get("runtime"), label="cache proof runtime")
    if (
        report.get("ok") is not True
        or producer.get("git_commit") != source_commit
        or producer.get("origin") != _CANONICAL_ORIGIN
        or producer.get("declared_container_image") != expected.image
        or trace.get("repository") != launch.TRACE_REPOSITORY
        or trace.get("revision") != launch.TRACE_REVISION
        or trace.get("artifact_path") != launch.TRACE_ARTIFACT_PATH
        or claim.get("hf_job_terminal_status_attested") is not False
    ):
        raise InputError("replayed cache proof does not bind the exact launch and trace contract")
    return _sha256_digest(runtime.get("runtime_hash"), label="cache proof runtime hash")


def _validate_receipt_semantics(payload: Mapping[str, object]) -> None:
    source = _mapping(payload.get("source"), label="receipt source")
    commit = _exact_commit(source.get("commit"), label="receipt source commit")
    attempt = _positive_integer(source.get("run_attempt"), label="receipt run attempt")
    expected = launch.build_launch_spec(source_commit=commit, run_attempt=attempt)
    proof = _mapping(payload.get("proof"), label="receipt proof")
    proof_namespace = expected_proof_namespace(commit, attempt)
    publication = _mapping(payload.get("publication"), label="receipt publication")
    job = _mapping(payload.get("job"), label="receipt job")
    expected_job_id = _job_id(job.get("id"))
    expected_job_url = f"https://huggingface.co/jobs/{expected.namespace}/{expected_job_id}"
    wanted_job: dict[str, object] = {
        "id": expected_job_id,
        "url": expected_job_url,
        "namespace": expected.namespace,
        "status": {"stage": "COMPLETED"},
        "image": expected.image,
        "space_id": None,
        "command": list(expected.command),
        "arguments": [],
        "environment": dict(expected.environment),
        "flavor": expected.flavor,
        "labels": dict(expected.labels),
        "secret_names": list(expected.secret_names),
        "volumes": [asdict(expected.volume)],
    }
    if dict(job) != wanted_job:
        raise InputError("terminal receipt JobInfo fields do not match the exact launch contract")
    if source.get("repository") != _CANONICAL_ORIGIN:
        raise InputError("terminal receipt source repository is not canonical")
    if (
        proof.get("repository") != PROOF_REPOSITORY
        or proof.get("namespace") != proof_namespace
        or proof.get("exact_revision_download_replayed") is not True
    ):
        raise InputError("terminal receipt proof binding is not the exact source/run namespace")
    _exact_commit(proof.get("revision"), label="receipt proof revision")
    _sha256_digest(proof.get("runtime_hash"), label="receipt proof runtime hash")
    _artifact_identity(proof.get("checksums"), expected_path=CHECKSUMS_NAME)
    _artifact_identity(
        proof.get("report"),
        expected_path="proof/cache-h200-proof.json",
    )
    if publication != {
        "repository": PROOF_REPOSITORY,
        "namespace": expected_receipt_namespace(proof_namespace),
        "sibling_of_proof_namespace": True,
    }:
        raise InputError("terminal receipt publication namespace is not the proof sibling")
    if payload.get("trace") != {
        "repository": launch.TRACE_REPOSITORY,
        "revision": launch.TRACE_REVISION,
        "artifact_path": launch.TRACE_ARTIFACT_PATH,
    }:
        raise InputError("terminal receipt trace binding drifted")


def _write_bundle_once(output_dir: Path, payload: Mapping[str, object]) -> None:
    _require_absent(output_dir, label="receipt output directory")
    parent = _require_directory(output_dir.parent, label="receipt output parent")
    if parent != output_dir.parent:
        raise InputError("receipt output parent did not resolve exactly")
    output_dir.mkdir(mode=0o700)
    try:
        schema_body = _read_regular_bytes(DEFAULT_SCHEMA_PATH, label="committed receipt schema")
        _schema_from_bytes(schema_body)
        _write_new(output_dir / SCHEMA_NAME, schema_body)
        _write_new(output_dir / RECEIPT_NAME, canonical_json_bytes(payload))
        checksums = "".join(
            f"{_sha256_hex(_read_regular_bytes(output_dir / name, label=name))}  {name}\n"
            for name in sorted((RECEIPT_NAME, SCHEMA_NAME))
        ).encode("ascii")
        _write_new(output_dir / CHECKSUMS_NAME, checksums)
    except BaseException:
        shutil.rmtree(output_dir)
        raise


def _verify_checksums(bundle: Path) -> None:
    body = _read_regular_bytes(bundle / CHECKSUMS_NAME, label="receipt checksums")
    try:
        text = body.decode("ascii")
    except UnicodeDecodeError as exc:
        raise InputError("terminal receipt checksums are not ASCII") from exc
    expected_names = {RECEIPT_NAME, SCHEMA_NAME}
    observed: dict[str, str] = {}
    for line in text.splitlines():
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None or match.group(2) in observed:
            raise InputError("terminal receipt checksum manifest is malformed")
        observed[match.group(2)] = match.group(1)
    if set(observed) != expected_names or not text.endswith("\n"):
        raise InputError("terminal receipt checksum manifest inventory is not exact")
    canonical = "".join(f"{observed[name]}  {name}\n" for name in sorted(observed))
    if text != canonical:
        raise InputError("terminal receipt checksum manifest is not canonical")
    for name, digest in observed.items():
        actual = _sha256_hex(_read_regular_bytes(bundle / name, label=f"receipt artifact {name}"))
        if actual != digest:
            raise InputError(f"terminal receipt checksum mismatch for {name}")


def _schema_from_bytes(body: bytes) -> Mapping[str, object]:
    schema = _json_object(body, label="terminal receipt schema")
    jsonschema = _jsonschema_module()
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise InputError("terminal receipt schema is invalid") from exc
    return schema


def _validate_payload(payload: Mapping[str, object], *, schema: Mapping[str, object]) -> None:
    jsonschema = _jsonschema_module()
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        raise InputError(
            "terminal receipt does not satisfy its schema",
            details={"path": list(errors[0].absolute_path), "message": errors[0].message},
        )


def _jsonschema_module() -> Any:
    try:
        return importlib.import_module("jsonschema")
    except ImportError as exc:
        raise ResourceError("jsonschema is required to validate terminal receipts") from exc


def _default_proof_verifier(path: Path) -> dict[str, object]:
    from tools.research.v03_cache_h200_proof import verify_existing_bundle

    return verify_existing_bundle(bundle_dir=path)


def _file_identity(path: Path, relative: str) -> dict[str, object]:
    body = _read_regular_bytes(path, label=f"proof artifact {relative}")
    return {"path": relative, "sha256": f"sha256:{_sha256_hex(body)}", "size_bytes": len(body)}


def _artifact_identity(value: object, *, expected_path: str) -> Mapping[str, object]:
    identity = _mapping(value, label=f"artifact identity {expected_path}")
    if set(identity) != {"path", "sha256", "size_bytes"} or identity.get("path") != expected_path:
        raise InputError(f"artifact identity for {expected_path} is not closed")
    digest = identity.get("sha256")
    size = identity.get("size_bytes")
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
    ):
        raise InputError(f"artifact identity for {expected_path} is invalid")
    return identity


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise InputError(f"{label} is missing") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise InputError(f"{label} must be a regular non-symlink file")
    try:
        body = path.read_bytes()
    except OSError as exc:
        raise InputError(f"{label} could not be read") from exc
    rebound = path.lstat()
    if (metadata.st_dev, metadata.st_ino, metadata.st_size) != (
        rebound.st_dev,
        rebound.st_ino,
        rebound.st_size,
    ):
        raise InputError(f"{label} changed while it was read")
    return body


def _write_new(path: Path, body: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(body)
    except OSError as exc:
        raise InputError(f"receipt artifact could not be installed: {path.name}") from exc


def _regular_inventory(root: Path) -> set[str]:
    inventory: set[str] = set()
    for path in root.iterdir():
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise InputError("terminal receipt bundle contains a non-regular artifact")
        inventory.add(path.name)
    return inventory


def _require_directory(path: Path, *, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise InputError(f"{label} is missing") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise InputError(f"{label} must be a non-symlink directory")
    return path


def _require_absent(path: Path, *, label: str) -> None:
    if os.path.lexists(path):
        raise InputError(f"{label} must not already exist")


def _json_object(body: bytes, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise InputError(f"{label} must be a JSON object")
    return cast(dict[str, object], payload)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise InputError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _safe_namespace(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip("/"):
        raise InputError(f"{label} must be a non-empty relative path without outer slashes")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise InputError(f"{label} contains an unsafe path component")
    if path.as_posix() != value:
        raise InputError(f"{label} is not canonical")
    return value


def _exact_commit(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise InputError(f"{label} must be a full lowercase 40-character SHA")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputError(f"{label} must be a positive integer")
    return value


def _job_id(value: object) -> str:
    if not isinstance(value, str) or _JOB_ID.fullmatch(value) is None:
        raise InputError("job id is not a safe canonical identifier")
    return value


def _sha256_hex(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _sha256_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise InputError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_argument(value: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("must be a positive canonical integer")
    return int(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    author = subparsers.add_parser(
        "author",
        help="inspect COMPLETED JobInfo and author a closed local receipt bundle",
    )
    author.add_argument("--output-dir", required=True, type=Path)
    author.add_argument("--proof-download-dir", required=True, type=Path)
    author.add_argument("--job-id", required=True)
    author.add_argument("--source-commit", required=True)
    author.add_argument("--run-attempt", required=True, type=_positive_argument)
    author.add_argument("--proof-revision", required=True)
    author.add_argument("--proof-namespace", required=True)

    verify = subparsers.add_parser(
        "verify-existing",
        help="read-only replay of one local closed receipt bundle",
    )
    verify.add_argument("--bundle-dir", required=True, type=Path)

    remote = subparsers.add_parser(
        "verify-remote",
        help="exact-download and replay a receipt plus its exact bound proof",
    )
    remote.add_argument("--receipt-revision", required=True)
    remote.add_argument("--receipt-namespace", required=True)
    remote.add_argument("--download-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the host-side terminal-receipt protocol without publishing anything."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "author":
            token = os.environ.get("HF_TOKEN", "")
            if not token:
                raise ResourceError("HF_TOKEN is required to inspect terminal JobInfo")
            payload = capture_terminal_receipt(
                output_dir=args.output_dir,
                proof_download_dir=args.proof_download_dir,
                job_id=args.job_id,
                source_commit=args.source_commit,
                run_attempt=args.run_attempt,
                proof_revision=args.proof_revision,
                proof_namespace=args.proof_namespace,
                client=HuggingFaceReadClient(token=token),
            )
        elif args.command == "verify-existing":
            payload = verify_existing_receipt(args.bundle_dir)
        else:
            payload = verify_remote_receipt(
                receipt_revision=args.receipt_revision,
                receipt_namespace=args.receipt_namespace,
                download_root=args.download_root,
                client=HuggingFaceReadClient(token=False),
            )
    except (InputError, ResourceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exit_code_for(exc)
    if payload.get("ok") is not True:
        raise AssertionError("terminal receipt command returned without a verified payload")
    print("GENO_LEWM_V03_CACHE_H200_JOB_RECEIPT_OK")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through host-side invocation.
    raise SystemExit(main())
