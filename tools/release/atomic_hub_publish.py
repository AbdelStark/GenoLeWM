# SPDX-License-Identifier: Apache-2.0
"""Conditionally publish and verify one immutable Hub success namespace."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol

from geno_lewm.errors import GenoLeWMError, InputError, ResourceError, exit_code_for
from tools.release.training_reproducibility import (
    GENERATED_BY as REPRODUCIBILITY_REPORT_GENERATED_BY,
    SCHEMA_VERSION as REPRODUCIBILITY_REPORT_SCHEMA_VERSION,
)
from tools.research.training_reproducibility_preflight import (
    EXPECTED_CARBON_RUNTIME_HASH,
    GENERATED_BY as JOB_PREFLIGHT_GENERATED_BY,
    SCHEMA_VERSION as JOB_PREFLIGHT_SCHEMA_VERSION,
)

SCHEMA_VERSION: Final = "1.0.0"
COMPLETION_SCHEMA_VERSION: Final = "2.0.0"
GENERATED_BY: Final = "tools.release.atomic_hub_publish"
CHECKSUMS_NAME: Final = "SHA256SUMS"
COMPLETION_NAME: Final = "completion.json"
REPORT_PATH: Final = "evidence/training_reproducibility_report.json"
JOB_PREFLIGHT_PATH: Final = "evidence/job_contract_preflight.json"
RUNTIME_PREFLIGHT_PATH: Final = "evidence/runtime_preflight.json"
RUNTIME_PREFLIGHT_SCHEMA_VERSION: Final = "1.0.0"
RUNTIME_PREFLIGHT_GENERATED_BY: Final = "tools.jobs.training_reproducibility_run"
EXPECTED_RUN_LABELS: Final = (
    "baseline_a",
    "deterministic_a",
    "deterministic_b",
    "baseline_b",
)
DEFAULT_MAX_ATTEMPTS: Final = 3
_REPO_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_CHECKSUM_RE: Final = re.compile(r"^([0-9a-f]{64})  (.+)$")


class StaleParentError(RuntimeError):
    """The Hub rejected a conditional commit because its parent moved."""


@dataclass(frozen=True, slots=True)
class HubParent:
    """One coherent view of a Hub branch head and its files."""

    commit_sha: str
    files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublishFile:
    """One local file included in the single Hub commit."""

    relative_path: str
    source_path: Path
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class AtomicPublishResult:
    """Receipt for a conditionally committed and independently verified bundle."""

    schema_version: str
    generated_by: str
    repo_id: str
    repo_type: str
    namespace: str
    source_commit_sha: str
    hub_parent_commit_sha: str
    hub_commit_sha: str
    attempts: int
    verified: bool
    files: tuple[PublishFile, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "namespace": self.namespace,
            "source_commit_sha": self.source_commit_sha,
            "hub_parent_commit_sha": self.hub_parent_commit_sha,
            "hub_commit_sha": self.hub_commit_sha,
            "attempts": self.attempts,
            "verified": self.verified,
            "files": [item.to_dict() for item in self.files],
        }


class HubClient(Protocol):
    """Minimal external boundary required by the atomic publication protocol."""

    def read_parent(self, *, repo_id: str, repo_type: str) -> HubParent: ...

    def create_commit(
        self,
        *,
        repo_id: str,
        repo_type: str,
        namespace: str,
        files: tuple[PublishFile, ...],
        parent_commit: str,
        commit_message: str,
    ) -> str: ...

    def download_namespace(
        self,
        *,
        repo_id: str,
        repo_type: str,
        namespace: str,
        revision: str,
        destination: Path,
    ) -> Path: ...


def publish_success_namespace(
    *,
    bundle_dir: Path,
    repo_id: str,
    run_name: str,
    source_commit_sha: str,
    verification_dir: Path,
    client: HubClient,
    repo_type: str = "model",
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    generated_at: str | None = None,
) -> AtomicPublishResult:
    """Publish a success bundle in one parent-conditional commit and verify it."""
    repo_id = _validate_repo_id(repo_id)
    run_name = _validate_relative_path("run_name", run_name)
    source_commit_sha = _validate_commit("source_commit_sha", source_commit_sha)
    max_attempts = _validate_max_attempts(max_attempts)
    namespace = f"{run_name}/success"
    if verification_dir.exists():
        raise InputError(
            "verification_dir must not exist before immutable download verification",
            details={"verification_dir": str(verification_dir)},
        )

    _write_completion_marker(
        bundle_dir=bundle_dir,
        repo_id=repo_id,
        source_commit_sha=source_commit_sha,
        run_name=run_name,
        generated_at=generated_at or _utc_now(),
    )
    files = _verify_bundle(
        bundle_dir,
        expected_repo_id=repo_id,
        expected_run_name=run_name,
        expected_source_commit=source_commit_sha,
    )
    expected_hashes = {item.relative_path: item.sha256 for item in files}

    winning_parent: str | None = None
    hub_commit_sha: str | None = None
    attempts = 0
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        parent = client.read_parent(repo_id=repo_id, repo_type=repo_type)
        parent_commit = _validate_commit("Hub parent commit", parent.commit_sha)
        if _namespace_present(namespace, parent.files):
            raise ResourceError(
                "immutable Hub success namespace already exists",
                details={
                    "repo_id": repo_id,
                    "namespace": namespace,
                    "parent_commit": parent_commit,
                },
            )
        try:
            observed_commit = client.create_commit(
                repo_id=repo_id,
                repo_type=repo_type,
                namespace=namespace,
                files=files,
                parent_commit=parent_commit,
                commit_message=f"Publish immutable training evidence for {run_name}",
            )
        except StaleParentError as exc:
            if attempt == max_attempts:
                raise ResourceError(
                    "conditional Hub commit exhausted the stale-parent retry budget",
                    details={
                        "repo_id": repo_id,
                        "namespace": namespace,
                        "max_attempts": max_attempts,
                    },
                ) from exc
            continue
        winning_parent = parent_commit
        hub_commit_sha = _validate_commit("Hub result commit", observed_commit)
        break

    if winning_parent is None or hub_commit_sha is None:  # pragma: no cover - loop invariant
        raise ResourceError("conditional Hub publication did not produce a commit")
    downloaded = client.download_namespace(
        repo_id=repo_id,
        repo_type=repo_type,
        namespace=namespace,
        revision=hub_commit_sha,
        destination=verification_dir,
    )
    try:
        downloaded_files = _verify_bundle(
            downloaded,
            expected_repo_id=repo_id,
            expected_run_name=run_name,
            expected_source_commit=source_commit_sha,
        )
    except InputError as exc:
        raise InputError(
            "immutable downloaded namespace failed checksum or marker verification",
            details={"hub_commit_sha": hub_commit_sha, "cause": exc.to_dict()},
        ) from exc
    observed_hashes = {item.relative_path: item.sha256 for item in downloaded_files}
    if observed_hashes != expected_hashes:
        raise InputError(
            "immutable downloaded namespace does not match the committed local bundle",
            details={"expected": expected_hashes, "observed": observed_hashes},
        )
    return AtomicPublishResult(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        repo_id=repo_id,
        repo_type=repo_type,
        namespace=namespace,
        source_commit_sha=source_commit_sha,
        hub_parent_commit_sha=winning_parent,
        hub_commit_sha=hub_commit_sha,
        attempts=attempts,
        verified=True,
        files=files,
    )


class HuggingFaceHubClient:
    """Lazy adapter around the optional ``huggingface_hub`` dependency."""

    def __init__(self) -> None:
        try:
            hub = importlib.import_module("huggingface_hub")
            hub_utils = importlib.import_module("huggingface_hub.utils")
            self._api: Any = hub.HfApi(token=True)
            self._operation_add: Any = hub.CommitOperationAdd
            self._snapshot_download: Any = hub.snapshot_download
            self._http_error_type: type[BaseException] = hub_utils.HfHubHTTPError
        except (AttributeError, ImportError) as exc:
            raise ResourceError(
                "huggingface_hub with HfApi.create_commit is required for publication"
            ) from exc

    def read_parent(self, *, repo_id: str, repo_type: str) -> HubParent:
        try:
            info = self._api.repo_info(
                repo_id=repo_id,
                repo_type=repo_type,
                revision="main",
                files_metadata=False,
                token=True,
            )
            commit_sha = getattr(info, "sha", None)
            siblings = getattr(info, "siblings", None)
            if not isinstance(commit_sha, str) or not isinstance(siblings, list | tuple):
                raise ResourceError(
                    "Hub repo_info did not return a coherent parent and file listing",
                    details={"repo_id": repo_id},
                )
            files: list[str] = []
            for sibling in siblings:
                path = getattr(sibling, "rfilename", None)
                if not isinstance(path, str):
                    raise ResourceError(
                        "Hub repo_info returned a file without a relative path",
                        details={"repo_id": repo_id},
                    )
                files.append(path)
            return HubParent(commit_sha=commit_sha, files=tuple(sorted(files)))
        except GenoLeWMError:
            raise
        except Exception as exc:
            raise ResourceError(
                "failed to fetch the current Hub parent and namespace listing",
                details={"repo_id": repo_id},
            ) from exc

    def create_commit(
        self,
        *,
        repo_id: str,
        repo_type: str,
        namespace: str,
        files: tuple[PublishFile, ...],
        parent_commit: str,
        commit_message: str,
    ) -> str:
        operations = [
            self._operation_add(
                path_in_repo=f"{namespace}/{item.relative_path}",
                path_or_fileobj=str(item.source_path),
            )
            for item in files
        ]
        try:
            result = self._api.create_commit(
                repo_id=repo_id,
                repo_type=repo_type,
                revision="main",
                create_pr=False,
                operations=operations,
                commit_message=commit_message,
                parent_commit=parent_commit,
                token=True,
            )
        except Exception as exc:
            if isinstance(exc, self._http_error_type) and _http_status(exc) == 412:
                raise StaleParentError(parent_commit) from exc
            raise ResourceError(
                "conditional Hub commit failed",
                details={"repo_id": repo_id, "parent_commit": parent_commit},
            ) from exc
        commit_sha = getattr(result, "oid", None)
        if not isinstance(commit_sha, str):
            raise ResourceError(
                "Hub create_commit response did not contain a commit SHA",
                details={"repo_id": repo_id},
            )
        return commit_sha

    def download_namespace(
        self,
        *,
        repo_id: str,
        repo_type: str,
        namespace: str,
        revision: str,
        destination: Path,
    ) -> Path:
        try:
            root = Path(
                self._snapshot_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                    allow_patterns=[f"{namespace}/**"],
                    local_dir=str(destination),
                    force_download=True,
                    token=True,
                )
            )
        except Exception as exc:
            raise ResourceError(
                "failed to download the immutable Hub success commit",
                details={"repo_id": repo_id, "revision": revision},
            ) from exc
        downloaded = root / namespace
        if not downloaded.is_dir():
            raise ResourceError(
                "immutable Hub commit did not contain the published success namespace",
                details={
                    "repo_id": repo_id,
                    "revision": revision,
                    "namespace": namespace,
                },
            )
        return downloaded


def main(argv: list[str] | None = None) -> int:
    """Run the atomic Hub publication protocol."""
    args = _parser().parse_args(argv)
    try:
        result = publish_success_namespace(
            bundle_dir=args.bundle_dir,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            run_name=args.run_name,
            source_commit_sha=args.source_commit_sha,
            verification_dir=args.verification_dir,
            max_attempts=args.max_attempts,
            client=_load_default_client(),
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    return 0


def _load_default_client() -> HubClient:
    return HuggingFaceHubClient()


def _write_completion_marker(
    *,
    bundle_dir: Path,
    repo_id: str,
    source_commit_sha: str,
    run_name: str,
    generated_at: str,
) -> None:
    checksums_path = bundle_dir / CHECKSUMS_NAME
    report_path = bundle_dir / REPORT_PATH
    _verify_checksum_file(bundle_dir)
    report = _load_json_object(report_path, label="training reproducibility report")
    _verify_evidence_identity(
        bundle_dir,
        report=report,
        expected_repo_id=repo_id,
        expected_run_name=run_name,
        expected_source_commit=source_commit_sha,
    )
    payload = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": generated_at,
        "ok": True,
        "commit_sha": source_commit_sha,
        "source_commit_sha": source_commit_sha,
        "run_name": run_name,
        "checksums": {
            "path": CHECKSUMS_NAME,
            "sha256": _sha256_uri(checksums_path),
        },
        "report": {
            "path": REPORT_PATH,
            "sha256": _sha256_uri(report_path),
        },
        "claim_boundary": (
            "This marker proves only the predeclared issue #47 H200 N-D-D-N engineering "
            "contract. It does not establish model quality or scientific validity."
        ),
    }
    (bundle_dir / COMPLETION_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_bundle(
    root: Path,
    *,
    expected_repo_id: str,
    expected_run_name: str,
    expected_source_commit: str,
) -> tuple[PublishFile, ...]:
    _verify_checksum_file(root)
    marker_path = root / COMPLETION_NAME
    marker = _load_json_object(marker_path, label="completion marker")
    if marker.get("ok") is not True:
        raise InputError(
            "completion marker must record ok=true", details={"path": str(marker_path)}
        )
    _expect_marker_value(marker, "run_name", expected_run_name, marker_path)
    _expect_marker_value(marker, "commit_sha", expected_source_commit, marker_path)
    _expect_marker_value(marker, "source_commit_sha", expected_source_commit, marker_path)
    _expect_marker_value(marker, "schema_version", COMPLETION_SCHEMA_VERSION, marker_path)
    _expect_marker_value(marker, "generated_by", GENERATED_BY, marker_path)
    _verify_marker_binding(marker, "checksums", CHECKSUMS_NAME, root / CHECKSUMS_NAME)
    _verify_marker_binding(marker, "report", REPORT_PATH, root / REPORT_PATH)
    report = _load_json_object(root / REPORT_PATH, label="training reproducibility report")
    _verify_evidence_identity(
        root,
        report=report,
        expected_repo_id=expected_repo_id,
        expected_run_name=expected_run_name,
        expected_source_commit=expected_source_commit,
    )
    return _collect_bundle_files(root)


def _verify_evidence_identity(
    root: Path,
    *,
    report: dict[str, Any],
    expected_repo_id: str,
    expected_run_name: str,
    expected_source_commit: str,
) -> None:
    """Bind every success-producing artifact to one immutable launch identity."""
    report_path = root / REPORT_PATH
    _expect_evidence_value(
        report,
        ("schema_version",),
        REPRODUCIBILITY_REPORT_SCHEMA_VERSION,
        report_path,
    )
    _expect_evidence_value(
        report,
        ("generated_by",),
        REPRODUCIBILITY_REPORT_GENERATED_BY,
        report_path,
    )
    for keys, expected in (
        (("ok",), True),
        (("run_contract", "ok"), True),
        (("deterministic_pair", "ok"), True),
        (("throughput", "ok"), True),
        (("throughput", "status"), "pass"),
    ):
        _expect_evidence_value(report, keys, expected, report_path)

    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != len(EXPECTED_RUN_LABELS):
        raise InputError(
            "training reproducibility report must bind exactly four ordered runs",
            details={"path": str(report_path)},
        )
    for index, (run, expected_label) in enumerate(zip(runs, EXPECTED_RUN_LABELS, strict=True)):
        if not isinstance(run, dict):
            raise InputError(
                "training reproducibility report contains an invalid run entry",
                details={"path": str(report_path), "index": index},
            )
        _expect_evidence_value(run, ("label",), expected_label, report_path)
        _expect_evidence_value(
            run,
            ("commit_sha",),
            expected_source_commit,
            report_path,
        )

    job_preflight_path = root / JOB_PREFLIGHT_PATH
    job_preflight = _load_json_object(job_preflight_path, label="job contract preflight")
    for keys, expected in (
        (("schema_version",), JOB_PREFLIGHT_SCHEMA_VERSION),
        (("generated_by",), JOB_PREFLIGHT_GENERATED_BY),
        (("ok",), True),
        (("repository", "expected_commit_sha"), expected_source_commit),
        (("repository", "observed_commit_sha"), expected_source_commit),
        (("repository", "worktree_clean"), True),
        (("job", "run_name"), expected_run_name),
        (("job", "upload_repo"), expected_repo_id),
        (("job", "expected_carbon_runtime_hash"), EXPECTED_CARBON_RUNTIME_HASH),
    ):
        _expect_evidence_value(job_preflight, keys, expected, job_preflight_path)

    runtime_preflight_path = root / RUNTIME_PREFLIGHT_PATH
    runtime_preflight = _load_json_object(
        runtime_preflight_path,
        label="runtime preflight",
    )
    for keys, expected in (
        (("schema_version",), RUNTIME_PREFLIGHT_SCHEMA_VERSION),
        (("generated_by",), RUNTIME_PREFLIGHT_GENERATED_BY),
        (("ok",), True),
        (("source_commit_sha",), expected_source_commit),
        (("run_name",), expected_run_name),
        (("carbon_runtime_hash",), EXPECTED_CARBON_RUNTIME_HASH),
        (("expected_carbon_runtime_hash",), EXPECTED_CARBON_RUNTIME_HASH),
    ):
        _expect_evidence_value(runtime_preflight, keys, expected, runtime_preflight_path)
    device_name = _nested_value(runtime_preflight, "accelerator", "device_name")
    if not isinstance(device_name, str) or "H200" not in device_name:
        raise InputError(
            "runtime preflight is not bound to an H200 accelerator",
            details={"path": str(runtime_preflight_path), "observed": device_name},
        )


def _expect_evidence_value(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    expected: object,
    path: Path,
) -> None:
    observed = _nested_value(payload, *keys)
    if observed != expected or type(observed) is not type(expected):
        raise InputError(
            "success evidence identity does not match the publication request",
            details={
                "path": str(path),
                "field": ".".join(keys),
                "expected": expected,
                "observed": observed,
            },
        )


def _verify_checksum_file(root: Path) -> None:
    checksum_path = root / CHECKSUMS_NAME
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise InputError(
            "success bundle SHA256SUMS is missing or unreadable",
            details={"path": str(checksum_path)},
        ) from exc
    if not lines:
        raise InputError("success bundle SHA256SUMS must not be empty")
    expected: dict[str, str] = {}
    for line in lines:
        match = _CHECKSUM_RE.fullmatch(line)
        if match is None:
            raise InputError("success bundle SHA256SUMS contains an invalid line")
        digest, raw_path = match.groups()
        relative = _validate_relative_path("SHA256SUMS path", raw_path)
        if relative in {CHECKSUMS_NAME, COMPLETION_NAME}:
            raise InputError(
                "SHA256SUMS must exclude itself and the completion marker",
                details={"path": relative},
            )
        if relative in expected:
            raise InputError("success bundle SHA256SUMS contains a duplicate path")
        expected[relative] = f"sha256:{digest}"

    observed_paths = {
        relative
        for path in root.rglob("*")
        if path.is_file()
        and (relative := path.relative_to(root).as_posix()) not in {CHECKSUMS_NAME, COMPLETION_NAME}
    }
    if set(expected) != observed_paths:
        raise InputError(
            "success bundle SHA256SUMS does not cover exactly the evidence files",
            details={
                "missing": sorted(observed_paths - set(expected)),
                "unexpected": sorted(set(expected) - observed_paths),
            },
        )
    for relative, wanted in expected.items():
        observed = _sha256_uri(root / relative)
        if observed != wanted:
            raise InputError(
                "success bundle file does not match SHA256SUMS",
                details={"path": relative, "expected": wanted, "observed": observed},
            )


def _collect_bundle_files(root: Path) -> tuple[PublishFile, ...]:
    files: list[PublishFile] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise InputError(
                "success bundle must not contain symbolic links",
                details={"path": str(path)},
            )
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        files.append(
            PublishFile(
                relative_path=relative,
                source_path=path,
                sha256=_sha256_uri(path),
                size_bytes=path.stat().st_size,
            )
        )
    if not files:
        raise InputError("success bundle must contain files")
    return tuple(files)


def _verify_marker_binding(marker: dict[str, Any], key: str, path: str, file: Path) -> None:
    binding = marker.get(key)
    expected = {"path": path, "sha256": _sha256_uri(file)}
    if binding != expected:
        raise InputError(
            "completion marker hash binding does not match the referenced artifact",
            details={"field": key, "expected": expected, "observed": binding},
        )


def _expect_marker_value(
    marker: dict[str, Any],
    key: str,
    expected: str,
    path: Path,
) -> None:
    observed = marker.get(key)
    if observed != expected:
        raise InputError(
            "completion marker identity does not match the publication request",
            details={"path": str(path), "field": key, "expected": expected, "observed": observed},
        )


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"{label} is missing or invalid", details={"path": str(path)}) from exc
    if not isinstance(payload, dict):
        raise InputError(f"{label} must be a JSON object", details={"path": str(path)})
    return payload


def _nested_value(payload: dict[str, Any], *keys: str) -> object:
    value: object = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _namespace_present(namespace: str, files: tuple[str, ...]) -> bool:
    prefix = f"{namespace}/"
    return any(path == namespace or path.startswith(prefix) for path in files)


def _validate_repo_id(value: str) -> str:
    if _REPO_ID_RE.fullmatch(value) is None:
        raise InputError("repo_id must be an owner/repository identifier")
    return value


def _validate_relative_path(label: str, value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or value.endswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise InputError(f"{label} must be a normalized relative POSIX path")
    return value


def _validate_commit(label: str, value: str) -> str:
    if _COMMIT_RE.fullmatch(value) is None:
        raise InputError(f"{label} must be a full lowercase 40-character Git SHA")
    return value


def _validate_max_attempts(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > 10:
        raise InputError("max_attempts must be an integer in [1, 10]")
    return value


def _http_status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _sha256_uri(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InputError("failed to hash success bundle file", details={"path": str(path)}) from exc
    return f"sha256:{digest.hexdigest()}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", choices=("model", "dataset", "space"), default="model")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-commit-sha", required=True)
    parser.add_argument("--verification-dir", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
