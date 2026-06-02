# SPDX-License-Identifier: Apache-2.0
"""Plan a Hugging Face Hub model release without uploading artifacts."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import unquote, urlparse

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import load_manifest, sha256_file
from geno_lewm.provenance.hashing import looks_like_sha256
from tools.demo.terminal_inference import DEMO_MANIFEST_NAME
from tools.release.paper_package import PackagePaths, verify_package
from tools.release.runtime_preflight import REPORT_NAME as RUNTIME_PREFLIGHT_REPORT_NAME
from tools.release.training_run import CHECKSUMS_NAME as TRAINING_RUN_CHECKSUMS_NAME

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.hub_release"
REPO_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
COMMIT_RE: Final = re.compile(r"^[0-9a-fA-F]{7,40}$")


@dataclass(frozen=True, slots=True)
class UploadFile:
    """One release-package file planned for upload."""

    source: str
    destination: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "destination": self.destination,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class HubReleasePlan:
    """Dry-run plan for publishing model, dataset, and demo artifacts."""

    schema_version: str
    generated_by: str
    generated_at: str
    ready: bool
    repo_id: str
    release_id: str
    model_id: str
    commit_sha: str
    dataset_url: str
    demo_url: str
    paper_url: str | None
    paper_file: UploadFile | None
    files: tuple[UploadFile, ...]
    dataset_files: tuple[UploadFile, ...]
    demo_files: tuple[UploadFile, ...]
    commands: tuple[str, ...]
    requirements: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "ready": self.ready,
            "repo_id": self.repo_id,
            "release_id": self.release_id,
            "model_id": self.model_id,
            "commit_sha": self.commit_sha,
            "dataset_url": self.dataset_url,
            "demo_url": self.demo_url,
            "paper_url": self.paper_url,
            "paper_file": None if self.paper_file is None else self.paper_file.to_dict(),
            "files": [file.to_dict() for file in self.files],
            "dataset_files": [file.to_dict() for file in self.dataset_files],
            "demo_files": [file.to_dict() for file in self.demo_files],
            "commands": list(self.commands),
            "requirements": list(self.requirements),
        }


def build_hub_release_plan(
    *,
    model_dir: Path,
    dataset_dir: Path,
    demo_dir: Path,
    repo_id: str,
    dataset_url: str,
    demo_url: str,
    commit_sha: str,
    paper_path: Path | None = None,
    paper_url: str | None = None,
    generated_at: str | None = None,
    allow_fixture_manifest: bool = False,
) -> HubReleasePlan:
    """Verify a release candidate and return the dry-run Hub upload plan."""
    repo_id = _validate_repo_id(repo_id)
    dataset_url = _validate_url("dataset_url", dataset_url)
    demo_url = _validate_url("demo_url", demo_url)
    paper_url = None if paper_url is None else _validate_url("paper_url", paper_url)
    if paper_path is not None and paper_url is None:
        raise InputError(
            "paper_url is required when paper_path is provided",
            details={"paper_path": str(paper_path)},
        )
    commit_sha = _validate_commit(commit_sha)

    report = verify_package(
        PackagePaths(
            model_dir=model_dir,
            dataset_dir=dataset_dir,
            demo_dir=demo_dir,
            paper_path=paper_path,
        ),
        allow_fixture_manifest=allow_fixture_manifest,
    )
    if not report.ok:
        raise InputError(
            "paper/demo release package is not valid",
            details={"issues": [issue.to_dict() for issue in report.issues]},
        )
    manifest = load_manifest(model_dir / "manifest.json")
    files = _collect_model_files(model_dir)
    dataset_files = _collect_dataset_files(dataset_dir)
    demo_files = _collect_demo_files(demo_dir)
    paper_file = None if paper_path is None else _paper_file(paper_path)
    commands = _commands(
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        demo_dir=demo_dir,
        paper_path=paper_path,
        paper_url=paper_url,
        paper_file=paper_file,
        repo_id=repo_id,
        dataset_url=dataset_url,
        demo_url=demo_url,
        release_id=manifest.release_id,
        model_id=manifest.model_id(),
        model_files=files,
        dataset_files=dataset_files,
        demo_files=demo_files,
    )
    requirements = (
        "A real trained checkpoint, not a fixture manifest.",
        "A Hugging Face account with write access to the target repo.",
        "HF_TOKEN exported in the release environment.",
        "Dataset snapshot, terminal demo transcript, and paper artifact URLs are public.",
        (
            "If paper_path is set, paper_url must serve the exact paper bytes; "
            "GitHub release download URLs are uploaded by tools.release.hub_publish."
        ),
    )
    return HubReleasePlan(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        generated_at=generated_at or _utc_now(),
        ready=True,
        repo_id=repo_id,
        release_id=manifest.release_id,
        model_id=manifest.model_id(),
        commit_sha=commit_sha,
        dataset_url=dataset_url,
        demo_url=demo_url,
        paper_url=paper_url,
        paper_file=paper_file,
        files=files,
        dataset_files=dataset_files,
        demo_files=demo_files,
        commands=commands,
        requirements=requirements,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        plan = build_hub_release_plan(
            model_dir=args.model_dir,
            dataset_dir=args.dataset_dir,
            demo_dir=args.demo_dir,
            repo_id=args.repo_id,
            dataset_url=args.dataset_url,
            demo_url=args.demo_url,
            commit_sha=args.commit_sha,
            paper_path=args.paper_path,
            paper_url=args.paper_url,
            allow_fixture_manifest=args.allow_fixture_manifest,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    payload = json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.write_text(payload, encoding="utf-8")
        sys.stdout.write(f"wrote {args.output}\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run a Hugging Face Hub model release after package verification.",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="Hub repo id, e.g. owner/model-name")
    parser.add_argument("--dataset-url", required=True)
    parser.add_argument("--demo-url", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--paper-path", type=Path)
    parser.add_argument("--paper-url")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-fixture-manifest",
        action="store_true",
        help="Allow fixture/test manifests for local verifier tests only.",
    )
    return parser


def _collect_model_files(model_dir: Path) -> tuple[UploadFile, ...]:
    return _merge_upload_files(
        _collect_checksum_files(model_dir, "SHA256SUMS", repo_type="model"),
        _collect_checksum_files(model_dir, TRAINING_RUN_CHECKSUMS_NAME, repo_type="model"),
    )


def _collect_dataset_files(dataset_dir: Path) -> tuple[UploadFile, ...]:
    return _collect_checksum_files(dataset_dir, "SHA256SUMS", repo_type="dataset")


def _merge_upload_files(*groups: tuple[UploadFile, ...]) -> tuple[UploadFile, ...]:
    merged: dict[str, UploadFile] = {}
    for group in groups:
        for file in group:
            existing = merged.get(file.destination)
            if existing is not None and existing.sha256 != file.sha256:
                raise InputError(
                    "planned upload file has conflicting hashes",
                    details={"destination": file.destination},
                )
            merged[file.destination] = file
    return tuple(merged[destination] for destination in sorted(merged))


def _collect_checksum_files(
    root: Path,
    checksums_name: str,
    *,
    repo_type: str,
) -> tuple[UploadFile, ...]:
    checksums_path = root / checksums_name
    try:
        text = checksums_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(
            f"{repo_type} checksum file is missing",
            details={"path": str(checksums_path)},
        ) from exc
    entries = _parse_sha256sums(text)
    files: list[UploadFile] = []
    for relative, expected_hash in entries.items():
        path = _safe_relative(root, relative, repo_type=repo_type)
        if not path.is_file():
            raise InputError(
                f"planned {repo_type} upload file is missing",
                details={"path": str(path)},
            )
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise InputError(
                f"planned {repo_type} upload file hash mismatch",
                details={"path": str(path), "expected": expected_hash, "observed": observed_hash},
            )
        files.append(_upload_file(path, destination=relative))
    files.append(_upload_file(checksums_path, destination=checksums_name))
    return tuple(files)


def _collect_demo_files(demo_dir: Path) -> tuple[UploadFile, ...]:
    manifest_path = demo_dir / DEMO_MANIFEST_NAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError("demo manifest is missing", details={"path": str(manifest_path)}) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "demo manifest JSON is invalid",
            details={"path": str(manifest_path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("demo manifest must be a JSON object")
    paths: list[Path] = [manifest_path]
    paths.extend(_demo_input_paths(payload.get("inputs"), demo_dir))
    paths.extend(_demo_artifact_paths(payload.get("artifacts"), demo_dir))
    paths.extend(
        demo_dir / required
        for required in (
            "terminal-demo-transcript.md",
            "scores.jsonl",
            "receipts.jsonl",
            RUNTIME_PREFLIGHT_REPORT_NAME,
            "batch_receipt_report.json",
        )
    )
    files: list[UploadFile] = []
    seen_destinations: set[str] = set()
    seen_assets: dict[str, str] = {}
    for path in paths:
        destination = _relative_destination(demo_dir, path)
        if destination in seen_destinations:
            continue
        asset_name = Path(destination).name
        previous_destination = seen_assets.get(asset_name)
        if previous_destination is not None:
            raise InputError(
                "planned demo upload asset names must be unique",
                details={
                    "asset": asset_name,
                    "first": previous_destination,
                    "duplicate": destination,
                },
            )
        seen_destinations.add(destination)
        seen_assets[asset_name] = destination
        files.append(_upload_file(path, destination=destination))
    return tuple(files)


def _paper_file(path: Path) -> UploadFile:
    return _upload_file(path, source=_public_source_path(path), destination=path.name)


def _demo_input_paths(raw: Any, demo_dir: Path) -> tuple[Path, ...]:
    if not isinstance(raw, dict):
        raise InputError("demo manifest inputs must be an object")
    paths: list[Path] = []
    for key in ("vcf", "fasta"):
        item = raw.get(key)
        if not isinstance(item, dict):
            raise InputError("demo manifest input identity is missing", details={"input": key})
        paths.append(_demo_identity_path(item, demo_dir, field=f"inputs.{key}"))
    return tuple(paths)


def _demo_artifact_paths(raw: Any, demo_dir: Path) -> tuple[Path, ...]:
    if not isinstance(raw, list) or not raw:
        raise InputError("demo manifest artifacts must be a non-empty list")
    paths: list[Path] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InputError("demo manifest artifact entries must be objects")
        paths.append(_demo_identity_path(item, demo_dir, field=f"artifacts[{index}]"))
    return tuple(paths)


def _demo_identity_path(item: dict[str, Any], demo_dir: Path, *, field: str) -> Path:
    path = _demo_file_path(item, demo_dir, field=f"{field}.path")
    if not path.is_file():
        raise InputError(
            "demo manifest file target is missing",
            details={"field": field, "path": str(path)},
        )
    expected_hash = item.get("sha256")
    if not isinstance(expected_hash, str) or not looks_like_sha256(expected_hash):
        raise InputError(
            "demo manifest file identity must include sha256:<64hex>",
            details={"field": f"{field}.sha256", "path": str(path)},
        )
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise InputError(
            "demo manifest file hash mismatch",
            details={
                "field": f"{field}.sha256",
                "path": str(path),
                "expected": expected_hash,
                "observed": observed_hash,
            },
        )
    expected_size = item.get("size_bytes")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size <= 0:
        raise InputError(
            "demo manifest file identity must include a positive size_bytes",
            details={"field": f"{field}.size_bytes", "path": str(path)},
        )
    observed_size = path.stat().st_size
    if observed_size != expected_size:
        raise InputError(
            "demo manifest file size mismatch",
            details={
                "field": f"{field}.size_bytes",
                "path": str(path),
                "expected": expected_size,
                "observed": observed_size,
            },
        )
    return path


def _demo_file_path(item: dict[str, Any], demo_dir: Path, *, field: str) -> Path:
    value = item.get("path")
    if not isinstance(value, str) or not value.strip():
        raise InputError("demo manifest file path is missing", details={"field": field})
    path = Path(value)
    if not path.is_absolute():
        path = _relative_demo_manifest_path(path, demo_dir)
    _ensure_under_root(path, demo_dir, field=field)
    return path


def _relative_demo_manifest_path(relative: Path, demo_dir: Path) -> Path:
    local_path = demo_dir / relative
    package_path = demo_dir.parent / relative
    if local_path.exists() or not package_path.exists():
        return local_path
    return package_path


def _upload_file(path: Path, *, destination: str, source: str | None = None) -> UploadFile:
    if not path.is_file():
        raise InputError("planned upload file is missing", details={"path": str(path)})
    return UploadFile(
        source=source or destination,
        destination=destination,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _relative_destination(root: Path, path: Path) -> str:
    _ensure_under_root(path, root, field="path")
    return path.resolve().relative_to(root.resolve()).as_posix()


def _ensure_under_root(path: Path, root: Path, *, field: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise InputError(
            "planned demo upload paths must stay inside demo_dir",
            details={"field": field, "path": str(path), "demo_dir": str(root)},
        ) from exc


def _parse_sha256sums(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise InputError("SHA256SUMS lines must be '<hex>  <path>'", details={"line": line_no})
        hex_digest, relative = parts
        expected_hash = "sha256:" + hex_digest.lower()
        if not looks_like_sha256(expected_hash):
            raise InputError(
                "SHA256SUMS contains an invalid SHA-256 digest",
                details={"line": line_no, "digest": hex_digest},
            )
        if relative in entries:
            raise InputError(
                "SHA256SUMS contains duplicate paths",
                details={"line": line_no, "path": relative},
            )
        entries[relative] = expected_hash
    if not entries:
        raise InputError("SHA256SUMS must contain at least one entry")
    return entries


def _commands(
    *,
    model_dir: Path,
    dataset_dir: Path,
    demo_dir: Path,
    paper_path: Path | None,
    paper_url: str | None,
    paper_file: UploadFile | None,
    repo_id: str,
    dataset_url: str,
    demo_url: str,
    release_id: str,
    model_id: str,
    model_files: tuple[UploadFile, ...],
    dataset_files: tuple[UploadFile, ...],
    demo_files: tuple[UploadFile, ...],
) -> tuple[str, ...]:
    package_cmd = [
        "python",
        "-m",
        "tools.release.paper_package",
        "--model-dir",
        _public_source_path(model_dir, default="model"),
        "--dataset-dir",
        _public_source_path(dataset_dir, default="dataset"),
        "--demo-dir",
        _public_source_path(demo_dir, default="demo"),
    ]
    if paper_path is not None:
        package_cmd.extend(["--paper-path", _public_source_path(paper_path)])
    commands = [
        _shell_join(package_cmd),
        *(
            _shell_join(command)
            for command in _hf_upload_commands(
                repo_id=repo_id,
                repo_type="model",
                files=_prefix_sources("model", model_files),
                commit_message=f"Release {release_id} ({model_id})",
            )
        ),
    ]
    dataset_repo_id = _dataset_repo_id_from_url(dataset_url)
    if dataset_repo_id is not None:
        commands.extend(
            _shell_join(command)
            for command in _hf_upload_commands(
                repo_id=dataset_repo_id,
                repo_type="dataset",
                files=_prefix_sources("dataset", dataset_files),
                commit_message=f"Release dataset for {release_id} ({model_id})",
            )
        )
    demo_upload_cmd = _demo_upload_command(demo_url, _prefix_sources("demo", demo_files))
    if demo_upload_cmd is not None:
        commands.append(_shell_join(demo_upload_cmd))
    paper_upload_cmd = _paper_upload_command(paper_url, paper_file)
    if paper_upload_cmd is not None:
        commands.append(_shell_join(paper_upload_cmd))
    return tuple(commands)


def _hf_upload_commands(
    *,
    repo_id: str,
    repo_type: str,
    files: tuple[UploadFile, ...],
    commit_message: str,
) -> tuple[list[str], ...]:
    return tuple(
        [
            "huggingface-cli",
            "upload",
            repo_id,
            file.source,
            file.destination,
            "--repo-type",
            repo_type,
            "--commit-message",
            commit_message,
        ]
        for file in files
    )


def _prefix_sources(prefix: str, files: tuple[UploadFile, ...]) -> tuple[UploadFile, ...]:
    return tuple(
        UploadFile(
            source=f"{prefix}/{file.source}",
            destination=file.destination,
            sha256=file.sha256,
            size_bytes=file.size_bytes,
        )
        for file in files
    )


def _public_source_path(path: Path, *, default: str | None = None) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        return default or path.name


def _dataset_repo_id_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    parts = tuple(part for part in parsed.path.strip("/").split("/") if part)
    if parsed.netloc != "huggingface.co" or len(parts) < 3 or parts[0] != "datasets":
        return None
    return f"{parts[1]}/{parts[2]}"


def _demo_upload_command(url: str, demo_files: tuple[UploadFile, ...]) -> list[str] | None:
    parsed = urlparse(url)
    parts = tuple(part for part in parsed.path.strip("/").split("/") if part)
    if parsed.netloc != "github.com" or len(parts) < 5 or parts[2:4] != ("releases", "tag"):
        return None
    owner, repo, _releases, _tag, tag = parts[:5]
    return [
        "gh",
        "release",
        "upload",
        tag,
        *(file.source for file in demo_files),
        "--repo",
        f"{owner}/{repo}",
        "--clobber",
    ]


def _paper_upload_command(url: str | None, paper_file: UploadFile | None) -> list[str] | None:
    if url is None or paper_file is None:
        return None
    target = _github_release_download_from_url(url)
    if target is None:
        return None
    repo, tag, asset_name = target
    if asset_name != paper_file.destination:
        return None
    return [
        "gh",
        "release",
        "upload",
        tag,
        paper_file.source,
        "--repo",
        repo,
        "--clobber",
    ]


def _github_release_download_from_url(url: str) -> tuple[str, str, str] | None:
    parsed = urlparse(url)
    parts = tuple(part for part in parsed.path.strip("/").split("/") if part)
    if parsed.netloc != "github.com" or len(parts) < 5 or parts[2] != "releases":
        return None
    if parts[3] != "download":
        return None
    owner, repo, _releases, _download, tag, *asset_parts = parts
    if not asset_parts:
        return None
    return f"{owner}/{repo}", tag, unquote("/".join(asset_parts))


def _validate_repo_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError("repo_id must be a non-empty string")
    value = value.strip()
    if "://" in value or not REPO_ID_RE.match(value):
        raise InputError("repo_id must look like 'owner/name'", details={"repo_id": value})
    return value


def _validate_url(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{name} must be a non-empty URL")
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise InputError(f"{name} must be an http(s) URL", details={name: value})
    return value


def _validate_commit(value: str) -> str:
    if not isinstance(value, str) or not COMMIT_RE.match(value.strip()):
        raise InputError("commit_sha must be a 7-40 character hex Git commit")
    return value.strip().lower()


def _safe_relative(root: Path, relative: str, *, repo_type: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise InputError(
            f"planned {repo_type} upload paths must be relative and stay inside the package root",
            details={"path": relative, "root": str(root)},
        )
    return root / candidate


def _shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
