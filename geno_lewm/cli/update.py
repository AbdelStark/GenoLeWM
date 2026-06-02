# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-update`` — explicit, user-initiated model updates.

The update command is the only CLI path that may contact the network.
It fetches a release index from the Hugging Face Hub, compares the
selected remote manifest against the local ``manifest.json``, displays
the delta, and installs the selected release only after explicit user
consent.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import urllib.error as _urllib_error
import urllib.parse as _urllib_parse
import urllib.request as _urllib_request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import typer

from geno_lewm.cli._dispatch import SharedOptions, finalize_shared, run_app, shared_option_decls
from geno_lewm.errors import InputError, ModelNotFoundError, RuntimeSetupError
from geno_lewm.provenance import Manifest, load_manifest, sha256_file, write_manifest

__all__ = [
    "app",
    "cli_main",
]


DEFAULT_UPDATE_INDEX_URL = (
    "https://huggingface.co/AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1-deploy/"
    "resolve/main/releases.json"
)
DEFAULT_MODEL_DIR = ".geno-lewm-models/current"
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class _ReleaseEntry:
    model_version: str
    release_id: str
    manifest_url: str
    artifact_base_url: str


@dataclass(frozen=True, slots=True)
class _ArtifactDownload:
    relative_path: Path
    expected_hash: str


@dataclass(frozen=True, slots=True)
class _UpdateSelection:
    local_manifest: Manifest
    remote_manifest: Manifest
    release: _ReleaseEntry

    @property
    def is_same_model(self) -> bool:
        return self.local_manifest.model_id() == self.remote_manifest.model_id()


@dataclass(frozen=True, slots=True)
class _InstallReport:
    target_dir: Path
    installed: bool
    artifact_count: int


app = typer.Typer(
    name="geno-lewm-update",
    help="Explicit, user-initiated model update (RFC-0010 §3.8 / RFC-0018 §3.3).",
    no_args_is_help=False,
    add_completion=True,
    pretty_exceptions_enable=False,
)

_S = shared_option_decls()


@app.callback(invoke_without_command=True)
def main(
    model_dir: Annotated[
        Path | None,
        typer.Option(
            "--model-dir",
            help=(
                "Existing model directory containing manifest.json; defaults to "
                "$GENO_LEWM_MODEL_DIR or .geno-lewm-models/current."
            ),
        ),
    ] = None,
    install_root: Annotated[
        Path | None,
        typer.Option(
            "--install-root",
            help="Directory where side-by-side release directories are installed.",
        ),
    ] = None,
    index_url: Annotated[
        str | None,
        typer.Option(
            "--index-url",
            help=("Hugging Face release-index URL; defaults to $GENO_LEWM_UPDATE_INDEX_URL."),
        ),
    ] = None,
    check_only: Annotated[
        bool,
        typer.Option("--check-only", help="Display the manifest delta without installing."),
    ] = False,
    target_version: Annotated[
        str | None,
        typer.Option(
            "--target-version",
            help="Exact remote model_version or release_id to install instead of latest.",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Apply the update without an interactive prompt."),
    ] = False,
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout", help="Network timeout in seconds for index and artifact fetches."
        ),
    ] = 30.0,
    config: Annotated[str | None, _S["config"]] = None,
    set_overrides: Annotated[list[str] | None, _S["set_overrides"]] = None,
    seed: Annotated[int | None, _S["seed"]] = None,
    deterministic: Annotated[bool, _S["deterministic"]] = False,
    log_level: Annotated[str, _S["log_level"]] = "info",
    log_dir: Annotated[str | None, _S["log_dir"]] = None,
    run_id: Annotated[str | None, _S["run_id"]] = None,
    wandb_project: Annotated[str | None, _S["wandb_project"]] = None,
    no_receipt: Annotated[bool, _S["no_receipt"]] = False,
    print_config: Annotated[bool, _S["print_config"]] = False,
    print_config_tree: Annotated[bool, _S["print_config_tree"]] = False,
    explain: Annotated[str | None, _S["explain"]] = None,
    quiet: Annotated[bool, _S["quiet"]] = False,
    no_banner: Annotated[bool, _S["no_banner"]] = False,
    version: Annotated[bool, _S["version"]] = False,
) -> None:
    opts: SharedOptions | None = finalize_shared(
        config=config,
        set_overrides=set_overrides,
        seed=seed,
        deterministic=deterministic,
        log_level=log_level,
        log_dir=log_dir,
        run_id=run_id,
        wandb_project=wandb_project,
        no_receipt=no_receipt,
        print_config=print_config,
        print_config_tree=print_config_tree,
        explain=explain,
        quiet=quiet,
        no_banner=no_banner,
        version=version,
        default_config_name="score",
    )
    if opts is None:
        return
    del opts

    if timeout <= 0:
        raise InputError("--timeout must be positive", details={"timeout": timeout})

    resolved_model_dir = _resolve_model_dir(model_dir)
    resolved_index_url = index_url or os.environ.get("GENO_LEWM_UPDATE_INDEX_URL")
    if resolved_index_url is None:
        resolved_index_url = DEFAULT_UPDATE_INDEX_URL

    selection = _select_update(
        model_dir=resolved_model_dir,
        index_url=resolved_index_url,
        target_version=target_version,
        timeout=timeout,
    )
    _print_update_summary(selection)

    if selection.is_same_model:
        typer.echo("Already up to date.")
        return
    if check_only:
        typer.echo("Check only: no files were installed.")
        return
    if not yes and not _confirm_update(selection):
        typer.echo("Update cancelled; no files were installed.")
        return

    root = (install_root if install_root is not None else resolved_model_dir.parent).expanduser()
    report = _install_release(selection.remote_manifest, selection.release, root, timeout=timeout)
    if report.installed:
        typer.echo(
            "Installed update "
            f"release_id={selection.remote_manifest.release_id} "
            f"artifacts={report.artifact_count} "
            f"path={report.target_dir}"
        )
    else:
        typer.echo(
            "Release already installed "
            f"release_id={selection.remote_manifest.release_id} "
            f"path={report.target_dir}"
        )
    typer.echo(f"Previous version preserved at {resolved_model_dir}")


def _resolve_model_dir(model_dir: Path | None) -> Path:
    if model_dir is not None:
        return model_dir.expanduser()
    configured = os.environ.get("GENO_LEWM_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(DEFAULT_MODEL_DIR).expanduser()


def _select_update(
    *,
    model_dir: Path,
    index_url: str,
    target_version: str | None,
    timeout: float,
) -> _UpdateSelection:
    manifest_path = model_dir / MANIFEST_NAME
    if not model_dir.is_dir() or not manifest_path.is_file():
        raise ModelNotFoundError(
            "model_dir must contain manifest.json before updates can be checked",
            details={"model_dir": str(model_dir), "manifest": str(manifest_path)},
            remediation="pass --model-dir or set GENO_LEWM_MODEL_DIR to an installed checkpoint",
        )

    local_manifest = load_manifest(manifest_path)
    releases = _fetch_release_index(index_url, timeout=timeout)
    release = _choose_release(releases, target_version=target_version)
    remote_manifest = _fetch_manifest(release, timeout=timeout)
    return _UpdateSelection(local_manifest, remote_manifest, release)


def _fetch_release_index(index_url: str, *, timeout: float) -> tuple[_ReleaseEntry, ...]:
    data = _fetch_bytes(index_url, timeout=timeout)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeSetupError(
            "release index is not valid UTF-8 JSON",
            details={"url": index_url, "error": str(exc)},
        ) from exc

    if not isinstance(payload, Mapping):
        raise RuntimeSetupError("release index must be a JSON object", details={"url": index_url})

    # Offline mirrors may point --index-url directly at a manifest.
    if "schema_version" in payload and "model_version" in payload and "release_id" in payload:
        manifest = _load_manifest_bytes(data, source=index_url)
        return (
            _ReleaseEntry(
                model_version=manifest.model_version,
                release_id=manifest.release_id,
                manifest_url=index_url,
                artifact_base_url=_parent_url(index_url),
            ),
        )

    raw_releases = payload.get("releases")
    if not isinstance(raw_releases, list) or not raw_releases:
        raise RuntimeSetupError(
            "release index must contain a non-empty releases list",
            details={"url": index_url},
        )

    entries: list[_ReleaseEntry] = []
    for idx, raw in enumerate(raw_releases):
        if not isinstance(raw, Mapping):
            raise RuntimeSetupError(
                "release index entries must be JSON objects",
                details={"url": index_url, "index": idx},
            )
        model_version = _required_str(raw, "model_version", source=index_url)
        release_id = _required_str(raw, "release_id", source=index_url)
        manifest_url = _required_str(raw, "manifest_url", source=index_url)
        artifact_base_url = raw.get("artifact_base_url") or raw.get("base_url")
        if artifact_base_url is None:
            artifact_base_url = _parent_url(manifest_url)
        if not isinstance(artifact_base_url, str) or not artifact_base_url:
            raise RuntimeSetupError(
                "artifact_base_url must be a non-empty string",
                details={"url": index_url, "release_id": release_id},
            )
        entries.append(
            _ReleaseEntry(
                model_version=model_version,
                release_id=release_id,
                manifest_url=manifest_url,
                artifact_base_url=artifact_base_url,
            )
        )
    return tuple(entries)


def _required_str(raw: Mapping[str, object], key: str, *, source: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeSetupError(
            "release index entry has an invalid field",
            details={"url": source, "field": key, "type": type(value).__name__},
        )
    return value


def _choose_release(
    releases: Sequence[_ReleaseEntry],
    *,
    target_version: str | None,
) -> _ReleaseEntry:
    if not releases:
        raise RuntimeSetupError("release index did not contain any releases")
    if target_version is not None:
        for release in releases:
            if target_version in {release.model_version, release.release_id}:
                return release
        raise InputError(
            "target version was not found in release index",
            details={
                "target_version": target_version,
                "available": [r.model_version for r in releases],
            },
        )
    return max(releases, key=lambda r: _version_key(r.model_version))


def _version_key(version: str) -> tuple[int, ...]:
    head = version.removeprefix("v").split("-", 1)[0]
    out: list[int] = []
    for part in head.split("."):
        if part.isdigit():
            out.append(int(part))
        else:
            out.append(0)
    return tuple(out)


def _fetch_manifest(release: _ReleaseEntry, *, timeout: float) -> Manifest:
    data = _fetch_bytes(release.manifest_url, timeout=timeout)
    manifest = _load_manifest_bytes(data, source=release.manifest_url)
    if manifest.model_version != release.model_version or manifest.release_id != release.release_id:
        raise RuntimeSetupError(
            "release index entry does not match fetched manifest",
            details={
                "index_model_version": release.model_version,
                "manifest_model_version": manifest.model_version,
                "index_release_id": release.release_id,
                "manifest_release_id": manifest.release_id,
            },
        )
    return manifest


def _load_manifest_bytes(data: bytes, *, source: str) -> Manifest:
    with tempfile.TemporaryDirectory(prefix="geno-lewm-manifest-") as tmp:
        path = Path(tmp) / MANIFEST_NAME
        path.write_bytes(data)
        try:
            return load_manifest(path)
        except Exception as exc:
            if hasattr(exc, "details"):
                raise
            raise RuntimeSetupError(
                "remote manifest could not be loaded",
                details={"source": source, "error": str(exc)},
            ) from exc


def _fetch_bytes(url: str, *, timeout: float) -> bytes:
    parsed = _urllib_parse.urlparse(url)
    if parsed.scheme not in {"https", "file"}:
        raise InputError(
            "update URLs must use https://",
            details={"url": url, "scheme": parsed.scheme},
            remediation="use a Hugging Face HTTPS URL; file:// is accepted for offline tests",
        )
    request = _urllib_request.Request(
        url,
        headers={"User-Agent": "geno-lewm-update"},
    )
    try:
        with _urllib_request.urlopen(request, timeout=timeout) as response:
            return cast(bytes, response.read())
    except (_urllib_error.URLError, OSError) as exc:
        raise RuntimeSetupError(
            "failed to fetch update resource",
            details={"url": url, "error": str(exc)},
        ) from exc


def _parent_url(url: str) -> str:
    return url.rsplit("/", 1)[0] + "/"


def _artifact_url(base_url: str, relative_path: Path) -> str:
    quoted = _urllib_parse.quote(relative_path.as_posix())
    return _urllib_parse.urljoin(base_url.rstrip("/") + "/", quoted)


def _manifest_artifacts(manifest: Manifest) -> tuple[_ArtifactDownload, ...]:
    raw = (
        (manifest.predictor.file, manifest.predictor.hash),
        (manifest.action_encoder.file, manifest.action_encoder.hash),
        (manifest.calibration.file, manifest.calibration.hash),
        (manifest.training.config_file, manifest.training.hash),
        (manifest.eval.file, manifest.eval.hash),
    )
    by_path: dict[Path, str] = {}
    for file_name, expected_hash in raw:
        relative = _safe_relative_path(file_name)
        existing = by_path.get(relative)
        if existing is not None and existing != expected_hash:
            raise InputError(
                "manifest references the same artifact path with conflicting hashes",
                details={"file": relative.as_posix(), "hashes": [existing, expected_hash]},
            )
        by_path[relative] = expected_hash
    return tuple(_ArtifactDownload(relative_path=p, expected_hash=h) for p, h in by_path.items())


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise InputError(
            "manifest artifact paths must be relative and stay inside the release directory",
            details={"path": value},
        )
    return path


def _safe_install_name(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise InputError("release_id is not a safe directory name", details={"release_id": value})
    return value


def _install_release(
    manifest: Manifest,
    release: _ReleaseEntry,
    install_root: Path,
    *,
    timeout: float,
) -> _InstallReport:
    install_root.mkdir(parents=True, exist_ok=True)
    release_name = _safe_install_name(manifest.release_id)
    target_dir = install_root / release_name
    artifacts = _manifest_artifacts(manifest)

    if target_dir.exists():
        _verify_existing_install(target_dir, manifest, artifacts)
        return _InstallReport(target_dir=target_dir, installed=False, artifact_count=len(artifacts))

    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{release_name}.tmp-", dir=str(install_root.resolve()))
    )
    try:
        write_manifest(manifest, temp_dir / MANIFEST_NAME)
        for artifact in artifacts:
            destination = temp_dir / artifact.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(
                _fetch_bytes(
                    _artifact_url(release.artifact_base_url, artifact.relative_path),
                    timeout=timeout,
                )
            )
            _verify_file_hash(destination, artifact.expected_hash)
        temp_dir.rename(target_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return _InstallReport(target_dir=target_dir, installed=True, artifact_count=len(artifacts))


def _verify_existing_install(
    target_dir: Path,
    manifest: Manifest,
    artifacts: Iterable[_ArtifactDownload],
) -> None:
    if not target_dir.is_dir():
        raise RuntimeSetupError(
            "target release path exists but is not a directory",
            details={"path": str(target_dir)},
        )
    existing_manifest_path = target_dir / MANIFEST_NAME
    if not existing_manifest_path.is_file():
        raise RuntimeSetupError(
            "target release directory exists but lacks manifest.json",
            details={"path": str(target_dir)},
        )
    existing_manifest = load_manifest(existing_manifest_path)
    if existing_manifest.model_id() != manifest.model_id():
        raise RuntimeSetupError(
            "target release directory exists with a different manifest",
            details={
                "path": str(target_dir),
                "existing_model_id": existing_manifest.model_id(),
                "remote_model_id": manifest.model_id(),
            },
        )
    for artifact in artifacts:
        path = target_dir / artifact.relative_path
        if not path.is_file():
            raise RuntimeSetupError(
                "target release directory is missing an artifact",
                details={"path": str(path)},
            )
        _verify_file_hash(path, artifact.expected_hash)


def _verify_file_hash(path: Path, expected_hash: str) -> None:
    observed = sha256_file(path)
    if observed != expected_hash:
        raise RuntimeSetupError(
            "downloaded artifact hash mismatch",
            details={"path": str(path), "expected": expected_hash, "observed": observed},
        )


def _print_update_summary(selection: _UpdateSelection) -> None:
    local = selection.local_manifest
    remote = selection.remote_manifest
    typer.echo("Local model:")
    typer.echo(f"  version={local.model_version} release_id={local.release_id}")
    typer.echo(f"  model_id={local.model_id()}")
    typer.echo("Remote model:")
    typer.echo(f"  version={remote.model_version} release_id={remote.release_id}")
    typer.echo(f"  model_id={remote.model_id()}")

    if selection.is_same_model:
        return
    typer.echo("Manifest diff:")
    for line in _format_manifest_diff(local, remote):
        typer.echo(f"  {line}")


def _format_manifest_diff(local: Manifest, remote: Manifest) -> tuple[str, ...]:
    rows = [
        ("model_version", local.model_version, remote.model_version),
        ("release_id", local.release_id, remote.release_id),
        ("encoder.revision", local.encoder.revision, remote.encoder.revision),
        ("encoder.hash", local.encoder.hash, remote.encoder.hash),
        ("predictor.file", local.predictor.file, remote.predictor.file),
        ("predictor.hash", local.predictor.hash, remote.predictor.hash),
        ("action_encoder.file", local.action_encoder.file, remote.action_encoder.file),
        ("action_encoder.hash", local.action_encoder.hash, remote.action_encoder.hash),
        ("calibration.file", local.calibration.file, remote.calibration.file),
        ("calibration.hash", local.calibration.hash, remote.calibration.hash),
        ("training.config_file", local.training.config_file, remote.training.config_file),
        ("training.hash", local.training.hash, remote.training.hash),
        ("eval.file", local.eval.file, remote.eval.file),
        ("eval.hash", local.eval.hash, remote.eval.hash),
    ]
    diff = [f"{name}: {old} -> {new}" for name, old, new in rows if old != new]
    if local.model_id() != remote.model_id() and not diff:
        diff.append(f"model_id: {local.model_id()} -> {remote.model_id()}")
    return tuple(diff)


def _confirm_update(selection: _UpdateSelection) -> bool:
    return typer.confirm(
        "Apply this model update? "
        f"{selection.local_manifest.model_version} -> {selection.remote_manifest.model_version}",
        default=False,
    )


def cli_main() -> int:
    return run_app(app)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli_main())
