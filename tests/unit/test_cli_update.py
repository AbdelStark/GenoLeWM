"""CLI tests for explicit model updates."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from geno_lewm.attestation import (
    SCHEMA_VERSION,
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    load_manifest,
    sha256_bytes,
    write_manifest,
)
from geno_lewm.cli import update
from geno_lewm.cli._dispatch import run_app
from geno_lewm.errors import InputError, RuntimeSetupError


def _write_checkpoint(root: Path, *, version: str, release_id: str, seed: str) -> Manifest:
    root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "predictor.safetensors": f"predictor-{seed}".encode(),
        "action_encoder.safetensors": f"action-{seed}".encode(),
        "calibration.parquet": f"calibration-{seed}".encode(),
        "train_config.yaml": f"seed: {seed}\n".encode(),
        "eval_report.md": f"# eval {seed}\n".encode(),
    }
    hashes: dict[str, str] = {}
    for name, body in artifacts.items():
        (root / name).write_bytes(body)
        hashes[name] = sha256_bytes(body)

    encoder_hash = sha256_bytes(f"encoder-{seed}".encode())
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        model_name="geno-lewm",
        model_version=version,
        release_id=release_id,
        encoder=ManifestEncoder(
            id="HuggingFaceBio/Carbon-500M",
            revision=f"main@{seed}",
            hash=encoder_hash,
        ),
        predictor=ManifestArtifact(
            file="predictor.safetensors",
            hash=hashes["predictor.safetensors"],
            dtype="bf16",
        ),
        action_encoder=ManifestArtifact(
            file="action_encoder.safetensors",
            hash=hashes["action_encoder.safetensors"],
            dtype="bf16",
        ),
        calibration=ManifestArtifact(
            file="calibration.parquet",
            hash=hashes["calibration.parquet"],
            version="1.0.0",
        ),
        training=ManifestTraining(
            config_file="train_config.yaml",
            hash=hashes["train_config.yaml"],
            data_snapshot={"fixture": seed},
        ),
        eval=ManifestArtifact(file="eval_report.md", hash=hashes["eval_report.md"]),
    )
    write_manifest(manifest, root / "manifest.json")
    return manifest


def _write_index(path: Path, *release_dirs: Path) -> Path:
    releases = []
    for release_dir in release_dirs:
        manifest = load_manifest(release_dir / "manifest.json")
        releases.append(
            {
                "model_version": manifest.model_version,
                "release_id": manifest.release_id,
                "manifest_url": (release_dir / "manifest.json").as_uri(),
                "artifact_base_url": release_dir.as_uri(),
            }
        )
    index_path = path / "releases.json"
    index_path.write_text(json.dumps({"releases": releases}), encoding="utf-8")
    return index_path


def _release_entry(release_dir: Path) -> update._ReleaseEntry:
    manifest = load_manifest(release_dir / "manifest.json")
    return update._ReleaseEntry(
        model_version=manifest.model_version,
        release_id=manifest.release_id,
        manifest_url=(release_dir / "manifest.json").as_uri(),
        artifact_base_url=release_dir.as_uri(),
    )


def test_update_check_only_displays_diff_without_installing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    models = tmp_path / "models"
    current = models / "current"
    _write_checkpoint(current, version="0.1.0", release_id="geno-lewm-v0.1.0-r1", seed="old")
    remote = tmp_path / "hub" / "v0.2"
    remote_manifest = _write_checkpoint(
        remote,
        version="0.2.0",
        release_id="geno-lewm-v0.2.0-r1",
        seed="new",
    )
    index = _write_index(tmp_path, remote)

    rc = run_app(
        update.app,
        argv=[
            "--quiet",
            "--no-banner",
            "--model-dir",
            str(current),
            "--index-url",
            index.as_uri(),
            "--check-only",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "Manifest diff:" in captured.out
    assert "model_version: 0.1.0 -> 0.2.0" in captured.out
    assert "Check only: no files were installed." in captured.out
    assert not (models / remote_manifest.release_id).exists()


def test_update_requires_confirmation_before_installing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    models = tmp_path / "models"
    current = models / "current"
    _write_checkpoint(current, version="0.1.0", release_id="geno-lewm-v0.1.0-r1", seed="old")
    remote = tmp_path / "hub" / "v0.2"
    remote_manifest = _write_checkpoint(
        remote,
        version="0.2.0",
        release_id="geno-lewm-v0.2.0-r1",
        seed="new",
    )
    index = _write_index(tmp_path, remote)

    monkeypatch.setattr(update, "_confirm_update", lambda _selection: False)
    rc = run_app(
        update.app,
        argv=[
            "--quiet",
            "--no-banner",
            "--model-dir",
            str(current),
            "--index-url",
            index.as_uri(),
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "Update cancelled" in captured.out
    assert not (models / remote_manifest.release_id).exists()


def test_update_yes_installs_side_by_side_and_preserves_previous_version(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    models = tmp_path / "models"
    current = models / "current"
    _write_checkpoint(current, version="0.1.0", release_id="geno-lewm-v0.1.0-r1", seed="old")
    (current / "user-marker.txt").write_text("do-not-delete", encoding="utf-8")
    remote = tmp_path / "hub" / "v0.2"
    remote_manifest = _write_checkpoint(
        remote,
        version="0.2.0",
        release_id="geno-lewm-v0.2.0-r1",
        seed="new",
    )
    index = _write_index(tmp_path, remote)

    rc = run_app(
        update.app,
        argv=[
            "--quiet",
            "--no-banner",
            "--model-dir",
            str(current),
            "--index-url",
            index.as_uri(),
            "--yes",
        ],
    )
    captured = capsys.readouterr()

    target = models / remote_manifest.release_id
    assert rc == 0
    assert target.is_dir()
    assert load_manifest(target / "manifest.json").model_id() == remote_manifest.model_id()
    assert (target / "predictor.safetensors").read_bytes() == (
        remote / "predictor.safetensors"
    ).read_bytes()
    assert (current / "user-marker.txt").read_text(encoding="utf-8") == "do-not-delete"
    assert "Installed update" in captured.out
    assert f"Previous version preserved at {current}" in captured.out


def test_update_rejects_artifact_hash_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    models = tmp_path / "models"
    current = models / "current"
    _write_checkpoint(current, version="0.1.0", release_id="geno-lewm-v0.1.0-r1", seed="old")
    remote = tmp_path / "hub" / "v0.2"
    remote_manifest = _write_checkpoint(
        remote,
        version="0.2.0",
        release_id="geno-lewm-v0.2.0-r1",
        seed="new",
    )
    (remote / "predictor.safetensors").write_bytes(b"tampered")
    index = _write_index(tmp_path, remote)

    rc = run_app(
        update.app,
        argv=[
            "--quiet",
            "--no-banner",
            "--model-dir",
            str(current),
            "--index-url",
            index.as_uri(),
            "--yes",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 4
    assert "downloaded artifact hash mismatch" in captured.err
    assert not (models / remote_manifest.release_id).exists()


def test_update_already_current_uses_env_model_dir_and_direct_manifest_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    current = tmp_path / "current"
    _write_checkpoint(current, version="0.2.0", release_id="geno-lewm-v0.2.0-r1", seed="same")
    monkeypatch.setenv("GENO_LEWM_MODEL_DIR", str(current))

    rc = run_app(
        update.app,
        argv=[
            "--quiet",
            "--no-banner",
            "--index-url",
            (current / "manifest.json").as_uri(),
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "Already up to date." in captured.out
    assert "Manifest diff:" not in captured.out


def test_update_reuses_existing_side_by_side_install(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    models = tmp_path / "models"
    current = models / "current"
    _write_checkpoint(current, version="0.1.0", release_id="geno-lewm-v0.1.0-r1", seed="old")
    remote = tmp_path / "hub" / "v0.2"
    _write_checkpoint(remote, version="0.2.0", release_id="geno-lewm-v0.2.0-r1", seed="new")
    index = _write_index(tmp_path, remote)
    argv = [
        "--quiet",
        "--no-banner",
        "--model-dir",
        str(current),
        "--index-url",
        index.as_uri(),
        "--yes",
    ]

    assert run_app(update.app, argv=argv) == 0
    capsys.readouterr()
    assert run_app(update.app, argv=argv) == 0
    captured = capsys.readouterr()

    assert "Release already installed" in captured.out


def test_update_target_version_selects_non_latest_release(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    current = tmp_path / "models" / "current"
    _write_checkpoint(current, version="0.1.0", release_id="geno-lewm-v0.1.0-r1", seed="old")
    remote_2 = tmp_path / "hub" / "v0.2"
    remote_3 = tmp_path / "hub" / "v0.3"
    _write_checkpoint(remote_2, version="0.2.0", release_id="geno-lewm-v0.2.0-r1", seed="two")
    _write_checkpoint(remote_3, version="0.3.0", release_id="geno-lewm-v0.3.0-r1", seed="three")
    index = _write_index(tmp_path, remote_2, remote_3)

    rc = run_app(
        update.app,
        argv=[
            "--quiet",
            "--no-banner",
            "--model-dir",
            str(current),
            "--index-url",
            index.as_uri(),
            "--target-version",
            "0.2.0",
            "--check-only",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "version=0.2.0 release_id=geno-lewm-v0.2.0-r1" in captured.out
    assert "version=0.3.0" not in captured.out


def test_update_helper_rejects_invalid_release_selection(tmp_path: Path) -> None:
    remote = tmp_path / "hub" / "v0.2"
    _write_checkpoint(remote, version="0.2.0", release_id="geno-lewm-v0.2.0-r1", seed="new")
    entry = _release_entry(remote)

    with pytest.raises(RuntimeSetupError):
        update._required_str({}, "model_version", source="file:///index.json")
    with pytest.raises(RuntimeSetupError):
        update._choose_release((), target_version=None)
    with pytest.raises(InputError):
        update._choose_release((entry,), target_version="0.9.0")


def test_update_rejects_bad_urls_and_unsafe_paths(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        update._fetch_bytes("http://example.com/releases.json", timeout=1.0)
    with pytest.raises(RuntimeSetupError):
        update._fetch_bytes((tmp_path / "missing.json").as_uri(), timeout=1.0)
    with pytest.raises(InputError):
        update._safe_relative_path("../predictor.safetensors")
    with pytest.raises(InputError):
        update._safe_install_name("bad/release")


def test_update_detects_release_index_manifest_mismatch(tmp_path: Path) -> None:
    remote = tmp_path / "hub" / "v0.2"
    _write_checkpoint(remote, version="0.2.0", release_id="geno-lewm-v0.2.0-r1", seed="new")
    entry = update._ReleaseEntry(
        model_version="9.9.9",
        release_id="geno-lewm-v0.2.0-r1",
        manifest_url=(remote / "manifest.json").as_uri(),
        artifact_base_url=remote.as_uri(),
    )

    with pytest.raises(RuntimeSetupError):
        update._fetch_manifest(entry, timeout=1.0)


def test_update_rejects_conflicting_manifest_artifact_paths(tmp_path: Path) -> None:
    remote = tmp_path / "hub" / "v0.2"
    manifest = _write_checkpoint(
        remote,
        version="0.2.0",
        release_id="geno-lewm-v0.2.0-r1",
        seed="new",
    )
    conflicting = Manifest(
        schema_version=manifest.schema_version,
        model_name=manifest.model_name,
        model_version=manifest.model_version,
        release_id=manifest.release_id,
        encoder=manifest.encoder,
        predictor=manifest.predictor,
        action_encoder=ManifestArtifact(
            file=manifest.predictor.file,
            hash=sha256_bytes(b"different"),
        ),
        calibration=manifest.calibration,
        training=manifest.training,
        eval=manifest.eval,
    )

    with pytest.raises(InputError):
        update._manifest_artifacts(conflicting)


def test_update_existing_install_validation_errors(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest = _write_checkpoint(
        source, version="0.2.0", release_id="geno-lewm-v0.2.0-r1", seed="new"
    )
    artifacts = update._manifest_artifacts(manifest)

    not_dir = tmp_path / "not-dir"
    not_dir.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeSetupError):
        update._verify_existing_install(not_dir, manifest, artifacts)

    no_manifest = tmp_path / "no-manifest"
    no_manifest.mkdir()
    with pytest.raises(RuntimeSetupError):
        update._verify_existing_install(no_manifest, manifest, artifacts)

    different = tmp_path / "different"
    _write_checkpoint(different, version="0.3.0", release_id="geno-lewm-v0.2.0-r1", seed="other")
    with pytest.raises(RuntimeSetupError):
        update._verify_existing_install(different, manifest, artifacts)

    missing_artifact = tmp_path / "missing-artifact"
    shutil.copytree(source, missing_artifact)
    (missing_artifact / "predictor.safetensors").unlink()
    with pytest.raises(RuntimeSetupError):
        update._verify_existing_install(missing_artifact, manifest, artifacts)


def test_update_load_manifest_bytes_wraps_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_value_error(_path: Path) -> Manifest:
        raise ValueError("boom")

    monkeypatch.setattr(update, "load_manifest", _raise_value_error)
    with pytest.raises(RuntimeSetupError):
        update._load_manifest_bytes(b"{}", source="file:///manifest.json")


def test_update_confirm_prompt_delegates_to_typer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_dir = tmp_path / "local"
    remote_dir = tmp_path / "remote"
    local = _write_checkpoint(
        local_dir, version="0.1.0", release_id="geno-lewm-v0.1.0-r1", seed="old"
    )
    remote = _write_checkpoint(
        remote_dir,
        version="0.2.0",
        release_id="geno-lewm-v0.2.0-r1",
        seed="new",
    )
    selection = update._UpdateSelection(
        local_manifest=local,
        remote_manifest=remote,
        release=_release_entry(remote_dir),
    )

    monkeypatch.setattr(
        update.typer,
        "confirm",
        lambda prompt, default: prompt.startswith("Apply this model update?") and default is False,
    )

    assert update._confirm_update(selection) is True
