"""CLI tests for ``geno-lewm-train`` fixture smoke mode."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import geno_lewm._atomic as atomic_module
from geno_lewm._source_provenance import SourceProvenance, resolve_package_source
from geno_lewm.cli import _dispatch, train as train_cli
from geno_lewm.cli.train import app
from geno_lewm.training import preflight as training_preflight
from geno_lewm.training.real import (
    CARBON_CHECKPOINT_NAME,
    CARBON_LOG_NAME,
    CARBON_METRICS_NAME,
    CARBON_TRAINING_METADATA_NAME,
    CarbonTrainingReport,
)
from tests.unit.test_training_preflight import (
    _available_accelerator,
    _available_dependency,
    _missing_dependency,
    _write_carbon_model_dir,
    _write_release_dataset,
    _write_training_config,
)
from tools.release.training_run import GENERATED_BY as TRAINING_RUN_GENERATED_BY

requires_secure_atomic_publication = pytest.mark.skipif(
    not atomic_module.supports_secure_atomic_publication(),
    reason=(
        "secure production checkpoint/report publication requires POSIX "
        "anchored directory operations"
    ),
)


@pytest.fixture(autouse=True)
def _stable_production_source(monkeypatch: pytest.MonkeyPatch) -> None:
    source = SourceProvenance(
        commit_sha="a" * 40,
        tree_sha="b" * 40,
        package_version="0.2.1",
    )
    monkeypatch.setattr(train_cli, "resolve_package_source", lambda **_kwargs: source)


def test_fixture_source_commit_uses_full_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    full_sha = "cd2bfccb33ec5a2df3c4707e8be8443f4682dad3"

    def fake_run(args, *, cwd, check, capture_output, text):
        assert args == ["git", "rev-parse", "HEAD"]
        assert cwd == tmp_path
        assert check is False
        assert capture_output is True
        assert text is True
        return SimpleNamespace(returncode=0, stdout=f"{full_sha}\n")

    monkeypatch.setattr(train_cli.subprocess, "run", fake_run)

    assert train_cli._fixture_source_commit(tmp_path) == full_sha


def test_train_requires_explicit_fixture_smoke(capsys) -> None:
    rc = _dispatch.run_app(app, argv=["--quiet", "--no-banner"])
    captured = capsys.readouterr()

    assert rc == 2
    assert "requires --fixture-smoke, --carbon-preflight, or --carbon-train" in captured.err
    assert "research tool" not in captured.err


def test_train_fixture_smoke_cli_writes_artifacts(tmp_path: Path, capsys) -> None:
    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--fixture-smoke",
            "--run-dir",
            str(tmp_path),
            "--steps",
            "4",
            "--seed",
            "11",
            "--deterministic",
            "--run-id",
            "fixture-cli",
            "--set",
            "data.batch_size=2",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["run_id"] == "fixture-cli"
    assert payload["steps_completed"] == 4
    assert (tmp_path / "fixture_predictor_checkpoint.json").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "train.log").is_file()
    assert "batch_size: 2" in (tmp_path / "config.resolved.yaml").read_text(encoding="utf-8")


def test_train_carbon_preflight_cli_writes_report(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    carbon_dir = _write_carbon_model_dir(tmp_path)
    config = _write_training_config(tmp_path)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(training_preflight, "_probe_dependency", _available_dependency)
    monkeypatch.setattr(training_preflight, "_probe_accelerator", _available_accelerator)

    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--carbon-preflight",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--carbon-model-dir",
            str(carbon_dir),
            "--training-config",
            str(config),
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["dataset_snapshot_id"] == "geno-lewm-data-v0.1.0-r1"
    assert (run_dir / "training_preflight_report.json").is_file()


def test_train_carbon_preflight_remains_available_without_atomic_dirfd_support(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    carbon_dir = _write_carbon_model_dir(tmp_path)
    config = _write_training_config(tmp_path)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(training_preflight, "_probe_dependency", _available_dependency)
    monkeypatch.setattr(training_preflight, "_probe_accelerator", _available_accelerator)
    monkeypatch.setattr(
        atomic_module,
        "_supports_anchored_directory_operations",
        lambda: False,
    )

    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--carbon-preflight",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--carbon-model-dir",
            str(carbon_dir),
            "--training-config",
            str(config),
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out)["ok"] is True
    assert (run_dir / "training_preflight_report.json").is_file()


@requires_secure_atomic_publication
def test_train_carbon_train_runs_preflight_before_launch(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    carbon_dir = _write_carbon_model_dir(tmp_path)
    config = _write_training_config(tmp_path)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(training_preflight, "_probe_dependency", _missing_dependency)
    monkeypatch.setattr(training_preflight, "_probe_accelerator", _available_accelerator)

    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--carbon-train",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--carbon-model-dir",
            str(carbon_dir),
            "--training-config",
            str(config),
            "--steps",
            "1",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 2
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert (run_dir / "training_preflight_report.json").is_file()
    assert not (run_dir / "metrics.json").exists()


@requires_secure_atomic_publication
@pytest.mark.parametrize("mode", ["--carbon-preflight", "--carbon-train"])
def test_train_carbon_mode_rejects_concurrent_writer_before_run_publication(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    carbon_dir = _write_carbon_model_dir(tmp_path)
    config = _write_training_config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock = run_dir / ".production-carbon-run.lock"
    lock.write_text("pid=another-writer\n", encoding="utf-8")
    monkeypatch.setattr(training_preflight, "_probe_dependency", _available_dependency)
    monkeypatch.setattr(training_preflight, "_probe_accelerator", _available_accelerator)
    monkeypatch.setattr(train_cli, "run_carbon_training", _fake_carbon_training)

    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            mode,
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--carbon-model-dir",
            str(carbon_dir),
            "--training-config",
            str(config),
            "--steps",
            "1",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert "another writer is already active" in captured.err
    assert lock.read_text(encoding="utf-8") == "pid=another-writer\n"
    assert not (run_dir / "training_config.effective.yaml").exists()
    assert not (run_dir / "training_preflight_report.json").exists()


@requires_secure_atomic_publication
def test_train_carbon_train_can_package_release_run_after_success(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    carbon_dir = _write_carbon_model_dir(tmp_path)
    config = _write_training_config(tmp_path)
    run_dir = tmp_path / "run"
    side_preflight = tmp_path / "preflight-sidecar.json"
    monkeypatch.setattr(training_preflight, "_probe_dependency", _available_dependency)
    monkeypatch.setattr(training_preflight, "_probe_accelerator", _available_accelerator)
    monkeypatch.setattr(train_cli, "run_carbon_training", _fake_carbon_training)

    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--carbon-train",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--carbon-model-dir",
            str(carbon_dir),
            "--training-config",
            str(config),
            "--preflight-output",
            str(side_preflight),
            "--package-release-run",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["steps_completed"] == 2
    assert payload["training_run_package"]["run_id"] == "first-snv-test"
    assert (run_dir / "training_preflight_report.json").is_file()
    assert (run_dir / "training_config.effective.yaml").is_file()
    assert side_preflight.is_file()
    assert (run_dir / "training_run_manifest.json").is_file()
    assert (run_dir / "training_run_card.md").is_file()
    assert (run_dir / "training_run_SHA256SUMS").is_file()
    metadata = json.loads((run_dir / CARBON_TRAINING_METADATA_NAME).read_text(encoding="utf-8"))
    assert "--package-release-run" in metadata["command"]
    assert "--steps" not in metadata["command"]
    assert metadata["training_config"] == "training_config.effective.yaml"
    assert metadata["training_preflight_report"] == "training_preflight_report.json"


@requires_secure_atomic_publication
def test_train_carbon_train_passes_resume_checkpoint_to_launcher(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    carbon_dir = _write_carbon_model_dir(tmp_path)
    config = _write_training_config(tmp_path)
    run_dir = tmp_path / "run"
    resume_path = tmp_path / "resume.pt"
    resume_path.write_bytes(b"checkpoint bytes\n")
    monkeypatch.setattr(training_preflight, "_probe_dependency", _available_dependency)
    monkeypatch.setattr(training_preflight, "_probe_accelerator", _available_accelerator)
    monkeypatch.setattr(train_cli, "run_carbon_training", _fake_carbon_training)

    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--carbon-train",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--carbon-model-dir",
            str(carbon_dir),
            "--training-config",
            str(config),
            "--steps",
            "4",
            "--resume-from",
            str(resume_path),
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["resumed_from_step"] == 1
    assert payload["resume_checkpoint_path"] == str(resume_path)
    metadata = json.loads((run_dir / CARBON_TRAINING_METADATA_NAME).read_text(encoding="utf-8"))
    assert "--resume-from" in metadata["command"]
    assert str(resume_path) in metadata["command"]
    assert metadata["resumed_from_step"] == 1
    assert metadata["resume_checkpoint"] == resume_path.name


@requires_secure_atomic_publication
def test_train_carbon_train_passes_stop_step_without_changing_target_horizon(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    carbon_dir = _write_carbon_model_dir(tmp_path)
    config = _write_training_config(tmp_path)
    run_dir = tmp_path / "run"
    observed: dict[str, object] = {}
    monkeypatch.setattr(training_preflight, "_probe_dependency", _available_dependency)
    monkeypatch.setattr(training_preflight, "_probe_accelerator", _available_accelerator)
    source = SourceProvenance(
        commit_sha="c" * 40,
        tree_sha="d" * 40,
        package_version="0.2.1",
    )
    monkeypatch.setattr(
        train_cli, "resolve_package_source", lambda **_kwargs: source, raising=False
    )

    def capture_training(**kwargs: object) -> CarbonTrainingReport:
        observed.update(kwargs)
        return _fake_carbon_training(**kwargs)

    monkeypatch.setattr(train_cli, "run_carbon_training", capture_training)

    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--carbon-train",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--carbon-model-dir",
            str(carbon_dir),
            "--training-config",
            str(config),
            "--steps",
            "10",
            "--stop-after-step",
            "3",
        ],
    )
    capsys.readouterr()

    assert rc == 0
    assert observed["steps"] == 10
    assert observed["stop_after_step"] == 3
    assert observed["commit_sha"] == source.commit_sha
    assert observed["source_tree"] == source.tree_sha
    metadata = json.loads((run_dir / CARBON_TRAINING_METADATA_NAME).read_text(encoding="utf-8"))
    assert "--steps 10" in metadata["command"]
    assert "--stop-after-step 3" in metadata["command"]


@requires_secure_atomic_publication
@pytest.mark.parametrize("caller_is_git", [False, True])
def test_public_carbon_cli_binds_package_source_outside_its_source_checkout(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    caller_is_git: bool,
) -> None:
    pytest.importorskip("pyarrow")
    source_repo, commit_sha, tree_sha = _write_source_repo(tmp_path / "source")
    caller = tmp_path / "caller"
    caller.mkdir()
    if caller_is_git:
        _git(caller, "init", "--quiet")
        _git(caller, "config", "user.name", "GenoLeWM test")
        _git(caller, "config", "user.email", "test@example.invalid")
        (caller / "unrelated.txt").write_text("unrelated\n", encoding="utf-8")
        _git(caller, "add", "unrelated.txt")
        _git(caller, "commit", "--quiet", "-m", "unrelated")
    dataset_dir = _write_release_dataset(tmp_path)
    carbon_dir = _write_carbon_model_dir(tmp_path)
    config = _write_training_config(tmp_path)
    run_dir = tmp_path / "run"
    observed: dict[str, object] = {}
    monkeypatch.setattr(training_preflight, "_probe_dependency", _available_dependency)
    monkeypatch.setattr(training_preflight, "_probe_accelerator", _available_accelerator)
    monkeypatch.setattr(
        train_cli,
        "resolve_package_source",
        lambda **kwargs: resolve_package_source(
            package_root=source_repo / "geno_lewm",
            **kwargs,
        ),
    )
    monkeypatch.setattr(
        train_cli,
        "run_carbon_training",
        lambda **kwargs: (observed.update(kwargs), _fake_carbon_training(**kwargs))[1],
    )
    monkeypatch.chdir(caller)

    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--carbon-train",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--carbon-model-dir",
            str(carbon_dir),
            "--training-config",
            str(config),
            "--steps",
            "2",
        ],
    )
    capsys.readouterr()

    assert rc == 0
    assert observed["commit_sha"] == commit_sha
    assert observed["source_tree"] == tree_sha


def test_train_rejects_packaging_an_early_stopped_carbon_run(capsys) -> None:
    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--carbon-train",
            "--package-release-run",
            "--steps",
            "10",
            "--stop-after-step",
            "3",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert "cannot be combined" in captured.err


@requires_secure_atomic_publication
def test_train_carbon_train_records_accelerator_preflight_overrides(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    carbon_dir = _write_carbon_model_dir(tmp_path)
    config = _write_training_config(tmp_path)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(training_preflight, "_probe_dependency", _available_dependency)
    monkeypatch.setattr(training_preflight, "_probe_accelerator", _available_accelerator)
    monkeypatch.setattr(train_cli, "run_carbon_training", _fake_carbon_training)

    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--carbon-train",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--carbon-model-dir",
            str(carbon_dir),
            "--training-config",
            str(config),
            "--no-require-accelerator",
            "--min-cuda-vram-gb",
            "24",
        ],
    )
    capsys.readouterr()

    assert rc == 0
    metadata = json.loads((run_dir / CARBON_TRAINING_METADATA_NAME).read_text(encoding="utf-8"))
    assert "--no-require-accelerator" in metadata["command"]
    assert "--min-cuda-vram-gb 24.0" in metadata["command"]


def _fake_carbon_training(
    *,
    config,
    dataset_dir: Path,
    carbon_model_dir: Path,
    run_dir: Path,
    steps: int,
    command: str,
    commit_sha: str,
    source_tree: str,
    package_version: str,
    preflight_report,
    resume_from: Path | None,
    stop_after_step: int | None = None,
) -> CarbonTrainingReport:
    del carbon_model_dir, commit_sha, source_tree, stop_after_step
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "training_config.effective.yaml"
    metrics_path = run_dir / CARBON_METRICS_NAME
    log_path = run_dir / CARBON_LOG_NAME
    checkpoint_path = run_dir / CARBON_CHECKPOINT_NAME
    metadata_path = run_dir / CARBON_TRAINING_METADATA_NAME
    dataset_manifest_path = run_dir / "dataset_manifest.json"
    assert config_path.is_file()
    dataset_manifest_path.write_text(
        (dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(
            {
                "sample_count": 4,
                "metrics": {"train_loss": 0.5},
                "schema_version": "1.0.0",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    log_path.write_text('{"event":"train.end","steps_completed":2}\n', encoding="utf-8")
    checkpoint_path.write_bytes(b"predictor checkpoint bytes\n")
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "run_id": config.run_id,
                "generated_by": TRAINING_RUN_GENERATED_BY,
                "command": command,
                "commit_sha": "abcdef123456",
                "package_version": package_version,
                "dataset_snapshot_id": "geno-lewm-data-v0.1.0-r1",
                "dataset_manifest": dataset_manifest_path.name,
                "training_config": config_path.name,
                "metrics": metrics_path.name,
                "logs": [log_path.name],
                "checkpoint_files": [checkpoint_path.name],
                "training_preflight_report": "training_preflight_report.json",
                "status": "completed",
                "hardware": ["CI CPU"],
                "runtime": ["GenoLeWM Carbon training test boundary"],
                "seeds": {"base": config.seed, "data": config.seed, "predictor": config.seed + 1},
                "determinism": '{"deterministic": true}',
                "monitoring": {"collapse_monitoring": True, "nan_monitoring": True},
                "resumed_from_step": 0 if resume_from is None else 1,
                "resume_checkpoint": None if resume_from is None else resume_from.name,
                "result_summary": "Completed Carbon-backed training boundary with loss 0.5.",
                "limitations": ["Release claims require the paired evaluation report."],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return CarbonTrainingReport(
        run_id=config.run_id,
        run_dir=run_dir,
        dataset_snapshot_id=preflight_report.dataset_snapshot_id or "geno-lewm-data-v0.1.0-r1",
        steps_requested=steps,
        steps_completed=steps,
        resumed_from_step=0 if resume_from is None else 1,
        sample_count=4,
        final_loss=0.5,
        checkpoint_path=checkpoint_path,
        resume_checkpoint_path=resume_from,
        metrics_path=metrics_path,
        log_path=log_path,
        config_path=config_path,
        preflight_path=run_dir / "training_preflight_report.json",
        training_metadata_path=metadata_path,
    )


def _write_source_repo(root: Path) -> tuple[Path, str, str]:
    (root / "geno_lewm" / "cli").mkdir(parents=True)
    (root / "geno_lewm" / "__init__.py").write_text("", encoding="utf-8")
    (root / "geno_lewm" / "cli" / "train.py").write_text("", encoding="utf-8")
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "GenoLeWM test")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "add", "geno_lewm")
    _git(root, "commit", "--quiet", "-m", "source")
    return root, _git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", "HEAD^{tree}")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
