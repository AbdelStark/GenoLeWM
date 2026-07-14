"""Tests for externally verified production resume-equivalence evidence."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from geno_lewm.training.resume import capture_rng_state, write_resume_checkpoint
from tools.research import production_resume_equivalence as resume_equivalence

_COMMIT = "a" * 40
_TREE = "b" * 40
_N = 4
_K = 2
_BATCH_SIZE = 2


def test_collect_and_verify_binds_expected_contract_to_raw_production_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dirs, processes = _write_run_artifacts(tmp_path)
    monkeypatch.setattr(
        resume_equivalence,
        "_git_identity",
        lambda _repo: (_COMMIT, _TREE),
    )
    report_path = tmp_path / "production_resume_equivalence.json"

    report = resume_equivalence.collect_production_resume_equivalence(
        report_path=report_path,
        repo_root=tmp_path / "repo",
        expected_source_commit=_COMMIT,
        expected_source_tree=_TREE,
        total_steps=_N,
        split_step=_K,
        run_dirs=run_dirs,
        processes=processes,
    )
    verified = resume_equivalence.verify_production_resume_equivalence(
        report_path,
        repo_root=tmp_path / "repo",
        expected_source_commit=_COMMIT,
        expected_source_tree=_TREE,
        total_steps=_N,
        split_step=_K,
    )

    assert report["claim_scope"]["software_only"] is True
    assert report["comparison"]["passed"] is True
    assert verified == report
    assert report["expected"] == {
        "source_commit": _COMMIT,
        "source_tree": _TREE,
        "total_steps": _N,
        "split_step": _K,
    }
    assert len({process["pid"] for process in report["processes"].values()}) == 3


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"expected_source_commit": "f" * 40}, "expected COMMIT"),
        ({"expected_source_tree": "e" * 40}, "expected TREE"),
        ({"total_steps": _N + 1}, "external expectations"),
        ({"split_step": _K - 1}, "external expectations"),
    ],
)
def test_verifier_requires_external_commit_tree_n_and_k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, object],
    message: str,
) -> None:
    report_path = _collect_fixture_report(tmp_path, monkeypatch)
    expected: dict[str, object] = {
        "repo_root": tmp_path / "repo",
        "expected_source_commit": _COMMIT,
        "expected_source_tree": _TREE,
        "total_steps": _N,
        "split_step": _K,
    }
    expected.update(override)

    with pytest.raises(InputError, match=message):
        resume_equivalence.verify_production_resume_equivalence(
            report_path,
            **expected,
        )


def test_verifier_rejects_raw_checkpoint_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = _collect_fixture_report(tmp_path, monkeypatch)
    with (tmp_path / "resumed" / "predictor_checkpoint.pt").open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(InputError, match="digest or size"):
        resume_equivalence.verify_production_resume_equivalence(
            report_path,
            repo_root=tmp_path / "repo",
            expected_source_commit=_COMMIT,
            expected_source_tree=_TREE,
            total_steps=_N,
            split_step=_K,
        )


def test_collector_rejects_reused_process_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dirs, processes = _write_run_artifacts(tmp_path)
    processes["resumed"]["pid"] = processes["prefix"]["pid"]
    monkeypatch.setattr(
        resume_equivalence,
        "_git_identity",
        lambda _repo: (_COMMIT, _TREE),
    )

    with pytest.raises(InputError, match="distinct processes"):
        resume_equivalence.collect_production_resume_equivalence(
            report_path=tmp_path / "production_resume_equivalence.json",
            repo_root=tmp_path / "repo",
            expected_source_commit=_COMMIT,
            expected_source_tree=_TREE,
            total_steps=_N,
            split_step=_K,
            run_dirs=run_dirs,
            processes=processes,
        )


def test_git_identity_rejects_dirty_tracked_or_untracked_source(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "geno_lewm" / "cli").mkdir(parents=True)
    (repo / "geno_lewm" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "geno_lewm" / "cli" / "train.py").write_text("", encoding="utf-8")
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "GenoLeWM test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt", "geno_lewm")
    _git(repo, "commit", "--quiet", "-m", "fixture")
    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(InputError, match="must be clean"):
        resume_equivalence._git_identity(repo)


def test_git_identity_rejects_unrelated_clean_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "GenoLeWM test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "fixture.txt").write_text("not GenoLeWM\n", encoding="utf-8")
    _git(repo, "add", "fixture.txt")
    _git(repo, "commit", "--quiet", "-m", "fixture")

    with pytest.raises(InputError, match="expected training package"):
        resume_equivalence._git_identity(repo)


def _collect_fixture_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    run_dirs, processes = _write_run_artifacts(tmp_path)
    monkeypatch.setattr(
        resume_equivalence,
        "_git_identity",
        lambda _repo: (_COMMIT, _TREE),
    )
    report_path = tmp_path / "production_resume_equivalence.json"
    resume_equivalence.collect_production_resume_equivalence(
        report_path=report_path,
        repo_root=tmp_path / "repo",
        expected_source_commit=_COMMIT,
        expected_source_tree=_TREE,
        total_steps=_N,
        split_step=_K,
        run_dirs=run_dirs,
        processes=processes,
    )
    return report_path


def _write_run_artifacts(
    root: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, object]]]:
    torch = pytest.importorskip("torch")
    rng_state = capture_rng_state()
    contract = {
        "target_steps": _N,
        "batch_size": _BATCH_SIZE,
        "config": {"seed": 17},
        "seeds": {"data": 17, "predictor": 18, "lora": 19},
    }
    identities = {
        "dataset_snapshot_id": "fixture",
        "dataset_manifest": "sha256:" + ("c" * 64),
        "encoder": "sha256:" + ("d" * 64),
        "membership_and_split": None,
    }
    history = [
        {
            "step": step,
            "lr_multiplier": float(step) / _N,
            "loss": 1.0 / step,
            "pred_loss": 1.0 / step,
            "kl_reg": 0.0,
            "action_count": 2,
            "pred_var_per_dim": 0.5,
        }
        for step in range(1, _N + 1)
    ]
    order = [f"window-{index}" for index in range(_N * _BATCH_SIZE)]
    final_states = {
        "predictor": {"weight": torch.tensor([4.0])},
        "action_encoder": {"weight": torch.tensor([5.0])},
        "optimizer": {"state": {0: {"step": torch.tensor(4.0)}}, "param_groups": []},
    }
    prefix_states = {
        "predictor": {"weight": torch.tensor([2.0])},
        "action_encoder": {"weight": torch.tensor([3.0])},
        "optimizer": {"state": {0: {"step": torch.tensor(2.0)}}, "param_groups": []},
    }
    run_dirs: dict[str, Path] = {}
    processes: dict[str, dict[str, object]] = {}
    for index, arm in enumerate(("uninterrupted", "prefix", "resumed"), start=1):
        run_dir = root / arm
        run_dir.mkdir()
        run_dirs[arm] = run_dir
        completed = _K if arm == "prefix" else _N
        resumed_from = _K if arm == "resumed" else 0
        write_resume_checkpoint(
            run_dir / "predictor_checkpoint.pt",
            source={"commit_sha": _COMMIT, "tree_sha": _TREE},
            training_contract=contract,
            identities=identities,
            progress={
                "steps_completed": completed,
                "samples_consumed": completed * _BATCH_SIZE,
                "consumed_window_ids": order[: completed * _BATCH_SIZE],
                "collapse_alert_count": 0,
            },
            states=prefix_states if arm == "prefix" else final_states,
            trainer_state={
                "schema_version": "fixture",
                "total_steps": _N,
                "monitor_step": completed,
            },
            rng_state=rng_state,
            metric_history=history[:completed],
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "steps_completed": completed,
                    "resumed_from_step": resumed_from,
                    "history": history[:completed],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "train.log").write_text(
            json.dumps({"event": "train.end", "steps_completed": completed}) + "\n",
            encoding="utf-8",
        )
        (run_dir / "training_run.json").write_text(
            json.dumps(
                {
                    "status": "stopped_early" if arm == "prefix" else "completed",
                    "target_steps": _N,
                    "steps_completed": completed,
                    "resumed_from_step": resumed_from,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        stdout = root / f"{arm}.stdout"
        stderr = root / f"{arm}.stderr"
        stdout.write_text("{}\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        argv = [
            "geno-lewm-train",
            "--carbon-train",
            "--steps",
            str(_N),
            "--run-dir",
            str(run_dir),
        ]
        if arm == "prefix":
            argv.extend(["--stop-after-step", str(_K)])
        if arm == "resumed":
            argv.extend(["--resume-from", str(run_dirs["prefix"] / "predictor_checkpoint.pt")])
        processes[arm] = {
            "pid": 1000 + index,
            "returncode": 0,
            "argv": argv,
            "stdout_path": stdout,
            "stderr_path": stderr,
        }
    return run_dirs, processes


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
