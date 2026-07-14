# SPDX-License-Identifier: Apache-2.0
"""Run and externally verify production Carbon resume equivalence.

The evidence in this module is deliberately software-scoped.  It binds three
real ``geno-lewm-train`` processes and their raw checkpoints to an immutable
Git commit/tree and an explicit ``N``/``K`` continuation contract.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from geno_lewm.errors import InputError, RuntimeSetupError
from geno_lewm.provenance import canonical_json_sha256, sha256_file
from geno_lewm.training.resume import load_resume_checkpoint

REPORT_NAME: Final = "production_resume_equivalence.json"
SCHEMA_VERSION: Final = "geno-lewm.production-resume-equivalence.v1"
GENERATED_BY: Final = "tools.research.production_resume_equivalence"
_ARMS: Final = ("uninterrupted", "prefix", "resumed")
_RAW_FILES: Final[dict[str, str]] = {
    "checkpoint": "predictor_checkpoint.pt",
    "metrics": "metrics.json",
    "training_log": "train.log",
    "training_metadata": "training_run.json",
}
_REPORT_KEYS: Final = frozenset(
    {
        "schema_version",
        "generated_by",
        "expected",
        "repository",
        "claim_scope",
        "processes",
        "artifacts",
        "comparison",
        "report_digest",
    }
)
_PROCESS_KEYS: Final = frozenset({"pid", "returncode", "argv", "stdout", "stderr"})
_IDENTITY_KEYS: Final = frozenset({"path", "sha256", "size_bytes"})
_SHA_RE: Final = re.compile(r"[0-9a-f]{40}")
_CLAIM_SCOPE: Final[dict[str, object]] = {
    "software_only": True,
    "establishes": [
        "single-process production checkpoint continuation is bit-equal for this fixture",
        "model, action encoder, AdamW, trainer monitor, RNG, cursor, metrics, and LR state close",
    ],
    "does_not_establish": [
        "accelerator or distributed resume equivalence",
        "hardware-independent floating-point equivalence",
        "model quality, biological utility, or clinical validity",
    ],
}


def run_production_resume_equivalence(
    *,
    repo_root: Path,
    output_dir: Path,
    dataset_dir: Path,
    carbon_model_dir: Path,
    training_config: Path,
    expected_source_commit: str,
    expected_source_tree: str,
    total_steps: int,
    split_step: int,
    train_executable: str = "geno-lewm-train",
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run uninterrupted, prefix, and fresh-process resumed production arms."""
    _validate_expected_contract(
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        total_steps=total_steps,
        split_step=split_step,
    )
    repo_root = repo_root.resolve()
    observed_commit, observed_tree = _git_identity(repo_root)
    _require_expected_repository(
        observed_commit,
        observed_tree,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
    )
    output_dir = output_dir.resolve()
    if output_dir == repo_root or repo_root in output_dir.parents:
        raise InputError(
            "production resume evidence output must be outside the source worktree",
            remediation="choose an empty output directory beside the repository",
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise InputError("production resume evidence output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = shutil.which(train_executable)
    if executable is None:
        raise RuntimeSetupError(
            "geno-lewm-train executable is unavailable",
            details={"executable": train_executable},
        )

    run_dirs = {arm: output_dir / arm for arm in _ARMS}
    common = [
        executable,
        "--quiet",
        "--no-banner",
        "--no-receipt",
        "--carbon-train",
        "--dataset-dir",
        str(dataset_dir.resolve()),
        "--carbon-model-dir",
        str(carbon_model_dir.resolve()),
        "--training-config",
        str(training_config.resolve()),
        "--steps",
        str(total_steps),
        "--allow-fixture-dataset",
        "--no-require-accelerator",
    ]
    commands = {
        "uninterrupted": [*common, "--run-dir", str(run_dirs["uninterrupted"])],
        "prefix": [
            *common,
            "--run-dir",
            str(run_dirs["prefix"]),
            "--stop-after-step",
            str(split_step),
        ],
        "resumed": [
            *common,
            "--run-dir",
            str(run_dirs["resumed"]),
            "--resume-from",
            str(run_dirs["prefix"] / _RAW_FILES["checkpoint"]),
        ],
    }
    process_environment = dict(os.environ)
    process_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "GENO_LEWM_CACHE": str(output_dir / "cache"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if environment is not None:
        process_environment.update(environment)
    inherited_pythonpath = process_environment.get("PYTHONPATH")
    process_environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(repo_root), inherited_pythonpath) if item
    )
    processes: dict[str, dict[str, object]] = {}
    for arm in _ARMS:
        processes[arm] = _run_cli_process(
            commands[arm],
            cwd=repo_root,
            stdout_path=output_dir / f"{arm}.stdout.jsonl",
            stderr_path=output_dir / f"{arm}.stderr.log",
            environment=process_environment,
        )
    return collect_production_resume_equivalence(
        report_path=output_dir / REPORT_NAME,
        repo_root=repo_root,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        total_steps=total_steps,
        split_step=split_step,
        run_dirs=run_dirs,
        processes=processes,
    )


def collect_production_resume_equivalence(
    *,
    report_path: Path,
    repo_root: Path,
    expected_source_commit: str,
    expected_source_tree: str,
    total_steps: int,
    split_step: int,
    run_dirs: Mapping[str, Path],
    processes: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Bind completed process records and raw production artifacts into one report."""
    _validate_expected_contract(
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        total_steps=total_steps,
        split_step=split_step,
    )
    observed_commit, observed_tree = _git_identity(repo_root.resolve())
    _require_expected_repository(
        observed_commit,
        observed_tree,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
    )
    if set(run_dirs) != set(_ARMS) or set(processes) != set(_ARMS):
        raise InputError("production resume evidence requires exactly three named arms")
    report_path = report_path.resolve()
    report_root = report_path.parent
    normalized_processes = _normalize_processes(processes, report_root=report_root)
    artifacts = {
        arm: {
            name: _artifact_identity(
                Path(run_dirs[arm]).resolve() / filename,
                root=report_root,
            )
            for name, filename in _RAW_FILES.items()
        }
        for arm in _ARMS
    }
    comparison = _compare_raw_runs(
        {arm: Path(run_dirs[arm]).resolve() for arm in _ARMS},
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        total_steps=total_steps,
        split_step=split_step,
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "expected": {
            "source_commit": expected_source_commit,
            "source_tree": expected_source_tree,
            "total_steps": total_steps,
            "split_step": split_step,
        },
        "repository": {
            "commit_sha": observed_commit,
            "tree_sha": observed_tree,
            "clean": True,
        },
        "claim_scope": dict(_CLAIM_SCOPE),
        "processes": normalized_processes,
        "artifacts": artifacts,
        "comparison": comparison,
    }
    report["report_digest"] = _report_digest(report)
    _write_json_atomic(report_path, report)
    return verify_production_resume_equivalence(
        report_path,
        repo_root=repo_root,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        total_steps=total_steps,
        split_step=split_step,
    )


def verify_production_resume_equivalence(
    report_path: Path,
    *,
    repo_root: Path,
    expected_source_commit: str,
    expected_source_tree: str,
    total_steps: int,
    split_step: int,
) -> dict[str, Any]:
    """Recompute a report from raw artifacts and explicit external expectations."""
    _validate_expected_contract(
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        total_steps=total_steps,
        split_step=split_step,
    )
    observed_commit, observed_tree = _git_identity(repo_root.resolve())
    _require_expected_repository(
        observed_commit,
        observed_tree,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
    )
    report_path = report_path.resolve()
    report = _load_json_object(report_path)
    if set(report) != _REPORT_KEYS:
        raise InputError("production resume report fields do not match the closed contract")
    if report.get("schema_version") != SCHEMA_VERSION or report.get("generated_by") != GENERATED_BY:
        raise InputError("production resume report schema identity is unsupported")
    if report.get("report_digest") != _report_digest(report):
        raise InputError("production resume report digest does not match its payload")
    expected = {
        "source_commit": expected_source_commit,
        "source_tree": expected_source_tree,
        "total_steps": total_steps,
        "split_step": split_step,
    }
    if report.get("expected") != expected:
        raise InputError("production resume report does not match external expectations")
    if report.get("repository") != {
        "commit_sha": observed_commit,
        "tree_sha": observed_tree,
        "clean": True,
    }:
        raise InputError("production resume report repository identity does not match")
    if report.get("claim_scope") != _CLAIM_SCOPE:
        raise InputError("production resume report claim scope is not the software-only contract")

    processes = _mapping(report.get("processes"), "report.processes")
    if set(processes) != set(_ARMS):
        raise InputError("production resume report process set is incomplete")
    pids: list[int] = []
    for arm in _ARMS:
        process = _mapping(processes[arm], f"processes.{arm}")
        if set(process) != _PROCESS_KEYS:
            raise InputError("production resume process fields do not match the closed contract")
        pid = _positive_int(process.get("pid"), f"processes.{arm}.pid")
        pids.append(pid)
        if process.get("returncode") != 0:
            raise InputError("production resume process did not exit successfully")
        argv = process.get("argv")
        if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
            raise InputError("production resume process argv must be a list of strings")
        _verify_process_command(arm, argv, total_steps=total_steps, split_step=split_step)
        _verify_artifact_identity(process.get("stdout"), root=report_path.parent)
        _verify_artifact_identity(process.get("stderr"), root=report_path.parent)
    if len(set(pids)) != len(_ARMS):
        raise InputError("production resume arms must execute in distinct processes")

    artifacts = _mapping(report.get("artifacts"), "report.artifacts")
    if set(artifacts) != set(_ARMS):
        raise InputError("production resume raw artifact set is incomplete")
    run_dirs: dict[str, Path] = {}
    for arm in _ARMS:
        arm_artifacts = _mapping(artifacts[arm], f"artifacts.{arm}")
        if set(arm_artifacts) != set(_RAW_FILES):
            raise InputError("production resume arm artifact set is incomplete")
        resolved = {
            name: _verify_artifact_identity(identity, root=report_path.parent)
            for name, identity in arm_artifacts.items()
        }
        parents = {path.parent for path in resolved.values()}
        if len(parents) != 1:
            raise InputError("production resume arm artifacts must share one run directory")
        run_dirs[arm] = parents.pop()
    for arm in _ARMS:
        process = _mapping(processes[arm], f"processes.{arm}")
        argv = process["argv"]
        assert isinstance(argv, list)
        if Path(_arg_text(argv, "--run-dir")).resolve() != run_dirs[arm]:
            raise InputError("production resume process run directory does not bind raw artifacts")
    resumed_process = _mapping(processes["resumed"], "processes.resumed")
    resumed_argv = resumed_process["argv"]
    assert isinstance(resumed_argv, list)
    expected_prefix_checkpoint = run_dirs["prefix"] / _RAW_FILES["checkpoint"]
    if Path(_arg_text(resumed_argv, "--resume-from")).resolve() != expected_prefix_checkpoint:
        raise InputError("resumed process command does not bind the raw prefix checkpoint")
    comparison = _compare_raw_runs(
        run_dirs,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        total_steps=total_steps,
        split_step=split_step,
    )
    if report.get("comparison") != comparison or comparison.get("passed") is not True:
        raise InputError("production resume comparison does not match raw artifacts")
    return report


def _compare_raw_runs(
    run_dirs: Mapping[str, Path],
    *,
    expected_source_commit: str,
    expected_source_tree: str,
    total_steps: int,
    split_step: int,
) -> dict[str, object]:
    checkpoints = {
        arm: load_resume_checkpoint(run_dirs[arm] / _RAW_FILES["checkpoint"]) for arm in _ARMS
    }
    source = {"commit_sha": expected_source_commit, "tree_sha": expected_source_tree}
    for arm, checkpoint in checkpoints.items():
        if checkpoint.get("source") != source:
            raise InputError(
                "production checkpoint source identity does not match expectations",
                details={"arm": arm},
            )
    uninterrupted = checkpoints["uninterrupted"]
    prefix = checkpoints["prefix"]
    resumed = checkpoints["resumed"]
    contract = uninterrupted.get("training_contract")
    if not isinstance(contract, dict) or contract.get("target_steps") != total_steps:
        raise InputError("production checkpoint target horizon does not match expected N")
    if prefix.get("training_contract") != contract or resumed.get("training_contract") != contract:
        raise InputError("production checkpoint training contracts diverge")
    if prefix.get("identities") != uninterrupted.get("identities") or resumed.get(
        "identities"
    ) != uninterrupted.get("identities"):
        raise InputError("production checkpoint data or encoder identities diverge")
    batch_size = _positive_int(contract.get("batch_size"), "training_contract.batch_size")
    _verify_checkpoint_progress(
        uninterrupted,
        expected_steps=total_steps,
        expected_samples=total_steps * batch_size,
    )
    _verify_checkpoint_progress(
        prefix,
        expected_steps=split_step,
        expected_samples=split_step * batch_size,
    )
    _verify_checkpoint_progress(
        resumed,
        expected_steps=total_steps,
        expected_samples=total_steps * batch_size,
    )
    full_history = uninterrupted.get("metric_history")
    prefix_history = prefix.get("metric_history")
    resumed_history = resumed.get("metric_history")
    if not isinstance(full_history, list) or len(full_history) != total_steps:
        raise InputError("uninterrupted checkpoint metric history does not span N")
    if prefix_history != full_history[:split_step]:
        raise InputError("prefix checkpoint metric history does not match steps 1..K")
    if resumed_history != full_history:
        raise InputError("resumed checkpoint metric history is not bit-equal")
    full_progress = _mapping(uninterrupted.get("progress"), "uninterrupted.progress")
    prefix_progress = _mapping(prefix.get("progress"), "prefix.progress")
    resumed_progress = _mapping(resumed.get("progress"), "resumed.progress")
    full_order = full_progress.get("consumed_window_ids")
    prefix_order = prefix_progress.get("consumed_window_ids")
    if not isinstance(full_order, list) or not isinstance(prefix_order, list):
        raise InputError("production checkpoint cursor order must be a list")
    if prefix_order != full_order[: split_step * batch_size]:
        raise InputError("prefix checkpoint cursor order does not match steps 1..K")
    if resumed_progress != full_progress:
        raise InputError("resumed checkpoint progress and cursor are not bit-equal")
    if resumed.get("state_digests") != uninterrupted.get("state_digests"):
        raise InputError("resumed model/action/AdamW/trainer state is not bit-equal")
    if resumed.get("rng_state_digests") != uninterrupted.get("rng_state_digests"):
        raise InputError("resumed Python/NumPy/Torch RNG state is not bit-equal")
    if resumed.get("payload_digest") != uninterrupted.get("payload_digest"):
        raise InputError("resumed final checkpoint payload is not bit-equal")

    metrics = {arm: _load_json_object(run_dirs[arm] / _RAW_FILES["metrics"]) for arm in _ARMS}
    if metrics["uninterrupted"].get("history") != full_history:
        raise InputError("uninterrupted public metrics do not bind checkpoint history")
    if metrics["prefix"].get("history") != prefix_history:
        raise InputError("prefix public metrics do not bind checkpoint history")
    if metrics["resumed"].get("history") != full_history:
        raise InputError("resumed public metrics do not bind cumulative checkpoint history")
    if metrics["prefix"].get("steps_completed") != split_step:
        raise InputError("prefix public metrics do not report K")
    if metrics["resumed"].get("resumed_from_step") != split_step:
        raise InputError("resumed public metrics do not report continuation from K")

    metadata = {
        arm: _load_json_object(run_dirs[arm] / _RAW_FILES["training_metadata"]) for arm in _ARMS
    }
    expected_completed = {
        "uninterrupted": total_steps,
        "prefix": split_step,
        "resumed": total_steps,
    }
    for arm, payload in metadata.items():
        if (
            payload.get("target_steps") != total_steps
            or payload.get("steps_completed") != expected_completed[arm]
        ):
            raise InputError("training metadata does not bind the expected N/K contract")
    if metadata["prefix"].get("status") != "stopped_early":
        raise InputError("prefix training metadata must not claim completion")
    if metadata["resumed"].get("resumed_from_step") != split_step:
        raise InputError("resumed training metadata does not bind K")
    return {
        "passed": True,
        "final_state_digests_equal": True,
        "final_rng_digests_equal": True,
        "final_payload_digest_equal": True,
        "metric_history_equal": True,
        "cursor_order_equal": True,
        "uninterrupted_payload_digest": uninterrupted["payload_digest"],
        "prefix_payload_digest": prefix["payload_digest"],
        "resumed_payload_digest": resumed["payload_digest"],
    }


def _verify_checkpoint_progress(
    checkpoint: Mapping[str, object],
    *,
    expected_steps: int,
    expected_samples: int,
) -> None:
    progress = _mapping(checkpoint.get("progress"), "checkpoint.progress")
    if progress.get("steps_completed") != expected_steps:
        raise InputError("production checkpoint step cursor does not match N/K")
    if progress.get("samples_consumed") != expected_samples:
        raise InputError("production checkpoint sample cursor does not match N/K")
    order = progress.get("consumed_window_ids")
    if not isinstance(order, list) or len(order) != expected_samples:
        raise InputError("production checkpoint sample order length does not match N/K")


def _normalize_processes(
    processes: Mapping[str, Mapping[str, object]],
    *,
    report_root: Path,
) -> dict[str, dict[str, object]]:
    normalized: dict[str, dict[str, object]] = {}
    pids: list[int] = []
    for arm in _ARMS:
        process = processes[arm]
        pid = _positive_int(process.get("pid"), f"processes.{arm}.pid")
        pids.append(pid)
        returncode = process.get("returncode")
        if returncode != 0:
            raise InputError(
                "production resume process failed",
                details={"arm": arm, "returncode": returncode},
            )
        argv = process.get("argv")
        if (
            not isinstance(argv, Sequence)
            or isinstance(argv, str | bytes)
            or any(not isinstance(item, str) for item in argv)
        ):
            raise InputError("production resume process argv must contain strings")
        argv_list = list(argv)
        _verify_process_command(
            arm, argv_list, total_steps=_arg_int(argv_list, "--steps"), split_step=None
        )
        stdout_path = process.get("stdout_path")
        stderr_path = process.get("stderr_path")
        if not isinstance(stdout_path, Path) or not isinstance(stderr_path, Path):
            raise InputError("production resume process stream paths must be pathlib Paths")
        normalized[arm] = {
            "pid": pid,
            "returncode": 0,
            "argv": argv_list,
            "stdout": _artifact_identity(stdout_path.resolve(), root=report_root),
            "stderr": _artifact_identity(stderr_path.resolve(), root=report_root),
        }
    if len(set(pids)) != len(_ARMS):
        raise InputError("production resume arms must execute in distinct processes")
    return normalized


def _verify_process_command(
    arm: str,
    argv: Sequence[str],
    *,
    total_steps: int,
    split_step: int | None,
) -> None:
    if not argv or Path(argv[0]).name != "geno-lewm-train":
        raise InputError("production resume evidence must invoke the public console script")
    if "--carbon-train" not in argv or _arg_int(argv, "--steps") != total_steps:
        raise InputError("production resume process command does not bind target N")
    if arm == "prefix":
        observed_k = _arg_int(argv, "--stop-after-step")
        if split_step is not None and observed_k != split_step:
            raise InputError("prefix process command does not bind split K")
    elif "--stop-after-step" in argv:
        raise InputError("only the prefix process may stop at K")
    if arm == "resumed":
        if "--resume-from" not in argv:
            raise InputError("resumed process command is missing its checkpoint")
    elif "--resume-from" in argv:
        raise InputError("only the resumed process may consume a checkpoint")


def _run_cli_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    environment: Mapping[str, str],
) -> dict[str, object]:
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
        )
        pid = process.pid
        returncode = process.wait()
    if returncode != 0:
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeSetupError(
            "production resume training arm failed",
            details={"pid": pid, "returncode": returncode, "stderr_tail": tail},
        )
    return {
        "pid": pid,
        "returncode": returncode,
        "argv": list(argv),
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
    }


def _git_identity(repo_root: Path) -> tuple[str, str]:
    top_level = Path(_git(repo_root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != repo_root.resolve():
        raise InputError("production resume repo root must be the Git top-level directory")
    for source_path in ("geno_lewm/__init__.py", "geno_lewm/cli/train.py"):
        if not (repo_root / source_path).is_file():
            raise InputError(
                "production resume repo does not contain the expected training package",
                details={"path": source_path},
            )
        _git(repo_root, "ls-files", "--error-unmatch", source_path)
    commit = _git(repo_root, "rev-parse", "HEAD")
    tree = _git(repo_root, "rev-parse", "HEAD^{tree}")
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise InputError(
            "source checkout must be clean before production resume evidence",
            details={"dirty_paths": status.splitlines()},
        )
    return commit, tree


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InputError("failed to resolve production resume Git identity") from exc
    return result.stdout.strip()


def _require_expected_repository(
    observed_commit: str,
    observed_tree: str,
    *,
    expected_source_commit: str,
    expected_source_tree: str,
) -> None:
    if observed_commit != expected_source_commit:
        raise InputError("live source commit does not match expected COMMIT")
    if observed_tree != expected_source_tree:
        raise InputError("live source tree does not match expected TREE")


def _validate_expected_contract(
    *,
    expected_source_commit: str,
    expected_source_tree: str,
    total_steps: int,
    split_step: int,
) -> None:
    if _SHA_RE.fullmatch(expected_source_commit) is None:
        raise InputError("expected COMMIT must be a full lowercase Git SHA")
    if _SHA_RE.fullmatch(expected_source_tree) is None:
        raise InputError("expected TREE must be a full lowercase Git SHA")
    _positive_int(total_steps, "total_steps")
    _positive_int(split_step, "split_step")
    if split_step >= total_steps:
        raise InputError("expected K must be less than N")


def _artifact_identity(path: Path, *, root: Path) -> dict[str, object]:
    if not path.is_file():
        raise InputError("production resume raw artifact is missing", details={"path": str(path)})
    try:
        relative = path.relative_to(root.resolve())
    except ValueError as exc:
        raise InputError(
            "production resume artifacts must stay inside the evidence directory"
        ) from exc
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_artifact_identity(value: object, *, root: Path) -> Path:
    identity = _mapping(value, "artifact identity")
    if set(identity) != _IDENTITY_KEYS:
        raise InputError("production resume artifact identity fields are not closed")
    relative = identity.get("path")
    if not isinstance(relative, str) or not relative:
        raise InputError("production resume artifact path must be non-empty text")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise InputError("production resume artifact path must stay inside evidence")
    path = (root / candidate).resolve()
    if root.resolve() not in path.parents:
        raise InputError("production resume artifact path escapes evidence")
    observed = _artifact_identity(path, root=root)
    if identity != observed:
        raise InputError("production resume raw artifact digest or size does not match")
    return path


def _report_digest(report: Mapping[str, object]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_digest"}
    )


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError("production resume JSON artifact is missing or invalid") from exc
    if not isinstance(payload, dict):
        raise InputError("production resume JSON artifact must be an object")
    return payload


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{label} must be a mapping")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(f"{label} must be a positive integer")
    return value


def _arg_int(argv: Sequence[str], option: str) -> int:
    raw = _arg_text(argv, option)
    try:
        value = int(raw)
    except ValueError as exc:
        raise InputError(f"production resume command is missing {option}") from exc
    return _positive_int(value, option)


def _arg_text(argv: Sequence[str], option: str) -> str:
    try:
        index = argv.index(option)
        return argv[index + 1]
    except (ValueError, IndexError) as exc:
        raise InputError(f"production resume command is missing {option}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--repo-root", type=Path, required=True)
        command.add_argument("--expected-source-commit", required=True)
        command.add_argument("--expected-source-tree", required=True)
        command.add_argument("--total-steps", type=int, required=True)
        command.add_argument("--split-step", type=int, required=True)
        if name == "run":
            command.add_argument("--output-dir", type=Path, required=True)
            command.add_argument("--dataset-dir", type=Path, required=True)
            command.add_argument("--carbon-model-dir", type=Path, required=True)
            command.add_argument("--training-config", type=Path, required=True)
            command.add_argument("--train-executable", default="geno-lewm-train")
        else:
            command.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "run":
            report = run_production_resume_equivalence(
                repo_root=args.repo_root,
                output_dir=args.output_dir,
                dataset_dir=args.dataset_dir,
                carbon_model_dir=args.carbon_model_dir,
                training_config=args.training_config,
                expected_source_commit=args.expected_source_commit,
                expected_source_tree=args.expected_source_tree,
                total_steps=args.total_steps,
                split_step=args.split_step,
                train_executable=args.train_executable,
            )
        else:
            report = verify_production_resume_equivalence(
                args.report,
                repo_root=args.repo_root,
                expected_source_commit=args.expected_source_commit,
                expected_source_tree=args.expected_source_tree,
                total_steps=args.total_steps,
                split_step=args.split_step,
            )
    except (InputError, RuntimeSetupError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI.
    raise SystemExit(main())
