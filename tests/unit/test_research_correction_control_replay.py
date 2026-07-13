"""Tests for correction-control deterministic replay evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.provenance import sha256_file
from geno_lewm.training.preflight import (
    GENERATED_BY as TRAINING_PREFLIGHT_GENERATED_BY,
    REPORT_NAME as TRAINING_PREFLIGHT_REPORT_NAME,
    SCHEMA_VERSION as TRAINING_PREFLIGHT_SCHEMA_VERSION,
)
from tools.release.training_run import (
    GENERATED_BY as TRAINING_RUN_GENERATED_BY,
    REQUIRED_PREFLIGHT_DATASET_CORE_FILES,
    build_training_run_package,
)
from tools.research.correction_control_postflight import (
    GENERATED_BY as CORRECTION_POSTFLIGHT_GENERATED_BY,
    SCHEMA_VERSION as CORRECTION_POSTFLIGHT_SCHEMA_VERSION,
)
from tools.research.correction_control_preflight import (
    GENERATED_BY as CORRECTION_PREFLIGHT_GENERATED_BY,
    SCHEMA_VERSION as CORRECTION_PREFLIGHT_SCHEMA_VERSION,
)
from tools.research.correction_control_replay import (
    CorrectionControlReplayRequest,
    build_correction_control_replay_report,
    main,
)

COMMIT_SHA = "a" * 40
RUN_PREFIX = f"geno-lewm-l2-p1-smoke-{COMMIT_SHA[:12]}-50-r"
RUN_ID = "correction-control-l2-p1-smoke-v1"
SNAPSHOT_ID = "geno-lewm-data-correction-control-l2-p1-proof-v1"


def test_replay_report_accepts_two_completed_bit_exact_runs(tmp_path: Path) -> None:
    reference = _write_completed_run(tmp_path / "reference" / "run", attempt=1)
    candidate = _write_completed_run(tmp_path / "candidate" / "run", attempt=2)

    report = build_correction_control_replay_report(
        CorrectionControlReplayRequest(
            reference_run_dir=reference,
            candidate_run_dir=candidate,
            reference_run_name=f"{RUN_PREFIX}1",
            candidate_run_name=f"{RUN_PREFIX}2",
            expected_commit_sha=COMMIT_SHA,
        ),
        generated_at="2026-07-13T10:00:00Z",
    )

    assert report["ok"] is True
    assert report["scope"] == "deterministic_pair"
    assert report["issue_47_complete"] is False
    assert report["throughput_evaluated"] is False
    assert report["postflights_ok"] is True
    assert report["reference"]["run_name"] == f"{RUN_PREFIX}1"
    assert report["reference"]["run_attempt"] == 1
    assert report["reference"]["correction_control_postflight"]["ok"] is True
    assert report["candidate"]["run_name"] == f"{RUN_PREFIX}2"
    assert report["candidate"]["run_attempt"] == 2
    assert report["candidate"]["correction_control_postflight"]["ok"] is True
    assert report["deterministic_pair"]["ok"] is True
    assert report["blockers"] == []
    assert "does not evaluate deterministic throughput" in report["claim_boundary"]


@pytest.mark.parametrize("failed_label", ["reference", "candidate"])
def test_replay_report_rejects_failed_postflight(
    tmp_path: Path,
    failed_label: str,
) -> None:
    reference = _write_completed_run(
        tmp_path / "reference" / "run",
        attempt=1,
        postflight_ok=failed_label != "reference",
    )
    candidate = _write_completed_run(
        tmp_path / "candidate" / "run",
        attempt=2,
        postflight_ok=failed_label != "candidate",
    )

    report = build_correction_control_replay_report(
        CorrectionControlReplayRequest(
            reference_run_dir=reference,
            candidate_run_dir=candidate,
            reference_run_name=f"{RUN_PREFIX}1",
            candidate_run_name=f"{RUN_PREFIX}2",
            expected_commit_sha=COMMIT_SHA,
        ),
    )

    assert report["ok"] is False
    assert report["postflights_ok"] is False
    assert report["deterministic_pair"]["status"] == "not_evaluated"
    assert [blocker["code"] for blocker in report["blockers"]] == [
        f"{failed_label}.postflight_not_ok"
    ]


def test_replay_report_never_accepts_run_partial(tmp_path: Path) -> None:
    reference = _write_completed_run(tmp_path / "reference" / "run-partial", attempt=1)
    candidate = _write_completed_run(tmp_path / "candidate" / "run", attempt=2)

    report = build_correction_control_replay_report(
        CorrectionControlReplayRequest(
            reference_run_dir=reference,
            candidate_run_dir=candidate,
            reference_run_name=f"{RUN_PREFIX}1",
            candidate_run_name=f"{RUN_PREFIX}2",
            expected_commit_sha=COMMIT_SHA,
        ),
    )

    assert report["ok"] is False
    assert [blocker["code"] for blocker in report["blockers"]] == [
        "reference.incomplete_run_directory"
    ]


def test_replay_report_rejects_checkpoint_mismatch(tmp_path: Path) -> None:
    reference = _write_completed_run(tmp_path / "reference" / "run", attempt=1)
    candidate = _write_completed_run(
        tmp_path / "candidate" / "run",
        attempt=2,
        checkpoint=b"different-checkpoint",
    )

    report = build_correction_control_replay_report(
        CorrectionControlReplayRequest(
            reference_run_dir=reference,
            candidate_run_dir=candidate,
            reference_run_name=f"{RUN_PREFIX}1",
            candidate_run_name=f"{RUN_PREFIX}2",
            expected_commit_sha=COMMIT_SHA,
        ),
    )

    assert report["ok"] is False
    assert report["deterministic_pair"]["ok"] is False
    assert [blocker["code"] for blocker in report["blockers"]] == [
        "deterministic_pair.artifact_mismatch"
    ]


def test_replay_report_rejects_launch_contract_drift(tmp_path: Path) -> None:
    reference = _write_completed_run(tmp_path / "reference" / "run", attempt=1)
    candidate = _write_completed_run(tmp_path / "candidate" / "run", attempt=2)
    preflight_path = candidate / "correction_control" / "job_contract_preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["config"]["sha256"] = "sha256:" + ("9" * 64)
    _write_json(preflight_path, preflight)

    report = build_correction_control_replay_report(
        CorrectionControlReplayRequest(
            reference_run_dir=reference,
            candidate_run_dir=candidate,
            reference_run_name=f"{RUN_PREFIX}1",
            candidate_run_name=f"{RUN_PREFIX}2",
            expected_commit_sha=COMMIT_SHA,
        ),
    )

    assert report["ok"] is False
    assert [blocker["code"] for blocker in report["blockers"]] == ["pair.contract_mismatch"]
    assert report["blockers"][0]["details"] == {"field": "config"}


def test_replay_report_rejects_source_identity_tampering(tmp_path: Path) -> None:
    reference = _write_completed_run(tmp_path / "reference" / "run", attempt=1)
    candidate = _write_completed_run(tmp_path / "candidate" / "run", attempt=2)
    source_path = candidate / "correction_control" / "source_identity_report.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["sources"]["carbon_corpus"]["revision"] = "unvalidated-revision"
    _write_json(source_path, source)

    report = build_correction_control_replay_report(
        CorrectionControlReplayRequest(
            reference_run_dir=reference,
            candidate_run_dir=candidate,
            reference_run_name=f"{RUN_PREFIX}1",
            candidate_run_name=f"{RUN_PREFIX}2",
            expected_commit_sha=COMMIT_SHA,
        ),
    )

    assert report["ok"] is False
    assert [blocker["code"] for blocker in report["blockers"]] == [
        "candidate.source_identity_artifact_mismatch"
    ]


def test_replay_report_requires_valid_training_preflight_archive(tmp_path: Path) -> None:
    reference = _write_completed_run(tmp_path / "reference" / "run", attempt=1)
    candidate = _write_completed_run(tmp_path / "candidate" / "run", attempt=2)
    (candidate / TRAINING_PREFLIGHT_REPORT_NAME).unlink()

    report = build_correction_control_replay_report(
        CorrectionControlReplayRequest(
            reference_run_dir=reference,
            candidate_run_dir=candidate,
            reference_run_name=f"{RUN_PREFIX}1",
            candidate_run_name=f"{RUN_PREFIX}2",
            expected_commit_sha=COMMIT_SHA,
        ),
    )

    assert report["ok"] is False
    assert [blocker["code"] for blocker in report["blockers"]] == [
        "candidate.invalid_training_archive"
    ]


def test_replay_cli_writes_pair_only_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference = _write_completed_run(tmp_path / "reference" / "run", attempt=1)
    candidate = _write_completed_run(tmp_path / "candidate" / "run", attempt=2)
    output = tmp_path / "deterministic_replay_report.json"

    rc = main(
        [
            "--reference-run-dir",
            str(reference),
            "--candidate-run-dir",
            str(candidate),
            "--reference-run-name",
            f"{RUN_PREFIX}1",
            "--candidate-run-name",
            f"{RUN_PREFIX}2",
            "--expected-commit-sha",
            COMMIT_SHA,
            "--output-json",
            str(output),
        ]
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scope"] == "deterministic_pair"
    assert payload["issue_47_complete"] is False
    assert payload["throughput_evaluated"] is False


def test_replay_cli_writes_failed_evidence_and_returns_two(tmp_path: Path) -> None:
    reference = _write_completed_run(
        tmp_path / "reference" / "run",
        attempt=1,
        postflight_ok=False,
    )
    candidate = _write_completed_run(tmp_path / "candidate" / "run", attempt=2)
    output = tmp_path / "deterministic_replay_report.json"

    rc = main(
        [
            "--reference-run-dir",
            str(reference),
            "--candidate-run-dir",
            str(candidate),
            "--reference-run-name",
            f"{RUN_PREFIX}1",
            "--candidate-run-name",
            f"{RUN_PREFIX}2",
            "--expected-commit-sha",
            COMMIT_SHA,
            "--output-json",
            str(output),
        ]
    )

    assert rc == 2
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is False


def _write_completed_run(
    run_dir: Path,
    *,
    attempt: int,
    checkpoint: bytes = b"bit-exact-checkpoint",
    postflight_ok: bool = True,
) -> Path:
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "dataset_manifest.json",
        {"schema_version": "1.0.0", "snapshot_id": SNAPSHOT_ID},
    )
    (run_dir / "training_config.effective.yaml").write_text(
        "deterministic: true\nrun_id: correction-control-l2-p1-smoke-v1\nseed: 104729\n",
        encoding="utf-8",
    )
    _write_json(
        run_dir / "metrics.json",
        {
            "sample_count": 400,
            "metrics": {"samples_per_second": 10.0, "train_loss": 0.1},
        },
    )
    (run_dir / "train.log").write_text("completed 50 steps\n", encoding="utf-8")
    (run_dir / "predictor_checkpoint.pt").write_bytes(checkpoint)
    _write_training_preflight(run_dir)
    metadata = {
        "schema_version": "1.0.0",
        "run_id": RUN_ID,
        "generated_by": TRAINING_RUN_GENERATED_BY,
        "generated_at": "2026-07-13T09:00:00Z",
        "command": "geno-lewm-train --carbon-train --steps 50",
        "commit_sha": COMMIT_SHA,
        "package_version": "0.2.1",
        "dataset_snapshot_id": SNAPSHOT_ID,
        "dataset_manifest": "dataset_manifest.json",
        "training_config": "training_config.effective.yaml",
        "metrics": "metrics.json",
        "training_preflight_report": TRAINING_PREFLIGHT_REPORT_NAME,
        "logs": ["train.log"],
        "checkpoint_files": ["predictor_checkpoint.pt"],
        "status": "completed",
        "hardware": ["Linux x86_64", "Python 3.13.11"],
        "runtime": ["torch==2.12.0", "transformers==4.57.6"],
        "seeds": {"base": 104729, "data": 104729, "predictor": 104730, "lora": 104731},
        "determinism": json.dumps(
            {
                "cublas_workspace_config": ":4096:8",
                "deterministic": True,
                "seed": 104730,
                "torch_deterministic_algorithms": True,
            },
            sort_keys=True,
        ),
        "monitoring": {"collapse_monitoring": True, "nan_monitoring": True},
        "result_summary": "Completed correction-control fixture replay run.",
        "limitations": ["Fixture archive for deterministic replay tool tests only."],
    }
    metadata_path = run_dir / "training_run.json"
    _write_json(metadata_path, metadata)
    build_training_run_package(run_dir, metadata_path)

    correction_dir = run_dir / "correction_control"
    correction_dir.mkdir()
    run_name = f"{RUN_PREFIX}{attempt}"
    _write_json(
        correction_dir / "job_contract_preflight.json",
        {
            "schema_version": CORRECTION_PREFLIGHT_SCHEMA_VERSION,
            "generated_by": CORRECTION_PREFLIGHT_GENERATED_BY,
            "generated_at": f"2026-07-13T09:0{attempt}:00Z",
            "ok": True,
            "repository": {
                "root": ".",
                "expected_commit_sha": COMMIT_SHA,
                "observed_commit_sha": COMMIT_SHA,
                "observed_git_root": ".",
                "worktree_clean": True,
                "dirty_paths": [],
            },
            "job": {
                "run_name": run_name,
                "run_attempt": attempt,
                "steps": 50,
                "container_image": "ghcr.io/astral-sh/uv@sha256:" + ("1" * 64),
                "carbon_model_dir": "/carbon",
            },
            "config": {
                "path": "configs/correction_control/train.yaml",
                "sha256": "sha256:" + ("2" * 64),
                "size_bytes": 100,
            },
            "snapshot": {
                "path": "configs/correction_control/snapshot.json",
                "sha256": "sha256:" + ("3" * 64),
                "size_bytes": 100,
            },
            "issues": [],
        },
    )
    source_identity_path = correction_dir / "source_identity_report.json"
    _write_json(
        source_identity_path,
        {
            "schema_version": "1.0.0",
            "generated_by": "tools.jobs.proof_run.source_identity",
            "generated_at": f"2026-07-13T09:0{attempt}:30Z",
            "ok": True,
            "commit_sha": COMMIT_SHA,
            "run_name": run_name,
            "dataset_snapshot_id": SNAPSHOT_ID,
            "training_contract": {
                "active_window_source": "carbon",
                "window_bp": 4096,
                "action_sub_encoders": ["snv"],
            },
            "sources": {"carbon_corpus": {"revision": "cb4c13a78102933b3a6ac65734d326f7b431d9b7"}},
        },
    )
    _write_json(
        correction_dir / "correction_control_postflight.json",
        {
            "schema_version": CORRECTION_POSTFLIGHT_SCHEMA_VERSION,
            "generated_by": CORRECTION_POSTFLIGHT_GENERATED_BY,
            "generated_at": f"2026-07-13T09:1{attempt}:00Z",
            "ok": postflight_ok,
            "expected": {
                "commit_sha": COMMIT_SHA,
                "run_id": RUN_ID,
                "dataset_snapshot_id": SNAPSHOT_ID,
                "steps_completed": 50,
                "sample_count": 400,
                "phase": "phase1",
                "state_contract_version": "l2_normalized_v2",
                "encoder_runtime_hash": "sha256:" + ("4" * 64),
            },
            "artifacts": {
                "source_identity_report": {
                    "path": source_identity_path.name,
                    "exists": True,
                    "sha256": sha256_file(source_identity_path),
                    "size_bytes": source_identity_path.stat().st_size,
                }
            },
            "blockers": [] if postflight_ok else [{"code": "fixture.failed"}],
        },
    )
    return run_dir


def _write_training_preflight(run_dir: Path) -> None:
    config = run_dir / "training_config.effective.yaml"
    _write_json(
        run_dir / TRAINING_PREFLIGHT_REPORT_NAME,
        {
            "schema_version": TRAINING_PREFLIGHT_SCHEMA_VERSION,
            "generated_by": TRAINING_PREFLIGHT_GENERATED_BY,
            "generated_at": "2026-07-13T09:00:00Z",
            "ok": True,
            "dataset_snapshot_id": SNAPSHOT_ID,
            "training_config": {
                "path": config.name,
                "sha256": sha256_file(config),
                "size_bytes": config.stat().st_size,
                "top_level_keys": ["deterministic", "run_id", "seed"],
                "resolved": {"run_id": RUN_ID},
            },
            "run_dir": {
                "path": "run",
                "exists": True,
                "preflight_report_path": TRAINING_PREFLIGHT_REPORT_NAME,
            },
            "dataset": {
                "path": "dataset",
                "snapshot_id": SNAPSHOT_ID,
                "core_files": {
                    relative: {
                        "path": relative,
                        "sha256": "sha256:" + ("5" * 64),
                        "size_bytes": 1,
                    }
                    for relative in REQUIRED_PREFLIGHT_DATASET_CORE_FILES
                },
                "files": [
                    {
                        "path": "carbon/windows.jsonl",
                        "split": "train",
                        "records": 512,
                        "sha256": "sha256:" + ("6" * 64),
                        "size_bytes": 1,
                    }
                ],
                "splits": {"train": {"records": 512}},
            },
            "carbon": {
                "path": "carbon",
                "local_files_only": True,
                "artifacts": {},
            },
            "dependencies": [],
            "issues": [],
        },
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
