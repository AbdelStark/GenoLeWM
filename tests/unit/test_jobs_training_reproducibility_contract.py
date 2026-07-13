# SPDX-License-Identifier: Apache-2.0
"""Contracts for the issue #47 H200 N-D-D-N job."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

DETERMINISTIC_CONFIG = Path("configs/reproducibility/train-carbon-500m-snv-deterministic-500.yaml")
BASELINE_CONFIG = Path("configs/reproducibility/train-carbon-500m-snv-baseline-500.yaml")
DATASET_REFERENCE = Path("configs/reproducibility/dataset-reference-r2.json")
JOB = Path("tools/jobs/training_reproducibility_run.sh")


def test_reproducibility_configs_differ_only_by_deterministic_flag() -> None:
    deterministic = yaml.safe_load(DETERMINISTIC_CONFIG.read_text(encoding="utf-8"))
    baseline = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))

    expected_baseline = dict(deterministic)
    expected_baseline["deterministic"] = False
    assert baseline == expected_baseline
    assert deterministic["deterministic"] is True
    assert deterministic["training"]["max_steps"] == 500
    assert deterministic["data"]["batch_size"] == 8
    assert deterministic["data"]["num_workers"] == 0
    assert deterministic["data"]["shuffle_buffer"] == 0


def test_reproducibility_dataset_reference_is_immutable_r2_snapshot() -> None:
    reference = json.loads(DATASET_REFERENCE.read_text(encoding="utf-8"))

    assert reference["repo_id"] == "abdelstark/geno-lewm-runs"
    assert reference["repo_type"] == "model"
    assert reference["revision"] == "1200467a6b940cb5b1230d9a7db0be74e51bd50d"
    assert reference["path"] == ("geno-lewm-l2-p1-smoke-304128e4d4f3-50-r2/dataset")
    assert reference["snapshot_id"] == "geno-lewm-data-correction-control-l2-p1-proof-v1"
    assert reference["dataset_manifest_sha256"] == (
        "sha256:8d60360f365185451ebac80cb8c37f8aa4324bb915e16243ac9ce661d6748621"
    )


def test_reproducibility_job_runs_fresh_h200_nddn_arms_and_fail_closed() -> None:
    script = JOB.read_text(encoding="utf-8")

    assert "tools.research.training_reproducibility_preflight" in script
    assert DETERMINISTIC_CONFIG.as_posix() in script
    assert BASELINE_CONFIG.as_posix() in script
    assert DATASET_REFERENCE.as_posix() in script
    assert "1200467a6b940cb5b1230d9a7db0be74e51bd50d" in script
    assert "geno-lewm-l2-p1-smoke-304128e4d4f3-50-r2/dataset" in script
    assert '"H200"' in script
    assert 'MIN_CUDA_VRAM_GB="120"' in script
    assert "sha256:71d27acc26bc809d850e9cd8cf558762c5bd4c1d611e2778c1614c0a8be77b38" in script
    assert "encoder_runtime_hash" in script
    assert '--upload-repo "$UPLOAD_REPO"' in script
    assert '"$WORK/evidence/runtime_preflight.json"' in script
    assert '"carbon_runtime_hash"' in script
    assert '"device_name"' in script
    assert 'cp "$WORK/evidence/runtime_preflight.json" "$WORK/public/evidence/"' in script

    baseline_a = script.index('run_arm "baseline-a"')
    deterministic_a = script.index('run_arm "deterministic-a"')
    deterministic_b = script.index('run_arm "deterministic-b"')
    baseline_b = script.index('run_arm "baseline-b"')
    assert baseline_a < deterministic_a < deterministic_b < baseline_b
    assert script.count("env -u CUBLAS_WORKSPACE_CONFIG") >= 2
    assert 'GENO_LEWM_CACHE="$WORK/cache/$label"' in script
    assert '"$WORK/runs/$label"' in script
    assert '--steps "$STEPS"' in script
    assert "--package-release-run" in script
    assert "--require-preflight" in script
    assert '--max-throughput-drop "$MAX_THROUGHPUT_DROP"' in script
    assert '--max-repeat-spread "$MAX_REPEAT_SPREAD"' in script
    assert '"$RUN_NAME/candidate-negative"' in script
    assert 'candidate_namespace="$candidate_root/$candidate_nonce"' in script
    assert "GENO_LEWM_TRAINING_REPRODUCIBILITY_OK" in script
    assert "trap upload_candidate_on_failure EXIT" in script
    assert script.index("trap upload_candidate_on_failure EXIT") < script.index(
        'CURRENT_STAGE="contract_preflight"'
    )
    assert '"$WORK/public/evidence/failure.json"' in script
    assert '"$RUN_NAME/candidate-negative"' in script
    assert "RUN_PROTECTED=1" in script
    failure_handler = script[
        script.index("upload_candidate_on_failure() {") : script.index(
            "trap upload_candidate_on_failure EXIT"
        )
    ]
    assert "trap - EXIT" in failure_handler
    assert "CURRENT_STAGE" in failure_handler
    assert "completion.json" not in failure_handler
    for stage in (
        "dataset_download",
        "dataset_integrity",
        "arm.$label.preflight",
        "arm.$label.train",
        "arm.$label.checksum",
        "reproducibility_verifier",
    ):
        assert stage in script
    assert "tools.release.atomic_hub_publish" in script
    assert '--bundle-dir "$WORK/public"' in script
    assert '--repo-id "$UPLOAD_REPO"' in script
    assert '--run-name "$RUN_NAME"' in script
    assert '--source-commit-sha "$COMMIT_SHA"' in script
    assert '--verification-dir "$WORK/verified-success"' in script
    assert 'hf upload "$UPLOAD_REPO" "$WORK/public" "$RUN_NAME/success"' not in script
    assert '"$RUN_NAME/success/completion.json"' not in script


def test_reproducibility_job_documents_exact_hf_submission_recipe() -> None:
    script = JOB.read_text(encoding="utf-8")

    for token in (
        "hf jobs run",
        "--flavor h200",
        "--volume hf://models/HuggingFaceBio/Carbon-500M:/carbon:ro",
        "--secrets HF_TOKEN",
        '--env COMMIT_SHA="$SHA"',
        '-- "$IMAGE"',
        'git checkout --detach "$COMMIT_SHA"',
        'test "$(git rev-parse HEAD)" = "$COMMIT_SHA"',
        "uv sync --frozen --extra train",
        "uv run --no-sync bash tools/jobs/training_reproducibility_run.sh",
    ):
        assert token in script


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="HF Jobs executes this Bash integration contract on Linux",
)
def test_exit_trap_publishes_distinct_download_failure_candidates_and_preserves_exit_code(
    tmp_path: Path,
) -> None:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    work = tmp_path / "work"
    carbon = tmp_path / "carbon"
    carbon.mkdir()
    upload_log = tmp_path / "hf-upload.log"
    _write_executable(
        mock_bin / "python",
        """#!/usr/bin/env bash
if [ "${1:-}" = "-m" ]; then
  printf '%s\n' '{"ok": true}'
  exit 0
fi
if [ "${1:-}" = "-" ] && [ "${2:-}" = "120" ]; then
  cat >/dev/null
  mkdir -p "$(dirname "$6")"
  printf '%s\n' '{"ok":true,"carbon_runtime_hash":"sha256:runtime","device_name":"NVIDIA H200"}' > "$6"
  exit 0
fi
exec """
        + shlex.quote(sys.executable)
        + ' "$@"\n',
    )
    _write_executable(
        mock_bin / "curl",
        """#!/usr/bin/env bash
printf '404'
""",
    )
    _write_executable(
        mock_bin / "nvidia-smi",
        """#!/usr/bin/env bash
exit 0
""",
    )
    _write_executable(
        mock_bin / "hf",
        """#!/usr/bin/env bash
if [ "${1:-}" = "download" ]; then
  exit 23
fi
printf '%s\n' "$*" >> "$MOCK_HF_LOG"
exit 0
""",
    )
    env = {
        **os.environ,
        "PATH": f"{mock_bin}:{os.environ['PATH']}",
        "COMMIT_SHA": "a" * 40,
        "HF_TOKEN": "test-token",
        "WORK": str(work),
        "CARBON_DIR": str(carbon),
        "MOCK_HF_LOG": str(upload_log),
    }

    result = subprocess.run(
        ["bash", JOB.as_posix()],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 23
    failure_path = work / "public/evidence/failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["ok"] is False
    assert failure["stage"] == "dataset_download"
    assert failure["exit_code"] == 23
    assert (work / "public/evidence/job_contract_preflight.json").is_file()
    assert (work / "public/evidence/runtime_preflight.json").is_file()
    assert (work / "public/SHA256SUMS").is_file()
    assert not (work / "public/completion.json").exists()
    assert "GENO_LEWM_TRAINING_REPRODUCIBILITY_NOT_PROVEN" in result.stderr

    second_work = tmp_path / "work-second"
    second_result = subprocess.run(
        ["bash", JOB.as_posix()],
        cwd=Path.cwd(),
        env={**env, "WORK": str(second_work)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert second_result.returncode == 23
    upload_lines = upload_log.read_text(encoding="utf-8").splitlines()
    assert len(upload_lines) == 2
    first_tokens, second_tokens = (shlex.split(line) for line in upload_lines)
    assert first_tokens[2] == str(work / "public")
    assert second_tokens[2] == str(second_work / "public")
    destinations = (first_tokens[3], second_tokens[3])
    prefix = "geno-lewm-repro-h200-aaaaaaaaaaaa-500-r1/candidate-negative/"
    assert all(destination.startswith(prefix) for destination in destinations)
    assert all(len(destination.removeprefix(prefix)) == 32 for destination in destinations)
    assert destinations[0] != destinations[1]


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
