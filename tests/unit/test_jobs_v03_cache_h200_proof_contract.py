# SPDX-License-Identifier: Apache-2.0
"""Static contracts for the exact-trace H200 cache interruption proof job."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROOF_JOB = Path("tools/jobs/v03_cache_h200_proof.sh")
PUBLIC_TRACE_PATH = (
    "training-traces/v0.3/"
    "geno-lewm-v03-training-trace-48b5bf71397f-712d612d85ea-"
    "job-6a55f38e85d9643ce16d29e7-r1/success"
)
pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="H200 proof job-control contracts require a POSIX bash runtime",
)


def _script() -> str:
    return PROOF_JOB.read_text(encoding="utf-8")


def test_job_requires_exact_public_inputs_and_a_clean_canonical_checkout() -> None:
    script = _script()

    for required_environment in (
        'COMMIT_SHA="${COMMIT_SHA:?COMMIT_SHA is required}"',
        'CONTAINER_IMAGE="${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"',
        'TRACE_REPOSITORY="${TRACE_REPOSITORY:?TRACE_REPOSITORY is required}"',
        'TRACE_REVISION="${TRACE_REVISION:?TRACE_REVISION is required}"',
        'TRACE_ARTIFACT_PATH="${TRACE_ARTIFACT_PATH:?TRACE_ARTIFACT_PATH is required}"',
        'RUN_ATTEMPT="${RUN_ATTEMPT:?RUN_ATTEMPT is required}"',
        'HF_TOKEN="${HF_TOKEN:?HF_TOKEN is required}"',
    ):
        assert required_environment in script
    assert 'export GENO_LEWM_CACHE_H200_PROOF_DECLARED_CONTAINER_IMAGE="$CONTAINER_IMAGE"' in script

    assert "set -euo pipefail" in script
    assert '[[ "$COMMIT_SHA" =~ ^[0-9a-f]{40}$ ]]' in script
    assert '[[ "$TRACE_REVISION" =~ ^[0-9a-f]{40}$ ]]' in script
    assert 'EXPECTED_TRACE_REVISION="da0d86cde7bf88de2015ab7c516f356e9ae89469"' in script
    assert 'test "$TRACE_REVISION" = "$EXPECTED_TRACE_REVISION"' in script
    assert f'EXPECTED_TRACE_ARTIFACT_PATH="{PUBLIC_TRACE_PATH}"' in script
    assert 'test "$TRACE_ARTIFACT_PATH" = "$EXPECTED_TRACE_ARTIFACT_PATH"' in script
    assert 'test "$TRACE_REPOSITORY" = "abdelstark/geno-lewm-data"' in script
    assert '[[ "$CONTAINER_IMAGE" =~ ^[^@[:space:]]+@sha256:[0-9a-f]{64}$ ]]' in script
    assert 'test "$(git rev-parse HEAD)" = "$COMMIT_SHA"' in script
    assert 'test -z "$(git status --porcelain=v1 --untracked-files=all)"' in script
    assert (
        'test "$(git remote get-url origin)" = '
        '"https://github.com/AbdelStark/GenoLeWM.git"' in script
    )
    assert "https://api.github.com/repos/AbdelStark/GenoLeWM/commits/$COMMIT_SHA" in script


def test_job_accepts_public_training_trace_namespace_and_rejects_candidates() -> None:
    script = _script()
    start = script.index("validate_trace_artifact_path() {")
    end = script.index("\n}\n", start) + len("\n}\n")
    function_body = script[start:end]
    program = (
        'fatal() { echo "FATAL: $*" >&2; exit 2; }\n'
        + function_body
        + '\nvalidate_trace_artifact_path "$1"'
    )

    accepted = subprocess.run(
        ["bash", "-c", program, "trace-test", PUBLIC_TRACE_PATH],
        capture_output=True,
        text=True,
        check=False,
    )
    rejected = subprocess.run(
        ["bash", "-c", program, "trace-test", "candidates/v0.3/fixture/success"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert accepted.returncode == 0, accepted.stderr
    assert rejected.returncode == 2
    assert "training-trace success namespace" in rejected.stderr


def test_job_pins_runtime_schema_carbon_and_cache_plan_arguments() -> None:
    script = _script()

    assert 'WORK_ROOT="/work/geno-lewm-v03-cache-h200-proof"' in script
    assert 'CARBON_DIR="/carbon"' in script
    assert 'RUNTIME_IDENTITY="configs/data_v03/carbon-500m-l2-runtime-identity.json"' in script
    assert 'PROOF_SCHEMA="configs/data_v03/cache-h200-proof.schema.json"' in script
    assert 'TRAINING_CONFIG="configs/data_v03/train-carbon-500m-snv-l2-epoch-r1.yaml"' in script
    assert 'CREATED_AT_NS="1783987200000000000"' in script
    assert 'BATCH_SIZE="8"' in script
    assert 'ROWS_PER_SHARD="256"' in script
    workspace_initialization = script[
        script.index("mkdir -p") : script.index('log "force-download')
    ]
    for directory in (
        '"$DOWNLOAD_ROOT"',
        '"$BUNDLE_DIR/cache"',
        '"$BUNDLE_DIR/evidence"',
        '"$BUNDLE_DIR/trace"',
        '"$BUNDLE_DIR/proof"',
        '"$LOG_ROOT"',
        '"$RUNTIME_ROOT"',
    ):
        assert directory in workspace_initialization
    for argument in (
        '"--cache-dir" "$BUNDLE_DIR/cache"',
        '"--requests-jsonl" "$BUNDLE_DIR/trace/cache_build_requests.jsonl"',
        '"--evidence-dir" "$BUNDLE_DIR/evidence"',
        '"--encoder-runtime-identity" "$RUNTIME_IDENTITY"',
        '"--carbon-model-dir" "$CARBON_DIR"',
        '"--config" "$TRAINING_CONFIG"',
        '"--created-at-ns" "$CREATED_AT_NS"',
        '"--batch-size" "$BATCH_SIZE"',
        '"--rows-per-shard" "$ROWS_PER_SHARD"',
        '"--device" "cuda"',
        '"--hardware" "$HARDWARE"',
    ):
        assert argument in script


def test_namespace_absence_and_trace_preflight_precede_cache_work() -> None:
    script = _script()

    first_absence_probe = script.index("python -m tools.data.v03_gnomad_lock probe-namespace")
    fixed_workspace_check = script.index('test ! -e "$WORK_ROOT"')
    first_workspace_write = script.index("mkdir -p")
    exact_trace_download = script.index('hf download "$TRACE_REPOSITORY"')
    trace_copy = script.index('cp -a "$DOWNLOADED_TRACE/." "$BUNDLE_DIR/trace/"')
    proof_preflight = script.index("python -m tools.research.v03_cache_h200_proof preflight")
    cache_launch = script.index('GENO_LEWM_RUN_ID="$ATTEMPT1_RUN_ID"')

    assert (
        first_absence_probe
        < fixed_workspace_check
        < first_workspace_write
        < exact_trace_download
        < trace_copy
        < proof_preflight
        < cache_launch
    )
    work_directory_initialization = script.index('ensure_physical_work_directory "/work"')
    assert first_absence_probe < work_directory_initialization < fixed_workspace_check
    preflight_block = script[proof_preflight:cache_launch]
    for flag in (
        "--bundle-dir",
        "--trace-dir",
        "--trace-repository",
        "--trace-revision",
        "--trace-artifact-path",
        "--runtime-identity",
        "--source-commit",
        "--container-image",
    ):
        assert flag in preflight_block


def test_work_directory_initializer_creates_a_missing_physical_directory(tmp_path: Path) -> None:
    script = _script()
    start = script.index("ensure_physical_work_directory() {")
    end = script.index("\n}\n", start) + len("\n}\n")
    function_body = script[start:end]
    work = tmp_path.resolve() / "missing-work"

    subprocess.run(
        [
            "bash",
            "-c",
            'fatal() { echo "FATAL: $*" >&2; exit 2; }\n'
            + function_body
            + '\nensure_physical_work_directory "$1"',
            "work-test",
            str(work),
        ],
        check=True,
    )

    assert work.is_dir()
    assert not work.is_symlink()


def test_attempt_is_stopped_captured_then_terminated_with_signal_status() -> None:
    script = _script()

    launch = script.index('GENO_LEWM_RUN_ID="$ATTEMPT1_RUN_ID"')
    background_pid = script.index("CACHE_PID=$!", launch)
    durable_poll = script.index('completed_shards "$CACHE_BUILD_STATE"', background_pid)
    stopped = script.index('kill -STOP "$CACHE_PID"', durable_poll)
    stopped_state = script.index('wait_until_stopped "$CACHE_PID"', stopped)
    capture = script.index("python -m tools.research.v03_cache_h200_proof capture-partial")
    terminated = script.index('kill -TERM "$CACHE_PID"', capture)
    continued = script.index('kill -CONT "$CACHE_PID"', terminated)
    waited = script.index('wait "$CACHE_PID"', continued)
    status_gate = script.index('[ "$ATTEMPT1_RC" -eq 143 ]', waited)
    finalized = script.index(
        "python -m tools.research.v03_cache_h200_proof finalize-interruption",
        status_gate,
    )
    resume = script.index('GENO_LEWM_RUN_ID="$RESUME_RUN_ID"', finalized)

    assert (
        launch
        < background_pid
        < durable_poll
        < stopped
        < stopped_state
        < capture
        < terminated
        < continued
        < waited
        < status_gate
        < finalized
        < resume
    )
    capture_block = script[capture:terminated]
    assert '"--bundle-dir" "$BUNDLE_DIR"' in capture_block
    assert '"--attempt-log" "$ATTEMPT1_LOG"' in capture_block
    assert '"--stopped-pid" "$CACHE_PID"' in capture_block
    finalize_block = script[finalized:resume]
    assert '"--bundle-dir" "$BUNDLE_DIR"' in finalize_block
    assert '"--attempt-log" "$ATTEMPT1_LOG"' in finalize_block
    assert '"--attempt-exit-code" "$ATTEMPT1_RC"' in finalize_block
    assert 'MIN_DURABLE_SHARDS="2"' in script


def test_resume_reuses_identical_argv_with_distinct_external_logs() -> None:
    script = _script()

    assert "CACHE_ARGV=(\n  geno-lewm-cache-windows" in script
    assert "CACHE_ARGV=(\n  uv run" not in script
    assert script.count('"${CACHE_ARGV[@]}"') == 2
    assert 'ATTEMPT1_RUN_ID="${RUN_NAME}-attempt1"' in script
    assert 'RESUME_RUN_ID="${RUN_NAME}-resume"' in script
    assert 'ATTEMPT1_LOG="$LOG_ROOT/${ATTEMPT1_RUN_ID}.jsonl"' in script
    assert 'RESUME_LOG="$LOG_ROOT/${RESUME_RUN_ID}.jsonl"' in script
    assert 'GENO_LEWM_RUN_ID="$ATTEMPT1_RUN_ID"' in script
    assert 'GENO_LEWM_RUN_ID="$RESUME_RUN_ID"' in script
    assert '"--log-dir" "$LOG_ROOT"' in script
    assert '"--run-id"' not in script


def test_hardware_receipt_is_closed_measured_h200_evidence() -> None:
    script = _script()

    assert (
        "nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version "
        "--format=csv,noheader,nounits" in script
    )
    assert '"schema_version": "geno-lewm.v03-h200-hardware.v1"' in script
    assert '"generated_by": "tools.jobs.v03_cache_h200_proof"' in script
    assert '"source_commit_sha": source_commit' in script
    assert '"container_image": container_image' in script
    assert '"nvidia_smi_query_raw": raw_query' in script
    assert '"name": properties.name' in script
    assert '"total_memory_bytes": properties.total_memory' in script
    assert '"compute_capability": compute_capability' in script
    assert '"python_version": platform.python_version()' in script
    assert '"torch_version": torch.__version__' in script
    assert '"cuda_version": torch.version.cuda' in script
    assert '"driver_version": driver_version' in script
    assert 'if "H200" not in properties.name:' in script


def test_author_verify_publish_and_exact_revision_replay_are_ordered() -> None:
    script = _script()

    resume = script.index('GENO_LEWM_RUN_ID="$RESUME_RUN_ID"')
    retire_lock = script.index(
        "python -m tools.research.v03_cache_h200_proof retire-runtime-lock",
        resume,
    )
    author = script.index("python -m tools.research.v03_cache_h200_proof author")
    local_verify = script.index(
        "python -m tools.research.v03_cache_h200_proof verify-existing", author
    )
    final_absence_probe = script.index(
        "python -m tools.data.v03_gnomad_lock probe-namespace", local_verify
    )
    publish = script.index("python -m tools.data.v03_gnomad_lock publish", final_absence_probe)
    exact_revision_download = script.index('hf download "$UPLOAD_REPOSITORY"', publish)
    remote_verify = script.index(
        "python -m tools.research.v03_cache_h200_proof verify-existing",
        exact_revision_download,
    )

    assert (
        resume
        < retire_lock
        < author
        < local_verify
        < final_absence_probe
        < publish
        < exact_revision_download
        < remote_verify
    )
    author_block = script[author:local_verify]
    for flag in (
        "--bundle-dir",
        "--trace-repository",
        "--trace-revision",
        "--trace-artifact-path",
        "--runtime-identity",
        "--source-commit",
        "--container-image",
        "--hardware-json",
        "--resume-log",
    ):
        assert flag in author_block
    assert '[[ "$HUB_REVISION" =~ ^[0-9a-f]{40}$ ]]' in script
    assert '"--revision" "$HUB_REVISION"' in script
    assert '"--include" "$PUBLISH_NAMESPACE/**"' in script
    assert 'cmp "$BUNDLE_DIR/SHA256SUMS" "$REMOTE_BUNDLE/SHA256SUMS"' in script


def test_job_header_documents_digest_pinned_h200_launch() -> None:
    script = _script()

    assert "hf jobs run" in script
    assert "host `hf` v1.8.0 or newer" in script
    assert "--namespace abdelstark" in script
    assert "--flavor h200" in script
    assert "--volume hf://models/HuggingFaceBio/Carbon-500M:/carbon:ro" in script
    assert "--secrets HF_TOKEN" in script
    assert 'test "$(git rev-parse HEAD)" = "$COMMIT_SHA"' in script
    assert "uv sync --frozen --extra train --extra evidence" in script
    assert "exec uv run --no-sync bash tools/jobs/v03_cache_h200_proof.sh" in script
    workspace_guard = script.index("test ! -L /workspace")
    workspace_create = script.index("mkdir /workspace", workspace_guard)
    workspace_physical = script.index('test "$(cd /workspace && pwd -P)" = /workspace')
    clone = script.index("git clone https://github.com/AbdelStark/GenoLeWM.git /workspace/GenoLeWM")
    assert workspace_guard < workspace_create < workspace_physical < clone


@pytest.mark.skipif(os.name == "nt", reason="POSIX job-control proof requires signals")
def test_direct_child_supervisor_freezes_bytes_and_observes_shell_rc143(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat.bin"
    child_pid_path = tmp_path / "child.pid"
    status_path = tmp_path / "status.txt"
    child_program = """\
import os
import sys
import time

with open(sys.argv[1], "ab", buffering=0) as stream:
    while True:
        stream.write(b"x")
        os.fsync(stream.fileno())
        time.sleep(0.01)
"""
    supervisor_program = """\
set +e
"$1" -c "$2" "$3" &
child_pid=$!
printf '%s\n' "$child_pid" > "$4"
wait "$child_pid"
rc=$?
printf '%s\n' "$rc" > "$5"
exit 0
"""
    supervisor = subprocess.Popen(
        [
            "bash",
            "-c",
            supervisor_program,
            "supervisor",
            sys.executable,
            child_program,
            str(heartbeat),
            str(child_pid_path),
            str(status_path),
        ]
    )
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if child_pid_path.is_file() and heartbeat.is_file() and heartbeat.stat().st_size >= 3:
                child_pid = int(child_pid_path.read_text(encoding="ascii").strip())
                break
            time.sleep(0.01)
        assert child_pid is not None, "direct child did not begin writing heartbeats"

        os.kill(child_pid, signal.SIGSTOP)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(child_pid)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if "T" in state or "t" in state:
                break
            time.sleep(0.01)
        else:
            pytest.fail("direct child never entered a stopped process state")

        frozen_size = heartbeat.stat().st_size
        time.sleep(0.2)
        assert heartbeat.stat().st_size == frozen_size

        os.kill(child_pid, signal.SIGTERM)
        os.kill(child_pid, signal.SIGCONT)
        assert supervisor.wait(timeout=5.0) == 0
        assert status_path.read_text(encoding="ascii").strip() == "143"
    finally:
        if supervisor.poll() is None:
            if child_pid is not None:
                try:
                    os.kill(child_pid, signal.SIGTERM)
                    os.kill(child_pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
            supervisor.kill()
            supervisor.wait(timeout=5.0)
