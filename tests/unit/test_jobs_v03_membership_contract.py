# SPDX-License-Identifier: Apache-2.0
"""Static contract for the exact-revision v0.3 membership HF Job."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

JOB = Path("tools/jobs/v03_build_membership_store.sh")
DOC = Path("docs/data-v03-membership-hf-job.md")


def test_membership_job_orders_exact_preflight_download_build_verify_and_publish() -> None:
    script = JOB.read_text(encoding="utf-8")

    commit_shape = script.index('[[ "$COMMIT_SHA" =~ ^[0-9a-f]{40}$ ]]')
    image_shape = script.index("@sha256:[0-9a-f]{64}")
    exact_head = script.index('OBSERVED_COMMIT_SHA="$(git rev-parse HEAD)"')
    clean_tree = script.index("git status --porcelain=v1 --untracked-files=all", exact_head)
    external_workspace = script.index(
        "fixed membership workspace must remain outside the checkout", exact_head
    )
    first_work_write = script.index('rm -rf "$WORK"')
    lineage_download = script.index('hf download "$LINEAGE_REPO"')
    download_plan = script.index("tools.data.v03_membership_job author-download-plan")
    gnomad_download = script.index('hf download "$DATA_REPO" "${GNOMAD_PATHS[@]}"')
    clinvar_download = script.index('hf download "$DATA_REPO" "$CLINVAR_PATH"')
    author_spec = script.index("tools.data.v03_membership_job author-spec")
    build = script.index("tools.data.v03_membership_store build")
    verify = script.index("tools.data.v03_membership_store verify")
    checksums = script.index("xargs -0 sha256sum > SHA256SUMS")
    publish = script.index("tools.data.v03_gnomad_lock publish")
    remote_download = script.index('hf download "$UPLOAD_REPO"')
    remote_verify = script.rindex("tools.data.v03_membership_store verify")

    assert (
        commit_shape
        < image_shape
        < exact_head
        < external_workspace
        < clean_tree
        < first_work_write
        < lineage_download
        < download_plan
        < gnomad_download
        < clinvar_download
        < author_spec
        < build
        < verify
        < checksums
        < publish
        < remote_download
        < remote_verify
    )
    assert "set -euo pipefail" in script
    assert 'COMMIT_SHA="${COMMIT_SHA:?COMMIT_SHA is required}"' in script
    assert 'CONTAINER_IMAGE="${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"' in script
    assert 'RUN_ATTEMPT="${RUN_ATTEMPT:?RUN_ATTEMPT is required}"' in script
    assert 'HF_TOKEN="${HF_TOKEN:?HF_TOKEN is required}"' in script
    assert 'WORK="/tmp/geno-lewm-v03-membership"' in script
    assert "${WORK:-" not in script
    assert 'REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"' in script
    assert 'GENO_LEWM_VERIFIED_BUILD_CONTAINER_IMAGE="$CONTAINER_IMAGE"' in script
    assert "git diff --quiet -- ." in script
    assert "git diff --cached --quiet -- ." in script
    assert "--force-download" in script
    assert 'test "${#GNOMAD_PATHS[@]}" -eq 22' in script
    assert '--expected-lineage-sha256 "$LINEAGE_SHA256"' in script
    assert '--expected-lineage-size-bytes "$LINEAGE_SIZE_BYTES"' in script
    assert "--repo-type dataset" in script
    assert '"$PUBLISH_NAMESPACE"' in script
    assert '"$WORK/verified-remote"' in script
    assert "sha256sum -c SHA256SUMS" in script


def test_membership_job_pins_audited_real_data_invariants() -> None:
    script = JOB.read_text(encoding="utf-8")

    for exact_value in (
        "sha256:d268f2e2b67cce56c5d5099ec1ddcbd810fbb5973e6c96a929fd2c99fbd25f68",
        '"row_count": 2_335_042',
        '"variant_count": 2_259_268',
        '"train": 2_251_087',
        '"validation": 53_002',
        '"evaluation": 30_953',
        '"source_kind_filtered_counts": {"gnomad": 0, "clinvar": 2_779_595}',
        '"lineage_evidence_profile": "official"',
        "membership {field} differs from the audited real-data invariant",
    ):
        assert exact_value in script

    assert '"membership": {' in script
    assert '"rowset_sha256"' in script
    assert '"clinvar_class_role_counts"' in script


def test_membership_job_pins_candidate_and_all_three_hub_revisions() -> None:
    script = JOB.read_text(encoding="utf-8")

    for exact_value in (
        "abdelstark/geno-lewm-data",
        "4e5c641d3720a28f28d0d3efb3c5969678e84fe3",
        "candidates/v0.3/geno-lewm-data-v0.3.0-r1/lineage/snapshot-lineage.json",
        "sha256:dcc7031bb1b409e55112c1f6576a878b9566b954d32ea75056a04b9ba1e95bea",
        'LINEAGE_SIZE_BYTES="195040"',
        "f3676763b3f7f71d0d0d098588e9bf377faa0c5c",
        "9e1a2b279681177a7ca00b30b9eb8048b511d1cb",
        "geno-lewm-data-v0.3.0-membership-r1",
    ):
        assert exact_value in script

    assert (
        'RUN_NAME="${RUN_NAME:-geno-lewm-v03-membership-${COMMIT_SHA:0:12}-r${RUN_ATTEMPT}}"'
        in script
    )
    assert (
        'PUBLISH_NAMESPACE="candidates/v0.3/${CANDIDATE_ID}/membership/${RUN_NAME}/success"'
        in script
    )


def test_membership_job_publishes_no_failure_or_partial_namespace() -> None:
    script = JOB.read_text(encoding="utf-8")

    assert script.count("tools.data.v03_gnomad_lock publish") == 1
    assert "candidate-negative" not in script
    assert "failure bundle" not in script
    assert "trap " not in script
    assert "|| true" not in script
    assert "upload_folder(" not in script
    assert "hf upload" not in script
    assert "GENO_LEWM_V03_MEMBERSHIP_OK" in script
    for excluded_claim in (
        "phased haplotypes",
        "released v0.3 snapshot",
        "dataset representativeness",
        "model quality",
        "benchmark performance",
        "clinical validity",
    ):
        assert excluded_claim in script


def test_membership_job_documents_exact_hf_submission_recipe() -> None:
    script = JOB.read_text(encoding="utf-8")

    for token in (
        "hf jobs run",
        "--flavor cpu-upgrade",
        "--secrets HF_TOKEN",
        '--env COMMIT_SHA="$SHA"',
        '--env RUN_ATTEMPT="$RUN_ATTEMPT"',
        '--env CONTAINER_IMAGE="$IMAGE"',
        "--timeout 4h",
        '-- "$IMAGE"',
        "git clone https://github.com/AbdelStark/GenoLeWM.git",
        'git checkout --detach "$COMMIT_SHA"',
        'test "$(git rev-parse HEAD)" = "$COMMIT_SHA"',
        'test -z "$(git status --porcelain=v1 --untracked-files=all)"',
        "uv sync --frozen --extra train",
        "uv run --no-sync bash tools/jobs/v03_build_membership_store.sh",
    ):
        assert token in script


def test_membership_job_operator_guide_preserves_evidence_and_claim_boundaries() -> None:
    guide = DOC.read_text(encoding="utf-8")
    navigation = Path("mkdocs.yml").read_text(encoding="utf-8")

    for token in (
        "tools/jobs/v03_build_membership_store.sh",
        "4e5c641d3720a28f28d0d3efb3c5969678e84fe3",
        "f3676763b3f7f71d0d0d098588e9bf377faa0c5c",
        "9e1a2b279681177a7ca00b30b9eb8048b511d1cb",
        "RUN_ATTEMPT",
        "GENO_LEWM_V03_MEMBERSHIP_OK",
        "exact Hub revision",
        "No failure path uploads",
        "not phased haplotypes",
        "not release the v0.3 snapshot",
    ):
        assert token in guide
    assert "data-v03-membership-hf-job.md" in navigation


@pytest.mark.skipif(os.name == "nt", reason="hosted job runner is a POSIX bash contract")
def test_membership_job_rejects_noncanonical_commit_before_workspace_writes(
    tmp_path: Path,
) -> None:
    work = tmp_path / "must-not-exist"
    result = subprocess.run(
        ["bash", JOB.as_posix()],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "COMMIT_SHA": "not-a-full-commit",
            "CONTAINER_IMAGE": "ghcr.io/astral-sh/uv@sha256:" + "b" * 64,
            "RUN_ATTEMPT": "1",
            "HF_TOKEN": "test-token",
            "WORK": str(work),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "COMMIT_SHA must be a full lowercase 40-character Git SHA" in result.stderr
    assert not work.exists()
