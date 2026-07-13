# SPDX-License-Identifier: Apache-2.0
"""Static contracts for the generation-pinned gnomAD v0.3 staging job."""

from __future__ import annotations

from pathlib import Path

STAGE_JOB = Path("tools/jobs/v03_stage_gnomad.sh")


def test_v03_gnomad_job_orders_preflight_transform_receipt_and_upload() -> None:
    script = STAGE_JOB.read_text(encoding="utf-8")

    exact_sha = script.index('OBSERVED_COMMIT_SHA="$(git rev-parse HEAD)"')
    lock_blob = script.index('git cat-file -e "$COMMIT_SHA:$SOURCE_LOCK"')
    select_lock = script.index("python -m tools.data.v03_gnomad_lock select")
    first_collision_check = script.index("python -m tools.data.v03_gnomad_lock probe-namespace")
    first_work_write = script.index('rm -rf "$WORK"')
    metadata_fetch = script.index('curl "$METADATA_URL"')
    metadata_verify = script.index("python -m tools.data.v03_gnomad_lock verify-metadata")
    source_fetch = script.index('curl "$MEDIA_URL"')
    source_hash = script.index("python -m tools.data.v03_gnomad_lock hash-source")
    transform = script.index("uv run geno-lewm-prepare-gnomad")
    receipt = script.index("python -m tools.data.v03_gnomad_lock author-receipt")
    upload = script.index("python -m tools.data.v03_gnomad_lock publish")

    assert (
        exact_sha
        < lock_blob
        < select_lock
        < first_collision_check
        < first_work_write
        < metadata_fetch
        < metadata_verify
        < source_fetch
        < source_hash
        < transform
        < receipt
        < upload
    )
    assert "set -euo pipefail" in script
    assert 'COMMIT_SHA="${COMMIT_SHA:?COMMIT_SHA is required}"' in script
    assert 'CHROMOSOME="${CHROMOSOME:?CHROMOSOME is required}"' in script
    assert 'CONTAINER_IMAGE="${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"' in script
    assert (
        'SOURCE_LOCK="${SOURCE_LOCK:-configs/data_v03/'
        'gnomad-v4.1-exomes-autosomes.source-lock.json}"'
    ) in script
    assert "git diff --quiet -- ." in script
    assert "git diff --cached --quiet -- ." in script
    assert 'EXPECTED_LOCK_BLOB="$(git rev-parse "$COMMIT_SHA:$SOURCE_LOCK")"' in script
    assert 'OBSERVED_LOCK_BLOB="$(git hash-object "$SOURCE_LOCK")"' in script
    assert (
        'SOURCE_LOCK_SCHEMA="$(json_field "$PREFLIGHT_SELECTION" source_lock.schema.path)"'
        in script
    )
    assert 'git cat-file -e "$COMMIT_SHA:$SOURCE_LOCK_SCHEMA"' in script
    assert 'cp "$SOURCE_LOCK_SCHEMA" "$EVIDENCE_DIR/source-lock.schema.json"' in script
    assert '--commit-message "stage gnomAD $RELEASE chr$CHROMOSOME at $COMMIT_SHA"' in script


def test_v03_gnomad_job_treats_namespace_and_claim_scope_as_hard_boundaries() -> None:
    script = STAGE_JOB.read_text(encoding="utf-8")

    assert script.count("python -m tools.data.v03_gnomad_lock publish") == 1
    assert script.count("python -m tools.data.v03_gnomad_lock probe-namespace") == 1
    assert "upload_folder(" not in script
    assert "delete_patterns" not in script
    assert "run-partial" not in script
    assert "|| true" not in script
    for excluded_claim in (
        "snapshot membership",
        "split leakage control",
        "dataset representativeness",
        "model quality",
        "benchmark performance",
        "clinical validity",
    ):
        assert excluded_claim in script


def test_v03_gnomad_job_documents_exact_hf_submission_recipe() -> None:
    script = STAGE_JOB.read_text(encoding="utf-8")

    for token in (
        "hf jobs run",
        "--flavor cpu-upgrade",
        "--secrets HF_TOKEN",
        '--env COMMIT_SHA="$SHA"',
        '--env CHROMOSOME="$CHROMOSOME"',
        '--env CONTAINER_IMAGE="$IMAGE"',
        "--timeout 8h",
        '-- "$IMAGE"',
        'git checkout --detach "$COMMIT_SHA"',
        'test "$(git rev-parse HEAD)" = "$COMMIT_SHA"',
        "uv sync --frozen --extra train",
        "uv run --no-sync bash tools/jobs/v03_stage_gnomad.sh",
    ):
        assert token in script
