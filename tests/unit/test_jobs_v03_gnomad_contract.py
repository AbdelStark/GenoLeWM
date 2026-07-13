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
    first_collision_check = script.index("check_remote_namespace_absent")
    first_work_write = script.index('rm -rf "$WORK"')
    metadata_fetch = script.index('curl "$METADATA_URL"')
    metadata_verify = script.index("python -m tools.data.v03_gnomad_lock verify-metadata")
    source_fetch = script.index('curl "$MEDIA_URL"')
    source_hash = script.index("python -m tools.data.v03_gnomad_lock hash-source")
    transform = script.index("uv run geno-lewm-prepare-gnomad")
    receipt = script.index("python -m tools.data.v03_gnomad_lock author-receipt")
    final_collision_check = script.rindex("check_remote_namespace_absent")
    upload = script.index("api.upload_folder(")

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
        < final_collision_check
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
    assert 'REMOTE_PARENT_COMMIT="$(check_remote_namespace_absent)"' in script
    assert "parent_commit=parent_commit" in script


def test_v03_gnomad_job_treats_namespace_and_claim_scope_as_hard_boundaries() -> None:
    script = STAGE_JOB.read_text(encoding="utf-8")

    assert script.count("api.upload_folder(") == 1
    assert "if exc.code == 404:" in script
    assert "cannot prove remote namespace absence" in script
    assert "immutable namespace already exists" in script
    assert script.count("check_remote_namespace_absent") == 3  # definition plus two gates
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
