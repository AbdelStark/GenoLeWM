# SPDX-License-Identifier: Apache-2.0
"""Static contract for the exact-revision v0.3 membership-split HF Job."""

import subprocess
from pathlib import Path

JOB = Path("tools/jobs/v03_publish_membership_splits.sh")


def test_split_job_uses_the_minimal_locked_evidence_dependency_contract() -> None:
    script = JOB.read_text(encoding="utf-8")
    project = Path("pyproject.toml").read_text(encoding="utf-8")

    assert (
        'evidence = [\n    "huggingface-hub>=0.36,<1",\n'
        '    "jsonschema>=4",\n'
        "    # Keep the store verifier on the typed PyArrow line used by development CI.\n"
        '    "pyarrow>=15,<25",\n]\n'
    ) in project
    assert "uv sync --frozen --extra evidence" in script
    assert "uv sync --frozen --extra dev" not in script
    assert "uv sync --frozen --extra train" not in script
    assert "command -v hf" in script
    for required_import in ("import huggingface_hub", "import jsonschema", "import pyarrow"):
        assert required_import in script


def test_split_job_orders_preflight_exact_inputs_export_verify_publish_and_reverify() -> None:
    script = JOB.read_text(encoding="utf-8")

    commit_shape = script.index('[[ "$COMMIT_SHA" =~ ^[0-9a-f]{40}$ ]]')
    image_shape = script.index("@sha256:[0-9a-f]{64}")
    exact_head = script.index('OBSERVED_COMMIT_SHA="$(git rev-parse HEAD)"')
    clean_tree = script.index("git status --porcelain=v1 --untracked-files=all", exact_head)
    first_work_write = script.index('rm -rf "$WORK"')
    membership_download = script.index('hf download "$DATA_REPO"', first_work_write)
    membership_checksums = script.index("sha256sum -c SHA256SUMS", membership_download)
    store_verify = script.index("tools.data.v03_membership_store verify", membership_checksums)
    training_download = script.index('hf download "$DATA_REPO"', store_verify)
    input_verify = script.index("verify the exact dataset manifest and placed-window bytes")
    export = script.index("tools.data.v03_membership_splits")
    output_verify = script.index("independently validate the complete split evidence bundle")
    publish = script.index("tools.data.v03_gnomad_lock publish")
    remote_download = script.index('hf download "$UPLOAD_REPO"', publish)
    remote_verify = script.index("reverify the exact published split bundle", remote_download)

    assert (
        commit_shape
        < image_shape
        < exact_head
        < clean_tree
        < first_work_write
        < membership_download
        < membership_checksums
        < store_verify
        < training_download
        < input_verify
        < export
        < output_verify
        < publish
        < remote_download
        < remote_verify
    )
    assert "set -euo pipefail" in script
    assert 'COMMIT_SHA="${COMMIT_SHA:?COMMIT_SHA is required}"' in script
    assert 'CONTAINER_IMAGE="${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"' in script
    assert 'RUN_ATTEMPT="${RUN_ATTEMPT:?RUN_ATTEMPT is required}"' in script
    assert 'HF_TOKEN="${HF_TOKEN:?HF_TOKEN is required}"' in script
    assert 'WORK="/tmp/geno-lewm-v03-membership-splits"' in script
    assert "${WORK:-" not in script
    assert 'GENO_LEWM_VERIFIED_SPLIT_CONTAINER_IMAGE="$CONTAINER_IMAGE"' in script
    assert "git diff --quiet -- ." in script
    assert "git diff --cached --quiet -- ." in script
    assert "--force-download" in script
    assert "sha256sum -c SHA256SUMS" in script


def test_split_job_pins_exact_membership_and_placed_window_inputs() -> None:
    script = JOB.read_text(encoding="utf-8")

    for exact_value in (
        "96e97a7ffe1e9ad8f9a98f690b220a32ac75ddc2",
        "candidates/v0.3/geno-lewm-data-v0.3.0-r1/membership/"
        "geno-lewm-v03-membership-fd7f4bbde476-r1/success",
        "geno-lewm-data-v0.3.0-membership-r1",
        "sha256:7fa661eefacf70258b8392aff88a6faea2749c812680d4a2bfc41376d061ff7a",
        "sha256:d7ea2c4b8413768c9128c70a299a11f4adf35140102778a71cf56e69fb4db536",
        "sha256:d268f2e2b67cce56c5d5099ec1ddcbd810fbb5973e6c96a929fd2c99fbd25f68",
        '"row_count": 2_335_042',
        '"variant_count": 2_259_268',
        '"train": 2_251_087',
        '"validation": 53_002',
        '"evaluation": 30_953',
        "dataset_manifest.json",
        "sha256:c3aa8f22b79e76fa5b6e3a43e02675cfc02d56dc7dc9fa36128c81874537016c",
        'DATASET_MANIFEST_SIZE_BYTES="5051"',
        "geno-lewm-data-v0.1.0-r1",
        "placed/gnomad-common-windows.jsonl",
        "sha256:ec76046771a163fbc22f326df26e2a332767eaa045dd919718c1cf86c4fbe0ac",
        'PLACED_WINDOWS_SIZE_BYTES="4186560"',
        'PLACED_WINDOWS_RECORD_COUNT="976"',
    ):
        assert exact_value in script

    assert '--revision "$INPUT_REVISION"' in script
    assert '"main"' not in script


def test_split_job_pins_export_arguments_and_independent_semantic_checks() -> None:
    script = JOB.read_text(encoding="utf-8")

    for argument in (
        "--store-dir",
        "--placed-windows-jsonl",
        "--dataset-manifest-json",
        "--output-dir",
        "--artifact-id",
        "--membership-repository",
        "--membership-revision",
        "--membership-artifact-path",
        "--training-windows-repository",
        "--training-windows-revision",
        "--training-windows-artifact-path",
        "--expected-store-content-identity",
        "--expected-store-physical-identity",
        "--expected-store-rowset-sha256",
        "--expected-dataset-manifest-sha256",
        "--expected-dataset-snapshot-id",
        "--expected-placed-windows-sha256",
        "--expected-placed-windows-size-bytes",
        "--expected-placed-windows-record-count",
        "--producer-git-commit",
        "--container-image",
        "--sample-seed 20260713",
        "--sample-size 128",
        "--report-schema-path",
    ):
        assert argument in script

    for invariant in (
        '"record_count": 34_657',
        '"record_count": 20_653',
        '"negative": 29_606',
        '"positive": 5_051',
        '"negative": 17_246',
        '"positive": 3_407',
        '"windows_scanned": 976',
        '"policy_exclusions": 0',
        '"indexed_overlaps": 0',
        '"observed_size": 128',
        "Draft202012Validator",
        "iter_labeled_clinvar(role)",
        "labels_jsonl",
        "vcf",
        "exact output inventory drifted",
        "bundled split schema differs from the exact checked-out contract",
        "JSONL rows or classes differ from the pinned membership store",
    ):
        assert invariant in script


def test_split_job_publishes_once_to_a_conflict_safe_child_namespace() -> None:
    script = JOB.read_text(encoding="utf-8")

    assert script.count("tools.data.v03_gnomad_lock publish") == 1
    assert (
        'RUN_NAME="${RUN_NAME:-geno-lewm-v03-membership-splits-'
        '${COMMIT_SHA:0:12}-r${RUN_ATTEMPT}}"' in script
    )
    assert 'PUBLISH_NAMESPACE="${MEMBERSHIP_CANDIDATE_ROOT}/splits/${RUN_NAME}/success"' in script
    assert "candidate-negative" not in script
    assert "failure bundle" not in script
    assert "trap " not in script
    assert "|| true" not in script
    assert "hf upload" not in script
    assert "GENO_LEWM_V03_SPLITS_OK $HUB_REVISION $PUBLISH_NAMESPACE" in script


def test_split_job_preserves_claim_boundary_and_exact_submission_recipe() -> None:
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
        "uv sync --frozen --extra evidence",
        "uv run --no-sync bash tools/jobs/v03_publish_membership_splits.sh",
        "variant memberships",
        "phased haplotypes",
        "released v0.3 snapshot",
        "dataset representativeness",
        "model quality",
        "benchmark performance",
        "clinical validity",
    ):
        assert token in script


def test_split_job_is_valid_bash() -> None:
    subprocess.run(("bash", "-n", str(JOB)), check=True)
