# SPDX-License-Identifier: Apache-2.0
"""Contracts for the immutable gnomAD v0.3 autosome source lock."""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tools.data.v03_gnomad_lock import (
    SourceLockError,
    audit_gnomad_parquet,
    is_hf_parent_head_conflict,
    main,
    select_source,
    upload_folder_with_parent_retry,
)

SOURCE_LOCK = Path("configs/data_v03/gnomad-v4.1-exomes-autosomes.source-lock.json")
SOURCE_LOCK_SCHEMA = Path("configs/data_v03/gnomad-v4.1-exomes-autosomes.source-lock.schema.json")


class _ParentConflict(RuntimeError):
    """Test-only stale-parent signal selected by an injected classifier."""


class _RecordingUploadApi:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = iter(outcomes)
        self.calls: list[dict[str, object]] = []

    def upload_folder(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        outcome = next(self._outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_upload_retries_only_parent_conflicts_with_fresh_absence_proofs() -> None:
    commit = SimpleNamespace(oid="f" * 40)
    api = _RecordingUploadApi([_ParentConflict(), _ParentConflict(), commit])
    parents = iter(["1" * 40, "2" * 40, "3" * 40])
    proof_calls: list[None] = []
    sleeps: list[float] = []

    def prove_namespace_absent() -> str:
        proof_calls.append(None)
        return next(parents)

    observed = upload_folder_with_parent_retry(
        api=api,
        repo_id="abdelstark/geno-lewm-data",
        repo_type="dataset",
        namespace="staging/v0.3/locked/chr22",
        publish_dir=Path("/tmp/publish"),
        commit_message="stage locked chr22",
        prove_namespace_absent=prove_namespace_absent,
        max_attempts=3,
        sleep=sleeps.append,
        is_parent_head_conflict=lambda exc: isinstance(exc, _ParentConflict),
    )

    assert observed is commit
    assert len(proof_calls) == 3
    assert [call["parent_commit"] for call in api.calls] == ["1" * 40, "2" * 40, "3" * 40]
    assert {call["path_in_repo"] for call in api.calls} == {"staging/v0.3/locked/chr22"}
    assert {call["folder_path"] for call in api.calls} == {Path("/tmp/publish")}
    assert all("delete_patterns" not in call for call in api.calls)
    assert len(sleeps) == 2
    assert 1.0 <= sleeps[0] < 1.5
    assert 2.0 <= sleeps[1] < 2.5


def test_upload_parent_conflict_retry_is_bounded() -> None:
    conflicts = [_ParentConflict() for _ in range(4)]
    api = _RecordingUploadApi(conflicts)
    proof_calls: list[None] = []

    def prove_namespace_absent() -> str:
        proof_calls.append(None)
        return "1" * 40

    with pytest.raises(_ParentConflict):
        upload_folder_with_parent_retry(
            api=api,
            repo_id="abdelstark/geno-lewm-data",
            repo_type="dataset",
            namespace="staging/v0.3/locked/chr22",
            publish_dir=Path("/tmp/publish"),
            commit_message="stage locked chr22",
            prove_namespace_absent=prove_namespace_absent,
            max_attempts=3,
            sleep=lambda _delay: None,
            is_parent_head_conflict=lambda exc: isinstance(exc, _ParentConflict),
        )

    assert len(api.calls) == 3
    assert len(proof_calls) == 3


def test_upload_fails_immediately_for_non_conflict_errors() -> None:
    auth_error = RuntimeError("401 unauthorized")
    api = _RecordingUploadApi([auth_error])
    proof_calls: list[None] = []

    def prove_namespace_absent() -> str:
        proof_calls.append(None)
        return "1" * 40

    with pytest.raises(RuntimeError, match="401 unauthorized"):
        upload_folder_with_parent_retry(
            api=api,
            repo_id="abdelstark/geno-lewm-data",
            repo_type="dataset",
            namespace="staging/v0.3/locked/chr22",
            publish_dir=Path("/tmp/publish"),
            commit_message="stage locked chr22",
            prove_namespace_absent=prove_namespace_absent,
            max_attempts=5,
            sleep=lambda _delay: pytest.fail("non-conflict errors must not back off"),
            is_parent_head_conflict=lambda exc: isinstance(exc, _ParentConflict),
        )

    assert len(api.calls) == 1
    assert len(proof_calls) == 1


def test_upload_aborts_if_namespace_appears_after_parent_conflict() -> None:
    api = _RecordingUploadApi([_ParentConflict(), SimpleNamespace(oid="f" * 40)])
    proofs = iter(["1" * 40, SourceLockError("immutable namespace already exists")])

    def prove_namespace_absent() -> str:
        outcome = next(proofs)
        if isinstance(outcome, SourceLockError):
            raise outcome
        return outcome

    with pytest.raises(SourceLockError, match="immutable namespace already exists"):
        upload_folder_with_parent_retry(
            api=api,
            repo_id="abdelstark/geno-lewm-data",
            repo_type="dataset",
            namespace="staging/v0.3/locked/chr22",
            publish_dir=Path("/tmp/publish"),
            commit_message="stage locked chr22",
            prove_namespace_absent=prove_namespace_absent,
            max_attempts=3,
            sleep=lambda _delay: None,
            is_parent_head_conflict=lambda exc: isinstance(exc, _ParentConflict),
        )

    assert len(api.calls) == 1


def test_hf_parent_conflict_classifier_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHfHubHTTPError(RuntimeError):
        def __init__(self, message: str, status_code: int) -> None:
            super().__init__(message)
            self.response = SimpleNamespace(status_code=status_code)

    original_import_module = importlib.import_module

    def fake_import_module(name: str) -> object:
        if name == "huggingface_hub.errors":
            return SimpleNamespace(HfHubHTTPError=FakeHfHubHTTPError)
        return original_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    exact = FakeHfHubHTTPError(
        "412 Client Error: A commit has happened since. Please refresh and try again.", 412
    )

    assert is_hf_parent_head_conflict(exact)
    assert not is_hf_parent_head_conflict(FakeHfHubHTTPError(str(exact), 409))
    assert not is_hf_parent_head_conflict(FakeHfHubHTTPError("412 validation failed", 412))
    assert not is_hf_parent_head_conflict(RuntimeError(str(exact)))


def test_audit_gnomad_parquet_scans_exact_schema_and_binds_positions(tmp_path: Path) -> None:
    parquet_path = _write_gnomad_parquet(
        tmp_path / "variants.parquet",
        [
            {"chrom": "22", "pos": 101, "ref": "A", "alt": "C", "af_global": 0.01},
            {"chrom": "22", "pos": 909, "ref": "AC", "alt": "GT", "af_global": 1.0},
        ],
    )

    audit = audit_gnomad_parquet(
        parquet_path,
        chromosome="22",
        expected_records=2,
        min_af=0.01,
        max_allele_len=16,
    )

    assert audit["audit_method"] == "pyarrow_metadata_and_full_iter_batches_scan_v1"
    assert audit["metadata_row_count"] == 2
    assert audit["scanned_row_count"] == 2
    assert audit["canonical_chromosome"] == "22"
    assert audit["position_min"] == 101
    assert audit["position_max"] == 909
    assert audit["schema_version"] == "2.0.0"
    assert audit["population_af_non_null_counts"]["af_mid"] == 2
    assert audit["population_af_non_null_counts"]["af_remaining"] == 2


@pytest.mark.parametrize(
    ("row_update", "error"),
    [
        ({"chrom": "21"}, "chromosome drifted"),
        ({"pos": 0}, "position must be a positive integer"),
        ({"ref": "N"}, "REF must be explicit uppercase ACGT"),
        ({"alt": "a"}, "ALT must be explicit uppercase ACGT"),
        ({"alt": "A" * 17}, "ALT must be explicit uppercase ACGT"),
        ({"alt": "A"}, "REF and ALT must differ"),
        ({"af_global": float("nan")}, "af_global must be finite"),
        ({"af_global": 0.009}, "af_global must be within"),
        ({"af_global": 1.1}, "af_global must be within"),
        ({"af_afr": float("nan")}, "af_afr must be finite"),
        ({"af_nfe": -0.1}, "af_nfe must be within"),
        ({"af_sas": 1.1}, "af_sas must be within"),
        ({"af_mid": None}, "required gnomAD v4.1 population AF columns"),
        ({"af_remaining": None}, "required gnomAD v4.1 population AF columns"),
        ({"filter": "LowQual"}, "filter must be 'PASS'"),
        ({"schema_version": "1.0.0"}, "schema_version must be '2.0.0'"),
    ],
)
def test_audit_gnomad_parquet_rejects_invalid_rows(
    tmp_path: Path, row_update: dict[str, object], error: str
) -> None:
    row = {"chrom": "22", "pos": 101, "ref": "A", "alt": "C", "af_global": 0.1}
    row.update(row_update)
    parquet_path = _write_gnomad_parquet(tmp_path / "variants.parquet", [row])

    with pytest.raises(SourceLockError, match=error):
        audit_gnomad_parquet(
            parquet_path,
            chromosome="22",
            expected_records=1,
            min_af=0.01,
            max_allele_len=16,
        )


def test_audit_gnomad_parquet_rejects_schema_drift(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    parquet_path = _write_gnomad_parquet(
        tmp_path / "variants.parquet",
        [{"chrom": "22", "pos": 101, "ref": "A", "alt": "C", "af_global": 0.1}],
    )
    table = pq.read_table(parquet_path).append_column("unexpected", pa.array([1], type=pa.int64()))
    pq.write_table(table, parquet_path)

    with pytest.raises(SourceLockError, match="Parquet schema drifted"):
        audit_gnomad_parquet(
            parquet_path,
            chromosome="22",
            expected_records=1,
            min_af=0.01,
            max_allele_len=16,
        )


def test_audit_gnomad_parquet_rejects_preparer_count_mismatch(tmp_path: Path) -> None:
    parquet_path = _write_gnomad_parquet(
        tmp_path / "variants.parquet",
        [{"chrom": "22", "pos": 101, "ref": "A", "alt": "C", "af_global": 0.1}],
    )

    with pytest.raises(SourceLockError, match="metadata/preparer row-count mismatch"):
        audit_gnomad_parquet(
            parquet_path,
            chromosome="22",
            expected_records=2,
            min_af=0.01,
            max_allele_len=16,
        )


def test_v03_gnomad_source_lock_covers_generation_pinned_autosomes() -> None:
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    objects = lock["objects"]

    assert lock["schema_version"] == "geno-lewm.gnomad-source-lock.v1"
    assert lock["release"] == "v4.1"
    assert lock["reference_genome"] == "GRCh38"
    assert lock["source"]["bucket"] == "gcp-public-data--gnomad"
    assert {entry["chromosome"] for entry in objects} == {
        str(chromosome) for chromosome in range(1, 23)
    }
    assert len(objects) == 22

    by_chromosome = {entry["chromosome"]: entry for entry in objects}
    assert {by_chromosome[str(chromosome)]["split_role"] for chromosome in range(1, 20)} == {
        "train"
    }
    assert by_chromosome["20"]["split_role"] == "validation"
    assert by_chromosome["21"]["split_role"] == "evaluation"
    assert by_chromosome["22"]["split_role"] == "train"

    for chromosome, entry in by_chromosome.items():
        assert entry["object"] == (
            f"release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr{chromosome}.vcf.bgz"
        )
        assert entry["generation"].isdigit()
        assert entry["size_bytes"] > 0
        assert len(base64.b64decode(entry["md5_base64"], validate=True)) == 16

    assert by_chromosome["22"] == {
        "chromosome": "22",
        "split_role": "train",
        "object": ("release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr22.vcf.bgz"),
        "generation": "1713312296186865",
        "size_bytes": 5_060_347_554,
        "md5_base64": "3PGRVj5pBUpxvU3HeGJ5mg==",
    }


def test_v03_gnomad_source_lock_has_a_closed_machine_readable_schema() -> None:
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    schema = json.loads(SOURCE_LOCK_SCHEMA.read_text(encoding="utf-8"))

    assert lock["$schema"] == f"./{SOURCE_LOCK_SCHEMA.name}"
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "$schema",
        "schema_version",
        "dataset_id",
        "release",
        "reference_genome",
        "source",
        "transform",
        "job",
        "objects",
        "claim_boundary",
    }
    objects_schema = schema["properties"]["objects"]
    assert objects_schema["minItems"] == 22
    assert objects_schema["maxItems"] == 22
    assert objects_schema["items"]["additionalProperties"] is False
    assert set(objects_schema["items"]["required"]) == {
        "chromosome",
        "split_role",
        "object",
        "generation",
        "size_bytes",
        "md5_base64",
    }


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_source_lock_rejects_non_finite_numeric_fields(tmp_path: Path, non_finite: float) -> None:
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    lock["transform"]["min_af"] = non_finite
    lock_path = tmp_path / SOURCE_LOCK.name
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    (tmp_path / SOURCE_LOCK_SCHEMA.name).write_bytes(SOURCE_LOCK_SCHEMA.read_bytes())

    with pytest.raises(SourceLockError, match=r"transform\.min_af must be finite"):
        select_source(
            lock_path,
            chromosome="22",
            commit_sha="a" * 40,
            container_image=lock["job"]["container_image"],
        )


def test_select_source_resolves_one_locked_object_and_immutable_namespace() -> None:
    commit_sha = "a" * 40
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    selection = select_source(
        SOURCE_LOCK,
        chromosome="22",
        commit_sha=commit_sha,
        container_image=lock["job"]["container_image"],
    )

    lock_sha256 = hashlib.sha256(SOURCE_LOCK.read_bytes()).hexdigest()
    selected_source = cast(Mapping[str, object], selection["source"])
    publication = cast(Mapping[str, object], selection["publication"])
    assert selection["source_lock"] == {
        "path": str(SOURCE_LOCK),
        "sha256": lock_sha256,
        "schema_version": "geno-lewm.gnomad-source-lock.v1",
        "schema": {
            "path": str(SOURCE_LOCK_SCHEMA),
            "sha256": hashlib.sha256(SOURCE_LOCK_SCHEMA.read_bytes()).hexdigest(),
            "draft": "https://json-schema.org/draft/2020-12/schema",
        },
    }
    assert selected_source["generation"] == "1713312296186865"
    assert selected_source["md5_hex"] == "dcf191563e69054a71bd4dc77862799a"
    assert cast(str, selected_source["metadata_url"]).endswith(
        "release%2F4.1%2Fvcf%2Fexomes%2Fgnomad.exomes.v4.1.sites.chr22.vcf.bgz"
        "?generation=1713312296186865"
    )
    assert cast(str, selected_source["media_url"]).endswith(
        "release%2F4.1%2Fvcf%2Fexomes%2Fgnomad.exomes.v4.1.sites.chr22.vcf.bgz"
        "?alt=media&generation=1713312296186865"
    )
    assert selection["execution"] == {
        "commit_sha": commit_sha,
        "container_image": lock["job"]["container_image"],
        "repository": "https://github.com/AbdelStark/GenoLeWM.git",
    }
    assert publication["namespace"] == (
        "staging/v0.3/gnomad-v4.1-exomes-autosomes/"
        f"lock-{lock_sha256[:12]}/chr22-g1713312296186865-{commit_sha[:12]}"
    )


def test_select_cli_writes_the_checked_entry_as_json(tmp_path: Path) -> None:
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    output = tmp_path / "selection.json"

    assert (
        main(
            [
                "select",
                "--lock-json",
                str(SOURCE_LOCK),
                "--chromosome",
                "20",
                "--commit-sha",
                "b" * 40,
                "--container-image",
                lock["job"]["container_image"],
                "--output-json",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source"]["chromosome"] == "20"
    assert payload["source"]["split_role"] == "validation"


def test_verify_metadata_cli_accepts_only_the_generation_pinned_gcs_object(
    tmp_path: Path,
) -> None:
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    selection_path = tmp_path / "selection.json"
    metadata_path = tmp_path / "gcs-metadata.json"
    verification_path = tmp_path / "metadata-verification.json"
    assert (
        main(
            [
                "select",
                "--lock-json",
                str(SOURCE_LOCK),
                "--chromosome",
                "22",
                "--commit-sha",
                "c" * 40,
                "--container-image",
                lock["job"]["container_image"],
                "--output-json",
                str(selection_path),
            ]
        )
        == 0
    )
    metadata_path.write_text(
        json.dumps(
            {
                "bucket": "gcp-public-data--gnomad",
                "name": ("release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr22.vcf.bgz"),
                "generation": "1713312296186865",
                "size": "5060347554",
                "md5Hash": "3PGRVj5pBUpxvU3HeGJ5mg==",
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "verify-metadata",
                "--selection-json",
                str(selection_path),
                "--metadata-json",
                str(metadata_path),
                "--output-json",
                str(verification_path),
            ]
        )
        == 0
    )
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    assert verification["ok"] is True
    assert verification["generation"] == "1713312296186865"
    assert verification["size_bytes"] == 5_060_347_554
    assert verification["md5_hex"] == "dcf191563e69054a71bd4dc77862799a"


def test_verify_metadata_cli_rejects_generation_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": "geno-lewm.gnomad-stage-selection.v1",
                "source": {
                    "bucket": "gcp-public-data--gnomad",
                    "object": "locked-object",
                    "generation": "123",
                    "size_bytes": 10,
                    "md5_base64": "AAAAAAAAAAAAAAAAAAAAAA==",
                },
            }
        ),
        encoding="utf-8",
    )
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "bucket": "gcp-public-data--gnomad",
                "name": "locked-object",
                "generation": "124",
                "size": "10",
                "md5Hash": "AAAAAAAAAAAAAAAAAAAAAA==",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "verification.json"

    assert (
        main(
            [
                "verify-metadata",
                "--selection-json",
                str(selection_path),
                "--metadata-json",
                str(metadata_path),
                "--output-json",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()
    assert "generation drifted" in capsys.readouterr().err


def test_hash_source_cli_verifies_md5_and_records_streamed_sha256(tmp_path: Path) -> None:
    source_path = tmp_path / "source.vcf.bgz"
    source_path.write_bytes(b"generation-pinned-gnomad-fixture\n")
    selection_path = tmp_path / "selection.json"
    selection = {
        "schema_version": "geno-lewm.gnomad-stage-selection.v1",
        "source": {
            "size_bytes": source_path.stat().st_size,
            "md5_base64": base64.b64encode(
                hashlib.md5(source_path.read_bytes(), usedforsecurity=False).digest()
            ).decode("ascii"),
        },
    }
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    output = tmp_path / "source-identity.json"

    assert (
        main(
            [
                "hash-source",
                "--selection-json",
                str(selection_path),
                "--input-vcf",
                str(source_path),
                "--output-json",
                str(output),
            ]
        )
        == 0
    )
    identity = json.loads(output.read_text(encoding="utf-8"))
    assert identity["ok"] is True
    assert identity["size_bytes"] == source_path.stat().st_size
    assert identity["sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert identity["hash_method"] == "single_pass_chunked_file_read"


def test_author_receipt_cli_reconciles_transform_and_output_evidence(tmp_path: Path) -> None:
    source_path = tmp_path / "gnomad.vcf.bgz"
    source_path.write_bytes(b"locked gnomad bytes\n")
    dataset_root = tmp_path / "publish" / "data"
    output_parquet = dataset_root / "gnomad" / "v4.1" / "variants.parquet"
    _write_gnomad_parquet(
        output_parquet,
        [
            {"chrom": "22", "pos": 101, "ref": "A", "alt": "C", "af_global": 0.01},
            {"chrom": "22", "pos": 202, "ref": "G", "alt": "T", "af_global": 0.2},
            {"chrom": "22", "pos": 303, "ref": "AC", "alt": "GT", "af_global": 1.0},
        ],
    )
    selection_path = tmp_path / "selection.json"
    source_md5_base64 = base64.b64encode(
        hashlib.md5(source_path.read_bytes(), usedforsecurity=False).digest()
    ).decode("ascii")
    selection = {
        "schema_version": "geno-lewm.gnomad-stage-selection.v1",
        "source_lock": {
            "path": str(SOURCE_LOCK),
            "sha256": "1" * 64,
            "schema_version": "geno-lewm.gnomad-source-lock.v1",
        },
        "dataset_id": "gnomad-v4.1-exomes-autosomes",
        "release": "v4.1",
        "reference_genome": "GRCh38",
        "source": {
            "bucket": "gcp-public-data--gnomad",
            "chromosome": "22",
            "split_role": "train",
            "object": "release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr22.vcf.bgz",
            "generation": "1713312296186865",
            "size_bytes": source_path.stat().st_size,
            "md5_base64": source_md5_base64,
            "md5_hex": hashlib.md5(source_path.read_bytes(), usedforsecurity=False).hexdigest(),
            "metadata_url": "https://example.test/metadata",
            "media_url": "https://example.test/media",
        },
        "transform": {
            "command": "geno-lewm-prepare-gnomad",
            "filter": "PASS",
            "min_af": 0.01,
            "max_allele_len": 16,
        },
        "execution": {
            "commit_sha": "d" * 40,
            "container_image": "example.test/image@sha256:" + "e" * 64,
            "repository": "https://github.com/AbdelStark/GenoLeWM.git",
        },
        "publication": {
            "repo": "abdelstark/geno-lewm-data",
            "repo_type": "dataset",
            "namespace": "staging/v0.3/test",
        },
        "claim_boundary": (
            "This staging receipt verifies only source and transform integrity; it is not "
            "evidence of snapshot membership, leakage control, model quality, or clinical validity."
        ),
    }
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    selection_sha256 = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    metadata_verification_path = tmp_path / "metadata-verification.json"
    metadata_verification_path.write_text(
        json.dumps(
            {
                "schema_version": "geno-lewm.gnomad-gcs-metadata-verification.v1",
                "ok": True,
                "selection_sha256": selection_sha256,
                "generation": "1713312296186865",
                "size_bytes": source_path.stat().st_size,
                "md5_base64": source_md5_base64,
            }
        ),
        encoding="utf-8",
    )
    source_identity_path = tmp_path / "source-identity.json"
    source_identity_path.write_text(
        json.dumps(
            {
                "schema_version": "geno-lewm.gnomad-stream-identity.v1",
                "ok": True,
                "selection_sha256": selection_sha256,
                "path": str(source_path),
                "size_bytes": source_path.stat().st_size,
                "md5_base64": source_md5_base64,
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    report_argv = [
        "geno-lewm-prepare-gnomad",
        "--input-vcf",
        str(source_path),
        "--output",
        str(dataset_root),
        "--release",
        "v4.1",
        "--min-af",
        "0.01",
        "--max-allele-len",
        "16",
    ]
    prepare_report_path = tmp_path / "prepare-report.json"
    prepare_report_path.write_text(
        json.dumps(
            {
                "command": shlex.join(report_argv),
                "release": "v4.1",
                "input_vcf": {
                    "path": str(source_path),
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    "size_bytes": source_path.stat().st_size,
                },
                "output_parquet": {
                    "path": str(output_parquet),
                    "sha256": hashlib.sha256(output_parquet.read_bytes()).hexdigest(),
                    "size_bytes": output_parquet.stat().st_size,
                },
                "records_read": 10,
                "allele_records_seen": 12,
                "records_written": 3,
                "skipped_filter": 2,
                "skipped_af": 6,
                "skipped_allele": 1,
                "already_exists": False,
                "runtime": {
                    "elapsed_seconds": 12.5,
                    "process_peak_rss_bytes": 123456,
                    "peak_memory_note": "fixture ru_maxrss",
                },
            }
        ),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "receipt.json"

    assert (
        main(
            [
                "author-receipt",
                "--selection-json",
                str(selection_path),
                "--metadata-verification-json",
                str(metadata_verification_path),
                "--source-identity-json",
                str(source_identity_path),
                "--prepare-report-json",
                str(prepare_report_path),
                "--input-vcf",
                str(source_path),
                "--dataset-root",
                str(dataset_root),
                "--output-parquet",
                str(output_parquet),
                "--output-json",
                str(receipt_path),
            ]
        )
        == 0
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["ok"] is True
    assert (
        receipt["source"]["streamed_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    )
    assert receipt["transform"]["argv"][:3] == [
        "uv",
        "run",
        "geno-lewm-prepare-gnomad",
    ]
    assert receipt["transform"]["runtime"]["process_peak_rss_bytes"] == 123456
    assert receipt["transform"]["counts"]["records_written"] == 3
    assert receipt["output"]["sha256"] == hashlib.sha256(output_parquet.read_bytes()).hexdigest()
    parquet_audit = receipt["output"]["parquet_audit"]
    assert parquet_audit["audit_method"] == "pyarrow_metadata_and_full_iter_batches_scan_v1"
    assert parquet_audit["metadata_row_count"] == 3
    assert parquet_audit["scanned_row_count"] == 3
    assert parquet_audit["position_min"] == 101
    assert parquet_audit["position_max"] == 303
    assert "not evidence" in receipt["claim_boundary"]


def _write_gnomad_parquet(path: Path, rows: list[dict[str, object]]) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    schema = pa.schema(
        [
            ("chrom", pa.string()),
            ("pos", pa.int64()),
            ("ref", pa.string()),
            ("alt", pa.string()),
            ("af_global", pa.float32()),
            ("af_afr", pa.float32()),
            ("af_ami", pa.float32()),
            ("af_amr", pa.float32()),
            ("af_asj", pa.float32()),
            ("af_eas", pa.float32()),
            ("af_fin", pa.float32()),
            ("af_mid", pa.float32()),
            ("af_nfe", pa.float32()),
            ("af_oth", pa.float32()),
            ("af_remaining", pa.float32()),
            ("af_sas", pa.float32()),
            ("filter", pa.string()),
            ("schema_version", pa.string()),
        ]
    )
    normalized = [
        {
            **dict.fromkeys(
                (
                    "af_afr",
                    "af_ami",
                    "af_amr",
                    "af_asj",
                    "af_eas",
                    "af_fin",
                    "af_mid",
                    "af_nfe",
                    "af_oth",
                    "af_remaining",
                    "af_sas",
                )
            ),
            "af_afr": 0.1,
            "af_amr": 0.1,
            "af_asj": 0.1,
            "af_eas": 0.1,
            "af_fin": 0.1,
            "af_mid": 0.1,
            "af_nfe": 0.1,
            "af_remaining": 0.1,
            "af_sas": 0.1,
            "filter": "PASS",
            "schema_version": "2.0.0",
            **row,
        }
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(normalized, schema=schema), path)
    return path
