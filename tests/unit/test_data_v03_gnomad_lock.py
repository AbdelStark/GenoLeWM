# SPDX-License-Identifier: Apache-2.0
"""Contracts for the immutable gnomAD v0.3 autosome source lock."""

from __future__ import annotations

import base64
import hashlib
import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from tools.data.v03_gnomad_lock import main, select_source

SOURCE_LOCK = Path("configs/data_v03/gnomad-v4.1-exomes-autosomes.source-lock.json")
SOURCE_LOCK_SCHEMA = Path("configs/data_v03/gnomad-v4.1-exomes-autosomes.source-lock.schema.json")


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
    output_parquet.parent.mkdir(parents=True)
    output_parquet.write_bytes(b"parquet fixture bytes\n")
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
    assert "not evidence" in receipt["claim_boundary"]
