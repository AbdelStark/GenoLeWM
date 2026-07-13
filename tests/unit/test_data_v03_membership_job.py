# SPDX-License-Identifier: Apache-2.0
"""Contracts for the hosted v0.3 membership-build inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.data.v03_membership_job import main


def test_author_spec_binds_all_23_downloaded_sources_to_lineage(tmp_path: Path) -> None:
    lineage_path, gnomad_root, clinvar_root = _write_download_fixture(tmp_path)
    lineage_sha256 = _sha256_uri(lineage_path)
    output = tmp_path / "membership-build.json"
    identities = tmp_path / "source-download-identities.json"

    rc = main(
        [
            "author-spec",
            "--lineage-json",
            str(lineage_path),
            "--expected-lineage-sha256",
            lineage_sha256,
            "--expected-lineage-size-bytes",
            str(lineage_path.stat().st_size),
            "--gnomad-download-root",
            str(gnomad_root),
            "--clinvar-download-root",
            str(clinvar_root),
            "--artifact-id",
            "geno-lewm-data-v0.3.0-membership-r1",
            "--builder-git-commit",
            "a" * 40,
            "--container-image",
            "ghcr.io/astral-sh/uv@sha256:" + "b" * 64,
            "--output-json",
            str(output),
            "--identity-report-json",
            str(identities),
        ]
    )

    assert rc == 0
    spec = json.loads(output.read_text(encoding="utf-8"))
    assert spec["$schema"] == "./membership-build-spec.schema.json"
    assert spec["schema_version"] == "geno-lewm.membership-build-spec.v1"
    assert spec["snapshot_lineage_sha256"] == lineage_sha256
    assert spec["builder"] == {
        "git_commit": "a" * 40,
        "container_image": "ghcr.io/astral-sh/uv@sha256:" + "b" * 64,
    }
    assert [source["chromosome"] for source in spec["sources"][:-1]] == [
        str(chromosome) for chromosome in range(1, 23)
    ]
    assert spec["sources"][-1]["kind"] == "clinvar"
    assert len(spec["sources"]) == 23

    report = json.loads(identities.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["source_count"] == 23
    assert report["lineage"]["sha256"] == lineage_sha256
    assert report["repositories"] == {
        "clinvar": {
            "repo_id": "abdelstark/geno-lewm-data",
            "repo_type": "dataset",
            "revision": "9e1a2b279681177a7ca00b30b9eb8048b511d1cb",
        },
        "gnomad": {
            "repo_id": "abdelstark/geno-lewm-data",
            "repo_type": "dataset",
            "revision": "f3676763b3f7f71d0d0d098588e9bf377faa0c5c",
        },
    }
    assert len(report["files"]) == 23


def test_author_download_plan_closes_exact_revisions_namespaces_and_paths(
    tmp_path: Path,
) -> None:
    lineage_path, _gnomad_root, _clinvar_root = _write_download_fixture(tmp_path)
    output = tmp_path / "download-plan.json"

    rc = main(
        [
            "author-download-plan",
            "--lineage-json",
            str(lineage_path),
            "--expected-lineage-sha256",
            _sha256_uri(lineage_path),
            "--expected-lineage-size-bytes",
            str(lineage_path.stat().st_size),
            "--output-json",
            str(output),
        ]
    )

    assert rc == 0
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["schema_version"] == "geno-lewm.membership-download-plan.v1"
    assert plan["candidate_snapshot_id"] == "geno-lewm-data-v0.3.0-r1"
    assert len(plan["downloads"]) == 23
    assert [entry["chromosome"] for entry in plan["downloads"][:-1]] == [
        str(chromosome) for chromosome in range(1, 23)
    ]
    assert {entry["revision"] for entry in plan["downloads"][:-1]} == {
        "f3676763b3f7f71d0d0d098588e9bf377faa0c5c"
    }
    assert plan["downloads"][-1]["revision"] == ("9e1a2b279681177a7ca00b30b9eb8048b511d1cb")
    assert all(entry["repo_id"] == "abdelstark/geno-lewm-data" for entry in plan["downloads"])
    assert all(entry["repo_type"] == "dataset" for entry in plan["downloads"])


def test_author_spec_rejects_tampered_download_before_writing_contract(
    tmp_path: Path,
) -> None:
    lineage_path, gnomad_root, clinvar_root = _write_download_fixture(tmp_path)
    first_path = next(gnomad_root.rglob("variants.parquet"))
    first_path.write_bytes(b"tampered\n")
    output = tmp_path / "membership-build.json"
    identities = tmp_path / "source-download-identities.json"

    rc = main(
        [
            "author-spec",
            "--lineage-json",
            str(lineage_path),
            "--expected-lineage-sha256",
            _sha256_uri(lineage_path),
            "--expected-lineage-size-bytes",
            str(lineage_path.stat().st_size),
            "--gnomad-download-root",
            str(gnomad_root),
            "--clinvar-download-root",
            str(clinvar_root),
            "--artifact-id",
            "geno-lewm-data-v0.3.0-membership-r1",
            "--builder-git-commit",
            "a" * 40,
            "--container-image",
            "ghcr.io/astral-sh/uv@sha256:" + "b" * 64,
            "--output-json",
            str(output),
            "--identity-report-json",
            str(identities),
        ]
    )

    assert rc == 2
    assert not output.exists()
    assert not identities.exists()


def test_download_plan_rejects_duplicate_lineage_keys_even_when_last_value_matches(
    tmp_path: Path,
) -> None:
    lineage_path, _gnomad_root, _clinvar_root = _write_download_fixture(tmp_path)
    original = lineage_path.read_text(encoding="utf-8")
    lineage_path.write_text(
        original.replace(
            "{",
            '{"schema_version":"attacker-controlled",',
            1,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "download-plan.json"

    rc = main(
        [
            "author-download-plan",
            "--lineage-json",
            str(lineage_path),
            "--expected-lineage-sha256",
            _sha256_uri(lineage_path),
            "--expected-lineage-size-bytes",
            str(lineage_path.stat().st_size),
            "--output-json",
            str(output),
        ]
    )

    assert rc == 2
    assert not output.exists()


def _write_download_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    gnomad_root = tmp_path / "gnomad"
    clinvar_root = tmp_path / "clinvar"
    shards: list[dict[str, object]] = []
    for chromosome in range(1, 23):
        namespace = f"staging/v0.3/gnomad/chr{chromosome}-fixture"
        artifact_path = f"{namespace}/data/gnomad/v4.1/variants.parquet"
        path = gnomad_root / artifact_path
        path.parent.mkdir(parents=True)
        path.write_bytes(f"gnomad-chr{chromosome}\n".encode())
        shards.append(
            {
                "chromosome": str(chromosome),
                "namespace": namespace,
                "revision": "f3676763b3f7f71d0d0d098588e9bf377faa0c5c",
                "output": {
                    "artifact_path": artifact_path,
                    "sha256": _sha256_uri(path),
                    "size_bytes": path.stat().st_size,
                },
            }
        )

    clinvar_namespace = "staging/clinvar-2026-04-15-fixture"
    clinvar_artifact = f"{clinvar_namespace}/clinvar/2026-04-15/variants.parquet"
    clinvar_path = clinvar_root / clinvar_artifact
    clinvar_path.parent.mkdir(parents=True)
    clinvar_path.write_bytes(b"clinvar\n")
    lineage = {
        "schema_version": "geno-lewm.v03-snapshot-lineage.v1",
        "lineage_id": "sha256:" + "c" * 64,
        "candidate_snapshot_id": "geno-lewm-data-v0.3.0-r1",
        "membership_status": "not_created",
        "gnomad": {
            "repo": "abdelstark/geno-lewm-data",
            "repo_type": "dataset",
            "shards": shards,
        },
        "clinvar": {
            "repo": "abdelstark/geno-lewm-data",
            "repo_type": "dataset",
            "revision": "9e1a2b279681177a7ca00b30b9eb8048b511d1cb",
            "namespace": clinvar_namespace,
            "output": {
                "artifact_path": clinvar_artifact,
                "sha256": _sha256_uri(clinvar_path),
                "size_bytes": clinvar_path.stat().st_size,
            },
        },
    }
    lineage_path = tmp_path / "snapshot-lineage.json"
    lineage_path.write_text(json.dumps(lineage, sort_keys=True) + "\n", encoding="utf-8")
    return lineage_path, gnomad_root, clinvar_root


def _sha256_uri(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
