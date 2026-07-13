"""Tests for the release dataset package builder."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.release.dataset_package as dataset_package_module
from geno_lewm.errors import InputError
from geno_lewm.provenance import sha256_file
from tools.release.dataset_integrity import DEFAULT_REPORT_NAME
from tools.release.dataset_package import (
    GENERATED_BY as DATASET_PACKAGE_GENERATED_BY,
    build_dataset_package,
    main,
    parse_dataset_package,
)
from tools.release.dataset_snapshot import (
    GENERATED_BY as DATASET_SNAPSHOT_GENERATED_BY,
    INPUT_CHECK_GENERATED_BY as DATASET_INPUT_CHECK_GENERATED_BY,
    INPUT_CHECK_REPORT_NAME as DATASET_INPUT_CHECK_REPORT_NAME,
    REPORT_NAME as DATASET_SNAPSHOT_REPORT_NAME,
)
from tools.release.paper_package import PackageIssue, _verify_dataset_dir


def test_build_dataset_package_writes_verifier_compatible_artifacts(tmp_path: Path) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)

    report = build_dataset_package(tmp_path, metadata_path)

    assert report.snapshot_id == "geno-lewm-data-v0.1.0-r1"
    assert (tmp_path / "dataset_package.json").is_file()
    assert (tmp_path / "dataset_manifest.json").is_file()
    assert (tmp_path / "data_card.md").is_file()
    assert (tmp_path / "split_integrity.json").is_file()
    assert (tmp_path / "SHA256SUMS").is_file()

    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["snapshot_id"] == "geno-lewm-data-v0.1.0-r1"
    assert manifest["files"][0]["path"] == "carbon/windows.jsonl"
    assert manifest["files"][0]["sha256"].startswith("sha256:")
    assert manifest["files"][0]["size_bytes"] > 0
    assert manifest["files"][0]["records"] == 1

    integrity = json.loads((tmp_path / "split_integrity.json").read_text(encoding="utf-8"))
    assert integrity["splits"]["train"]["observed_records"] == 1
    assert integrity["splits"]["train"]["label_counts"] == {}
    assert integrity["splits"]["eval_clinvar_coding"]["label_counts"] == {"P": 1}
    assert integrity["splits"]["eval_clinvar_coding"]["labelled_records"] == 1

    data_card = (tmp_path / "data_card.md").read_text(encoding="utf-8")
    assert "## Sources" in data_card
    assert "## License" in data_card
    assert "## Preprocessing" in data_card
    assert "## Splits" in data_card
    assert "## Class Balance" in data_card
    assert "| train | 1 | 0 | 1 | none observed |" in data_card
    assert "| eval_clinvar_coding | 1 | 1 | 0 | P=1 |" in data_card
    assert "split_integrity.json" in data_card
    assert "## Limitations" in data_card

    _write_dataset_snapshot_report(tmp_path)
    issues: list[PackageIssue] = []
    _verify_dataset_dir(tmp_path, issues)
    assert issues == []


def test_dataset_verifier_requires_metadata_json(tmp_path: Path) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    build_dataset_package(tmp_path, metadata_path)
    (tmp_path / "dataset_package.json").unlink()
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert "dataset.metadata_missing" in {issue.code for issue in issues}
    assert "dataset.checksums.file_missing" in {issue.code for issue in issues}


def test_dataset_verifier_rejects_stale_manifest_from_metadata(tmp_path: Path) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    build_dataset_package(tmp_path, metadata_path)
    manifest_path = tmp_path / "dataset_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["snapshot_id"] = "stale-snapshot"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert "dataset.manifest_stale" in {issue.code for issue in issues}


def test_dataset_verifier_rejects_stale_data_card_from_metadata(tmp_path: Path) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    build_dataset_package(tmp_path, metadata_path)
    card_path = tmp_path / "data_card.md"
    card_path.write_text(
        card_path.read_text(encoding="utf-8").replace("First GenoLeWM", "Changed GenoLeWM", 1),
        encoding="utf-8",
    )
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert "dataset.card.stale" in {issue.code for issue in issues}


def test_dataset_verifier_rejects_stale_data_card_class_balance(tmp_path: Path) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    build_dataset_package(tmp_path, metadata_path)
    card_path = tmp_path / "data_card.md"
    card_path.write_text(
        card_path.read_text(encoding="utf-8").replace(
            "| eval_clinvar_coding | 1 | 1 | 0 | P=1 |",
            "| eval_clinvar_coding | 1 | 0 | 1 | none observed |",
        ),
        encoding="utf-8",
    )
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert "dataset.card.stale" in {issue.code for issue in issues}


def test_dataset_verifier_rejects_hand_authored_split_integrity_report(
    tmp_path: Path,
) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    build_dataset_package(tmp_path, metadata_path)
    _write_dataset_snapshot_report(tmp_path)
    integrity_path = tmp_path / DEFAULT_REPORT_NAME
    payload = json.loads(integrity_path.read_text(encoding="utf-8"))
    payload["generated_by"] = "manual-editor"
    integrity_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert "dataset.integrity.generated_by" in {issue.code for issue in issues}


def test_dataset_package_main_outputs_json_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)

    rc = main(["--dataset-dir", str(tmp_path), "--metadata-json", str(metadata_path)])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == "1.0.0"
    assert payload["generated_by"] == DATASET_PACKAGE_GENERATED_BY
    assert payload["snapshot_id"] == "geno-lewm-data-v0.1.0-r1"
    assert str(tmp_path) not in json.dumps(payload)
    assert payload["manifest_path"] == "dataset_manifest.json"
    assert payload["data_card_path"] == "data_card.md"
    assert payload["integrity_path"] == DEFAULT_REPORT_NAME
    assert payload["checksums_path"] == "SHA256SUMS"
    assert len(payload["files"]) == 2


def test_parse_dataset_package_rejects_missing_file(tmp_path: Path) -> None:
    payload = _metadata()
    payload["files"][0]["path"] = "missing.jsonl"

    with pytest.raises(InputError, match="dataset file is missing"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_parse_dataset_package_rejects_unsafe_file_path(tmp_path: Path) -> None:
    payload = _metadata()
    payload["files"][0]["path"] = "../outside.jsonl"

    with pytest.raises(InputError, match="dataset paths must be relative"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_parse_dataset_package_rejects_placeholder_text(tmp_path: Path) -> None:
    _write_data_files(tmp_path)
    payload = _metadata()
    payload["limitations"] = ["TODO"]

    with pytest.raises(InputError, match="placeholder text is not allowed"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_parse_dataset_package_rejects_unexpected_generator(tmp_path: Path) -> None:
    _write_data_files(tmp_path)
    payload = _metadata()
    payload["generated_by"] = "manual-editor"

    with pytest.raises(InputError, match="generated_by"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_parse_dataset_package_rejects_undeclared_file_split(tmp_path: Path) -> None:
    _write_data_files(tmp_path)
    payload = _metadata()
    payload["files"][0]["split"] = "holdout"

    with pytest.raises(InputError, match="file split must be declared"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_parse_dataset_package_rejects_generated_file_as_input(tmp_path: Path) -> None:
    _write_data_files(tmp_path)
    (tmp_path / "data_card.md").write_text("# Data Card\n", encoding="utf-8")
    payload = _metadata()
    payload["files"].append({"path": "data_card.md", "split": "train"})

    with pytest.raises(InputError, match="generated dataset package files cannot be listed"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_parse_dataset_package_schema_1_1_emits_explicit_split_data_roles(
    tmp_path: Path,
) -> None:
    _write_data_files(tmp_path)
    payload = _metadata()
    payload["schema_version"] = "1.1.0"
    for file in payload["files"]:
        file["artifact_role"] = "split_data"

    package = parse_dataset_package(payload, dataset_dir=tmp_path)

    assert [file["artifact_role"] for file in package.manifest()["files"]] == [
        "split_data",
        "split_data",
    ]


def test_parse_dataset_package_schema_1_1_requires_explicit_roles(tmp_path: Path) -> None:
    _write_data_files(tmp_path)
    payload = _metadata()
    payload["schema_version"] = "1.1.0"

    with pytest.raises(InputError, match="artifact_role must be declared"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_parse_dataset_package_schema_1_1_rejects_unknown_role(tmp_path: Path) -> None:
    _write_data_files(tmp_path)
    payload = _metadata()
    payload["schema_version"] = "1.1.0"
    for file in payload["files"]:
        file["artifact_role"] = "split_data"
    payload["files"][0]["artifact_role"] = "sidecar"

    with pytest.raises(InputError, match="artifact_role is invalid"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_parse_dataset_package_schema_1_1_split_data_forbids_companion_of_key(
    tmp_path: Path,
) -> None:
    _write_data_files(tmp_path)
    payload = _metadata()
    payload["schema_version"] = "1.1.0"
    for file in payload["files"]:
        file["artifact_role"] = "split_data"
    payload["files"][0]["companion_of"] = None

    with pytest.raises(InputError, match="split_data files forbid companion_of"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


@pytest.mark.parametrize("field", ["artifact_role", "companion_of"])
def test_parse_dataset_package_schema_1_0_rejects_role_fields_even_when_null(
    tmp_path: Path,
    field: str,
) -> None:
    _write_data_files(tmp_path)
    payload = _metadata()
    payload["files"][0][field] = None

    with pytest.raises(InputError, match=r"schema 1\.0\.0 forbids"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_parse_dataset_package_schema_1_1_accepts_unsplit_evidence(tmp_path: Path) -> None:
    _write_data_files(tmp_path)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text('{"verified":true}\n', encoding="utf-8")
    payload = _metadata()
    payload["schema_version"] = "1.1.0"
    for file in payload["files"]:
        file["artifact_role"] = "split_data"
    payload["files"].append(
        {
            "path": evidence_path.name,
            "artifact_role": "evidence",
            "description": "release evidence",
        }
    )

    package = parse_dataset_package(payload, dataset_dir=tmp_path)

    assert package.manifest()["files"][-1] == {
        "path": "evidence.json",
        "sha256": sha256_file(evidence_path),
        "size_bytes": evidence_path.stat().st_size,
        "artifact_role": "evidence",
        "description": "release evidence",
    }


@pytest.mark.parametrize("field", ["split", "records", "companion_of"])
def test_parse_dataset_package_schema_1_1_evidence_forbids_split_fields_even_when_null(
    tmp_path: Path,
    field: str,
) -> None:
    _write_data_files(tmp_path)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text('{"verified":true}\n', encoding="utf-8")
    payload = _metadata()
    payload["schema_version"] = "1.1.0"
    for file in payload["files"]:
        file["artifact_role"] = "split_data"
    payload["files"].append({"path": evidence_path.name, "artifact_role": "evidence", field: None})

    with pytest.raises(InputError, match="evidence files forbid"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_parse_dataset_package_schema_1_1_accepts_split_companion(
    tmp_path: Path,
) -> None:
    _write_data_files(tmp_path)
    companion_path = tmp_path / "carbon" / "windows.labels.jsonl"
    companion_path.write_text('{"record_id":"r1"}\n', encoding="utf-8")
    payload = _metadata()
    payload["schema_version"] = "1.1.0"
    for file in payload["files"]:
        file["artifact_role"] = "split_data"
    payload["files"].append(
        {
            "path": "carbon/windows.labels.jsonl",
            "artifact_role": "split_companion",
            "split": "train",
            "records": 1,
            "companion_of": "carbon/windows.jsonl",
        }
    )

    package = parse_dataset_package(payload, dataset_dir=tmp_path)

    assert package.manifest()["files"][-1]["companion_of"] == "carbon/windows.jsonl"


@pytest.mark.parametrize(
    ("companion_of", "split", "records", "message"),
    [
        ("carbon/missing.jsonl", "train", 1, "exactly one split_data"),
        ("carbon/windows.jsonl", "eval_clinvar_coding", 1, "match companion_of"),
        ("carbon/windows.jsonl", "train", 2, "match companion_of"),
    ],
)
def test_parse_dataset_package_schema_1_1_rejects_invalid_companion_binding(
    tmp_path: Path,
    companion_of: str,
    split: str,
    records: int,
    message: str,
) -> None:
    _write_data_files(tmp_path)
    companion_path = tmp_path / "carbon" / "windows.labels.jsonl"
    companion_path.write_text('{"record_id":"r1"}\n', encoding="utf-8")
    payload = _metadata()
    payload["schema_version"] = "1.1.0"
    for file in payload["files"]:
        file["artifact_role"] = "split_data"
    payload["files"].append(
        {
            "path": "carbon/windows.labels.jsonl",
            "artifact_role": "split_companion",
            "split": split,
            "records": records,
            "companion_of": companion_of,
        }
    )

    with pytest.raises(InputError, match=message):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_build_dataset_package_schema_1_1_reports_schema_version(tmp_path: Path) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "1.1.0"
    for file in payload["files"]:
        file["artifact_role"] = "split_data"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_dataset_package(tmp_path, metadata_path)

    assert report.to_dict()["schema_version"] == "1.1.0"


def test_schema_1_0_rejects_membership_and_split_evidence_binding(tmp_path: Path) -> None:
    _write_data_files(tmp_path)
    payload = _metadata()
    payload["membership_and_split_evidence"] = {}

    with pytest.raises(InputError, match=r"schema 1\.0\.0 forbids"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_schema_1_1_binds_verified_membership_store_and_split_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, manifest = _write_role_bound_inputs(tmp_path)
    _patch_membership_store(monkeypatch, manifest)

    package = parse_dataset_package(payload, dataset_dir=tmp_path)

    binding = package.manifest()["membership_and_split_evidence"]
    assert binding["membership_store"]["path"] == "membership/store"
    assert binding["membership_store"]["content_identity"] == manifest.content_identity
    assert binding["report"] == {
        "path": "evidence/membership-split-evidence.json",
        "schema_path": "contract/membership-split-evidence.schema.json",
        "artifact_id": "geno-lewm-v03-membership-splits-fixture",
        "schema_version": "geno-lewm.membership-split-evidence.v1",
    }
    card = dataset_package_module.render_data_card(package)
    assert "## Membership and Split Evidence" in card
    assert "Deterministic unphased variant membership" in card
    assert "not phased-haplotype membership" in card
    assert "| Artifact role |" in card


def test_schema_1_1_rejects_membership_binding_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, manifest = _write_role_bound_inputs(tmp_path)
    binding = payload["membership_and_split_evidence"]
    assert isinstance(binding, dict)
    membership = binding["membership_store"]
    assert isinstance(membership, dict)
    membership["content_identity"] = "sha256:" + "f" * 64
    _patch_membership_store(monkeypatch, manifest)

    with pytest.raises(InputError, match="membership store binding identity mismatch"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_schema_1_1_rejects_self_authored_split_evidence_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, manifest = _write_role_bound_inputs(tmp_path)
    schema_path = tmp_path / "contract" / "membership-split-evidence.schema.json"
    schema_path.write_text('{"type":"object"}\n', encoding="utf-8")
    _patch_membership_store(monkeypatch, manifest)

    with pytest.raises(InputError, match="tracked schema identity"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_schema_1_1_rejects_nonpublication_split_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, manifest = _write_role_bound_inputs(tmp_path)
    report_path = tmp_path / "evidence" / "membership-split-evidence.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["producer"]["invocation_verified"] = False
    report["membership_store"]["lineage"]["evidence_profile"] = "synthetic_fixture"
    report["claim_boundary"]["publication_eligible"] = False
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _patch_membership_store(monkeypatch, manifest)

    with pytest.raises(InputError, match="not publication eligible"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_schema_1_1_rejects_split_report_stream_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, manifest = _write_role_bound_inputs(tmp_path)
    report_path = tmp_path / "evidence" / "membership-split-evidence.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["streams"]["evaluation"]["vcf"]["sha256"] = "sha256:" + "f" * 64
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _patch_membership_store(monkeypatch, manifest)

    with pytest.raises(InputError, match="split evidence file identity mismatch"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_schema_1_1_rejects_ambiguous_training_window_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, manifest = _write_role_bound_inputs(tmp_path)
    duplicate = tmp_path / "carbon" / "windows-copy.jsonl"
    duplicate.write_bytes((tmp_path / "carbon" / "windows.jsonl").read_bytes())
    files = payload["files"]
    assert isinstance(files, list)
    files.append(
        {
            "path": "carbon/windows-copy.jsonl",
            "artifact_role": "split_data",
            "split": "train_placed_gnomad_common",
            "records": 1,
        }
    )
    _patch_membership_store(monkeypatch, manifest)

    with pytest.raises(InputError, match="exactly one split_data file by identity"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_schema_1_1_rejects_noncanonical_artifact_paths(tmp_path: Path) -> None:
    _write_data_files(tmp_path)
    payload = _metadata()
    payload["schema_version"] = "1.1.0"
    files = payload["files"]
    assert isinstance(files, list)
    for file in files:
        file["artifact_role"] = "split_data"
    files[0]["path"] = "carbon//windows.jsonl"

    with pytest.raises(InputError, match="canonical relative POSIX"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_schema_1_1_rejects_symlinked_artifacts(tmp_path: Path) -> None:
    _write_data_files(tmp_path)
    alias = tmp_path / "carbon" / "windows-alias.jsonl"
    try:
        alias.symlink_to("windows.jsonl")
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")
    payload = _metadata()
    payload["schema_version"] = "1.1.0"
    files = payload["files"]
    assert isinstance(files, list)
    for file in files:
        file["artifact_role"] = "split_data"
    files[0]["path"] = "carbon/windows-alias.jsonl"

    with pytest.raises(InputError, match="must not traverse symbolic links"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def _write_dataset_inputs(root: Path) -> Path:
    _write_data_files(root)
    metadata_path = root / "dataset_package.json"
    metadata_path.write_text(json.dumps(_metadata(), indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def _write_data_files(root: Path) -> None:
    (root / "carbon").mkdir()
    (root / "clinvar").mkdir()
    (root / "carbon" / "windows.jsonl").write_text(
        '{"record_id":"r1","source":"mrna","start_bp":0,"end_bp":12}\n',
        encoding="utf-8",
    )
    (root / "clinvar" / "eval.vcf").write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t10\t.\tA\tT\t.\tPASS\tCLNSIG=Pathogenic\n",
        encoding="utf-8",
    )


def _write_role_bound_inputs(root: Path) -> tuple[dict[str, object], SimpleNamespace]:
    _write_data_files(root)
    stream_files = {
        role: _write_split_stream_fixture(root, role=role, chrom=chrom)
        for role, chrom in (("validation", "20"), ("evaluation", "21"))
    }
    store_root = root / "membership" / "store"
    store_root.mkdir(parents=True)
    for name in (
        "manifest.json",
        "memberships.parquet",
        "lookup.sqlite",
        "snapshot-lineage.json",
        "build-receipt.json",
    ):
        (store_root / name).write_text(name + "\n", encoding="utf-8")
    schema_path = root / "contract" / "membership-split-evidence.schema.json"
    schema_path.parent.mkdir()
    schema_path.write_bytes(
        Path("configs/data_v03/membership-split-evidence.schema.json").read_bytes()
    )
    manifest = SimpleNamespace(
        artifact_id="fixture-membership",
        content_identity="sha256:" + "1" * 64,
        physical_identity="sha256:" + "2" * 64,
        rowset_sha256="sha256:" + "3" * 64,
    )
    report_path = root / "evidence" / "membership-split-evidence.json"
    report_path.parent.mkdir()
    source_revision = "a" * 40
    report_path.write_text(
        json.dumps(
            {
                "$schema": "../contract/membership-split-evidence.schema.json",
                "artifact_id": "geno-lewm-v03-membership-splits-fixture",
                "schema_version": "geno-lewm.membership-split-evidence.v1",
                "assembly": "GRCh38",
                "ok": True,
                "producer": {
                    "generated_by": "tools.data.v03_membership_splits",
                    "git_commit": "b" * 40,
                    "container_image": "ghcr.io/example/geno-lewm@sha256:" + "c" * 64,
                    "invocation_verified": True,
                },
                "membership_store": {
                    "repository": "abdelstark/geno-lewm-data",
                    "revision": source_revision,
                    "artifact_path": "candidates/v0.3/membership/success/store",
                    "artifact_id": manifest.artifact_id,
                    "content_identity": manifest.content_identity,
                    "physical_identity": manifest.physical_identity,
                    "rowset_sha256": manifest.rowset_sha256,
                    "lineage": {
                        "candidate_snapshot_id": "geno-lewm-data-v0.3.0-r1",
                        "evidence_profile": "official",
                        "lineage_id": "sha256:" + "4" * 64,
                        "sha256": "sha256:" + "5" * 64,
                    },
                    "chromosome_roles": {
                        "train": [*(str(value) for value in range(1, 20)), "22"],
                        "validation": ["20"],
                        "evaluation": ["21"],
                    },
                },
                "training_windows": {
                    "source": {
                        "repository": "abdelstark/geno-lewm-data",
                        "revision": source_revision,
                        "artifact_path": (
                            "candidates/v0.3/geno-lewm-data-v0.3.0-r1/"
                            "placed/geno-lewm-v03-placed-windows-r1/success/"
                            "placed/windows.jsonl"
                        ),
                    },
                    "sha256": sha256_file(root / "carbon" / "windows.jsonl"),
                    "size_bytes": (root / "carbon" / "windows.jsonl").stat().st_size,
                    "record_count": 1,
                    "assembly": "GRCh38",
                    "role": "train",
                    "split": "train_placed_gnomad_common",
                    "chromosomes": ["22"],
                    "dataset_manifest": {
                        "path": "dataset_manifest.json",
                        "sha256": "sha256:" + "6" * 64,
                        "size_bytes": 1,
                        "snapshot_id": "geno-lewm-data-v0.3.0-r1",
                    },
                    "record_fields": [
                        "record_id",
                        "source",
                        "variant_source",
                        "chrom",
                        "start_bp",
                        "end_bp",
                        "sequence",
                        "variant_count",
                    ],
                },
                "streams": stream_files,
                "audits": {
                    "exhaustive": {
                        "windows_scanned": 1,
                        "policy_exclusions": 0,
                        "indexed_overlaps": 0,
                        "status": "passed",
                    },
                    "deterministic_sample": {
                        "algorithm": "sha256-priority-v1",
                        "seed": 20260713,
                        "requested_size": 1,
                        "observed_size": 1,
                        "sample_digest": "sha256:" + "7" * 64,
                        "policy_exclusions": 0,
                        "indexed_overlaps": 0,
                        "status": "passed",
                    },
                },
                "claim_boundary": {
                    "variant_membership": True,
                    "phased_haplotype_membership": False,
                    "released_v03_snapshot": False,
                    "publication_eligible": True,
                    "limitations": [
                        "This evidence covers deterministic unphased variant memberships and placed-window nonintersection only.",
                        "It does not establish phased-haplotype membership, a released v0.3 snapshot, dataset representativeness, model quality, benchmark performance, or clinical validity.",
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    payload = _metadata()
    payload["schema_version"] = "1.1.0"
    splits = payload["splits"]
    assert isinstance(splits, dict)
    splits.clear()
    splits.update(
        {
            "train_placed_gnomad_common": {"records": 1},
            "validation": {"records": 2},
            "evaluation": {"records": 2},
        }
    )
    files = [
        {
            "path": "carbon/windows.jsonl",
            "artifact_role": "split_data",
            "split": "train_placed_gnomad_common",
            "records": 1,
        }
    ]
    for role, stream in stream_files.items():
        labels = stream["labels_jsonl"]
        vcf = stream["vcf"]
        assert isinstance(labels, dict)
        assert isinstance(vcf, dict)
        files.extend(
            (
                {
                    "path": labels["path"],
                    "artifact_role": "split_data",
                    "split": role,
                    "records": 2,
                },
                {
                    "path": vcf["path"],
                    "artifact_role": "split_companion",
                    "split": role,
                    "records": 2,
                    "companion_of": labels["path"],
                },
            )
        )
    evidence_paths = [
        *(
            f"membership/store/{name}"
            for name in (
                "manifest.json",
                "memberships.parquet",
                "lookup.sqlite",
                "snapshot-lineage.json",
                "build-receipt.json",
            )
        ),
        "evidence/membership-split-evidence.json",
        "contract/membership-split-evidence.schema.json",
    ]
    files.extend({"path": path, "artifact_role": "evidence"} for path in evidence_paths)
    payload["files"] = files
    payload["membership_and_split_evidence"] = {
        "membership_store": {
            "path": "membership/store",
            "artifact_id": manifest.artifact_id,
            "content_identity": manifest.content_identity,
            "physical_identity": manifest.physical_identity,
            "rowset_sha256": manifest.rowset_sha256,
        },
        "report": {
            "path": "evidence/membership-split-evidence.json",
            "schema_path": "contract/membership-split-evidence.schema.json",
            "artifact_id": "geno-lewm-v03-membership-splits-fixture",
            "schema_version": "geno-lewm.membership-split-evidence.v1",
        },
    }
    return payload, manifest


def _write_split_stream_fixture(root: Path, *, role: str, chrom: str) -> dict[str, object]:
    stem = f"splits/{role}/clinvar-chr{chrom}"
    labels_relative = f"{stem}.labels.jsonl"
    vcf_relative = f"{stem}.vcf"
    labels_path = root / labels_relative
    vcf_path = root / vcf_relative
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        {"chrom": chrom, "pos": 100, "ref": "A", "alt": "C", "clinical_significance": "B"},
        {"chrom": chrom, "pos": 200, "ref": "G", "alt": "T", "clinical_significance": "P"},
    )
    labels_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    vcf_path.write_text(
        "##fileformat=VCFv4.3\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        f"{chrom}\t100\t.\tA\tC\t.\tPASS\tCLNSIG=B;ROLE={role};LABEL=0\n"
        f"{chrom}\t200\t.\tG\tT\t.\tPASS\tCLNSIG=P;ROLE={role};LABEL=1\n",
        encoding="utf-8",
    )
    return {
        "role": role,
        "chromosome": chrom,
        "record_count": 2,
        "class_counts": {"B": 1, "LB": 0, "LP": 0, "P": 1},
        "binary_counts": {"negative": 1, "positive": 1},
        "keyset_sha256": "sha256:" + ("8" if role == "validation" else "9") * 64,
        "labels_jsonl": _file_identity(root, labels_relative),
        "vcf": _file_identity(root, vcf_relative),
    }


def _patch_membership_store(
    monkeypatch: pytest.MonkeyPatch,
    manifest: SimpleNamespace,
) -> None:
    class _Store:
        def __enter__(self) -> _Store:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __init__(self) -> None:
            self.manifest = manifest

    monkeypatch.setattr(
        dataset_package_module,
        "MembershipStore",
        SimpleNamespace(open=lambda *_args, **_kwargs: _Store()),
        raising=False,
    )


def _write_dataset_snapshot_report(root: Path) -> Path:
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    files = []
    for item in manifest["files"]:
        path = root / item["path"]
        files.append(
            {
                "path": item["path"],
                "source_path": f"sources/{Path(item['path']).name}",
                "source_sha256": sha256_file(path),
                "source_size_bytes": path.stat().st_size,
                "split": item["split"],
                "records": item["records"],
                "description": item["description"],
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "already_exists": False,
            }
        )
    payload = {
        "schema_version": "1.0.0",
        "generated_by": DATASET_SNAPSHOT_GENERATED_BY,
        "generated_at": manifest["generated_at"],
        "snapshot_id": manifest["snapshot_id"],
        "report_path": DATASET_SNAPSHOT_REPORT_NAME,
        "snapshot_spec": {
            "path": "dataset_snapshot.json",
            "sha256": "sha256:" + "1" * 64,
            "size_bytes": 1024,
        },
        "input_check_path": DATASET_INPUT_CHECK_REPORT_NAME,
        "input_check": _file_identity(root, _write_dataset_input_check_report(root, files)),
        "metadata_path": "dataset_package.json",
        "package": {
            "snapshot_id": manifest["snapshot_id"],
            "metadata": _file_identity(root, "dataset_package.json"),
            "manifest_path": "dataset_manifest.json",
            "manifest": _file_identity(root, "dataset_manifest.json"),
            "data_card_path": "data_card.md",
            "data_card": _file_identity(root, "data_card.md"),
            "integrity_path": DEFAULT_REPORT_NAME,
            "integrity": _file_identity(root, DEFAULT_REPORT_NAME),
            "checksums_path": "SHA256SUMS",
            "files": [
                {
                    key: file[key]
                    for key in ("path", "sha256", "size_bytes", "split", "records", "description")
                    if key in file
                }
                for file in files
            ],
        },
        "files": files,
    }
    report_path = root / DATASET_SNAPSHOT_REPORT_NAME
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_sha256sums(
        root,
        (
            "data_card.md",
            "dataset_package.json",
            "dataset_manifest.json",
            DEFAULT_REPORT_NAME,
            DATASET_INPUT_CHECK_REPORT_NAME,
            DATASET_SNAPSHOT_REPORT_NAME,
            *(file["path"] for file in manifest["files"]),
        ),
    )
    return report_path


def _write_dataset_input_check_report(root: Path, files: list[dict[str, object]]) -> str:
    inputs = []
    total_size_bytes = 0
    for item in files:
        size_bytes = item["source_size_bytes"]
        assert isinstance(size_bytes, int)
        total_size_bytes += size_bytes
        inputs.append(
            {
                "kind": "carbon" if str(item["path"]).startswith("carbon/") else "clinvar",
                "source_path": item["source_path"],
                "staged_path": item["path"],
                "split": item["split"],
                "description": item["description"],
                "sha256": item["source_sha256"],
                "size_bytes": size_bytes,
            }
        )
    payload = {
        "ok": True,
        "schema_version": "1.0.0",
        "generated_by": DATASET_INPUT_CHECK_GENERATED_BY,
        "snapshot_id": "geno-lewm-data-v0.1.0-r1",
        "snapshot_spec": {
            "path": "dataset_snapshot.json",
            "sha256": "sha256:" + "1" * 64,
            "size_bytes": 1024,
        },
        "source_count": len(inputs),
        "total_size_bytes": total_size_bytes,
        "inputs": inputs,
    }
    (root / DATASET_INPUT_CHECK_REPORT_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return DATASET_INPUT_CHECK_REPORT_NAME


def _file_identity(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_sha256sums(root: Path, files: tuple[str, ...]) -> None:
    lines = []
    for relative in files:
        digest = sha256_file(root / relative).removeprefix("sha256:")
        lines.append(f"{digest}  {relative}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metadata() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "snapshot_id": "geno-lewm-data-v0.1.0-r1",
        "generated_by": DATASET_PACKAGE_GENERATED_BY,
        "generated_at": "2026-06-01T00:00:00Z",
        "sources": [
            {
                "name": "Carbon pretraining corpus",
                "revision": "2026-04-15",
                "url": "https://huggingface.co/collections/HuggingFaceBio/carbon",
                "license": "upstream Carbon corpus terms",
                "notes": "deterministic Phase 1 source mix",
            },
            {
                "name": "ClinVar",
                "revision": "2026-04-15",
                "url": "https://www.ncbi.nlm.nih.gov/clinvar/",
                "license": "NCBI public data terms",
            },
        ],
        "license": "Apache-2.0 for package metadata; upstream data licenses apply.",
        "preprocessing": [
            "Select the pinned Carbon source mix and deterministic subset seed.",
            "Normalize VCF alleles to uppercase ACGT and split ClinVar coding/non-coding rows.",
        ],
        "split_policy": "Training windows exclude all ClinVar evaluation variants by genomic locus.",
        "splits": {
            "train": {"records": 1, "description": "Carbon corpus source-mix windows"},
            "eval_clinvar_coding": {"records": 1, "description": "held-out ClinVar coding SNVs"},
        },
        "leakage_checks": [
            "No ClinVar evaluation locus may appear in training-window positive labels.",
            "Split manifests are checked before publishing the dataset package.",
        ],
        "intended_use": "First GenoLeWM paper/demo release experiments and reproducibility checks.",
        "limitations": [
            "This package records the selected snapshot; it is not a clinical dataset.",
            "ClinVar labels may change after the pinned upstream release.",
        ],
        "files": [
            {
                "path": "carbon/windows.jsonl",
                "split": "train",
                "records": 1,
                "description": "sampled Carbon source-mix windows",
            },
            {
                "path": "clinvar/eval.vcf",
                "split": "eval_clinvar_coding",
                "records": 1,
                "description": "held-out ClinVar coding variants",
            },
        ],
    }
