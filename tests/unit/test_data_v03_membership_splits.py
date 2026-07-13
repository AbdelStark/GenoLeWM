# SPDX-License-Identifier: Apache-2.0
"""Contracts for checksum-closed v0.3 membership split evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from geno_lewm.data import _membership_store_writer as membership_store_writer
from geno_lewm.data.membership_store import (
    build_membership_store,
    verify_membership_store,
)
from geno_lewm.errors import InputError
from geno_lewm.provenance import canonical_json_sha256, sha256_file
from geno_lewm.provenance.hashing import canonical_json_bytes
from tests.unit.test_data_membership_store import (
    BUILDER_CONTAINER_IMAGE,
    BUILDER_GIT_COMMIT,
    _write_source_bundle,
)
from tools.data import v03_membership_splits as membership_splits
from tools.data.v03_membership_splits import build_membership_splits, main

EXPECTED_FILES = {
    "SHA256SUMS",
    "contract/membership-split-evidence.schema.json",
    "evidence/membership-split-evidence.json",
    "splits/evaluation/clinvar-chr21.labels.jsonl",
    "splits/evaluation/clinvar-chr21.vcf",
    "splits/validation/clinvar-chr20.labels.jsonl",
    "splits/validation/clinvar-chr20.vcf",
}
REPORT_PATH = Path("evidence/membership-split-evidence.json")
SCHEMA_PATH = Path("contract/membership-split-evidence.schema.json")
DATASET_SNAPSHOT_ID = "geno-lewm-data-v0.3.0-r1"
PLACED_WINDOWS_ARTIFACT_PATH = (
    "candidates/v0.3/geno-lewm-data-v0.3.0-r1/placed/gnomad-windows.jsonl"
)


@pytest.fixture(scope="module")
def built_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the production-shaped store around the repository's synthetic source fixture."""
    root = tmp_path_factory.mktemp("membership-split-sources")
    lineage_path, sources = _write_source_bundle(root)
    output = tmp_path_factory.mktemp("membership-split-store") / "store"

    # The production builder correctly requires a clean checkout. This local source fixture
    # is built while the test/implementation changes are intentionally uncommitted, so bypass
    # only that launcher-boundary check; the resulting store is still fully verified below.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv(
            "GENO_LEWM_VERIFIED_BUILD_CONTAINER_IMAGE",
            BUILDER_CONTAINER_IMAGE,
        )
        monkeypatch.setattr(
            membership_store_writer,
            "_verify_build_invocation",
            lambda _commit, _image: None,
        )
        build_membership_store(
            artifact_id="geno-lewm-v0.3-membership-split-fixture",
            snapshot_lineage_path=lineage_path,
            expected_snapshot_lineage_sha256=sha256_file(lineage_path),
            builder_git_commit=BUILDER_GIT_COMMIT,
            container_image=BUILDER_CONTAINER_IMAGE,
            sources=sources,
            output_dir=output,
        )

    assert verify_membership_store(output).ok is True
    return output


def test_builder_writes_deterministic_checksum_closed_evidence(
    tmp_path: Path,
    built_store: Path,
) -> None:
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl")
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_membership_splits(
        **_build_kwargs(built_store, windows, first),
        allow_fixture=True,
    )
    build_membership_splits(
        **_build_kwargs(built_store, windows, second),
        allow_fixture=True,
    )

    assert _relative_files(first) == EXPECTED_FILES
    assert _relative_files(second) == EXPECTED_FILES
    assert all(
        (first / relative).is_file() and not (first / relative).is_symlink()
        for relative in EXPECTED_FILES
    )
    for relative in sorted(EXPECTED_FILES):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()

    checksums = _read_sha256sums(first / "SHA256SUMS")
    assert set(checksums) == EXPECTED_FILES - {"SHA256SUMS"}
    assert list(checksums) == sorted(checksums)
    for relative, digest in checksums.items():
        assert digest == sha256_file(first / relative).removeprefix("sha256:")

    report = _read_json(first / REPORT_PATH)
    schema = _read_json(first / SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    assert not list(Draft202012Validator(schema).iter_errors(report))
    assert report["$schema"] == "../contract/membership-split-evidence.schema.json"
    assert report["schema_version"] == "geno-lewm.membership-split-evidence.v1"
    assert report["artifact_id"] == "geno-lewm-v03-membership-splits-fixture"
    assert report["assembly"] == "GRCh38"
    assert report["ok"] is True
    assert report["claim_boundary"]["variant_membership"] is True
    assert report["claim_boundary"]["phased_haplotype_membership"] is False
    assert report["claim_boundary"]["released_v03_snapshot"] is False
    assert report["claim_boundary"]["publication_eligible"] is False


def test_builder_rejects_a_report_that_does_not_match_the_supplied_schema(
    tmp_path: Path,
    built_store: Path,
) -> None:
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl")
    output = tmp_path / "output"
    schema = _read_json(Path("configs/data_v03/membership-split-evidence.schema.json"))
    schema["properties"]["artifact_id"] = {"const": "a-different-artifact"}
    schema_path = tmp_path / "drifted.schema.json"
    schema_path.write_text(json.dumps(schema, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="does not satisfy its bundled schema"):
        build_membership_splits(
            **_build_kwargs(built_store, windows, output),
            allow_fixture=True,
            report_schema_path=schema_path,
        )

    _assert_no_partial_output(output)


def test_labels_and_vcfs_have_exact_key_parity_and_report_real_class_counts(
    tmp_path: Path,
    built_store: Path,
) -> None:
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl")
    output = tmp_path / "output"
    build_membership_splits(
        **_build_kwargs(built_store, windows, output),
        allow_fixture=True,
    )

    expected = {
        "validation": [
            {
                "alt": "T",
                "chrom": "20",
                "clinical_significance": "B",
                "pos": 1020,
                "ref": "C",
            },
            {
                "alt": "T",
                "chrom": "20",
                "clinical_significance": "LP",
                "pos": 1120,
                "ref": "C",
            },
        ],
        "evaluation": [
            {
                "alt": "T",
                "chrom": "21",
                "clinical_significance": "P",
                "pos": 1021,
                "ref": "C",
            },
            {
                "alt": "T",
                "chrom": "21",
                "clinical_significance": "LB",
                "pos": 1121,
                "ref": "C",
            },
        ],
    }
    paths = {
        "validation": "splits/validation/clinvar-chr20",
        "evaluation": "splits/evaluation/clinvar-chr21",
    }
    for role, stem in paths.items():
        labels = _read_jsonl(output / f"{stem}.labels.jsonl")
        assert labels == expected[role]
        label_keys = [(row["chrom"], row["pos"], row["ref"], row["alt"]) for row in labels]
        vcf_path = output / f"{stem}.vcf"
        vcf_text = vcf_path.read_text(encoding="utf-8")
        assert "##fileformat=VCFv4.3\n" in vcf_text
        assert "##reference=GRCh38\n" in vcf_text
        assert "##source=GenoLeWM-v0.3-membership-splits\n" in vcf_text
        assert _read_vcf_keys(vcf_path) == label_keys
        assert len(label_keys) == len(set(label_keys))

    report = _read_json(output / REPORT_PATH)
    expected_counts = {
        "validation": {
            "binary": {"negative": 1, "positive": 1},
            "chromosome": "20",
            "classes": {"B": 1, "LB": 0, "LP": 1, "P": 0},
        },
        "evaluation": {
            "binary": {"negative": 1, "positive": 1},
            "chromosome": "21",
            "classes": {"B": 0, "LB": 1, "LP": 0, "P": 1},
        },
    }
    for role, stem in paths.items():
        stream = report["streams"][role]
        assert stream["role"] == role
        assert stream["chromosome"] == expected_counts[role]["chromosome"]
        assert stream["record_count"] == 2
        assert stream["class_counts"] == expected_counts[role]["classes"]
        assert stream["binary_counts"] == expected_counts[role]["binary"]
        assert stream["keyset_sha256"].startswith("sha256:")
        assert stream["labels_jsonl"] == _file_identity(output, f"{stem}.labels.jsonl")
        assert stream["vcf"] == _file_identity(output, f"{stem}.vcf")


def test_report_binds_exact_membership_and_training_window_provenance(
    tmp_path: Path,
    built_store: Path,
) -> None:
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl")
    output = tmp_path / "output"
    kwargs = _build_kwargs(built_store, windows, output)
    build_membership_splits(**kwargs, allow_fixture=True)

    report = _read_json(output / REPORT_PATH)
    assert report["producer"] == {
        "generated_by": "tools.data.v03_membership_splits",
        "git_commit": kwargs["producer_git_commit"],
        "container_image": kwargs["container_image"],
        "invocation_verified": False,
    }
    membership = report["membership_store"]
    assert membership["repository"] == kwargs["membership_repository"]
    assert membership["revision"] == kwargs["membership_revision"]
    assert membership["artifact_path"] == kwargs["membership_artifact_path"]
    assert membership["content_identity"] == kwargs["expected_store_content_identity"]
    assert membership["physical_identity"] == kwargs["expected_store_physical_identity"]
    assert membership["rowset_sha256"] == kwargs["expected_store_rowset_sha256"]
    assert membership["lineage"]["evidence_profile"] == "synthetic_fixture"
    assert membership["chromosome_roles"] == {
        "train": [*map(str, range(1, 20)), "22"],
        "validation": ["20"],
        "evaluation": ["21"],
    }

    training = report["training_windows"]
    assert training["source"] == {
        "artifact_path": kwargs["training_windows_artifact_path"],
        "repository": kwargs["training_windows_repository"],
        "revision": kwargs["training_windows_revision"],
    }
    assert training["sha256"] == kwargs["expected_placed_windows_sha256"]
    assert training["size_bytes"] == kwargs["expected_placed_windows_size_bytes"]
    assert training["record_count"] == len(_placed_window_rows())
    assert training["assembly"] == "GRCh38"
    assert training["role"] == "train"
    assert training["split"] == "train_placed_gnomad_common"
    dataset_manifest_json = Path(kwargs["dataset_manifest_json"])
    assert training["dataset_manifest"] == {
        "path": "dataset_manifest.json",
        "sha256": kwargs["expected_dataset_manifest_sha256"],
        "size_bytes": dataset_manifest_json.stat().st_size,
        "snapshot_id": kwargs["expected_dataset_snapshot_id"],
    }


def test_window_audit_is_exhaustive_and_sample_is_digest_stable(
    tmp_path: Path,
    built_store: Path,
) -> None:
    rows = _placed_window_rows()
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl", rows)
    seed = 73
    sample_size = 4
    expected_digest = _expected_sample_digest(
        rows,
        seed=seed,
        sample_size=sample_size,
    )

    outputs = (tmp_path / "first", tmp_path / "second")
    for output in outputs:
        build_membership_splits(
            **_build_kwargs(built_store, windows, output),
            sample_seed=seed,
            sample_size=sample_size,
            allow_fixture=True,
        )

    first = _read_json(outputs[0] / REPORT_PATH)["audits"]
    second = _read_json(outputs[1] / REPORT_PATH)["audits"]
    assert first == second
    assert first["exhaustive"] == {
        "indexed_overlaps": 0,
        "policy_exclusions": 0,
        "status": "passed",
        "windows_scanned": len(rows),
    }
    assert first["deterministic_sample"] == {
        "algorithm": "sha256-priority-v1",
        "indexed_overlaps": 0,
        "observed_size": sample_size,
        "policy_exclusions": 0,
        "requested_size": sample_size,
        "sample_digest": expected_digest,
        "seed": seed,
        "status": "passed",
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("expected_store_content_identity", "sha256:" + "0" * 64),
        ("expected_store_physical_identity", "sha256:" + "1" * 64),
        ("expected_store_rowset_sha256", "sha256:" + "2" * 64),
        ("expected_placed_windows_sha256", "sha256:" + "3" * 64),
        ("expected_placed_windows_size_bytes", -1),
    ],
)
def test_builder_rejects_pinned_identity_drift_without_partial_output(
    tmp_path: Path,
    built_store: Path,
    field: str,
    bad_value: object,
) -> None:
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl")
    output = tmp_path / "output"
    kwargs = _build_kwargs(built_store, windows, output)
    kwargs[field] = (
        windows.stat().st_size + 1 if field == "expected_placed_windows_size_bytes" else bad_value
    )

    with pytest.raises(InputError, match=r"identity|SHA-256|sha256|size"):
        build_membership_splits(**kwargs, allow_fixture=True)

    _assert_no_partial_output(output)


def test_builder_rejects_dataset_manifest_sha_drift_without_partial_output(
    tmp_path: Path,
    built_store: Path,
) -> None:
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl")
    output = tmp_path / "output"
    kwargs = _build_kwargs(built_store, windows, output)
    kwargs["expected_dataset_manifest_sha256"] = "sha256:" + "4" * 64

    with pytest.raises(InputError, match=r"dataset manifest|manifest.*identity|SHA-256"):
        build_membership_splits(**kwargs, allow_fixture=True)

    _assert_no_partial_output(output)


@pytest.mark.parametrize(
    "artifact_path",
    [
        "membership store",
        "membership/évidence",
        "membership/./store",
        "membership/../store",
        "/membership/store",
        "membership/store/",
        r"membership\store",
    ],
)
def test_builder_rejects_unsafe_artifact_paths_at_input_boundary(
    tmp_path: Path,
    built_store: Path,
    artifact_path: str,
) -> None:
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl")
    output = tmp_path / "output"
    kwargs = _build_kwargs(built_store, windows, output)
    kwargs["membership_artifact_path"] = artifact_path

    with pytest.raises(InputError, match=r"safe relative POSIX path"):
        build_membership_splits(**kwargs, allow_fixture=True)

    _assert_no_partial_output(output)


@pytest.mark.parametrize(
    "drift",
    [
        "snapshot_id",
        "path",
        "sha256",
        "size_bytes",
        "records",
        "split",
    ],
)
def test_builder_rejects_dataset_manifest_binding_drift_without_partial_output(
    tmp_path: Path,
    built_store: Path,
    drift: str,
) -> None:
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl")
    output = tmp_path / "output"
    kwargs = _build_kwargs(built_store, windows, output)
    manifest_path = Path(kwargs["dataset_manifest_json"])
    payload = _read_json(manifest_path)
    file_binding = payload["files"][0]
    if drift == "snapshot_id":
        payload["snapshot_id"] = "geno-lewm-data-v0.3.0-drifted"
    elif drift == "path":
        file_binding["path"] = "placed/a-different-window-artifact.jsonl"
    elif drift == "sha256":
        file_binding["sha256"] = "sha256:" + "5" * 64
    elif drift == "size_bytes":
        file_binding["size_bytes"] += 1
    elif drift == "records":
        file_binding["records"] += 1
    else:
        file_binding["split"] = "eval_clinvar"
    _write_json(manifest_path, payload)
    kwargs["expected_dataset_manifest_sha256"] = sha256_file(manifest_path)

    with pytest.raises(
        InputError,
        match=r"dataset|manifest|placed|record|snapshot|split|identity",
    ):
        build_membership_splits(**kwargs, allow_fixture=True)

    _assert_no_partial_output(output)


def test_builder_rejects_expected_placed_window_record_count_drift(
    tmp_path: Path,
    built_store: Path,
) -> None:
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl")
    output = tmp_path / "output"
    kwargs = _build_kwargs(built_store, windows, output)
    kwargs["expected_placed_windows_record_count"] += 1

    with pytest.raises(InputError, match=r"record.*count|count.*record|placed-window binding"):
        build_membership_splits(**kwargs, allow_fixture=True)

    _assert_no_partial_output(output)


def test_builder_rejects_synthetic_lineage_without_explicit_fixture_opt_in(
    tmp_path: Path,
    built_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl")
    blocked = tmp_path / "blocked"
    monkeypatch.setattr(
        membership_splits,
        "_verify_producer_invocation",
        lambda **_kwargs: None,
    )

    with pytest.raises(
        InputError,
        match=r"official lineage|synthetic.*fixture|fixture.*lineage",
    ):
        build_membership_splits(**_build_kwargs(built_store, windows, blocked))
    _assert_no_partial_output(blocked)

    allowed = tmp_path / "allowed"
    build_membership_splits(
        **_build_kwargs(built_store, windows, allowed),
        allow_fixture=True,
    )
    assert (
        _read_json(allowed / REPORT_PATH)["membership_store"]["lineage"]["evidence_profile"]
        == "synthetic_fixture"
    )


def test_fixture_mode_bypasses_invocation_gate_and_is_never_publication_eligible(
    tmp_path: Path,
    built_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl")
    output = tmp_path / "output"

    def _unexpected_invocation_check(**_kwargs: object) -> None:
        raise AssertionError("fixture mode must not claim an independently verified invocation")

    monkeypatch.setattr(
        membership_splits,
        "_verify_producer_invocation",
        _unexpected_invocation_check,
    )
    build_membership_splits(
        **_build_kwargs(built_store, windows, output),
        allow_fixture=True,
    )

    report = _read_json(output / REPORT_PATH)
    assert report["producer"]["invocation_verified"] is False
    assert report["claim_boundary"]["publication_eligible"] is False


def test_non_fixture_publication_fails_closed_when_invocation_cannot_be_verified(
    tmp_path: Path,
    built_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl")
    output = tmp_path / "output"

    def _reject_invocation(**_kwargs: object) -> None:
        raise InputError("producer invocation is not independently verified")

    monkeypatch.setattr(
        membership_splits,
        "_verify_producer_invocation",
        _reject_invocation,
    )
    with pytest.raises(InputError, match="invocation is not independently verified"):
        build_membership_splits(**_build_kwargs(built_store, windows, output))

    _assert_no_partial_output(output)


def test_non_fixture_publication_rejects_an_untracked_schema_override(
    tmp_path: Path,
    built_store: Path,
) -> None:
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl")
    output = tmp_path / "output"
    schema_path = tmp_path / "untracked.schema.json"
    schema_path.write_bytes(
        Path("configs/data_v03/membership-split-evidence.schema.json").read_bytes()
    )

    with pytest.raises(InputError, match=r"official.*tracked report schema"):
        build_membership_splits(
            **_build_kwargs(built_store, windows, output),
            report_schema_path=schema_path,
        )

    _assert_no_partial_output(output)


def test_official_invocation_gate_checks_exact_clean_canonical_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_image = "ghcr.io/astral-sh/uv@sha256:" + "c" * 64
    producer_commit = "d" * 40
    observed: list[tuple[str, ...]] = []

    def _git_output(_root: Path, *arguments: str) -> str:
        observed.append(arguments)
        if arguments == ("rev-parse", "HEAD"):
            return producer_commit
        if arguments == ("remote", "get-url", "origin"):
            return "https://github.com/AbdelStark/GenoLeWM.git"
        return ""

    monkeypatch.setenv(
        "GENO_LEWM_VERIFIED_SPLIT_CONTAINER_IMAGE",
        container_image,
    )
    monkeypatch.setattr(membership_splits, "_git_output", _git_output)

    membership_splits._verify_producer_invocation(
        producer_git_commit=producer_commit,
        container_image=container_image,
    )

    assert ("status", "--porcelain=v1", "--untracked-files=all") in observed
    assert ("remote", "get-url", "origin") in observed
    assert sum(arguments[:2] == ("cat-file", "-e") for arguments in observed) == 2


def test_official_invocation_gate_rejects_unbound_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENO_LEWM_VERIFIED_SPLIT_CONTAINER_IMAGE", raising=False)

    with pytest.raises(InputError, match="trusted launcher binding"):
        membership_splits._verify_producer_invocation(
            producer_git_commit="d" * 40,
            container_image="ghcr.io/astral-sh/uv@sha256:" + "c" * 64,
        )


@pytest.mark.parametrize(
    ("case", "mutate"),
    [
        ("missing chromosome", lambda row: row.pop("chrom")),
        ("unplaced chromosome", lambda row: row.__setitem__("chrom", None)),
        ("unassigned chromosome", lambda row: row.__setitem__("chrom", "X")),
        ("noncanonical chromosome", lambda row: row.__setitem__("chrom", "chr22")),
        ("empty record id", lambda row: row.__setitem__("record_id", "")),
        ("negative start", lambda row: row.__setitem__("start_bp", -1)),
        ("reversed interval", lambda row: row.__setitem__("end_bp", row["start_bp"])),
        ("sequence length drift", lambda row: row.__setitem__("sequence", "ACGT")),
        ("unexpected field", lambda row: row.__setitem__("manual_split", True)),
    ],
)
def test_builder_rejects_malformed_placed_window_rows(
    tmp_path: Path,
    built_store: Path,
    case: str,
    mutate: Callable[[dict[str, object]], object],
) -> None:
    rows = _placed_window_rows()
    mutate(rows[0])
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl", rows)
    output = tmp_path / "output"

    with pytest.raises(InputError, match=r"placed|window|chromosome|record|sequence"):
        build_membership_splits(
            **_build_kwargs(built_store, windows, output),
            allow_fixture=True,
        )

    _assert_no_partial_output(output)


def test_builder_rejects_duplicate_placed_window_identity(
    tmp_path: Path,
    built_store: Path,
) -> None:
    rows = _placed_window_rows()
    rows.append(dict(rows[0]))
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl", rows)
    output = tmp_path / "output"

    with pytest.raises(InputError, match=r"duplicate.*window|window.*duplicate"):
        build_membership_splits(
            **_build_kwargs(built_store, windows, output),
            allow_fixture=True,
        )

    _assert_no_partial_output(output)


def test_builder_negative_control_rejects_held_chromosome_training_window(
    tmp_path: Path,
    built_store: Path,
) -> None:
    rows = _placed_window_rows()
    rows[0]["chrom"] = "20"
    rows[0]["record_id"] = "gnomad:20:1-64"
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl", rows)
    output = tmp_path / "output"

    with pytest.raises(
        InputError,
        match=r"holdout|nonintersection|train.*role|chromosome",
    ):
        build_membership_splits(
            **_build_kwargs(built_store, windows, output),
            allow_fixture=True,
        )

    _assert_no_partial_output(output)


def test_builder_never_clobbers_an_existing_output_directory(
    tmp_path: Path,
    built_store: Path,
) -> None:
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl")
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "owner-data.txt"
    sentinel.write_bytes(b"do not replace\n")

    with pytest.raises(InputError, match=r"already exists|output"):
        build_membership_splits(
            **_build_kwargs(built_store, windows, output),
            allow_fixture=True,
        )

    assert _relative_files(output) == {"owner-data.txt"}
    assert sentinel.read_bytes() == b"do not replace\n"


def test_cli_returns_typed_input_error_without_publishing_partial_output(
    tmp_path: Path,
    built_store: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    windows = _write_placed_windows(tmp_path / "placed-windows.jsonl")
    output = tmp_path / "output"
    kwargs = _build_kwargs(built_store, windows, output)

    rc = main([*_cli_args(kwargs), "--sample-size", "0", "--allow-fixture"])
    captured = capsys.readouterr()

    assert rc == 2
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "sample_size must be a positive integer" in captured.err
    _assert_no_partial_output(output)


def _build_kwargs(
    built_store: Path,
    placed_windows_jsonl: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = verify_membership_store(built_store).manifest
    placed_window_records = len(_read_jsonl(placed_windows_jsonl))
    dataset_manifest_json = placed_windows_jsonl.with_name("dataset_manifest.json")
    _write_dataset_manifest(
        dataset_manifest_json,
        placed_windows_jsonl=placed_windows_jsonl,
        record_count=placed_window_records,
    )
    return {
        "store_dir": built_store,
        "placed_windows_jsonl": placed_windows_jsonl,
        "dataset_manifest_json": dataset_manifest_json,
        "output_dir": output_dir,
        "artifact_id": "geno-lewm-v03-membership-splits-fixture",
        "membership_repository": "abdelstark/geno-lewm-data",
        "membership_revision": "a" * 40,
        "membership_artifact_path": (
            "candidates/v0.3/geno-lewm-data-v0.3.0-r1/membership/fixture/store"
        ),
        "training_windows_repository": "abdelstark/geno-lewm-data",
        "training_windows_revision": "b" * 40,
        "training_windows_artifact_path": PLACED_WINDOWS_ARTIFACT_PATH,
        "expected_store_content_identity": manifest.content_identity,
        "expected_store_physical_identity": manifest.physical_identity,
        "expected_store_rowset_sha256": manifest.rowset_sha256,
        "expected_dataset_manifest_sha256": sha256_file(dataset_manifest_json),
        "expected_dataset_snapshot_id": DATASET_SNAPSHOT_ID,
        "expected_placed_windows_sha256": sha256_file(placed_windows_jsonl),
        "expected_placed_windows_size_bytes": placed_windows_jsonl.stat().st_size,
        "expected_placed_windows_record_count": placed_window_records,
        "producer_git_commit": BUILDER_GIT_COMMIT,
        "container_image": BUILDER_CONTAINER_IMAGE,
    }


def _cli_args(kwargs: Mapping[str, object]) -> list[str]:
    flags = (
        ("--store-dir", "store_dir"),
        ("--placed-windows-jsonl", "placed_windows_jsonl"),
        ("--dataset-manifest-json", "dataset_manifest_json"),
        ("--output-dir", "output_dir"),
        ("--artifact-id", "artifact_id"),
        ("--membership-repository", "membership_repository"),
        ("--membership-revision", "membership_revision"),
        ("--membership-artifact-path", "membership_artifact_path"),
        ("--training-windows-repository", "training_windows_repository"),
        ("--training-windows-revision", "training_windows_revision"),
        ("--training-windows-artifact-path", "training_windows_artifact_path"),
        ("--expected-store-content-identity", "expected_store_content_identity"),
        ("--expected-store-physical-identity", "expected_store_physical_identity"),
        ("--expected-store-rowset-sha256", "expected_store_rowset_sha256"),
        ("--expected-dataset-manifest-sha256", "expected_dataset_manifest_sha256"),
        ("--expected-dataset-snapshot-id", "expected_dataset_snapshot_id"),
        ("--expected-placed-windows-sha256", "expected_placed_windows_sha256"),
        ("--expected-placed-windows-size-bytes", "expected_placed_windows_size_bytes"),
        ("--expected-placed-windows-record-count", "expected_placed_windows_record_count"),
        ("--producer-git-commit", "producer_git_commit"),
        ("--container-image", "container_image"),
    )
    return [item for flag, key in flags for item in (flag, str(kwargs[key]))]


def _placed_window_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, chromosome in enumerate(("1", "2", "3", "4", "5", "6", "7", "8", "9", "22")):
        start = index * 100
        end = start + 64
        rows.append(
            {
                "record_id": f"gnomad:{chromosome}:{start + 1}-{end}",
                "source": "gnomad_common",
                "variant_source": "gnomad",
                "chrom": chromosome,
                "start_bp": start,
                "end_bp": end,
                "sequence": "ACGT" * 16,
                "variant_count": 1,
            }
        )
    return rows


def _write_placed_windows(
    path: Path,
    rows: list[dict[str, object]] | None = None,
) -> Path:
    values = _placed_window_rows() if rows is None else rows
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in values),
        encoding="utf-8",
    )
    return path


def _write_dataset_manifest(
    path: Path,
    *,
    placed_windows_jsonl: Path,
    record_count: int,
) -> Path:
    _write_json(
        path,
        {
            "schema_version": "1.0.0",
            "snapshot_id": DATASET_SNAPSHOT_ID,
            "generated_by": "tools.release.dataset_package",
            "files": [
                {
                    "path": PLACED_WINDOWS_ARTIFACT_PATH,
                    "sha256": sha256_file(placed_windows_jsonl),
                    "size_bytes": placed_windows_jsonl.stat().st_size,
                    "records": record_count,
                    "split": "train_placed_gnomad_common",
                }
            ],
            "splits": {
                "train_placed_gnomad_common": {
                    "records": record_count,
                }
            },
        },
    )
    return path


def _expected_sample_digest(
    rows: list[dict[str, object]],
    *,
    seed: int,
    sample_size: int,
) -> str:
    ranked: list[tuple[str, dict[str, object]]] = []
    for row in rows:
        identity = {
            "record_id": row["record_id"],
            "chrom": row["chrom"],
            "start_bp": row["start_bp"],
            "end_bp": row["end_bp"],
            "window_sha256": "sha256:" + hashlib.sha256(str(row["sequence"]).encode()).hexdigest(),
        }
        priority = hashlib.sha256(
            str(seed).encode() + b"\x00" + canonical_json_bytes(identity)
        ).hexdigest()
        ranked.append((priority, identity))
    sample_payload = [
        {"priority_sha256": "sha256:" + priority, **identity}
        for priority, identity in sorted(ranked)[:sample_size]
    ]
    return canonical_json_sha256(sample_payload)


def _file_identity(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _relative_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }


def _read_sha256sums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)
        assert relative not in entries
        entries[relative] = digest
    return entries


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _read_vcf_keys(path: Path) -> list[tuple[str, int, str, str]]:
    return [
        (fields[0], int(fields[1]), fields[3], fields[4])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
        for fields in (line.split("\t"),)
    ]


def _assert_no_partial_output(output: Path) -> None:
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.tmp-*"))
