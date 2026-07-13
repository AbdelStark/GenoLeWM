# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the canonical v0.3 dataset snapshot assembler."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.release.dataset_package as dataset_package_module
import tools.release.v03_dataset_snapshot as snapshot_module
from geno_lewm.errors import InputError, RuntimeSetupError
from geno_lewm.provenance import sha256_file
from geno_lewm.training.preflight import TrainingPreflightIssue, _inspect_dataset
from tools.release.paper_package import PackageIssue, _verify_dataset_dir
from tools.release.v03_dataset_snapshot import (
    assemble_v03_dataset_snapshot,
    filter_membership_parquet,
    verify_v03_dataset_snapshot,
)


def test_filter_membership_parquet_preserves_rows_schema_and_order(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source.parquet"
    output = tmp_path / "filtered.parquet"
    schema = pa.schema(
        [
            ("chrom", pa.string()),
            ("pos", pa.int64()),
            ("ref", pa.string()),
            ("alt", pa.string()),
            ("payload", pa.string()),
        ],
        metadata={b"source-release": b"fixture-r1"},
    )
    rows = [
        {"chrom": "1", "pos": 10, "ref": "A", "alt": "C", "payload": "keep-a"},
        {"chrom": "1", "pos": 20, "ref": "G", "alt": "T", "payload": "drop"},
        {"chrom": "1", "pos": 30, "ref": "C", "alt": "A", "payload": "keep-b"},
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), source)

    report = filter_membership_parquet(
        source,
        output,
        kind="gnomad",
        expected_source_row_ids={"1:10:A:C", "1:30:C:A"},
    )

    filtered = pq.read_table(output)
    assert filtered.to_pylist() == [rows[0], rows[2]]
    assert filtered.schema.equals(schema, check_metadata=True)
    assert report.records == 2
    assert report.source_rows == 3


def test_filter_membership_parquet_closes_footer_reader_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source.parquet"
    output = tmp_path / "filtered.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"chrom": "1", "pos": 10, "ref": "A", "alt": "C"}]),
        source,
    )
    opened: list[object] = []
    parquet_file = pq.ParquetFile

    class TrackingParquetFile:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._inner = parquet_file(*args, **kwargs)
            opened.append(self)

        @property
        def closed(self) -> bool:
            return bool(self._inner.closed)

        def close(self) -> None:
            self._inner.close()

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    parquet_proxy = SimpleNamespace(
        ParquetFile=TrackingParquetFile,
        ParquetWriter=pq.ParquetWriter,
    )
    monkeypatch.setattr(snapshot_module, "_require_pyarrow", lambda: (pa, parquet_proxy))
    path_replace = Path.replace

    def guarded_replace(path: Path, target: str | Path) -> Path:
        assert len(opened) == 2
        assert opened[-1].closed is True
        return path_replace(path, target)

    monkeypatch.setattr(Path, "replace", guarded_replace)

    filter_membership_parquet(
        source,
        output,
        kind="gnomad",
        expected_source_row_ids={"1:10:A:C"},
    )

    assert output.is_file()


def test_filter_membership_parquet_rejects_missing_membership_row(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"chrom": "1", "pos": 10, "ref": "A", "alt": "C"}]),
        source,
    )

    with pytest.raises(InputError, match="membership rows were absent"):
        filter_membership_parquet(
            source,
            tmp_path / "filtered.parquet",
            kind="gnomad",
            expected_source_row_ids={"1:99:A:G"},
        )

    assert not (tmp_path / "filtered.parquet").exists()


def test_filter_membership_parquet_rejects_duplicate_selected_source_row(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source.parquet"
    row = {"chrom": "1", "pos": 10, "ref": "A", "alt": "C"}
    pq.write_table(pa.Table.from_pylist([row, row]), source)

    with pytest.raises(InputError, match="duplicate selected source row"):
        filter_membership_parquet(
            source,
            tmp_path / "filtered.parquet",
            kind="gnomad",
            expected_source_row_ids={"1:10:A:C"},
        )


def test_filter_membership_parquet_rejects_symlink_source(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source.parquet"
    alias = tmp_path / "alias.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"chrom": "1", "pos": 10, "ref": "A", "alt": "C"}]),
        source,
    )
    try:
        alias.symlink_to(source.name)
    except OSError:
        pytest.skip("symbolic links are unavailable")

    with pytest.raises(InputError, match="regular non-symlink"):
        filter_membership_parquet(
            alias,
            tmp_path / "filtered.parquet",
            kind="gnomad",
            expected_source_row_ids={"1:10:A:C"},
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes are not portable to Windows")
def test_snapshot_copy_creates_private_staging_file(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    target = tmp_path / "stage" / "target.txt"
    source.write_text("public upstream bytes\n", encoding="utf-8")
    source.chmod(0o644)

    snapshot_module._copy_regular_file(source, target)

    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_mode & 0o077 == 0


def test_snapshot_copy_opens_source_and_target_in_binary_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.txt"
    target = tmp_path / "stage" / "target.txt"
    source.write_bytes(b"canonical\nbytes\n")
    binary_flag = 1 << 30
    observed_flags: list[int] = []
    os_open = os.open

    def tracking_open(path: str | os.PathLike[str], flags: int, mode: int = 0o600) -> int:
        observed_flags.append(flags)
        return os_open(path, flags & ~binary_flag, mode)

    monkeypatch.setattr(snapshot_module.os, "O_BINARY", binary_flag, raising=False)
    monkeypatch.setattr(snapshot_module.os, "open", tracking_open)

    snapshot_module._copy_regular_file(source, target)

    assert len(observed_flags) == 2
    assert all(flags & binary_flag for flags in observed_flags)
    assert target.read_bytes() == b"canonical\nbytes\n"


def test_clinvar_filter_uses_the_exact_membership_source_row_identity(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "clinvar.parquet"
    rows = [
        {
            "chrom": "chr1",
            "pos": 10,
            "ref": "A",
            "alt": "C",
            "clinvar_id": 42,
            "clinical_significance": "P",
        },
        {
            "chrom": "chr20",
            "pos": 20,
            "ref": "G",
            "alt": "T",
            "clinvar_id": 43,
            "clinical_significance": "B",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)

    filter_membership_parquet(
        source,
        tmp_path / "train.parquet",
        kind="clinvar",
        expected_source_row_ids={"42:chr1:10:A:C"},
    )

    assert pq.read_table(tmp_path / "train.parquet").to_pylist() == [rows[0]]


def test_assembler_builds_and_reverifies_a_closed_role_bound_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    fixture = _write_assembly_fixture(tmp_path)
    _patch_membership_store(monkeypatch, fixture.manifest, fixture.train_rows)
    output = tmp_path / "published-snapshot"

    report = assemble_v03_dataset_snapshot(
        membership_bundle_dir=fixture.membership_bundle,
        split_bundle_dir=fixture.split_bundle,
        gnomad_root=fixture.gnomad_root,
        clinvar_root=fixture.clinvar_root,
        training_windows_path=fixture.training_windows,
        dataset_dir=output,
        split_repository="owner/data",
        split_revision="d" * 40,
        split_artifact_path="candidates/v0.3/splits/r1/success",
        snapshot_id="geno-lewm-data-v0.3.0-fixture-r1",
        generated_at="2026-07-13T18:00:00Z",
        producer_git_commit="e" * 40,
        container_image="ghcr.io/example/uv@sha256:" + "f" * 64,
    )
    verified = verify_v03_dataset_snapshot(output)
    strict_verified = verify_v03_dataset_snapshot(
        output,
        gnomad_root=fixture.gnomad_root,
        clinvar_root=fixture.clinvar_root,
    )

    assert report.to_dict() == verified.to_dict()
    assert strict_verified.to_dict() == verified.to_dict()
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    snapshot_report = json.loads(
        (output / "dataset_snapshot_report.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "1.1.0"
    assert manifest["splits"]["train_gnomad_common"]["records"] == 21
    assert manifest["splits"]["train_clinvar"]["records"] == 1
    assert manifest["splits"]["validation"]["records"] == 2
    assert manifest["splits"]["evaluation"]["records"] == 2
    assert snapshot_report["v03"]["observed_splits"]["validation"]["observed_records"] == 2
    assert snapshot_report["v03"]["observed_splits"]["validation"]["label_counts"] == {
        "B": 1,
        "P": 1,
    }
    assert snapshot_report["v03"]["observed_splits"]["evaluation"]["observed_records"] == 2
    assert snapshot_report["v03"]["observed_splits"]["evaluation"]["label_counts"] == {
        "B": 1,
        "P": 1,
    }
    claim_boundary = snapshot_report["v03"]["claim_boundary"]
    assert claim_boundary["variant_membership"] is True
    assert claim_boundary["standalone_upstream_nonidentity_value_replay"] is False
    assert any(
        "does not re-download prepared upstream Parquets" in limitation
        for limitation in claim_boundary["limitations"]
    )
    clinvar_rows = pq.read_table(output / "clinvar/2026-04-15/train.variants.parquet").to_pylist()
    assert [(row["chrom"], row["clinvar_id"]) for row in clinvar_rows] == [("1", 9001)]
    assert "## Membership and Split Evidence" in (output / "data_card.md").read_text(
        encoding="utf-8"
    )
    assert not any(path.is_symlink() for path in output.rglob("*"))
    paper_package_issues: list[PackageIssue] = []
    _verify_dataset_dir(output, paper_package_issues)
    assert paper_package_issues == []
    training_preflight_issues: list[TrainingPreflightIssue] = []
    _inspect_dataset(output, True, training_preflight_issues)
    assert training_preflight_issues == []


@pytest.mark.parametrize(
    ("held_split", "chrom", "position"),
    (("validation", "20", 20001), ("evaluation", "21", 21001)),
)
def test_assembler_rejects_training_window_intersecting_held_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    held_split: str,
    chrom: str,
    position: int,
) -> None:
    fixture = _write_assembly_fixture(tmp_path)
    _patch_membership_store(monkeypatch, fixture.manifest, fixture.train_rows)
    window = {
        "record_id": "overlapping-window",
        "source": "reference",
        "variant_source": "gnomad",
        "chrom": chrom,
        "start_bp": position - 1,
        "end_bp": position + 1,
        "sequence": "AC",
        "variant_count": 0,
    }
    fixture.training_windows.write_text(json.dumps(window, sort_keys=True) + "\n", encoding="utf-8")
    report_path = fixture.split_bundle / "evidence/membership-split-evidence.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["training_windows"].update(
        {
            "sha256": sha256_file(fixture.training_windows),
            "size_bytes": fixture.training_windows.stat().st_size,
        }
    )
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    _write_checksums(fixture.split_bundle)

    with pytest.raises(InputError, match="dataset split leakage check failed") as excinfo:
        _assemble_fixture_snapshot(fixture, tmp_path / f"overlap-{held_split}")

    assert excinfo.value.details["failure_reason"] == "intersecting_genomic_regions"
    assert excinfo.value.details["split_b"] == held_split
    assert excinfo.value.details["region_overlap_count"] == 2


def test_assembler_rejects_split_lineage_for_a_different_candidate_snapshot(
    tmp_path: Path,
) -> None:
    fixture = _write_assembly_fixture(tmp_path)
    report_path = fixture.split_bundle / "evidence/membership-split-evidence.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["membership_store"]["lineage"]["candidate_snapshot_id"] = (
        "geno-lewm-data-v0.3.0-other-r1"
    )
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    _write_checksums(fixture.split_bundle)

    with pytest.raises(InputError, match="candidate_snapshot_id"):
        assemble_v03_dataset_snapshot(
            membership_bundle_dir=fixture.membership_bundle,
            split_bundle_dir=fixture.split_bundle,
            gnomad_root=fixture.gnomad_root,
            clinvar_root=fixture.clinvar_root,
            training_windows_path=fixture.training_windows,
            dataset_dir=tmp_path / "output",
            split_repository="owner/data",
            split_revision="d" * 40,
            split_artifact_path="candidates/v0.3/splits/r1/success",
            snapshot_id="geno-lewm-data-v0.3.0-fixture-r1",
            generated_at="2026-07-13T18:00:00Z",
            producer_git_commit="e" * 40,
            container_image="ghcr.io/example/uv@sha256:" + "f" * 64,
        )


def test_assembler_rejects_a_structurally_valid_but_impossible_utc_timestamp(
    tmp_path: Path,
) -> None:
    fixture = _write_assembly_fixture(tmp_path)

    with pytest.raises(InputError, match="valid calendar timestamp"):
        assemble_v03_dataset_snapshot(
            membership_bundle_dir=fixture.membership_bundle,
            split_bundle_dir=fixture.split_bundle,
            gnomad_root=fixture.gnomad_root,
            clinvar_root=fixture.clinvar_root,
            training_windows_path=fixture.training_windows,
            dataset_dir=tmp_path / "output",
            split_repository="owner/data",
            split_revision="d" * 40,
            split_artifact_path="candidates/v0.3/splits/r1/success",
            snapshot_id="geno-lewm-data-v0.3.0-fixture-r1",
            generated_at="2026-99-99T99:99:99Z",
            producer_git_commit="e" * 40,
            container_image="ghcr.io/example/uv@sha256:" + "f" * 64,
        )


@pytest.mark.parametrize(
    "timestamp",
    ("2026-07-13T18:00:00Z", "2026-07-13T18:00:00.1Z", "2026-07-13T18:00:00.123456Z"),
)
def test_utc_timestamp_accepts_canonical_whole_and_fractional_seconds(timestamp: str) -> None:
    assert snapshot_module._utc_timestamp(timestamp, "generated_at") == timestamp


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    (
        ("missing", "do not exactly match bound store memberships"),
        ("duplicate", "duplicate source_row_id"),
        ("substituted", "do not exactly match bound store memberships"),
    ),
)
def test_standalone_verifier_rejects_coherently_repackaged_train_row_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    expected_error: str,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    fixture = _write_assembly_fixture(tmp_path)
    _patch_membership_store(monkeypatch, fixture.manifest, fixture.train_rows)
    output = tmp_path / "published-snapshot"
    _assemble_fixture_snapshot(fixture, output)
    shard = output / "gnomad/v4.1/train/chr1.variants.parquet"
    table = pq.read_table(shard)
    rows = table.to_pylist()
    if tamper == "missing":
        rows = rows[:1]
    elif tamper == "duplicate":
        rows = [rows[0], rows[0]]
    else:
        substituted = dict(rows[0])
        substituted["pos"] = int(substituted["pos"]) + 1
        rows = [substituted, *rows[1:]]
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), shard)
    _rewrite_snapshot_identities_after_tamper(output)

    with pytest.raises(InputError, match=expected_error):
        verify_v03_dataset_snapshot(output)


def test_strict_upstream_replay_rejects_nonidentity_value_tampering_that_default_declares(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    fixture = _write_assembly_fixture(tmp_path)
    _patch_membership_store(monkeypatch, fixture.manifest, fixture.train_rows)
    output = tmp_path / "published-snapshot"
    _assemble_fixture_snapshot(fixture, output)
    shard = output / "gnomad/v4.1/train/chr1.variants.parquet"
    table = pq.read_table(shard)
    rows = table.to_pylist()
    rows[0] = {**rows[0], "af_global": 0.999}
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), shard)
    _rewrite_snapshot_identities_after_tamper(output)

    verify_v03_dataset_snapshot(output)
    with pytest.raises(InputError, match="strict upstream replay output differs"):
        verify_v03_dataset_snapshot(
            output,
            gnomad_root=fixture.gnomad_root,
            clinvar_root=fixture.clinvar_root,
        )


def test_strict_upstream_replay_requires_both_prepared_source_roots(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="requires both"):
        verify_v03_dataset_snapshot(tmp_path, gnomad_root=tmp_path)


def test_json_object_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"snapshot_id":"first","snapshot_id":"second"}\n', encoding="utf-8")

    with pytest.raises(InputError, match="duplicate JSON key"):
        snapshot_module._json_object(path, "duplicate fixture")


def test_atomic_publication_fails_closed_without_noreplace_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "target"
    monkeypatch.setattr(sys, "platform", "unsupported")
    monkeypatch.setattr(
        ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    with pytest.raises(RuntimeSetupError, match="no-replace"):
        snapshot_module._publish_directory_noreplace(source, target)

    assert source.is_dir()
    assert not target.exists()


def test_assembler_rejects_an_extra_membership_bundle_file(tmp_path: Path) -> None:
    fixture = _write_assembly_fixture(tmp_path)
    (fixture.membership_bundle / "unexpected.txt").write_text("drift\n", encoding="utf-8")

    with pytest.raises(InputError, match="exact inventory drifted"):
        assemble_v03_dataset_snapshot(
            membership_bundle_dir=fixture.membership_bundle,
            split_bundle_dir=fixture.split_bundle,
            gnomad_root=fixture.gnomad_root,
            clinvar_root=fixture.clinvar_root,
            training_windows_path=fixture.training_windows,
            dataset_dir=tmp_path / "output",
            split_repository="owner/data",
            split_revision="d" * 40,
            split_artifact_path="candidates/v0.3/splits/r1/success",
            snapshot_id="fixture-r1",
            generated_at="2026-07-13T18:00:00Z",
            producer_git_commit="e" * 40,
            container_image="ghcr.io/example/uv@sha256:" + "f" * 64,
        )


def test_snapshot_job_is_exact_revision_fail_closed_and_remotely_reverified() -> None:
    script = Path("tools/jobs/v03_publish_dataset_snapshot.sh").read_text(encoding="utf-8")

    for token in (
        "set -euo pipefail",
        'COMMIT_SHA="${COMMIT_SHA:?COMMIT_SHA is required}"',
        'CONTAINER_IMAGE="${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"',
        'GENERATED_AT="${GENERATED_AT:?GENERATED_AT is required}"',
        '[[ "$COMMIT_SHA" =~ ^[0-9a-f]{40}$ ]]',
        "git status --porcelain=v1 --untracked-files=all",
        "--force-download",
        "96e97a7ffe1e9ad8f9a98f690b220a32ac75ddc2",
        "6d2ec7dd68af636ba8c594774c3c55a236c0995f",
        "f3676763b3f7f71d0d0d098588e9bf377faa0c5c",
        "9e1a2b279681177a7ca00b30b9eb8048b511d1cb",
        "tools.release.v03_dataset_snapshot assemble",
        "tools.release.v03_dataset_snapshot verify",
        '--gnomad-root "$GNOMAD_ROOT"',
        '--clinvar-root "$CLINVAR_ROOT"',
        "LOCAL_STRICT_UPSTREAM_REPLAY_OK",
        "REMOTE_STRICT_UPSTREAM_REPLAY_OK",
        "tools.data.v03_gnomad_lock publish",
        'HUB_REVISION="${PUBLISH_REPORT#uploaded commit: }"',
        '--revision "$HUB_REVISION"',
        "GENO_LEWM_V03_SNAPSHOT_OK",
    ):
        assert token in script
    assert '"main"' not in script
    assert "|| true" not in script
    assert "hf upload" not in script


def test_snapshot_module_cli_entrypoint_is_wired() -> None:
    completed = subprocess.run(
        (sys.executable, "-m", "tools.release.v03_dataset_snapshot", "--help"),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "assemble" in completed.stdout
    assert "verify" in completed.stdout


@pytest.mark.skipif(os.name == "nt", reason="HF Jobs executes this contract on Linux")
def test_snapshot_job_is_valid_bash() -> None:
    subprocess.run(("bash", "-n", "tools/jobs/v03_publish_dataset_snapshot.sh"), check=True)


def _write_assembly_fixture(root: Path) -> SimpleNamespace:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    membership_bundle = root / "membership-bundle"
    split_bundle = root / "split-bundle"
    gnomad_root = root / "gnomad-download"
    clinvar_root = root / "clinvar-download"
    windows = root / "windows.jsonl"
    for directory in (membership_bundle, split_bundle, gnomad_root, clinvar_root):
        directory.mkdir(parents=True)

    roles = SimpleNamespace(
        to_dict=lambda: {
            "train": [*(str(value) for value in range(1, 20)), "22"],
            "validation": ["20"],
            "evaluation": ["21"],
        }
    )
    manifest = SimpleNamespace(
        artifact_id="fixture-membership",
        content_identity="sha256:" + "1" * 64,
        physical_identity="sha256:" + "2" * 64,
        rowset_sha256="sha256:" + "3" * 64,
        chromosome_roles=roles,
        snapshot_lineage=SimpleNamespace(candidate_snapshot_id="geno-lewm-data-v0.3.0-fixture-r1"),
        role_counts={"train": 22, "validation": 2, "evaluation": 2},
    )
    store = membership_bundle / "store"
    store.mkdir()
    (store / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": manifest.artifact_id,
                "content_identity": manifest.content_identity,
                "physical_identity": manifest.physical_identity,
                "rowset_sha256": manifest.rowset_sha256,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for name in ("build-receipt.json", "snapshot-lineage.json"):
        (store / name).write_text("{}\n", encoding="utf-8")
    (store / "memberships.parquet").write_bytes(b"fixture-memberships\n")
    (store / "lookup.sqlite").write_bytes(b"fixture-index\n")

    gnomad_schema = pa.schema(
        [
            ("chrom", pa.string()),
            ("pos", pa.int64()),
            ("ref", pa.string()),
            ("alt", pa.string()),
            ("af_global", pa.float64()),
            ("filter", pa.string()),
            ("schema_version", pa.string()),
        ],
        metadata={b"release": b"v4.1-fixture"},
    )
    source_entries: list[dict[str, object]] = []
    spec_sources: list[dict[str, object]] = []
    train_rows: list[SimpleNamespace] = []
    for chromosome in map(str, range(1, 23)):
        artifact_path = f"staging/gnomad/chr{chromosome}/data/gnomad/v4.1/variants.parquet"
        path = gnomad_root / artifact_path
        path.parent.mkdir(parents=True)
        pos = int(chromosome) * 100
        shard_rows = [
            {
                "chrom": chromosome,
                "pos": pos,
                "ref": "A",
                "alt": "C",
                "af_global": 0.1,
                "filter": "PASS",
                "schema_version": "2.0.0",
            }
        ]
        if chromosome == "1":
            shard_rows.append({**shard_rows[0], "pos": pos + 1, "alt": "G"})
        pq.write_table(pa.Table.from_pylist(shard_rows, schema=gnomad_schema), path)
        source_entries.append(
            {
                "kind": "gnomad",
                "chromosome": chromosome,
                "artifact_path": artifact_path,
                "revision": "a" * 40,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
        spec_sources.append(
            {
                "kind": "gnomad",
                "chromosome": chromosome,
                "path": f"gnomad/{artifact_path}",
            }
        )
        if chromosome not in {"20", "21"}:
            train_rows.extend(
                SimpleNamespace(
                    role="train",
                    source=f"gnomad-v4.1-chr{chromosome}",
                    source_row_id=f"{row['chrom']}:{row['pos']}:{row['ref']}:{row['alt']}",
                )
                for row in shard_rows
            )

    clinvar_artifact = "staging/clinvar/clinvar/2026-04-15/variants.parquet"
    clinvar_path = clinvar_root / clinvar_artifact
    clinvar_path.parent.mkdir(parents=True)
    clinvar_schema = pa.schema(
        [
            ("chrom", pa.string()),
            ("pos", pa.int64()),
            ("ref", pa.string()),
            ("alt", pa.string()),
            ("clinical_significance", pa.string()),
            ("review_status", pa.string()),
            ("gene_symbol", pa.string()),
            ("clinvar_id", pa.int64()),
            ("schema_version", pa.string()),
        ],
        metadata={b"release": b"2026-04-15-fixture"},
    )
    clinvar_rows = [
        {
            "chrom": chrom,
            "pos": pos,
            "ref": "G",
            "alt": "T",
            "clinical_significance": label,
            "review_status": "criteria_provided",
            "gene_symbol": "GENE",
            "clinvar_id": identifier,
            "schema_version": "1.0.0",
        }
        for chrom, pos, label, identifier in (
            ("1", 9999, "P", 9001),
            ("20", 20001, "B", 9002),
            ("21", 21001, "P", 9003),
        )
    ]
    pq.write_table(pa.Table.from_pylist(clinvar_rows, schema=clinvar_schema), clinvar_path)
    source_entries.append(
        {
            "kind": "clinvar",
            "artifact_path": clinvar_artifact,
            "revision": "b" * 40,
            "sha256": sha256_file(clinvar_path),
            "size_bytes": clinvar_path.stat().st_size,
        }
    )
    spec_sources.append({"kind": "clinvar", "path": f"clinvar/{clinvar_artifact}"})
    train_rows.append(
        SimpleNamespace(
            role="train",
            source="clinvar-2026-04-15",
            source_row_id="9001:1:9999:G:T",
        )
    )

    evidence = membership_bundle / "evidence"
    contract = membership_bundle / "contract"
    evidence.mkdir()
    contract.mkdir()
    source_payload = {
        "ok": True,
        "candidate_snapshot_id": "geno-lewm-data-v0.3.0-fixture-r1",
        "source_count": 23,
        "files": source_entries,
        "repositories": {
            "gnomad": {"repo_id": "owner/data", "repo_type": "dataset", "revision": "a" * 40},
            "clinvar": {"repo_id": "owner/data", "repo_type": "dataset", "revision": "b" * 40},
        },
    }
    (evidence / "source-download-identities.json").write_text(
        json.dumps(source_payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    (evidence / "job-summary.json").write_text(
        json.dumps(
            {"inputs": {"gnomad_revision": "a" * 40, "clinvar_revision": "b" * 40}},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (contract / "membership-build.json").write_text(
        json.dumps({"sources": spec_sources}, sort_keys=True) + "\n", encoding="utf-8"
    )
    for path in (
        contract / "membership-build-receipt.schema.json",
        contract / "membership-build-spec.schema.json",
        contract / "membership-store.schema.json",
        evidence / "download-plan.json",
        evidence / "membership-build-report.json",
        evidence / "membership-verify-report.json",
    ):
        path.write_text("{}\n", encoding="utf-8")
    _write_checksums(membership_bundle)

    window_row = {
        "record_id": "window-1",
        "source": "reference",
        "variant_source": "gnomad",
        "chrom": "22",
        "start_bp": 0,
        "end_bp": 8,
        "sequence": "ACGTACGT",
        "variant_count": 0,
    }
    windows.write_text(json.dumps(window_row, sort_keys=True) + "\n", encoding="utf-8")
    stream_payloads = {
        role: _write_stream(split_bundle, role=role, chrom=chrom, pos=pos, label=label)
        for role, chrom, pos, label in (
            ("validation", "20", 20001, "B"),
            ("evaluation", "21", 21001, "P"),
        )
    }
    split_contract = split_bundle / "contract"
    split_evidence = split_bundle / "evidence"
    split_contract.mkdir()
    split_evidence.mkdir()
    (split_contract / "membership-split-evidence.schema.json").write_bytes(
        Path("configs/data_v03/membership-split-evidence.schema.json").read_bytes()
    )
    report = {
        "$schema": "../contract/membership-split-evidence.schema.json",
        "schema_version": "geno-lewm.membership-split-evidence.v1",
        "artifact_id": "fixture-splits",
        "assembly": "GRCh38",
        "ok": True,
        "producer": {
            "generated_by": "tools.data.v03_membership_splits",
            "git_commit": "c" * 40,
            "container_image": "ghcr.io/example/uv@sha256:" + "4" * 64,
            "invocation_verified": True,
        },
        "membership_store": {
            "repository": "owner/data",
            "revision": "5" * 40,
            "artifact_path": "candidates/v0.3/membership/r1/success",
            "artifact_id": manifest.artifact_id,
            "content_identity": manifest.content_identity,
            "physical_identity": manifest.physical_identity,
            "rowset_sha256": manifest.rowset_sha256,
            "lineage": {
                "lineage_id": "sha256:" + "6" * 64,
                "sha256": "sha256:" + "7" * 64,
                "candidate_snapshot_id": "geno-lewm-data-v0.3.0-fixture-r1",
                "evidence_profile": "official",
            },
            "chromosome_roles": roles.to_dict(),
        },
        "training_windows": {
            "source": {
                "repository": "owner/data",
                "revision": "5" * 40,
                "artifact_path": "placed/windows.jsonl",
            },
            "dataset_manifest": {
                "path": "dataset_manifest.json",
                "sha256": "sha256:" + "8" * 64,
                "size_bytes": 1,
                "snapshot_id": "earlier-snapshot",
            },
            "sha256": sha256_file(windows),
            "size_bytes": windows.stat().st_size,
            "record_count": 1,
            "assembly": "GRCh38",
            "split": "train_placed_gnomad_common",
            "role": "train",
            "chromosomes": ["22"],
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
        "streams": stream_payloads,
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
                "sample_digest": "sha256:" + "9" * 64,
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
    }
    (split_evidence / "membership-split-evidence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_checksums(split_bundle)
    return SimpleNamespace(
        membership_bundle=membership_bundle,
        split_bundle=split_bundle,
        gnomad_root=gnomad_root,
        clinvar_root=clinvar_root,
        training_windows=windows,
        manifest=manifest,
        train_rows=tuple(train_rows),
    )


def _assemble_fixture_snapshot(fixture: SimpleNamespace, output: Path) -> None:
    assemble_v03_dataset_snapshot(
        membership_bundle_dir=fixture.membership_bundle,
        split_bundle_dir=fixture.split_bundle,
        gnomad_root=fixture.gnomad_root,
        clinvar_root=fixture.clinvar_root,
        training_windows_path=fixture.training_windows,
        dataset_dir=output,
        split_repository="owner/data",
        split_revision="d" * 40,
        split_artifact_path="candidates/v0.3/splits/r1/success",
        snapshot_id="geno-lewm-data-v0.3.0-fixture-r1",
        generated_at="2026-07-13T18:00:00Z",
        producer_git_commit="e" * 40,
        container_image="ghcr.io/example/uv@sha256:" + "f" * 64,
    )


def _rewrite_snapshot_identities_after_tamper(root: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    metadata_path = root / "dataset_package.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tampered_relative = "gnomad/v4.1/train/chr1.variants.parquet"
    tampered_records = int(pq.ParquetFile(root / tampered_relative).metadata.num_rows)
    for item in metadata["files"]:
        if item["path"] == tampered_relative:
            previous_records = int(item["records"])
            item["records"] = tampered_records
            break
    else:
        raise AssertionError("tampered fixture shard is absent from package metadata")
    metadata["splits"]["train_gnomad_common"]["records"] += tampered_records - previous_records
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    dataset_package_module.build_dataset_package(root, root / "dataset_package.json")
    integrity_path = root / "split_integrity.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["generated_at"] = metadata["generated_at"]
    integrity_path.write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    package_model = dataset_package_module.load_dataset_package(root, metadata_path)
    (root / "data_card.md").write_text(
        dataset_package_module.render_data_card(package_model, integrity_report=integrity),
        encoding="utf-8",
    )
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    report_path = root / snapshot_module.REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["v03"]["observed_splits"] = integrity["splits"]
    old_files = {item["path"]: item for item in report["files"]}
    report["files"] = [
        {
            **manifest_file,
            **{
                key: old_files[manifest_file["path"]][key]
                for key in (
                    "source_path",
                    "source_sha256",
                    "source_size_bytes",
                    "already_exists",
                )
            },
        }
        for manifest_file in manifest["files"]
    ]
    package = report["package"]
    package["schema_version"] = manifest["schema_version"]
    package["metadata"] = _identity(root / "dataset_package.json", "dataset_package.json")
    package["manifest"] = _identity(root / "dataset_manifest.json", "dataset_manifest.json")
    package["data_card"] = _identity(root / "data_card.md", "data_card.md")
    package["integrity"] = _identity(root / "split_integrity.json", "split_integrity.json")
    package["files"] = manifest["files"]
    package["membership_and_split_evidence"] = manifest["membership_and_split_evidence"]
    manifest_index = {item["path"]: item for item in manifest["files"]}
    for transformation in report["v03"]["transformations"]:
        output = transformation["output"]
        manifest_file = manifest_index[output["path"]]
        output["sha256"] = manifest_file["sha256"]
        output["size_bytes"] = manifest_file["size_bytes"]
        output["records"] = manifest_file["records"]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    snapshot_module._write_snapshot_checksums(root)


def _write_stream(
    root: Path,
    *,
    role: str,
    chrom: str,
    pos: int,
    label: str,
) -> dict[str, object]:
    del label
    stem = f"splits/{role}/clinvar-chr{chrom}"
    labels_relative = f"{stem}.labels.jsonl"
    vcf_relative = f"{stem}.vcf"
    labels = root / labels_relative
    vcf = root / vcf_relative
    labels.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"chrom": chrom, "pos": pos, "ref": "G", "alt": "T", "clinical_significance": "B"},
        {
            "chrom": chrom,
            "pos": pos + 1,
            "ref": "C",
            "alt": "A",
            "clinical_significance": "P",
        },
    ]
    labels.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    vcf.write_text(
        "##fileformat=VCFv4.3\n"
        "##reference=GRCh38\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        f"{chrom}\t{pos}\t.\tG\tT\t.\tPASS\tCLNSIG=B;ROLE={role};LABEL=0\n"
        f"{chrom}\t{pos + 1}\t.\tC\tA\t.\tPASS\tCLNSIG=P;ROLE={role};LABEL=1\n",
        encoding="utf-8",
    )
    return {
        "role": role,
        "chromosome": chrom,
        "record_count": 2,
        "class_counts": {"B": 1, "LB": 0, "LP": 0, "P": 1},
        "binary_counts": {"negative": 1, "positive": 1},
        "keyset_sha256": "sha256:" + ("a" if role == "validation" else "b") * 64,
        "labels_jsonl": _identity(labels, labels_relative),
        "vcf": _identity(vcf, vcf_relative),
    }


def _patch_membership_store(
    monkeypatch: pytest.MonkeyPatch,
    manifest: SimpleNamespace,
    train_rows: tuple[SimpleNamespace, ...],
) -> None:
    class _Store:
        def __init__(self) -> None:
            self.manifest = manifest

        def __enter__(self) -> _Store:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def iter_role(self, role: str) -> tuple[SimpleNamespace, ...]:
            assert role == "train"
            return train_rows

    replacement = SimpleNamespace(open=lambda *_args, **_kwargs: _Store())
    monkeypatch.setattr(snapshot_module, "MembershipStore", replacement)
    monkeypatch.setattr(dataset_package_module, "MembershipStore", replacement)


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(root / relative).removeprefix('sha256:')}  {relative}\n"
            for relative in paths
        ),
        encoding="utf-8",
    )


def _identity(path: Path, relative: str) -> dict[str, object]:
    return {"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
