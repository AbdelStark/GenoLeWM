"""Contract tests for the scalable v0.3 membership store."""

from __future__ import annotations

import importlib
import json
import multiprocessing
import os
import pickle
import random
import shutil
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, strategies as st

from geno_lewm.action import EditType, RelEdit
from geno_lewm.data.builder import EditSourceCount, WindowContext, build_training_tuples
from geno_lewm.data.membership import V03_CHROMOSOME_ROLES
from geno_lewm.data.membership_store import (
    MEMBERSHIP_STORE_SCHEMA_VERSION,
    MembershipSourceInput,
    MembershipStore,
    MembershipStoreHoldoutPolicy,
    build_membership_store,
    verify_membership_store,
)
from geno_lewm.data.variant_identity import CanonicalVariant
from geno_lewm.errors import InputError, RuntimeSetupError
from geno_lewm.provenance import canonical_json_sha256, sha256_file
from tools.data.v03_membership_store import main

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

GNOMAD_REMOTE_FILES = (
    "data/gnomad/v4.1/variants.parquet",
    "evidence/gcs-metadata-verification.json",
    "evidence/gcs-object-metadata.json",
    "evidence/prepare-report.json",
    "evidence/receipt.json",
    "evidence/selection.json",
    "evidence/source-lock.json",
    "evidence/source-lock.schema.json",
    "evidence/source-stream-identity.json",
)
GNOMAD_REMOTE_CHECKS = (
    "exact_hub_revision_resolved",
    "complete_namespace_file_set",
    "source_lock_and_schema_bound",
    "source_lock_and_schema_match_source_commit",
    "selection_rederived_from_source_lock",
    "metadata_verification_recomputed",
    "receipt_evidence_identities_recomputed",
    "parquet_sha256_and_size_recomputed",
    "parquet_full_scan_recomputed",
)
CLINVAR_REMOTE_FILES = (
    "clinvar/2026-04-15/variants.parquet",
    "evidence/audit.json",
    "evidence/prepare_report.json",
    "evidence/runtime_report.json",
)
CLINVAR_REMOTE_CHECKS = (
    "exact_hub_revision_resolved",
    "complete_namespace_file_set",
    "source_contract_loaded_from_exact_git_commit",
    "source_contract_derived_from_ast",
    "audit_prepare_runtime_reconciled",
    "source_release_sha256_and_size_reconciled",
    "parquet_sha256_and_size_recomputed",
    "parquet_schema_derived_from_source_commit",
    "parquet_full_scan_recomputed",
)
BUILDER_GIT_COMMIT = "a" * 40
BUILDER_CONTAINER_IMAGE = "ghcr.io/abdelstark/geno-lewm@sha256:" + "b" * 64
_FORKED_POLICIES: list[MembershipStoreHoldoutPolicy] = []


def _spawned_membership_policy_query(
    policy: MembershipStoreHoldoutPolicy,
) -> tuple[int, bool, bool]:
    window = WindowContext(
        record_id="spawned-window",
        source="fixture",
        sequence="A" * 32,
        chrom="1",
        start_bp=990,
    )
    return (
        os.getpid(),
        policy.excludes_window(window),
        policy.store.contains_variant(
            "GRCh38:1:1001:C:T",
            roles=("train",),
            sources=("clinvar-2026-04-15",),
        ),
    )


def _forked_membership_policy_query(queue: Any) -> None:
    if not _FORKED_POLICIES:
        raise AssertionError("forked policy was not initialized")
    queue.put(_spawned_membership_policy_query(_FORKED_POLICIES[0]))


@pytest.fixture(scope="module")
def source_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, tuple[MembershipSourceInput, ...]]:
    root = tmp_path_factory.mktemp("membership-sources")
    return _write_source_bundle(root)


@pytest.fixture(scope="module")
def built_store(
    tmp_path_factory: pytest.TempPathFactory,
    source_bundle: tuple[Path, tuple[MembershipSourceInput, ...]],
) -> Path:
    lineage_path, sources = source_bundle
    output = tmp_path_factory.mktemp("membership-output") / "store"
    build_membership_store(
        artifact_id="geno-lewm-v0.3-membership-fixture",
        snapshot_lineage_path=lineage_path,
        expected_snapshot_lineage_sha256=sha256_file(lineage_path),
        builder_git_commit=BUILDER_GIT_COMMIT,
        container_image=BUILDER_CONTAINER_IMAGE,
        sources=sources,
        output_dir=output,
    )
    return output


def test_builder_streams_sources_into_closed_manifest_parquet_and_lookup(
    built_store: Path,
) -> None:
    report = verify_membership_store(built_store)
    manifest = report.manifest

    assert report.ok is True
    assert manifest.schema_version == MEMBERSHIP_STORE_SCHEMA_VERSION
    assert manifest.snapshot_lineage.candidate_snapshot_id == "geno-lewm-data-v0.3.0-r1"
    assert manifest.row_count == 25
    assert manifest.variant_count == 25
    assert manifest.role_counts == {"train": 21, "validation": 2, "evaluation": 2}
    assert manifest.source_counts["clinvar-2026-04-15"] == 3
    clinvar_binding = next(source for source in manifest.sources if source.kind == "clinvar")
    assert clinvar_binding.filtered_row_count == 3
    assert manifest.content_identity.startswith("sha256:")
    assert manifest.rowset_sha256.startswith("sha256:")
    assert [binding.path for binding in manifest.files] == [
        "build-receipt.json",
        "lookup.sqlite",
        "memberships.parquet",
        "snapshot-lineage.json",
    ]
    assert {path.name for path in built_store.iterdir()} == {
        "build-receipt.json",
        "lookup.sqlite",
        "manifest.json",
        "memberships.parquet",
        "snapshot-lineage.json",
    }
    assert manifest.physical_identity.startswith("sha256:")
    assert manifest.physical_identity != manifest.content_identity
    receipt = json.loads((built_store / "build-receipt.json").read_text(encoding="utf-8"))
    assert receipt["builder"] == {
        "container_image": BUILDER_CONTAINER_IMAGE,
        "geno_lewm_version": "0.2.1",
        "git_commit": BUILDER_GIT_COMMIT,
    }
    assert receipt["content_identity"] == manifest.content_identity
    assert sha256_file(built_store / "snapshot-lineage.json") == manifest.snapshot_lineage.sha256

    table = pq.read_table(built_store / "memberships.parquet")
    chromosomes = table.column("chrom").to_pylist()
    assert chromosomes == sorted(chromosomes, key=lambda value: (int(value),))
    assert set(table.schema.names) == {
        "schema_version",
        "variant_key",
        "variant_digest",
        "chrom",
        "pos",
        "ref",
        "alt",
        "role",
        "reason_mask",
        "source",
        "source_row_id",
    }


@given(chromosome=st.integers(min_value=1, max_value=22))
def test_indexed_lookup_matches_canonical_role_without_materializing_keys(
    built_store: Path,
    chromosome: int,
) -> None:
    chrom = str(chromosome)
    variant = CanonicalVariant("GRCh38", chrom, 100 + chromosome, "A", "G")
    expected_role = V03_CHROMOSOME_ROLES.role_for(chrom)

    with MembershipStore.open(built_store, verify=False) as store:
        assert store.contains_variant(variant, roles=(expected_role,))
        assert not store.contains_variant(
            variant,
            roles=tuple(
                role for role in ("train", "validation", "evaluation") if role != expected_role
            ),
        )
        assert store.overlaps_interval(
            chrom,
            start_bp=variant.pos - 1,
            end_bp=variant.pos,
            roles=(expected_role,),
        )
        assert store.contains_variant(
            variant,
            roles=(expected_role,),
            sources=(f"gnomad-v4.1-chr{chrom}",),
        )


def test_store_streams_eval_rows_and_adapts_to_existing_builder_holdouts(
    built_store: Path,
) -> None:
    with MembershipStore.open(built_store) as store:
        evaluation = tuple(store.iter_role("evaluation", batch_size=1))
        assert len(evaluation) == 2
        assert {row.variant.chrom for row in evaluation} == {"21"}

        policy = MembershipStoreHoldoutPolicy(store)
        eval_window = WindowContext(
            record_id="eval-window",
            source="fixture",
            sequence="A" * 32,
            chrom="21",
            start_bp=100,
        )
        train_window = WindowContext(
            record_id="train-window",
            source="fixture",
            sequence="A" * 32,
            chrom="1",
            start_bp=0,
        )
        gnomad_only_window = WindowContext(
            record_id="gnomad-only-window",
            source="fixture",
            sequence="A" * 32,
            chrom="1",
            start_bp=90,
        )
        clinvar_holdout_window = WindowContext(
            record_id="clinvar-holdout-window",
            source="fixture",
            sequence="A" * 32,
            chrom="1",
            start_bp=990,
        )
        assert policy.excludes_window(eval_window)
        assert policy.excludes_edit(
            eval_window,
            RelEdit(
                rel_pos=20,
                edit_type=EditType.SNV,
                ref_bases="A",
                alt_bases="G",
            ),
        )
        assert not policy.excludes_window(train_window)
        assert not policy.excludes_window(gnomad_only_window)
        assert policy.excludes_window(clinvar_holdout_window)
        assert policy.excludes_edit(
            clinvar_holdout_window,
            RelEdit(
                rel_pos=10,
                edit_type=EditType.SNV,
                ref_bases="C",
                alt_bases="T",
            ),
        )
        assert not store.contains_variant(
            "GRCh38:2:1002:C:T",
            roles=("train",),
            sources=("clinvar-2026-04-15",),
        )

        provider_called = False

        def _provider(window: WindowContext, count: int, rng: random.Random) -> tuple[RelEdit, ...]:
            del window, count, rng
            nonlocal provider_called
            provider_called = True
            return ()

        assert (
            build_training_tuples(
                eval_window,
                {"must-not-run": _provider},
                rng=random.Random(0),
                mix=(EditSourceCount("must-not-run", 1),),
                holdouts=policy,
                fallback_sources={},
            )
            == ()
        )
        assert (
            build_training_tuples(
                clinvar_holdout_window,
                {"must-not-run": _provider},
                rng=random.Random(0),
                mix=(EditSourceCount("must-not-run", 1),),
                holdouts=policy,
                fallback_sources={},
            )
            == ()
        )
        assert provider_called is False


@pytest.mark.parametrize("chrom", [None, "X", "Y", "MT"])
def test_holdout_policy_rejects_unplaced_or_unassigned_windows(
    built_store: Path,
    chrom: str | None,
) -> None:
    window = WindowContext(
        record_id="unsupported-window",
        source="fixture",
        sequence="A" * 32,
        chrom=chrom,
    )
    edit = RelEdit(
        rel_pos=1,
        edit_type=EditType.SNV,
        ref_bases="A",
        alt_bases="G",
    )

    with MembershipStore.open(built_store) as store:
        policy = MembershipStoreHoldoutPolicy(store)
        with pytest.raises(InputError, match="placed and assigned"):
            policy.excludes_window(window)
        with pytest.raises(InputError, match="placed and assigned"):
            policy.excludes_edit(window, edit)


def test_store_and_policy_pickle_by_content_and_reopen_in_spawned_process(
    built_store: Path,
) -> None:
    store = MembershipStore.open(built_store, verify=False)
    assert store.contains_variant("GRCh38:1:101:A:G", roles=("train",))
    restored = pickle.loads(pickle.dumps(store))
    policy = MembershipStoreHoldoutPolicy(store)
    restored_policy = pickle.loads(pickle.dumps(policy))

    assert restored == store
    assert hash(restored) == hash(store)
    assert restored_policy == policy
    assert hash(restored_policy) == hash(policy)

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        child_pid, excluded, found = executor.submit(
            _spawned_membership_policy_query,
            restored_policy,
        ).result(timeout=30)

    assert child_pid != os.getpid()
    assert excluded is True
    assert found is True
    restored_policy.store.close()
    restored.close()
    store.close()


def test_store_uses_distinct_lazy_connections_across_threads_and_fork(
    built_store: Path,
) -> None:
    store = MembershipStore.open(built_store, verify=False)
    assert store.contains_variant("GRCh38:1:101:A:G", roles=("train",))

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = tuple(
            executor.map(
                lambda _index: store.overlaps_interval(
                    "1",
                    start_bp=90,
                    end_bp=110,
                    roles=("train",),
                    sources=("gnomad-v4.1-chr1",),
                ),
                range(8),
            )
        )
    assert results == (True,) * 8

    if "fork" in multiprocessing.get_all_start_methods():
        _FORKED_POLICIES.append(MembershipStoreHoldoutPolicy(store))
        context = multiprocessing.get_context("fork")
        queue = context.Queue()
        process = context.Process(target=_forked_membership_policy_query, args=(queue,))
        process.start()
        child_pid, excluded, found = queue.get(timeout=30)
        process.join(timeout=30)
        assert process.exitcode == 0
        assert child_pid != os.getpid()
        assert excluded is True
        assert found is True
        queue.close()
        _FORKED_POLICIES.clear()
    store.close()


def test_runtime_queries_use_indexes_without_role_stream_resort(built_store: Path) -> None:
    connection = sqlite3.connect(built_store / "lookup.sqlite")
    try:
        role_plan = " ".join(
            str(row[-1])
            for row in connection.execute(
                "EXPLAIN QUERY PLAN "
                "SELECT schema_version, variant_key, variant_digest, chrom, pos, ref, alt, "
                "role, reason_mask, source, source_row_id FROM memberships WHERE role = ? "
                "ORDER BY chrom_rank, pos, ref, alt, role_rank, source, source_row_id, "
                "reason_mask",
                ("evaluation",),
            )
        )
        interval_plan = " ".join(
            str(row[-1])
            for row in connection.execute(
                "EXPLAIN QUERY PLAN SELECT 1 FROM membership_intervals AS intervals "
                "CROSS JOIN memberships AS memberships "
                "ON memberships.membership_id = intervals.membership_id "
                "WHERE intervals.chrom_min <= ? AND intervals.chrom_max >= ? "
                "AND intervals.start_min < ? AND intervals.end_max > ? "
                "AND memberships.role IN (?, ?) AND memberships.source IN (?) LIMIT 1",
                (
                    20,
                    20,
                    200,
                    100,
                    "validation",
                    "evaluation",
                    "clinvar-2026-04-15",
                ),
            )
        )
    finally:
        connection.close()

    assert "memberships_role_order" in role_plan
    assert "TEMP B-TREE" not in role_plan
    assert "intervals VIRTUAL TABLE INDEX" in interval_plan
    assert "INTEGER PRIMARY KEY" in interval_plan


def test_verifier_rejects_manifest_shape_and_bound_file_tampering(
    tmp_path: Path,
    built_store: Path,
) -> None:
    malformed = tmp_path / "malformed"
    shutil.copytree(built_store, malformed)
    manifest_path = malformed / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        json.dumps({**payload, "unexpected": True}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(InputError, match="manifest keys do not match"):
        verify_membership_store(malformed)

    tampered = tmp_path / "tampered"
    shutil.copytree(built_store, tampered)
    with (tampered / "lookup.sqlite").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(InputError, match="file identity mismatch"):
        verify_membership_store(tampered)


def test_verifier_rejects_extra_files_symlinks_and_duplicate_bindings(
    tmp_path: Path,
    built_store: Path,
) -> None:
    extra = tmp_path / "extra"
    shutil.copytree(built_store, extra)
    (extra / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(InputError, match="exact layout"):
        verify_membership_store(extra)

    linked = tmp_path / "linked"
    shutil.copytree(built_store, linked)
    (linked / "build-receipt.json").unlink()
    try:
        (linked / "build-receipt.json").symlink_to("manifest.json")
    except OSError:
        # Unprivileged Windows runners may not permit symlink creation.
        shutil.rmtree(linked)
    else:
        with pytest.raises(InputError, match="must not contain symlinks"):
            verify_membership_store(linked)

    duplicated = tmp_path / "duplicated-binding"
    shutil.copytree(built_store, duplicated)
    manifest_path = duplicated / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["files"].append(dict(payload["files"][0]))
    payload["physical_identity"] = canonical_json_sha256(
        {"content_identity": payload["content_identity"], "files": payload["files"]}
    )
    _write_json(manifest_path, payload)
    with pytest.raises(InputError, match="file bindings do not match"):
        verify_membership_store(duplicated)

    receipt_drift = tmp_path / "receipt-drift"
    shutil.copytree(built_store, receipt_drift)
    receipt_path = receipt_drift / "build-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["unexpected"] = True
    _write_json(receipt_path, receipt)
    _rebind_manifest_file(receipt_drift, "build-receipt.json")
    with pytest.raises(InputError, match="build receipt keys do not match"):
        verify_membership_store(receipt_drift)


def test_verifier_rejects_tampered_lookup_columns_even_when_file_is_rebound(
    tmp_path: Path,
    built_store: Path,
) -> None:
    tampered = tmp_path / "tampered-derived-column"
    shutil.copytree(built_store, tampered)
    index_path = tampered / "lookup.sqlite"
    connection = sqlite3.connect(index_path)
    try:
        connection.execute(
            "UPDATE memberships SET chrom_rank = 21 WHERE variant_key = ?",
            ("GRCh38:1:101:A:G",),
        )
        connection.commit()
    finally:
        connection.close()

    _rebind_manifest_file(tampered, "lookup.sqlite")

    with pytest.raises(InputError, match="derived columns are inconsistent"):
        verify_membership_store(tampered)


def test_builder_rejects_source_identity_drift_before_atomic_publication(
    tmp_path: Path,
    source_bundle: tuple[Path, tuple[MembershipSourceInput, ...]],
) -> None:
    lineage_path, sources = source_bundle
    copied = tmp_path / "source.parquet"
    shutil.copyfile(sources[0].path, copied)
    with copied.open("ab") as handle:
        handle.write(b"tamper")
    drifted_sources = (MembershipSourceInput("gnomad", copied, chromosome="1"), *sources[1:])
    output = tmp_path / "store"

    with pytest.raises(InputError, match="source artifact identity mismatch"):
        build_membership_store(
            artifact_id="drifted",
            snapshot_lineage_path=lineage_path,
            expected_snapshot_lineage_sha256=sha256_file(lineage_path),
            builder_git_commit=BUILDER_GIT_COMMIT,
            container_image=BUILDER_CONTAINER_IMAGE,
            sources=drifted_sources,
            output_dir=output,
        )
    assert not output.exists()


def test_builder_rejects_unexpected_lineage_identity_before_publication(
    tmp_path: Path,
    source_bundle: tuple[Path, tuple[MembershipSourceInput, ...]],
) -> None:
    lineage_path, sources = source_bundle
    output = tmp_path / "store"

    with pytest.raises(InputError, match="expected byte identity"):
        build_membership_store(
            artifact_id="wrong-lineage",
            snapshot_lineage_path=lineage_path,
            expected_snapshot_lineage_sha256="sha256:" + "0" * 64,
            builder_git_commit=BUILDER_GIT_COMMIT,
            container_image=BUILDER_CONTAINER_IMAGE,
            sources=sources,
            output_dir=output,
        )
    assert not output.exists()


def test_builder_derives_rows_from_captured_bytes_when_source_path_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_bundle: tuple[Path, tuple[MembershipSourceInput, ...]],
) -> None:
    lineage_path, sources = source_bundle
    original_clinvar = next(source for source in sources if source.kind == "clinvar")
    captured_target = tmp_path / "clinvar.parquet"
    shutil.copyfile(original_clinvar.path, captured_target)
    clinvar = MembershipSourceInput("clinvar", captured_target)
    sources = tuple(clinvar if source.kind == "clinvar" else source for source in sources)
    replacement = tmp_path / "replacement.parquet"
    replacement_rows = [
        _clinvar_row("21", 2021, 21),
        _clinvar_row("1", 2001, 1),
        _clinvar_row("X", 2023, 23),
        _clinvar_row("20", 2020, 20),
        _clinvar_row("GL000220.1", 2024, 24),
        _clinvar_row("2", 2002, 25, significance="VUS"),
    ]
    pq.write_table(pa.Table.from_pylist(replacement_rows, schema=_clinvar_schema()), replacement)
    real_parquet_file = pq.ParquetFile
    replaced = False

    def _replace_before_scan(path: object, *args: object, **kwargs: object) -> Any:
        nonlocal replaced
        if not replaced:
            replacement.replace(clinvar.path)
            replaced = True
        return real_parquet_file(path, *args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", _replace_before_scan)
    output = tmp_path / "store"
    build_membership_store(
        artifact_id="captured-source",
        snapshot_lineage_path=lineage_path,
        expected_snapshot_lineage_sha256=sha256_file(lineage_path),
        builder_git_commit=BUILDER_GIT_COMMIT,
        container_image=BUILDER_CONTAINER_IMAGE,
        sources=sources,
        output_dir=output,
    )

    with MembershipStore.open(output, verify=False) as store:
        assert store.contains_variant(
            "GRCh38:1:1001:C:T",
            roles=("train",),
            sources=("clinvar-2026-04-15",),
        )
        assert not store.contains_variant(
            "GRCh38:1:2001:C:T",
            roles=("train",),
            sources=("clinvar-2026-04-15",),
        )


def test_builder_rejects_audit_only_clinvar_lineage(
    tmp_path: Path,
    source_bundle: tuple[Path, tuple[MembershipSourceInput, ...]],
) -> None:
    lineage_path, sources = source_bundle
    payload = json.loads(lineage_path.read_text(encoding="utf-8"))
    del payload["clinvar"]["remote_postflight"]
    payload["lineage_id"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "lineage_id"}
    )
    audit_only = tmp_path / "snapshot-lineage.json"
    _write_json(audit_only, payload)

    with pytest.raises(InputError, match="ClinVar remote_postflight must be an object"):
        build_membership_store(
            artifact_id="audit-only",
            snapshot_lineage_path=audit_only,
            expected_snapshot_lineage_sha256=sha256_file(audit_only),
            builder_git_commit=BUILDER_GIT_COMMIT,
            container_image=BUILDER_CONTAINER_IMAGE,
            sources=sources,
            output_dir=tmp_path / "store",
        )
    assert not (tmp_path / "store").exists()


def test_builder_rejects_duplicate_source_membership_and_leaves_no_partial_store(
    tmp_path: Path,
) -> None:
    lineage_path, sources = _write_source_bundle(tmp_path / "duplicate-source")
    first = sources[0]
    table = pq.read_table(first.path)
    pq.write_table(pa.concat_tables([table, table]), first.path)
    _refresh_lineage_source_identities(lineage_path, sources)
    output = tmp_path / "store"

    with pytest.raises(InputError, match="duplicate membership identity"):
        build_membership_store(
            artifact_id="duplicate",
            snapshot_lineage_path=lineage_path,
            expected_snapshot_lineage_sha256=sha256_file(lineage_path),
            builder_git_commit=BUILDER_GIT_COMMIT,
            container_image=BUILDER_CONTAINER_IMAGE,
            sources=sources,
            output_dir=output,
        )
    assert not output.exists()


def test_builder_independently_verifies_before_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage_path, sources = _write_source_bundle(tmp_path / "corrupt-build-sources")
    writer = importlib.import_module("geno_lewm.data._membership_store_writer")
    real_write = writer._write_membership_parquet

    def _write_corrupt_parquet(connection: sqlite3.Connection, path: Path) -> str:
        rowset_sha256 = real_write(connection, path)
        path.write_bytes(b"corrupt parquet")
        return rowset_sha256

    monkeypatch.setattr(writer, "_write_membership_parquet", _write_corrupt_parquet)
    output = tmp_path / "store"
    with pytest.raises(InputError, match="Parquet scan failed"):
        build_membership_store(
            artifact_id="corrupt-before-publication",
            snapshot_lineage_path=lineage_path,
            expected_snapshot_lineage_sha256=sha256_file(lineage_path),
            builder_git_commit=BUILDER_GIT_COMMIT,
            container_image=BUILDER_CONTAINER_IMAGE,
            sources=sources,
            output_dir=output,
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".store.tmp-*"))
    assert not tuple(tmp_path.glob(".store.sources-*"))


def test_builder_does_not_publish_when_durability_barrier_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_bundle: tuple[Path, tuple[MembershipSourceInput, ...]],
) -> None:
    lineage_path, sources = source_bundle
    writer = importlib.import_module("geno_lewm.data._membership_store_writer")

    def _fail_fsync(_root: Path) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(writer, "_fsync_artifact", _fail_fsync)
    output = tmp_path / "store"
    with pytest.raises(InputError, match="failed before publication"):
        build_membership_store(
            artifact_id="fsync-failure",
            snapshot_lineage_path=lineage_path,
            expected_snapshot_lineage_sha256=sha256_file(lineage_path),
            builder_git_commit=BUILDER_GIT_COMMIT,
            container_image=BUILDER_CONTAINER_IMAGE,
            sources=sources,
            output_dir=output,
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".store.tmp-*"))
    assert not tuple(tmp_path.glob(".store.sources-*"))


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("memberships.parquet", "membership Parquet scan failed"),
        ("lookup.sqlite", "membership SQLite scan failed"),
    ],
)
def test_verify_cli_reports_typed_storage_corruption(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    built_store: Path,
    filename: str,
    message: str,
) -> None:
    corrupted = tmp_path / filename.replace(".", "-")
    shutil.copytree(built_store, corrupted)
    (corrupted / filename).write_bytes(b"corrupt storage")
    _rebind_manifest_file(corrupted, filename)

    assert main(["verify", "--store-dir", str(corrupted)]) == 2
    assert message in capsys.readouterr().err


def test_lookup_has_no_pyarrow_runtime_dependency_but_file_adapter_fails_typed(
    monkeypatch: pytest.MonkeyPatch,
    built_store: Path,
) -> None:
    real_import = importlib.import_module

    def _blocked(name: str, package: str | None = None) -> Any:
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("blocked for adapter-boundary test")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _blocked)
    with MembershipStore.open(built_store, verify=False) as store:
        assert store.contains_variant("GRCh38:21:121:A:G", roles=("evaluation",))
    with pytest.raises(RuntimeSetupError, match="requires pyarrow"):
        verify_membership_store(built_store)


def test_verify_cli_and_checked_schemas_are_closed(
    capsys: pytest.CaptureFixture[str],
    built_store: Path,
) -> None:
    assert main(["verify", "--store-dir", str(built_store)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["row_count"] == 25

    root = Path(__file__).resolve().parents[2]
    for name in (
        "membership-build-receipt.schema.json",
        "membership-build-spec.schema.json",
        "membership-store.schema.json",
    ):
        schema = json.loads((root / "configs" / "data_v03" / name).read_text(encoding="utf-8"))
        assert schema["additionalProperties"] is False
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_build_cli_consumes_closed_relative_spec(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    source_bundle: tuple[Path, tuple[MembershipSourceInput, ...]],
    built_store: Path,
) -> None:
    lineage_path, sources = source_bundle
    bundle = tmp_path / "bundle"
    shutil.copytree(lineage_path.parent, bundle)
    source_entries: list[dict[str, str]] = []
    for source in sources:
        entry = {"kind": source.kind, "path": source.path.name}
        if source.chromosome is not None:
            entry["chromosome"] = source.chromosome
        source_entries.append(entry)
    spec = {
        "$schema": "./membership-build-spec.schema.json",
        "schema_version": "geno-lewm.membership-build-spec.v1",
        "artifact_id": "geno-lewm-v0.3-membership-fixture",
        "snapshot_lineage": "snapshot-lineage.json",
        "snapshot_lineage_sha256": sha256_file(bundle / "snapshot-lineage.json"),
        "builder": {
            "git_commit": BUILDER_GIT_COMMIT,
            "container_image": BUILDER_CONTAINER_IMAGE,
        },
        "sources": source_entries,
    }
    spec_path = bundle / "membership-build.json"
    _write_json(spec_path, spec)
    output = tmp_path / "store"

    assert main(["build", "--spec-json", str(spec_path), "--output-dir", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["row_count"] == 25
    rebuilt = verify_membership_store(output)
    original = verify_membership_store(built_store)
    assert rebuilt.ok is True
    assert rebuilt.manifest.content_identity == original.manifest.content_identity
    assert rebuilt.manifest.physical_identity != original.manifest.physical_identity
    assert rebuilt.manifest.rowset_sha256 == original.manifest.rowset_sha256


def _write_source_bundle(root: Path) -> tuple[Path, tuple[MembershipSourceInput, ...]]:
    root.mkdir(parents=True, exist_ok=True)
    sources: list[MembershipSourceInput] = []
    shards: list[dict[str, object]] = []
    for chromosome in range(1, 23):
        chrom = str(chromosome)
        path = root / f"gnomad-chr{chrom}.parquet"
        pq.write_table(
            pa.Table.from_pylist([_gnomad_row(chromosome)], schema=_gnomad_schema()),
            path,
        )
        source = MembershipSourceInput("gnomad", path, chromosome=chrom)
        sources.append(source)
        role = V03_CHROMOSOME_ROLES.role_for(chrom)
        namespace = f"staging/v0.3/gnomad-v4.1/chr{chrom}"
        shards.append(
            {
                "chromosome": chrom,
                "split_role": role,
                "revision": f"{chromosome:040x}",
                "namespace": namespace,
                "receipt": {
                    "artifact_path": f"{namespace}/evidence/receipt.json",
                    "sha256": "sha256:" + f"{chromosome + 40:064x}",
                    "size_bytes": 100,
                },
                "remote_postflight": {
                    "schema_version": "geno-lewm.gnomad-remote-postflight.v1",
                    "sha256": "sha256:" + f"{chromosome + 80:064x}",
                    "size_bytes": 200,
                    "verified_files": list(GNOMAD_REMOTE_FILES),
                    "checks": list(GNOMAD_REMOTE_CHECKS),
                },
                "source": {"fixture": True},
                "transform": {"command": "geno-lewm-prepare-gnomad"},
                "output": {
                    "artifact_path": f"{namespace}/data/gnomad/v4.1/variants.parquet",
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "records": 1,
                    "schema_version": "2.0.0",
                },
            }
        )

    clinvar_path = root / "clinvar.parquet"
    clinvar_rows = [
        _clinvar_row("21", 1021, 21),
        _clinvar_row("1", 1001, 1),
        _clinvar_row("X", 1023, 23),
        _clinvar_row("20", 1020, 20),
        _clinvar_row("GL000220.1", 1024, 24),
        _clinvar_row("2", 1002, 25, significance="VUS"),
    ]
    pq.write_table(pa.Table.from_pylist(clinvar_rows, schema=_clinvar_schema()), clinvar_path)
    sources.append(MembershipSourceInput("clinvar", clinvar_path))

    lineage = {
        "schema_version": "geno-lewm.v03-snapshot-lineage.v1",
        "generated_by": "tools.data.v03_snapshot_lineage",
        "candidate_snapshot_id": "geno-lewm-data-v0.3.0-r1",
        "reference_genome": "GRCh38",
        "membership_status": "not_created",
        "assembly_inputs": {"fixture": True},
        "gnomad": {
            "dataset_id": "gnomad-v4.1-exomes-autosomes",
            "release": "v4.1",
            "repo": "abdelstark/geno-lewm-data",
            "repo_type": "dataset",
            "data_use": {"fixture": True},
            "source_lock": {"fixture": True},
            "transform": {"command": "geno-lewm-prepare-gnomad"},
            "common_execution": {"fixture": True},
            "split_policy": V03_CHROMOSOME_ROLES.to_dict(),
            "total_records": 22,
            "total_size_bytes": sum(source.path.stat().st_size for source in sources),
            "shards": shards,
        },
        "clinvar": {
            "release": "2026-04-15",
            "reference_genome": "GRCh38",
            "repo": "abdelstark/geno-lewm-data",
            "repo_type": "dataset",
            "data_use": {"fixture": True},
            "revision": "d" * 40,
            "namespace": "staging/clinvar-2026-04-15-corrected-r1",
            "audit": {
                "artifact_path": "staging/clinvar-2026-04-15-corrected-r1/evidence/audit.json",
                "sha256": "sha256:" + "e" * 64,
                "size_bytes": 300,
            },
            "remote_postflight": {
                "schema_version": "geno-lewm.clinvar-remote-postflight.v1",
                "sha256": "sha256:" + "c" * 64,
                "size_bytes": 400,
                "verified_files": list(CLINVAR_REMOTE_FILES),
                "checks": list(CLINVAR_REMOTE_CHECKS),
                "parquet_audit": _clinvar_parquet_audit(clinvar_rows),
            },
            "source": {"fixture": True},
            "output": {
                "artifact_path": (
                    "staging/clinvar-2026-04-15-corrected-r1/clinvar/2026-04-15/variants.parquet"
                ),
                "sha256": sha256_file(clinvar_path),
                "size_bytes": clinvar_path.stat().st_size,
                "records": len(clinvar_rows),
                "class_balance": _clinvar_class_balance(clinvar_rows),
            },
            "execution": {"fixture": True},
            "evidence_claim_boundary": "fixture",
        },
        "claim_boundary": "fixture source lineage; no real membership claim",
    }
    lineage["lineage_id"] = canonical_json_sha256(lineage)
    lineage_path = root / "snapshot-lineage.json"
    _write_json(lineage_path, lineage)
    return lineage_path, tuple(sources)


def _refresh_lineage_source_identities(
    lineage_path: Path, sources: tuple[MembershipSourceInput, ...]
) -> None:
    payload = json.loads(lineage_path.read_text(encoding="utf-8"))
    by_chrom = {source.chromosome: source for source in sources if source.kind == "gnomad"}
    for shard in payload["gnomad"]["shards"]:
        source = by_chrom[shard["chromosome"]]
        parquet = pq.ParquetFile(source.path)
        shard["output"]["sha256"] = sha256_file(source.path)
        shard["output"]["size_bytes"] = source.path.stat().st_size
        shard["output"]["records"] = parquet.metadata.num_rows
    payload["lineage_id"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "lineage_id"}
    )
    _write_json(lineage_path, payload)


def _gnomad_row(chromosome: int) -> dict[str, object]:
    return {
        "chrom": str(chromosome),
        "pos": 100 + chromosome,
        "ref": "A",
        "alt": "G",
        "af_global": 0.1,
        "af_afr": 0.1,
        "af_ami": None,
        "af_amr": 0.1,
        "af_asj": 0.1,
        "af_eas": 0.1,
        "af_fin": 0.1,
        "af_mid": 0.1,
        "af_nfe": 0.1,
        "af_oth": None,
        "af_remaining": 0.1,
        "af_sas": 0.1,
        "filter": "PASS",
        "schema_version": "2.0.0",
    }


def _gnomad_schema() -> Any:
    return pa.schema(
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


def _clinvar_row(
    chrom: str, pos: int, clinvar_id: int, *, significance: str = "P"
) -> dict[str, object]:
    return {
        "chrom": chrom,
        "pos": pos,
        "ref": "C",
        "alt": "T",
        "clinical_significance": significance,
        "review_status": "criteria_provided",
        "gene_symbol": "FIXTURE",
        "clinvar_id": clinvar_id,
        "schema_version": "1.0.0",
    }


def _clinvar_schema() -> Any:
    return pa.schema(
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
        ]
    )


def _clinvar_parquet_audit(rows: list[dict[str, object]]) -> dict[str, object]:
    positions = [int(row["pos"]) for row in rows]
    identifiers = [int(row["clinvar_id"]) for row in rows]
    chromosomes: dict[str, int] = {}
    for row in rows:
        chromosome = str(row["chrom"])
        chromosomes[chromosome] = chromosomes.get(chromosome, 0) + 1
    schema = _clinvar_schema()
    return {
        "metadata_row_count": len(rows),
        "scanned_row_count": len(rows),
        "class_balance": _clinvar_class_balance(rows),
        "chromosome_balance": dict(sorted(chromosomes.items())),
        "schema_version_balance": {"1.0.0": len(rows)},
        "null_counts": dict.fromkeys(schema.names, 0),
        "position_range": {"min": min(positions), "max": max(positions)},
        "clinvar_id_range": {"min": min(identifiers), "max": max(identifiers)},
        "schema": [{"name": field.name, "type": str(field.type)} for field in schema],
    }


def _clinvar_class_balance(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        significance = str(row["clinical_significance"])
        counts[significance] = counts.get(significance, 0) + 1
    return dict(sorted(counts.items()))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rebind_manifest_file(root: Path, filename: str) -> None:
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = next(binding for binding in payload["files"] if binding["path"] == filename)
    path = root / filename
    binding["sha256"] = sha256_file(path)
    binding["size_bytes"] = path.stat().st_size
    payload["physical_identity"] = canonical_json_sha256(
        {"content_identity": payload["content_identity"], "files": payload["files"]}
    )
    _write_json(manifest_path, payload)
