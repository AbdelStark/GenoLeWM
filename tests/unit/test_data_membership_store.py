"""Contract tests for the scalable v0.3 membership store."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import multiprocessing
import os
import pickle
import random
import shutil
import sqlite3
import subprocess
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
from hypothesis import given, strategies as st

from geno_lewm.action import EditType, RelEdit
from geno_lewm.data import _membership_store_lineage as membership_store_lineage
from geno_lewm.data.builder import EditSourceCount, WindowContext, build_training_tuples
from geno_lewm.data.membership import V03_CHROMOSOME_ROLES
from geno_lewm.data.membership_store import (
    MEMBERSHIP_STORE_SCHEMA_VERSION,
    LabeledClinVarMembership,
    MembershipSourceInput,
    MembershipStore,
    MembershipStoreHoldoutPolicy,
    build_membership_store,
    verify_membership_store,
)
from geno_lewm.data.variant_identity import CanonicalVariant
from geno_lewm.errors import InputError, ResourceError, RuntimeSetupError
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
BUILDER_GIT_COMMIT = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
BUILDER_CONTAINER_IMAGE = "ghcr.io/abdelstark/geno-lewm@sha256:" + "b" * 64
_FORKED_POLICIES: list[MembershipStoreHoldoutPolicy] = []


@pytest.fixture(scope="module", autouse=True)
def _verified_builder_environment() -> Any:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv(
            "GENO_LEWM_VERIFIED_BUILD_CONTAINER_IMAGE",
            BUILDER_CONTAINER_IMAGE,
        )
        yield


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


def _create_empty_directory(path: str) -> None:
    Path(path).mkdir()


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
    assert manifest.snapshot_lineage.evidence_profile == "synthetic_fixture"
    assert manifest.row_count == 28
    assert manifest.variant_count == 28
    assert manifest.role_counts == {"train": 22, "validation": 3, "evaluation": 3}
    assert manifest.source_counts["clinvar-2026-04-15"] == 6
    assert manifest.source_kind_role_counts == {
        "gnomad": {"train": 20, "validation": 1, "evaluation": 1},
        "clinvar": {"train": 2, "validation": 2, "evaluation": 2},
    }
    assert manifest.source_role_counts["clinvar-2026-04-15"] == {
        "train": 2,
        "validation": 2,
        "evaluation": 2,
    }
    assert manifest.source_role_counts["gnomad-v4.1-chr1"] == {
        "train": 1,
        "validation": 0,
        "evaluation": 0,
    }
    assert manifest.source_role_counts["gnomad-v4.1-chr20"] == {
        "train": 0,
        "validation": 1,
        "evaluation": 0,
    }
    assert manifest.clinvar_class_role_counts == {
        "B": {"train": 1, "validation": 1, "evaluation": 0},
        "LB": {"train": 0, "validation": 0, "evaluation": 1},
        "LP": {"train": 1, "validation": 1, "evaluation": 0},
        "P": {"train": 0, "validation": 0, "evaluation": 1},
    }
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
        "verification": {
            "container_image": ("environment:GENO_LEWM_VERIFIED_BUILD_CONTAINER_IMAGE"),
            "git_commit": "git-rev-parse-head-clean-tree",
        },
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
        "clinical_significance",
    }


def test_clinvar_labels_follow_chromosome_roles_symmetrically_and_keep_train_anchors(
    built_store: Path,
) -> None:
    with MembershipStore.open(built_store) as store:
        labeled_rows = tuple(
            row
            for role in ("train", "validation", "evaluation")
            for row in store.iter_labeled_clinvar(role, batch_size=1)
        )
        observed = {
            (row.membership.variant.chrom, row.clinical_significance, row.is_pathogenic)
            for row in labeled_rows
        }

        assert observed == {
            ("1", "LP", True),
            ("3", "B", False),
            ("20", "B", False),
            ("20", "LP", True),
            ("21", "LB", False),
            ("21", "P", True),
        }
        assert all(isinstance(row, LabeledClinVarMembership) for row in labeled_rows)

        policy = MembershipStoreHoldoutPolicy(store)
        train_anchor = WindowContext(
            record_id="train-pathogenic-anchor",
            source="fixture",
            sequence="A" * 32,
            chrom="1",
            start_bp=990,
        )
        validation_anchor = WindowContext(
            record_id="validation-same-coordinate",
            source="fixture",
            sequence="A" * 32,
            chrom="20",
            start_bp=990,
        )
        assert store.contains_variant(
            "GRCh38:1:1001:C:T",
            roles=("train",),
            sources=("clinvar-2026-04-15",),
        )
        assert not policy.excludes_window(train_anchor)
        assert not policy.excludes_edit(
            train_anchor,
            RelEdit(
                rel_pos=10,
                edit_type=EditType.SNV,
                ref_bases="C",
                alt_bases="T",
            ),
        )
        assert policy.excludes_window(validation_anchor)
        assert policy.excludes_edit(
            validation_anchor,
            RelEdit(
                rel_pos=10,
                edit_type=EditType.SNV,
                ref_bases="C",
                alt_bases="T",
            ),
        )


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
        assert len(evaluation) == 3
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
        assert not policy.excludes_window(clinvar_holdout_window)
        assert not policy.excludes_edit(
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
            return (
                RelEdit(
                    rel_pos=0,
                    edit_type=EditType.SNV,
                    ref_bases="A",
                    alt_bases="G",
                ),
            )

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
            len(
                build_training_tuples(
                    clinvar_holdout_window,
                    {"must-not-run": _provider},
                    rng=random.Random(0),
                    mix=(EditSourceCount("must-not-run", 1),),
                    holdouts=policy,
                    fallback_sources={},
                )
            )
            == 1
        )
        assert provider_called is True


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
    assert excluded is False
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
        assert excluded is False
        assert found is True
        queue.close()
        _FORKED_POLICIES.clear()
    store.close()


def test_store_reclaims_connections_from_short_lived_threads(built_store: Path) -> None:
    store = MembershipStore.open(built_store, verify=False)

    def _run_queries() -> None:
        threads = [
            threading.Thread(
                target=store.contains_variant,
                args=("GRCh38:1:101:A:G",),
                kwargs={"roles": ("train",)},
            )
            for _ in range(32)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()

    _run_queries()
    gc.collect()

    assert len(store._connections) == 0
    store.close()


def test_verified_store_queries_the_exact_lookup_snapshot_verified_at_open(
    tmp_path: Path,
    built_store: Path,
) -> None:
    published = tmp_path / "replace-after-open"
    shutil.copytree(built_store, published)
    store = MembershipStore.open(published, verify=True)
    original_identity = store.manifest.content_identity
    (published / "lookup.sqlite").unlink()
    (published / "lookup.sqlite").write_bytes(b"replacement after verified open")
    try:
        assert store.manifest.content_identity == original_identity
        assert store.contains_variant("GRCh38:1:101:A:G", roles=("train",))
    finally:
        store.close()


def test_verifier_hashes_and_scans_one_private_capture_per_published_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    built_store: Path,
) -> None:
    published = tmp_path / "single-capture"
    shutil.copytree(built_store, published)
    snapshot = importlib.import_module("geno_lewm.data._membership_store_snapshot")
    verifier = importlib.import_module("geno_lewm.data._membership_store_verifier")
    real_capture_file = snapshot._capture_file
    real_scan_parquet = verifier._scan_parquet
    captured_names: list[str] = []

    def _count_capture(root_fd: int, name: str, destination: Path) -> Any:
        captured_names.append(name)
        return real_capture_file(root_fd, name, destination)

    def _replace_published_lookup_after_capture(path: Path, manifest: Any) -> Any:
        (published / "lookup.sqlite").unlink()
        (published / "lookup.sqlite").write_bytes(b"A-B-A path replacement")
        return real_scan_parquet(path, manifest)

    monkeypatch.setattr(snapshot, "_capture_file", _count_capture)
    monkeypatch.setattr(verifier, "_scan_parquet", _replace_published_lookup_after_capture)

    assert verify_membership_store(published).ok is True
    assert sorted(captured_names) == [
        "build-receipt.json",
        "lookup.sqlite",
        "manifest.json",
        "memberships.parquet",
        "snapshot-lineage.json",
    ]


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


def test_verifier_rejects_stale_clinvar_class_role_evidence(
    tmp_path: Path,
    built_store: Path,
) -> None:
    stale = tmp_path / "stale-class-crosstab"
    shutil.copytree(built_store, stale)
    manifest_path = stale / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["clinvar_class_role_counts"]["B"]["evaluation"] = 1
    payload["clinvar_class_role_counts"]["P"]["evaluation"] = 0
    identity_keys = {
        "schema_version",
        "artifact_id",
        "assembly",
        "chromosome_roles",
        "snapshot_lineage",
        "sources",
        "row_count",
        "variant_count",
        "role_counts",
        "source_counts",
        "source_role_counts",
        "source_kind_role_counts",
        "clinvar_class_role_counts",
        "rowset_sha256",
    }
    payload["content_identity"] = canonical_json_sha256(
        {key: payload[key] for key in identity_keys}
    )
    receipt_path = stale / "build-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["content_identity"] = payload["content_identity"]
    _write_json(receipt_path, receipt)
    connection = sqlite3.connect(stale / "lookup.sqlite")
    try:
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'content_identity'",
            (payload["content_identity"],),
        )
        connection.commit()
    finally:
        connection.close()
    for binding in payload["files"]:
        path = stale / binding["path"]
        binding["sha256"] = sha256_file(path)
        binding["size_bytes"] = path.stat().st_size
    payload["physical_identity"] = canonical_json_sha256(
        {"content_identity": payload["content_identity"], "files": payload["files"]}
    )
    _write_json(manifest_path, payload)

    with pytest.raises(InputError, match="semantic scan does not match manifest"):
        verify_membership_store(stale)


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
        connection.execute("PRAGMA ignore_check_constraints=ON")
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


def test_lookup_uses_strict_checked_tables_and_rejects_typed_corruption(
    tmp_path: Path,
    built_store: Path,
) -> None:
    checked = tmp_path / "strict-lookup"
    shutil.copytree(built_store, checked)
    connection = sqlite3.connect(checked / "lookup.sqlite")
    try:
        strict_by_table = {
            str(row[1]): int(row[5])
            for row in connection.execute("PRAGMA table_list")
            if row[1] in {"memberships", "metadata"}
        }
        assert strict_by_table == {"memberships": 1, "metadata": 1}

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE memberships SET pos = ? WHERE variant_key = ?",
                (101.5, "GRCh38:1:101:A:G"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE memberships SET clinical_significance = NULL WHERE variant_key = ?",
                ("GRCh38:1:1001:C:T",),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE memberships SET clinical_significance = TRUE WHERE variant_key = ?",
                ("GRCh38:1:1001:C:T",),
            )
    finally:
        connection.close()


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


def test_builder_fails_closed_when_declared_provenance_is_not_the_current_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_bundle: tuple[Path, tuple[MembershipSourceInput, ...]],
) -> None:
    lineage_path, sources = source_bundle
    output = tmp_path / "store"

    with pytest.raises(InputError, match="does not match the current checkout"):
        build_membership_store(
            artifact_id="wrong-builder-commit",
            snapshot_lineage_path=lineage_path,
            expected_snapshot_lineage_sha256=sha256_file(lineage_path),
            builder_git_commit="a" * 40,
            container_image=BUILDER_CONTAINER_IMAGE,
            sources=sources,
            output_dir=output,
        )
    monkeypatch.delenv("GENO_LEWM_VERIFIED_BUILD_CONTAINER_IMAGE")
    with pytest.raises(InputError, match="container provenance is unverifiable"):
        build_membership_store(
            artifact_id="unverified-container",
            snapshot_lineage_path=lineage_path,
            expected_snapshot_lineage_sha256=sha256_file(lineage_path),
            builder_git_commit=BUILDER_GIT_COMMIT,
            container_image=BUILDER_CONTAINER_IMAGE,
            sources=sources,
            output_dir=output,
        )
    monkeypatch.setenv(
        "GENO_LEWM_VERIFIED_BUILD_CONTAINER_IMAGE",
        "ghcr.io/abdelstark/other@sha256:" + "c" * 64,
    )
    with pytest.raises(InputError, match="does not match the verified invocation"):
        build_membership_store(
            artifact_id="wrong-container",
            snapshot_lineage_path=lineage_path,
            expected_snapshot_lineage_sha256=sha256_file(lineage_path),
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


def test_nonfixture_build_fails_closed_when_official_lineage_verification_fails(
    tmp_path: Path,
    source_bundle: tuple[Path, tuple[MembershipSourceInput, ...]],
) -> None:
    lineage_path, sources = source_bundle
    payload = json.loads(lineage_path.read_text(encoding="utf-8"))
    payload["assembly_inputs"] = {"fixture": False}
    payload["lineage_id"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "lineage_id"}
    )
    blocked = tmp_path / "snapshot-lineage.json"
    _write_json(blocked, payload)

    with pytest.raises(InputError, match="failed official verification"):
        build_membership_store(
            artifact_id="real-build-blocked",
            snapshot_lineage_path=blocked,
            expected_snapshot_lineage_sha256=sha256_file(blocked),
            builder_git_commit=BUILDER_GIT_COMMIT,
            container_image=BUILDER_CONTAINER_IMAGE,
            sources=sources,
            output_dir=tmp_path / "store",
        )


def test_nonfixture_lineage_consumes_one_exact_official_capture_without_rereading(
    tmp_path: Path,
    source_bundle: tuple[Path, tuple[MembershipSourceInput, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage_path, _sources = source_bundle
    verified_lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    verified_lineage["assembly_inputs"] = {
        "spec": {"sha256": "sha256:" + "1" * 64, "size_bytes": 1},
        "gnomad_source_lock": {"sha256": "sha256:" + "2" * 64, "size_bytes": 1},
    }
    for shard in verified_lineage["gnomad"]["shards"]:
        shard["remote_postflight"]["parquet_audit"] = {"officially_verified": True}
    verified_lineage["lineage_id"] = canonical_json_sha256(
        {key: value for key, value in verified_lineage.items() if key != "lineage_id"}
    )
    verified_payload = (
        "\n" + json.dumps(verified_lineage, separators=(",", ":"), sort_keys=False) + "\n"
    ).encode()
    verified_sha256 = "sha256:" + hashlib.sha256(verified_payload).hexdigest()

    def _freeze(value: object) -> object:
        if isinstance(value, dict):
            return MappingProxyType({key: _freeze(item) for key, item in value.items()})
        if isinstance(value, list):
            return tuple(_freeze(item) for item in value)
        return value

    immutable_lineage = _freeze(verified_lineage)
    replacement = b'{"replaced_after_official_capture":true}\n'
    captured_path = tmp_path / "snapshot-lineage.json"
    captured_path.write_bytes(b'{"unverified_path_bytes":true}\n')
    capture_calls: list[Path] = []

    def _capture(path: Path) -> SimpleNamespace:
        capture_calls.append(path)
        captured_path.write_bytes(replacement)
        return SimpleNamespace(
            payload=verified_payload,
            lineage=immutable_lineage,
            payload_sha256=verified_sha256,
            size_bytes=len(verified_payload),
        )

    def _forbid_reread(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("verified non-fixture lineage path was reopened")

    monkeypatch.setattr(membership_store_lineage, "capture_verified_snapshot_lineage", _capture)
    monkeypatch.setattr(membership_store_lineage, "_read_json_mapping", _forbid_reread)

    binding, expected_sources, bundled_payload = membership_store_lineage._load_snapshot_lineage(
        captured_path
    )

    assert capture_calls == [captured_path]
    assert captured_path.read_bytes() == replacement
    assert bundled_payload == verified_payload
    assert binding.sha256 == verified_sha256
    assert binding.size_bytes == len(verified_payload)
    assert binding.lineage_id == verified_lineage["lineage_id"]
    assert binding.evidence_profile == "official"
    assert set(expected_sources) == {
        "clinvar-2026-04-15",
        *(f"gnomad-v4.1-chr{chromosome}" for chromosome in range(1, 23)),
    }


def test_official_lineage_capture_adapter_rejects_incoherent_returned_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = SimpleNamespace(
        payload=b"{}",
        lineage=MappingProxyType({}),
        payload_sha256="sha256:" + "0" * 64,
        size_bytes=2,
    )
    monkeypatch.setattr(
        membership_store_lineage,
        "capture_verified_snapshot_lineage",
        lambda _path: capture,
    )

    with pytest.raises(InputError, match="official snapshot-lineage capture identity mismatch"):
        membership_store_lineage._capture_official_snapshot_lineage(tmp_path / "lineage.json")


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


def test_labeled_clinvar_stream_collapses_same_target_assertions_by_variant(
    tmp_path: Path,
) -> None:
    lineage_path, sources = _write_source_bundle(tmp_path / "same-target")
    clinvar = next(source for source in sources if source.kind == "clinvar")
    rows = pq.read_table(clinvar.path).to_pylist()
    rows.append(_clinvar_row("21", 1021, 99, significance="LP"))
    pq.write_table(pa.Table.from_pylist(rows, schema=_clinvar_schema()), clinvar.path)
    _refresh_clinvar_lineage_identity(lineage_path, clinvar.path, rows)

    output = tmp_path / "same-target-store"
    manifest = build_membership_store(
        artifact_id="same-target",
        snapshot_lineage_path=lineage_path,
        expected_snapshot_lineage_sha256=sha256_file(lineage_path),
        builder_git_commit=BUILDER_GIT_COMMIT,
        container_image=BUILDER_CONTAINER_IMAGE,
        sources=sources,
        output_dir=output,
    )

    assert manifest.source_kind_role_counts["clinvar"]["evaluation"] == 3
    assert sum(counts["evaluation"] for counts in manifest.clinvar_class_role_counts.values()) == 2
    with MembershipStore.open(output) as store:
        rows = tuple(store.iter_labeled_clinvar("evaluation"))
    assert len(rows) == 2
    selected = next(row for row in rows if row.membership.variant.pos == 1021)
    assert selected.clinical_significance == "P"


def test_builder_rejects_conflicting_binary_clinvar_targets(
    tmp_path: Path,
) -> None:
    lineage_path, sources = _write_source_bundle(tmp_path / "conflicting-target")
    clinvar = next(source for source in sources if source.kind == "clinvar")
    rows = pq.read_table(clinvar.path).to_pylist()
    rows.append(_clinvar_row("21", 1021, 99, significance="B"))
    pq.write_table(pa.Table.from_pylist(rows, schema=_clinvar_schema()), clinvar.path)
    _refresh_clinvar_lineage_identity(lineage_path, clinvar.path, rows)
    output = tmp_path / "conflicting-target-store"

    with pytest.raises(InputError, match="conflicting binary targets"):
        build_membership_store(
            artifact_id="conflicting-target",
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


def test_builder_reports_post_publication_parent_fsync_failure_honestly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_bundle: tuple[Path, tuple[MembershipSourceInput, ...]],
) -> None:
    lineage_path, sources = source_bundle
    writer = importlib.import_module("geno_lewm.data._membership_store_writer")
    real_fsync_directory = writer._fsync_directory
    calls = 0

    def _fail_after_publication(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected parent fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(writer, "_fsync_directory", _fail_after_publication)
    output = tmp_path / "store"
    with pytest.raises(ResourceError) as raised:
        build_membership_store(
            artifact_id="post-publication-fsync-failure",
            snapshot_lineage_path=lineage_path,
            expected_snapshot_lineage_sha256=sha256_file(lineage_path),
            builder_git_commit=BUILDER_GIT_COMMIT,
            container_image=BUILDER_CONTAINER_IMAGE,
            sources=sources,
            output_dir=output,
        )

    error = raised.value
    assert "was published" in error.message
    assert "before publication" not in error.message
    assert error.details == {
        "path": str(output),
        "publication_state": "published_durability_uncertain",
    }
    assert error.remediation is not None
    assert "Do not rerun against the same output path" in error.remediation
    assert calls == 2
    assert output.is_dir()
    assert {path.name for path in output.iterdir()} == {
        "build-receipt.json",
        "lookup.sqlite",
        "manifest.json",
        "memberships.parquet",
        "snapshot-lineage.json",
    }
    assert verify_membership_store(output).ok is True
    assert not tuple(tmp_path.glob(".store.tmp-*"))
    assert not tuple(tmp_path.glob(".store.sources-*"))


def test_atomic_publication_never_replaces_a_concurrently_created_empty_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_bundle: tuple[Path, tuple[MembershipSourceInput, ...]],
) -> None:
    lineage_path, sources = source_bundle
    output = tmp_path / "store"
    real_exists = Path.exists
    output_checks = 0
    creator: multiprocessing.Process | None = None

    def _exists_with_racing_creator(path: Path) -> bool:
        nonlocal creator, output_checks
        observed = real_exists(path)
        if path == output:
            output_checks += 1
            if output_checks == 2:
                context = multiprocessing.get_context("spawn")
                creator = context.Process(target=_create_empty_directory, args=(str(output),))
                creator.start()
                creator.join(timeout=30)
                assert creator.exitcode == 0
                assert observed is False
        return observed

    monkeypatch.setattr(Path, "exists", _exists_with_racing_creator)
    with pytest.raises(InputError, match="appeared before publication"):
        build_membership_store(
            artifact_id="publication-race",
            snapshot_lineage_path=lineage_path,
            expected_snapshot_lineage_sha256=sha256_file(lineage_path),
            builder_git_commit=BUILDER_GIT_COMMIT,
            container_image=BUILDER_CONTAINER_IMAGE,
            sources=sources,
            output_dir=output,
        )

    assert creator is not None
    assert output.is_dir()
    assert not tuple(output.iterdir())
    assert not tuple(tmp_path.glob(".store.tmp-*"))


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
    assert payload["row_count"] == 28
    assert payload["lineage_evidence_profile"] == "synthetic_fixture"
    assert payload["source_kind_filtered_counts"] == {"clinvar": 3, "gnomad": 0}
    assert payload["source_kind_role_counts"]["gnomad"]["train"] == 20
    assert payload["clinvar_class_role_counts"]["LB"]["evaluation"] == 1
    assert payload["source_kind_role_counts"]["clinvar"] == {
        "train": 2,
        "validation": 2,
        "evaluation": 2,
    }
    assert payload["clinvar_class_role_counts"]["P"]["evaluation"] == 1

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
    assert payload["row_count"] == 28
    assert payload["variant_count"] == 28
    assert payload["lineage_evidence_profile"] == "synthetic_fixture"
    assert payload["source_kind_filtered_counts"] == {"clinvar": 3, "gnomad": 0}
    assert payload["source_kind_role_counts"]["clinvar"]["validation"] == 2
    assert payload["clinvar_class_role_counts"]["LP"]["validation"] == 1
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
        _clinvar_row("21", 1021, 21, significance="P"),
        _clinvar_row("1", 1001, 1, significance="LP"),
        _clinvar_row("X", 1023, 23),
        _clinvar_row("20", 1020, 20, significance="B"),
        _clinvar_row("GL000220.1", 1024, 24),
        _clinvar_row("2", 1002, 25, significance="VUS"),
        _clinvar_row("21", 1121, 26, significance="LB"),
        _clinvar_row("20", 1120, 27, significance="LP"),
        _clinvar_row("3", 1003, 28, significance="B"),
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


def _refresh_clinvar_lineage_identity(
    lineage_path: Path, clinvar_path: Path, rows: list[dict[str, object]]
) -> None:
    payload = json.loads(lineage_path.read_text(encoding="utf-8"))
    output = payload["clinvar"]["output"]
    output["sha256"] = sha256_file(clinvar_path)
    output["size_bytes"] = clinvar_path.stat().st_size
    output["records"] = len(rows)
    output["class_balance"] = _clinvar_class_balance(rows)
    payload["clinvar"]["remote_postflight"]["parquet_audit"] = _clinvar_parquet_audit(rows)
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
