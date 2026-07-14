"""Behavior tests for the prepared real-training stream boundary."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import geno_lewm.training.real as real_module
import tools.data.v03_training_trace as trace_module
from geno_lewm.action import EditSpec
from geno_lewm.config import load_config
from geno_lewm.data import (
    EditSourceCount,
    GenoLeWMDataset,
    HoldoutPolicy,
    TrainingDatasetItem,
    WindowContext,
    synthetic_indel_provider,
    synthetic_snv_provider,
    variant_provider,
)
from geno_lewm.errors import InputError
from geno_lewm.provenance import canonical_json_sha256
from geno_lewm.training._data_stream import PreparedTrainingStream
from geno_lewm.training.real import _dataset_fallback_sources, _training_edit_contract
from tests.unit.test_training_preflight import _write_training_config
from tools.data.v03_training_trace import (
    author_training_trace,
    verify_training_trace_evidence,
)

_DATASET_REPOSITORY = "abdelstark/geno-lewm-data"
_DATASET_REVISION = "c" * 40
_DATASET_ARTIFACT_PATH = "candidates/v0.3/fixture/success"


def _membership_identity() -> dict[str, object]:
    content_identity = "sha256:" + "1" * 64
    policy = {
        "schema_version": "geno-lewm.membership-store.v1",
        "membership_content_identity": content_identity,
        "excluded_chromosomes": ["20", "21"],
        "selection": "chromosome_roles",
        "lookup": "lookup.sqlite",
    }
    return {
        "membership_store": {
            "path": "membership/store",
            "artifact_id": "fixture-membership-store",
            "content_identity": content_identity,
            "physical_identity": "sha256:" + "2" * 64,
            "rowset_sha256": "sha256:" + "3" * 64,
        },
        "report": {
            "path": "evidence/membership-split-evidence.json",
            "schema_path": "contract/membership-split-evidence.schema.json",
            "artifact_id": "fixture-membership-report",
            "schema_version": "geno-lewm.membership-split-evidence.v1",
        },
        "holdout_policy": policy,
        "holdout_policy_identity": canonical_json_sha256(policy),
    }


def _write_trace_manifest(path: Path, *, schema_version: str = "1.1.0") -> None:
    membership = _membership_identity()
    payload = {
        "schema_version": schema_version,
        "snapshot_id": "fixture-v03",
        "membership_and_split_evidence": {
            "membership_store": membership["membership_store"],
            "report": membership["report"],
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_trace_config(
    root: Path,
    *,
    batch_size: int,
    max_steps: int,
) -> Path:
    path = _write_training_config(root)
    body = path.read_text(encoding="utf-8")
    body = body.replace("schema_version: 1.0.0", "schema_version: 1.1.0")
    body = body.replace(
        "  revision: main\n",
        "  revision: " + "a" * 40 + "\n",
        1,
    )
    body = body.replace(
        "  normalize: true\n",
        "  normalize: true\n"
        "  state_contract_version: l2_normalized_v2\n"
        "  trust_remote_code: false\n",
        1,
    )
    body = body.replace("  batch_size: 8", f"  batch_size: {batch_size}")
    body = body.replace("  max_steps: 2", f"  max_steps: {max_steps}")
    path.write_text(body, encoding="utf-8")
    return path


def _minimal_trace_inputs(
    tmp_path: Path,
) -> tuple[PreparedTrainingStream, Path, Path]:
    config_path = _write_trace_config(tmp_path, batch_size=1, max_steps=1)
    config = load_config(config_path)
    window = WindowContext(
        record_id="placed-window",
        source="fixture",
        sequence="ACGT" * 64,
        chrom="1",
        start_bp=100,
    )
    stream = PreparedTrainingStream.from_components(
        dataset_snapshot_id="fixture-v03",
        schema_version="1.1.0",
        windows=(window,),
        providers={"synthetic_snv": synthetic_snv_provider},
        mix=(EditSourceCount("synthetic_snv", 1),),
        fallback_sources={},
        holdouts=HoldoutPolicy(),
        membership_identity=_membership_identity(),
        seed=config.seed,
    )
    manifest_path = tmp_path / "dataset_manifest.json"
    _write_trace_manifest(manifest_path)
    (tmp_path / "SHA256SUMS").write_text("fixture checksum closure\n", encoding="utf-8")
    return stream, manifest_path, config_path


def _author_kwargs(
    *, stream: PreparedTrainingStream, manifest_path: Path, config_path: Path
) -> dict[str, object]:
    return {
        "stream": stream,
        "dataset_manifest_path": manifest_path,
        "training_config_path": config_path,
        "producer_git_commit": "a" * 40,
        "container_image": "ghcr.io/abdelstark/geno-lewm@sha256:" + "b" * 64,
        "dataset_repository": _DATASET_REPOSITORY,
        "dataset_revision": _DATASET_REVISION,
        "dataset_artifact_path": _DATASET_ARTIFACT_PATH,
    }


def test_prepared_stream_filters_nonfallback_windows_before_training_rng(
    tmp_path: Path,
) -> None:
    config = load_config(_write_training_config(tmp_path))
    unusable = WindowContext(
        record_id="insufficient-gnomad",
        source="gnomad_common",
        sequence="A" * 256,
        chrom="1",
        start_bp=100,
    )
    usable = WindowContext(
        record_id="usable-gnomad",
        source="gnomad_common",
        sequence="A" * 256,
        chrom="2",
        start_bp=200,
    )
    gnomad = (
        EditSpec("1", 101, "A", "C"),
        EditSpec("1", 102, "A", "G"),
        EditSpec("2", 201, "A", "C"),
        EditSpec("2", 202, "A", "G"),
        EditSpec("2", 203, "A", "T"),
    )
    clinvar = (EditSpec("2", 204, "A", "C"),)
    providers, mix = _training_edit_contract(
        config,
        gnomad_edits=gnomad,
        clinvar_edits=clinvar,
    )
    fallbacks = _dataset_fallback_sources((unusable, usable))

    stream = PreparedTrainingStream.from_components(
        dataset_snapshot_id="fixture-v03",
        schema_version="1.1.0",
        windows=(unusable, usable),
        providers=providers,
        mix=mix,
        fallback_sources=fallbacks,
        holdouts=HoldoutPolicy(),
        membership_identity=_membership_identity(),
        seed=config.seed,
    )

    observed = tuple(stream.iter_epoch(0))
    expected = tuple(
        GenoLeWMDataset(
            (usable,),
            providers,
            seed=config.seed,
            mix=mix,
            fallback_sources=fallbacks,
            holdouts=HoldoutPolicy(),
        ).iter_with_source_windows()
    )

    assert observed == expected
    assert len(observed) == 8
    assert stream.input_window_count == 2
    assert stream.usable_window_count == 1
    assert [item.record_id for item in stream.exclusions] == ["insufficient-gnomad"]
    assert stream.exclusions[0].required_count == 3
    assert stream.exclusions[0].available_count == 2


def test_training_trace_authors_closed_immutable_cache_request_evidence(
    tmp_path: Path,
) -> None:
    config_path = _write_trace_config(tmp_path, batch_size=8, max_steps=1)
    config = load_config(config_path)
    usable = WindowContext(
        record_id="usable-gnomad",
        source="gnomad_common",
        sequence="A" * 256,
        chrom="2",
        start_bp=200,
    )
    providers, mix = _training_edit_contract(
        config,
        gnomad_edits=(
            EditSpec("2", 201, "A", "C"),
            EditSpec("2", 202, "A", "G"),
            EditSpec("2", 203, "A", "T"),
        ),
        clinvar_edits=(),
    )
    stream = PreparedTrainingStream.from_components(
        dataset_snapshot_id="fixture-v03",
        schema_version="1.1.0",
        windows=(usable,),
        providers=providers,
        mix=mix,
        fallback_sources={"clinvar": "synthetic_snv"},
        holdouts=HoldoutPolicy(),
        membership_identity=_membership_identity(),
        seed=config.seed,
    )
    manifest_path = tmp_path / "dataset_manifest.json"
    _write_trace_manifest(manifest_path)
    (tmp_path / "SHA256SUMS").write_text("fixture checksum closure\n", encoding="utf-8")
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = author_training_trace(
        stream=stream,
        dataset_manifest_path=manifest_path,
        training_config_path=config_path,
        output_dir=first_output,
        producer_git_commit="a" * 40,
        container_image="ghcr.io/abdelstark/geno-lewm@sha256:" + "b" * 64,
        dataset_repository=_DATASET_REPOSITORY,
        dataset_revision=_DATASET_REVISION,
        dataset_artifact_path=_DATASET_ARTIFACT_PATH,
    )
    second = author_training_trace(
        stream=stream,
        dataset_manifest_path=manifest_path,
        training_config_path=config_path,
        output_dir=second_output,
        producer_git_commit="a" * 40,
        container_image="ghcr.io/abdelstark/geno-lewm@sha256:" + "b" * 64,
        dataset_repository=_DATASET_REPOSITORY,
        dataset_revision=_DATASET_REVISION,
        dataset_artifact_path=_DATASET_ARTIFACT_PATH,
    )

    expected_files = {
        "cache_build_requests.jsonl",
        "training_config.yaml",
        "training_trace_report.json",
        "training_trace.schema.json",
        "SHA256SUMS",
    }
    assert {path.name for path in first_output.iterdir()} == expected_files
    assert {path.name for path in second_output.iterdir()} == expected_files
    assert first == second
    assert all(
        (first_output / name).read_bytes() == (second_output / name).read_bytes()
        for name in expected_files
    )
    requests = [
        json.loads(line)
        for line in (first_output / "cache_build_requests.jsonl").read_text().splitlines()
    ]
    assert len(requests) == 8
    assert all(
        set(row) == {"request_id", "chrom", "start_bp", "end_bp", "window", "edit_locus"}
        for row in requests
    )
    assert first["trace"]["request_rows"] == 8
    assert first["trace"]["source_counts"] == {
        "clinvar": 0,
        "gnomad_common": 3,
        "synthetic_snv": 5,
    }
    assert first["trace"]["fallback_counts"] == {"clinvar->synthetic_snv": 1}
    assert first["producer"] == {
        "container_binding": "launcher_environment_declaration",
        "declared_container_image": "ghcr.io/abdelstark/geno-lewm@sha256:" + "b" * 64,
        "git_commit": "a" * 40,
        "origin": "https://github.com/AbdelStark/GenoLeWM.git",
        "source_publication": {
            "endpoint": "https://api.github.com",
            "method": "unauthenticated_exact_commit_lookup",
        },
    }
    assert first["dataset"] == {
        "artifact_path": _DATASET_ARTIFACT_PATH,
        "manifest": first["dataset"]["manifest"],
        "membership_and_split_evidence": _membership_identity(),
        "publication_binding": first["dataset"]["publication_binding"],
        "repository": _DATASET_REPOSITORY,
        "revision": _DATASET_REVISION,
        "schema_version": "1.1.0",
        "snapshot_id": "fixture-v03",
    }
    checksums = (first_output / "SHA256SUMS").read_text().splitlines()
    assert [line.split("  ", 1)[1] for line in checksums] == [
        "cache_build_requests.jsonl",
        "training_config.yaml",
        "training_trace.schema.json",
        "training_trace_report.json",
    ]
    assert (
        verify_training_trace_evidence(
            stream=stream,
            dataset_manifest_path=manifest_path,
            training_config_path=config_path,
            evidence_dir=first_output,
            producer_git_commit="a" * 40,
            container_image="ghcr.io/abdelstark/geno-lewm@sha256:" + "b" * 64,
            dataset_repository=_DATASET_REPOSITORY,
            dataset_revision=_DATASET_REVISION,
            dataset_artifact_path=_DATASET_ARTIFACT_PATH,
        )
        == first
    )

    requests_path = first_output / "cache_build_requests.jsonl"
    requests_path.chmod(0o600)
    requests_path.write_bytes(requests_path.read_bytes() + b"tampered\n")
    with pytest.raises(InputError, match="does not match exact re-authoring"):
        verify_training_trace_evidence(
            stream=stream,
            dataset_manifest_path=manifest_path,
            training_config_path=config_path,
            evidence_dir=first_output,
            producer_git_commit="a" * 40,
            container_image="ghcr.io/abdelstark/geno-lewm@sha256:" + "b" * 64,
            dataset_repository=_DATASET_REPOSITORY,
            dataset_revision=_DATASET_REVISION,
            dataset_artifact_path=_DATASET_ARTIFACT_PATH,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("request_rows", "batch cardinalities do not cover"),
        ("source_counts", "source_counts do not sum"),
        ("policy_identity", "holdout-policy identity is not canonical"),
    ],
)
def test_training_trace_report_rejects_semantic_inconsistencies(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    stream, manifest_path, config_path = _minimal_trace_inputs(tmp_path)
    report = author_training_trace(
        output_dir=tmp_path / "valid-trace",
        **_author_kwargs(
            stream=stream,
            manifest_path=manifest_path,
            config_path=config_path,
        ),
    )
    mutated: dict[str, Any] = json.loads(json.dumps(report))
    if mutation == "request_rows":
        mutated["trace"]["request_rows"] = 2
    elif mutation == "source_counts":
        mutated["trace"]["source_counts"]["synthetic_snv"] = 2
    else:
        mutated["dataset"]["membership_and_split_evidence"]["holdout_policy_identity"] = (
            "sha256:" + "f" * 64
        )
    schema = json.loads(trace_module.DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))

    with pytest.raises(InputError, match=message):
        trace_module._validate_report(mutated, schema)


def test_training_trace_rejects_stream_membership_not_bound_by_manifest(
    tmp_path: Path,
) -> None:
    stream, manifest_path, config_path = _minimal_trace_inputs(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["membership_and_split_evidence"]["membership_store"]["content_identity"] = (
        "sha256:" + "9" * 64
    )
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="membership binding does not match"):
        author_training_trace(
            output_dir=tmp_path / "trace",
            **_author_kwargs(
                stream=stream,
                manifest_path=manifest_path,
                config_path=config_path,
            ),
        )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "state_contract_version: l2_normalized_v2",
            "state_contract_version: legacy_raw_v1",
            "requires corrected l2_normalized_v2 states",
        ),
        ("max_steps: 1", "max_steps: 2", "max_steps must consume exactly one"),
    ],
)
def test_training_trace_rejects_noncanonical_training_contract(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    stream, manifest_path, config_path = _minimal_trace_inputs(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )

    with pytest.raises(InputError, match=message):
        author_training_trace(
            output_dir=tmp_path / "trace",
            **_author_kwargs(
                stream=stream,
                manifest_path=manifest_path,
                config_path=config_path,
            ),
        )


def test_prepared_stream_open_owns_and_closes_verified_membership_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = WindowContext(
        record_id="legacy",
        source="fixture",
        sequence="ACGT" * 64,
    )
    events: list[object] = []

    class HoldoutContext:
        def __enter__(self) -> HoldoutPolicy:
            events.append("enter")
            return HoldoutPolicy()

        def __exit__(self, *args: object) -> None:
            events.append(("exit", *args))

    monkeypatch.setattr(
        real_module,
        "_load_dataset_manifest",
        lambda _root: {
            "snapshot_id": "legacy-fixture",
            "schema_version": "1.0.0",
        },
    )
    monkeypatch.setattr(real_module, "_dataset_files", lambda _manifest: ())
    monkeypatch.setattr(real_module, "_load_windows", lambda *_args, **_kwargs: iter((window,)))
    monkeypatch.setattr(
        real_module,
        "_load_gnomad_edits",
        lambda *_args, **_kwargs: iter((EditSpec("1", 1, "A", "C"),)),
    )
    monkeypatch.setattr(
        real_module,
        "_load_clinvar_edits",
        lambda *_args, **_kwargs: iter(()),
    )
    monkeypatch.setattr(
        real_module,
        "_membership_holdout_policy",
        lambda *_args: HoldoutContext(),
    )
    monkeypatch.setattr(
        real_module,
        "_membership_runtime_identity",
        lambda *_args: {"verified": True},
    )
    monkeypatch.setattr(
        real_module,
        "_training_edit_contract",
        lambda *_args, **_kwargs: (
            {"synthetic_snv": synthetic_snv_provider},
            (EditSourceCount("synthetic_snv", 1),),
        ),
    )
    monkeypatch.setattr(real_module, "_dataset_fallback_sources", lambda _windows: {})

    with PreparedTrainingStream.open(
        dataset_dir=tmp_path,
        config=SimpleNamespace(seed=7),
        require_membership=True,
    ) as stream:
        assert stream.dataset_snapshot_id == "legacy-fixture"
        assert stream.usable_window_count == 1
        assert events == ["enter"]

    assert events == ["enter", ("exit", None, None, None)]
    stream.close()
    with pytest.raises(InputError, match="closed"):
        stream.__enter__()
    with pytest.raises(InputError, match="closed"):
        tuple(stream.iter_epoch(0))


def test_prepared_stream_open_closes_membership_context_on_required_binding_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = WindowContext(
        record_id="placed",
        source="fixture",
        sequence="ACGT" * 64,
        chrom="1",
    )
    exits: list[tuple[object, ...]] = []

    class HoldoutContext:
        def __enter__(self) -> HoldoutPolicy:
            return HoldoutPolicy()

        def __exit__(self, *args: object) -> None:
            exits.append(args)

    monkeypatch.setattr(
        real_module,
        "_load_dataset_manifest",
        lambda _root: {"snapshot_id": "v03", "schema_version": "1.1.0"},
    )
    monkeypatch.setattr(real_module, "_dataset_files", lambda _manifest: ())
    monkeypatch.setattr(real_module, "_load_windows", lambda *_args, **_kwargs: iter((window,)))
    monkeypatch.setattr(
        real_module,
        "_load_gnomad_edits",
        lambda *_args, **_kwargs: iter((EditSpec("1", 1, "A", "C"),)),
    )
    monkeypatch.setattr(real_module, "_load_clinvar_edits", lambda *_args, **_kwargs: iter(()))
    monkeypatch.setattr(
        real_module,
        "_membership_holdout_policy",
        lambda *_args: HoldoutContext(),
    )
    monkeypatch.setattr(real_module, "_membership_runtime_identity", lambda *_args: None)

    with pytest.raises(InputError, match="requires verified membership"):
        PreparedTrainingStream.open(
            dataset_dir=tmp_path,
            config=SimpleNamespace(seed=7),
            require_membership=True,
        )

    assert len(exits) == 1
    assert exits[0][0] is InputError


def test_prepared_stream_rejects_invalid_epochs_and_unusable_release_windows(
    tmp_path: Path,
) -> None:
    stream, _manifest_path, _config_path = _minimal_trace_inputs(tmp_path)
    with pytest.raises(InputError, match="non-negative integer"):
        tuple(stream.iter_epoch(True))

    repeated = stream.iter_repeated()
    assert next(repeated).source_window.record_id == "placed-window"
    assert next(repeated).source_window.record_id == "placed-window"

    window = stream.usable_windows[0]
    with pytest.raises(InputError, match="no usable source windows"):
        PreparedTrainingStream.from_components(
            dataset_snapshot_id="fixture-v03",
            schema_version="1.1.0",
            windows=(window,),
            providers={"gnomad_common": variant_provider(())},
            mix=(EditSourceCount("gnomad_common", 1),),
            fallback_sources={},
            holdouts=HoldoutPolicy(),
            membership_identity=_membership_identity(),
            seed=7,
        )

    with pytest.raises(InputError, match="provider is missing"):
        PreparedTrainingStream.from_components(
            dataset_snapshot_id="fixture-v03",
            schema_version="1.1.0",
            windows=(window,),
            providers={"synthetic_snv": synthetic_snv_provider},
            mix=(EditSourceCount("gnomad_common", 1),),
            fallback_sources={},
            holdouts=HoldoutPolicy(),
            membership_identity=_membership_identity(),
            seed=7,
        )


def test_prepared_stream_eligibility_is_seed_independent_with_edit_holdouts() -> None:
    window = WindowContext(
        record_id="placed",
        source="gnomad_common",
        sequence="A" * 256,
        chrom="1",
        start_bp=100,
    )
    provider = variant_provider(
        (
            EditSpec("1", 101, "A", "C"),
            EditSpec("1", 102, "A", "G"),
            EditSpec("1", 103, "A", "T"),
        )
    )
    holdouts = HoldoutPolicy(edit_keys=("1:101:A:C",))

    for seed in range(10):
        stream = PreparedTrainingStream.from_components(
            dataset_snapshot_id="fixture-v03",
            schema_version="1.1.0",
            windows=(window,),
            providers={"gnomad_common": provider},
            mix=(EditSourceCount("gnomad_common", 2),),
            fallback_sources={},
            holdouts=holdouts,
            membership_identity=_membership_identity(),
            seed=seed,
        )
        observed = tuple(stream.iter_epoch(0))
        assert len(observed) == 2
        assert {item.training_tuple.rel_edits[0].rel_pos for item in observed} == {1, 2}


def test_prepared_stream_synthetic_sampling_refills_after_edit_holdouts() -> None:
    window = WindowContext(
        record_id="placed",
        source="fixture",
        sequence="A" * 256,
        chrom="1",
        start_bp=100,
    )
    blocked = synthetic_snv_provider(window, 4, random.Random(0))[0]
    blocked_key = (
        f"1:{window.start_bp + blocked.rel_pos + 1}:{blocked.ref_bases}:{blocked.alt_bases}"
    )
    holdouts = HoldoutPolicy(edit_keys=(blocked_key,))

    for seed in range(10):
        stream = PreparedTrainingStream.from_components(
            dataset_snapshot_id="fixture-v03",
            schema_version="1.1.0",
            windows=(window,),
            providers={"synthetic_snv": synthetic_snv_provider},
            mix=(EditSourceCount("synthetic_snv", 4),),
            fallback_sources={},
            holdouts=holdouts,
            membership_identity=_membership_identity(),
            seed=seed,
        )
        observed = tuple(stream.iter_epoch(0))
        assert len(observed) == 4
        assert all(
            not holdouts.excludes_edit(window, item.training_tuple.rel_edits[0])
            for item in observed
        )


def test_prepared_stream_rejects_synthetic_indel_edit_key_holdouts_before_rng() -> None:
    window = WindowContext(
        record_id="placed",
        source="fixture",
        sequence="A" * 256,
        chrom="1",
        start_bp=100,
    )

    with pytest.raises(InputError, match="synthetic-indel edit-key holdouts"):
        PreparedTrainingStream.from_components(
            dataset_snapshot_id="fixture-v03",
            schema_version="1.1.0",
            windows=(window,),
            providers={"synthetic_indel": synthetic_indel_provider},
            mix=(EditSourceCount("synthetic_indel", 1),),
            fallback_sources={},
            holdouts=HoldoutPolicy(edit_keys=("1:165:A:AGTT",)),
            membership_identity=_membership_identity(),
            seed=0,
        )


def test_prepared_stream_excludes_synthetic_windows_without_editable_bases() -> None:
    window = WindowContext(
        record_id="ambiguous-only",
        source="fixture",
        sequence="N" * 256,
        chrom="1",
        start_bp=100,
    )

    with pytest.raises(InputError, match="no usable source windows"):
        PreparedTrainingStream.from_components(
            dataset_snapshot_id="fixture-v03",
            schema_version="1.1.0",
            windows=(window,),
            providers={"synthetic_snv": synthetic_snv_provider},
            mix=(EditSourceCount("synthetic_snv", 1),),
            fallback_sources={},
            holdouts=HoldoutPolicy(),
            membership_identity=_membership_identity(),
            seed=0,
        )


@pytest.mark.parametrize(
    ("validator", "value", "message"),
    [
        (trace_module._require_commit, "A" * 40, "producer_git_commit"),
        (trace_module._require_container_image, "image:latest", "digest-pinned"),
        (trace_module._require_dataset_repository, "someone/else", "canonical public dataset"),
        (trace_module._require_dataset_revision, "main", "exact lowercase 40-hex"),
        (
            trace_module._require_dataset_artifact_path,
            "releases/v0.3/success",
            "successful v0.3 candidate",
        ),
        (
            trace_module._require_dataset_artifact_path,
            "candidates/v0.3/fixture/../success",
            "unsafe path component",
        ),
        (
            trace_module._require_dataset_artifact_path,
            "candidates/v0.3//fixture/success",
            "normalized relative Hub path",
        ),
    ],
)
def test_training_trace_rejects_ambiguous_producer_and_dataset_bindings(
    validator: object,
    value: object,
    message: str,
) -> None:
    assert callable(validator)
    with pytest.raises(InputError, match=message):
        validator(value)


def test_training_trace_build_and_verify_wrappers_replay_the_exact_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_stream, _manifest_path, config_path = _minimal_trace_inputs(tmp_path)
    second_stream, _manifest_path, _config_path = _minimal_trace_inputs(tmp_path)
    streams = iter((first_stream, second_stream))
    monkeypatch.setattr(trace_module, "_verify_producer_invocation", lambda **_kwargs: None)
    monkeypatch.setattr(
        trace_module,
        "_verify_dataset_publication_binding",
        lambda **_kwargs: b"fixture checksum closure\n",
    )
    monkeypatch.setattr(
        PreparedTrainingStream,
        "open",
        staticmethod(lambda **_kwargs: next(streams)),
    )
    common = {
        "dataset_dir": tmp_path,
        "training_config_path": config_path,
        "producer_git_commit": "a" * 40,
        "container_image": "ghcr.io/abdelstark/geno-lewm@sha256:" + "b" * 64,
        "dataset_repository": _DATASET_REPOSITORY,
        "dataset_revision": _DATASET_REVISION,
        "dataset_artifact_path": _DATASET_ARTIFACT_PATH,
    }
    output_dir = tmp_path / "trace"

    built = trace_module.build_training_trace(output_dir=output_dir, **common)
    verified = trace_module.verify_training_trace(evidence_dir=output_dir, **common)

    assert verified == built


def test_training_trace_build_does_not_promote_dataset_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, _manifest_path, config_path = _minimal_trace_inputs(tmp_path)
    receipts = iter((b"fixture checksum closure\n", b"drifted checksum closure\n"))
    monkeypatch.setattr(trace_module, "_verify_producer_invocation", lambda **_kwargs: None)
    monkeypatch.setattr(
        trace_module,
        "_verify_dataset_publication_binding",
        lambda **_kwargs: next(receipts),
    )
    monkeypatch.setattr(
        PreparedTrainingStream,
        "open",
        staticmethod(lambda **_kwargs: stream),
    )
    output_dir = tmp_path / "trace"

    with pytest.raises(InputError, match="publication binding changed"):
        trace_module.build_training_trace(
            dataset_dir=tmp_path,
            training_config_path=config_path,
            output_dir=output_dir,
            producer_git_commit="a" * 40,
            container_image="ghcr.io/abdelstark/geno-lewm@sha256:" + "b" * 64,
            dataset_repository=_DATASET_REPOSITORY,
            dataset_revision=_DATASET_REVISION,
            dataset_artifact_path=_DATASET_ARTIFACT_PATH,
        )

    assert not output_dir.exists()


def test_training_trace_evidence_accepts_transport_modes_and_rejects_inventory_drift(
    tmp_path: Path,
) -> None:
    stream, manifest_path, config_path = _minimal_trace_inputs(tmp_path)
    kwargs = _author_kwargs(
        stream=stream,
        manifest_path=manifest_path,
        config_path=config_path,
    )
    evidence_dir = tmp_path / "trace"
    trace_module.author_training_trace(output_dir=evidence_dir, **kwargs)

    requests_path = evidence_dir / trace_module.REQUESTS_NAME
    requests_path.chmod(0o600)
    assert trace_module.verify_training_trace_evidence(evidence_dir=evidence_dir, **kwargs)

    extra = evidence_dir / "unexpected.txt"
    extra.write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(InputError, match="inventory is not closed"):
        trace_module.verify_training_trace_evidence(evidence_dir=evidence_dir, **kwargs)


def test_training_trace_author_rejects_metadata_drift(tmp_path: Path) -> None:
    stream, manifest_path, config_path = _minimal_trace_inputs(tmp_path)
    kwargs = _author_kwargs(
        stream=stream,
        manifest_path=manifest_path,
        config_path=config_path,
    )

    stream.schema_version = "1.0.0"
    with pytest.raises(InputError, match=r"requires dataset schema 1\.1\.0"):
        trace_module.author_training_trace(output_dir=tmp_path / "schema", **kwargs)
    stream.schema_version = "1.1.0"

    stream.membership_identity = None
    with pytest.raises(InputError, match="requires verified membership"):
        trace_module.author_training_trace(output_dir=tmp_path / "membership", **kwargs)
    stream.membership_identity = _membership_identity()

    manifest_path.write_text(
        '{"schema_version":"1.0.0","snapshot_id":"fixture-v03"}\n',
        encoding="utf-8",
    )
    with pytest.raises(InputError, match="schema does not match"):
        trace_module.author_training_trace(output_dir=tmp_path / "manifest-schema", **kwargs)
    manifest_path.write_text(
        '{"schema_version":"1.1.0","snapshot_id":"other"}\n',
        encoding="utf-8",
    )
    with pytest.raises(InputError, match="snapshot does not match"):
        trace_module.author_training_trace(output_dir=tmp_path / "manifest-snapshot", **kwargs)
    _write_trace_manifest(manifest_path)

    stream.seed += 1
    with pytest.raises(InputError, match="data seed does not match"):
        trace_module.author_training_trace(output_dir=tmp_path / "seed", **kwargs)


def test_training_trace_request_contract_rejects_stream_drift(tmp_path: Path) -> None:
    stream, _manifest_path, _config_path = _minimal_trace_inputs(tmp_path)
    item = next(stream.iter_epoch(0))

    empty_mix = SimpleNamespace(
        mix=(EditSourceCount("synthetic_snv", 0),),
        fallback_sources={},
    )
    with pytest.raises(InputError, match="at least one row"):
        trace_module._request_artifact(empty_mix)

    too_many = SimpleNamespace(
        mix=stream.mix,
        fallback_sources=stream.fallback_sources,
        usable_window_count=0,
        usable_windows=(),
        iter_epoch=lambda _epoch: iter((item,)),
    )
    with pytest.raises(InputError, match="more rows"):
        trace_module._request_artifact(too_many)

    other_window = WindowContext(
        record_id="other",
        source="fixture",
        sequence="ACGT" * 64,
        chrom="1",
        start_bp=100,
    )
    wrong_order = SimpleNamespace(
        mix=stream.mix,
        fallback_sources=stream.fallback_sources,
        usable_window_count=1,
        usable_windows=(other_window,),
        iter_epoch=lambda _epoch: iter((item,)),
    )
    with pytest.raises(InputError, match="order drifted"):
        trace_module._request_artifact(wrong_order)

    undeclared_source = SimpleNamespace(
        mix=(EditSourceCount("gnomad_common", 1),),
        fallback_sources={},
        usable_window_count=1,
        usable_windows=(item.source_window,),
        iter_epoch=lambda _epoch: iter((item,)),
    )
    with pytest.raises(InputError, match="undeclared source substitution"):
        trace_module._request_artifact(undeclared_source)

    unplaced = WindowContext(
        record_id="unplaced",
        source="fixture",
        sequence=item.source_window.sequence,
    )
    unplaced_item = TrainingDatasetItem(
        source_window=unplaced,
        training_tuple=item.training_tuple,
    )
    unplaced_stream = SimpleNamespace(
        mix=stream.mix,
        fallback_sources=stream.fallback_sources,
        usable_window_count=1,
        usable_windows=(unplaced,),
        iter_epoch=lambda _epoch: iter((unplaced_item,)),
    )
    with pytest.raises(InputError, match="require placed source windows"):
        trace_module._request_artifact(unplaced_stream)

    missing_edit_item = SimpleNamespace(
        source_window=item.source_window,
        training_tuple=SimpleNamespace(edit_source="synthetic_snv", rel_edits=()),
    )
    missing_edit_stream = SimpleNamespace(
        mix=stream.mix,
        fallback_sources=stream.fallback_sources,
        usable_window_count=1,
        usable_windows=(item.source_window,),
        iter_epoch=lambda _epoch: iter((missing_edit_item,)),
    )
    with pytest.raises(InputError, match="contains no edit locus"):
        trace_module._request_artifact(missing_edit_stream)

    too_few = SimpleNamespace(
        mix=stream.mix,
        fallback_sources=stream.fallback_sources,
        usable_window_count=1,
        usable_windows=(item.source_window,),
        iter_epoch=lambda _epoch: iter(()),
    )
    with pytest.raises(InputError, match="row count does not match"):
        trace_module._request_artifact(too_few)


def test_training_trace_filesystem_guards_reject_unsafe_evidence_paths(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(InputError, match="already exists"):
        trace_module._publish_closed_directory(existing, {"one": b"1"})

    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("file\n", encoding="utf-8")
    with pytest.raises(InputError, match="directories only"):
        trace_module._reject_symlink_ancestors(parent_file)

    symlink = tmp_path / "symlink"
    symlink.symlink_to(existing, target_is_directory=True)
    with pytest.raises(InputError, match="symlink components"):
        trace_module._reject_symlink_ancestors(symlink)

    with pytest.raises(InputError, match="directory is missing"):
        trace_module._capture_evidence_directory(tmp_path / "missing")
    with pytest.raises(InputError, match="non-symlink directory"):
        trace_module._capture_evidence_directory(parent_file)

    nonregular = tmp_path / "nonregular"
    nonregular.mkdir()
    (nonregular / "child").mkdir()
    with pytest.raises(InputError, match="contains a non-regular file"):
        trace_module._capture_evidence_directory(nonregular)


def test_training_trace_json_and_regular_file_guards(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="is missing"):
        trace_module._read_regular_bytes(tmp_path / "missing", label="input")

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(InputError, match="regular non-symlink"):
        trace_module._read_regular_bytes(directory, label="input")
    target = tmp_path / "target"
    target.write_bytes(b"target")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(InputError, match="regular non-symlink"):
        trace_module._read_regular_bytes(link, label="input")

    with pytest.raises(InputError, match="JSON is invalid"):
        trace_module._json_object(b"not-json", label="input")
    with pytest.raises(InputError, match="must be a JSON object"):
        trace_module._json_object(b"[]", label="input")


def test_training_trace_binds_verified_local_inventory_to_remote_checksums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "SHA256SUMS").write_bytes(b"closed inventory\n")
    (tmp_path / "artifact.txt").write_bytes(b"artifact\n")
    verified: list[Path] = []
    remote_calls: list[dict[str, object]] = []

    def _record_local_verification(root: Path) -> None:
        verified.append(root)

    monkeypatch.setattr(
        trace_module,
        "verify_v03_dataset_snapshot",
        _record_local_verification,
    )
    monkeypatch.setattr(
        trace_module,
        "_verify_remote_dataset_namespace",
        lambda **kwargs: remote_calls.append(kwargs),
    )

    checksums = trace_module._verify_dataset_publication_binding(
        dataset_dir=tmp_path,
        repository=_DATASET_REPOSITORY,
        revision=_DATASET_REVISION,
        artifact_path=_DATASET_ARTIFACT_PATH,
    )

    assert verified == [tmp_path]
    assert checksums == b"closed inventory\n"
    assert remote_calls == [
        {
            "repository": _DATASET_REPOSITORY,
            "revision": _DATASET_REVISION,
            "artifact_path": _DATASET_ARTIFACT_PATH,
            "local_root": tmp_path,
            "expected_inventory": {"SHA256SUMS", "artifact.txt"},
            "expected_checksums": b"closed inventory\n",
        }
    ]


def _remote_binding_case(tmp_path: Path) -> tuple[bytes, list[SimpleNamespace]]:
    artifact_bytes = b"artifact payload\n"
    metadata_bytes = b'{"snapshot":"fixture"}\n'
    artifact_path = tmp_path / "artifact.bin"
    metadata_path = tmp_path / "metadata.json"
    artifact_path.write_bytes(artifact_bytes)
    metadata_path.write_bytes(metadata_bytes)

    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    expected_checksums = (
        f"{artifact_sha256}  artifact.bin\n{metadata_sha256}  metadata.json\n"
    ).encode()
    checksums_path = tmp_path / "SHA256SUMS"
    checksums_path.write_bytes(expected_checksums)

    prefix = f"{_DATASET_ARTIFACT_PATH}/"
    siblings = [
        SimpleNamespace(rfilename="unrelated.txt"),
        SimpleNamespace(
            rfilename=f"{prefix}SHA256SUMS",
            size=checksums_path.stat().st_size,
            blob_id=trace_module._git_blob_sha1(checksums_path),
            lfs=None,
        ),
        SimpleNamespace(
            rfilename=f"{prefix}artifact.bin",
            size=artifact_path.stat().st_size,
            blob_id="unused-for-lfs",
            lfs=SimpleNamespace(sha256=artifact_sha256),
        ),
        SimpleNamespace(
            rfilename=f"{prefix}metadata.json",
            size=metadata_path.stat().st_size,
            blob_id=trace_module._git_blob_sha1(metadata_path),
            lfs=None,
        ),
    ]
    return expected_checksums, siblings


def _install_remote_binding_hub_fake(
    monkeypatch: pytest.MonkeyPatch,
    *,
    resolved_revision: str,
    siblings: list[SimpleNamespace],
) -> tuple[list[tuple[object, str]], list[dict[str, object]]]:
    constructor_calls: list[tuple[object, str]] = []
    repo_info_calls: list[dict[str, object]] = []

    class FakeApi:
        def __init__(self, *, token: object, endpoint: str) -> None:
            constructor_calls.append((token, endpoint))

        def repo_info(self, **kwargs: object) -> SimpleNamespace:
            repo_info_calls.append(kwargs)
            return SimpleNamespace(sha=resolved_revision, siblings=siblings)

    fake_hub = SimpleNamespace(HfApi=FakeApi)
    monkeypatch.setattr(
        trace_module.importlib,
        "import_module",
        lambda name: fake_hub if name == "huggingface_hub" else None,
    )
    return constructor_calls, repo_info_calls


def _remote_binding_kwargs(
    tmp_path: Path,
    expected_checksums: bytes,
) -> dict[str, object]:
    return {
        "repository": _DATASET_REPOSITORY,
        "revision": _DATASET_REVISION,
        "artifact_path": _DATASET_ARTIFACT_PATH,
        "local_root": tmp_path,
        "expected_inventory": {"SHA256SUMS", "artifact.bin", "metadata.json"},
        "expected_checksums": expected_checksums,
    }


def test_training_trace_remote_binding_resolves_exact_namespace_and_pins_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_checksums, siblings = _remote_binding_case(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_ENDPOINT", "https://hostile.invalid")
    constructor_calls, repo_info_calls = _install_remote_binding_hub_fake(
        monkeypatch,
        resolved_revision=_DATASET_REVISION,
        siblings=siblings,
    )

    kwargs = _remote_binding_kwargs(tmp_path, expected_checksums)
    trace_module._verify_remote_dataset_namespace(**kwargs)

    assert constructor_calls == [(False, "https://huggingface.co")]
    assert repo_info_calls == [
        {
            "repo_id": _DATASET_REPOSITORY,
            "repo_type": "dataset",
            "revision": _DATASET_REVISION,
            "files_metadata": True,
        }
    ]


def test_training_trace_remote_binding_rejects_resolved_revision_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_checksums, siblings = _remote_binding_case(tmp_path)
    _install_remote_binding_hub_fake(
        monkeypatch,
        resolved_revision="d" * 40,
        siblings=siblings,
    )

    with pytest.raises(InputError, match="resolved revision differs"):
        trace_module._verify_remote_dataset_namespace(
            **_remote_binding_kwargs(tmp_path, expected_checksums)
        )


def test_training_trace_remote_binding_rejects_namespace_inventory_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_checksums, siblings = _remote_binding_case(tmp_path)
    siblings = [
        sibling
        for sibling in siblings
        if getattr(sibling, "rfilename", None) != f"{_DATASET_ARTIFACT_PATH}/artifact.bin"
    ]
    _install_remote_binding_hub_fake(
        monkeypatch,
        resolved_revision=_DATASET_REVISION,
        siblings=siblings,
    )

    with pytest.raises(InputError, match="namespace inventory differs"):
        trace_module._verify_remote_dataset_namespace(
            **_remote_binding_kwargs(tmp_path, expected_checksums)
        )


@pytest.mark.parametrize(
    ("identity_kind", "message"),
    [
        ("lfs", "LFS identity differs"),
        ("git", "Git blob identity differs"),
    ],
)
def test_training_trace_remote_binding_rejects_content_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_kind: str,
    message: str,
) -> None:
    expected_checksums, siblings = _remote_binding_case(tmp_path)
    if identity_kind == "lfs":
        artifact = next(
            sibling
            for sibling in siblings
            if getattr(sibling, "rfilename", None) == f"{_DATASET_ARTIFACT_PATH}/artifact.bin"
        )
        artifact.lfs = SimpleNamespace(sha256="0" * 64)
    else:
        metadata = next(
            sibling
            for sibling in siblings
            if getattr(sibling, "rfilename", None) == f"{_DATASET_ARTIFACT_PATH}/metadata.json"
        )
        metadata.blob_id = "0" * 40
    _install_remote_binding_hub_fake(
        monkeypatch,
        resolved_revision=_DATASET_REVISION,
        siblings=siblings,
    )

    with pytest.raises(InputError, match=message):
        trace_module._verify_remote_dataset_namespace(
            **_remote_binding_kwargs(tmp_path, expected_checksums)
        )


def test_training_trace_remote_binding_wraps_hub_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trace_module.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("missing hub client")),
    )
    with pytest.raises(InputError, match="cannot verify the exact-revision"):
        trace_module._verify_remote_dataset_namespace(
            repository=_DATASET_REPOSITORY,
            revision=_DATASET_REVISION,
            artifact_path=_DATASET_ARTIFACT_PATH,
            local_root=tmp_path,
            expected_inventory={"SHA256SUMS"},
            expected_checksums=b"closure\n",
        )


def test_training_trace_cli_routes_build_and_verify_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def _record(mode: str, **kwargs: object) -> dict[str, object]:
        calls.append((mode, kwargs))
        return {"mode": mode}

    monkeypatch.setattr(
        trace_module,
        "build_training_trace",
        lambda **kwargs: _record("build", **kwargs),
    )
    monkeypatch.setattr(
        trace_module,
        "verify_training_trace",
        lambda **kwargs: _record("verify", **kwargs),
    )
    base_args = [
        "--dataset-dir",
        str(tmp_path / "dataset"),
        "--training-config",
        str(tmp_path / "train.yaml"),
        "--output-dir",
        str(tmp_path / "trace"),
        "--producer-git-commit",
        "a" * 40,
        "--container-image",
        "image@sha256:" + "b" * 64,
        "--dataset-repository",
        _DATASET_REPOSITORY,
        "--dataset-revision",
        _DATASET_REVISION,
        "--dataset-artifact-path",
        _DATASET_ARTIFACT_PATH,
    ]

    assert trace_module.main(base_args) == 0
    assert json.loads(capsys.readouterr().out) == {"mode": "build"}
    assert trace_module.main([*base_args, "--verify-existing"]) == 0
    assert json.loads(capsys.readouterr().out) == {"mode": "verify"}
    assert [mode for mode, _kwargs in calls] == ["build", "verify"]


def test_training_trace_producer_gate_binds_clean_exact_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40
    image = "image@sha256:" + "b" * 64
    monkeypatch.setenv("GENO_LEWM_TRAINING_TRACE_DECLARED_CONTAINER_IMAGE", image)
    public_commits: list[str] = []

    def _git_output(*arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return commit
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        if arguments == ("remote", "get-url", "origin"):
            return "https://github.com/AbdelStark/GenoLeWM.git"
        if arguments[:2] == ("cat-file", "-e"):
            return ""
        raise AssertionError(arguments)

    def _record_public_commit(value: str) -> None:
        public_commits.append(value)

    monkeypatch.setattr(trace_module, "_git_output", _git_output)
    monkeypatch.setattr(
        trace_module,
        "_verify_public_git_commit",
        _record_public_commit,
    )
    trace_module._verify_producer_invocation(
        producer_git_commit=commit,
        container_image=image,
    )
    assert public_commits == [commit]

    monkeypatch.setenv("GENO_LEWM_TRAINING_TRACE_DECLARED_CONTAINER_IMAGE", "wrong")
    with pytest.raises(InputError, match="launcher declaration"):
        trace_module._verify_producer_invocation(
            producer_git_commit=commit,
            container_image=image,
        )


def test_training_trace_public_git_binding_is_exact_and_unauthenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40
    expected_url = "https://api.github.com/repos/AbdelStark/GenoLeWM/commits/" + commit
    state = {"sha": commit}

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return expected_url

        def read(self) -> bytes:
            return json.dumps(state).encode("utf-8")

    def _urlopen(request: object, *, timeout: int) -> Response:
        assert timeout == 30
        assert request.full_url == expected_url
        assert request.get_header("Authorization") is None
        return Response()

    monkeypatch.setattr(trace_module.urllib_request, "urlopen", _urlopen)
    trace_module._verify_public_git_commit(commit)

    state["sha"] = "b" * 40
    with pytest.raises(InputError, match="different source identity"):
        trace_module._verify_public_git_commit(commit)
