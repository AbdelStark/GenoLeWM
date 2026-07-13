# SPDX-License-Identifier: Apache-2.0
"""Remote postflight contracts for immutable gnomAD v0.3 staging shards."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools.data._immutable_json import supports_secure_immutable_json_publication
from tools.data.v03_gnomad_lock import (
    audit_gnomad_parquet,
    main,
    select_source,
    verify_gcs_metadata,
)

SOURCE_LOCK = Path("configs/data_v03/gnomad-v4.1-exomes-autosomes.source-lock.json")
SOURCE_LOCK_SCHEMA = Path("configs/data_v03/gnomad-v4.1-exomes-autosomes.source-lock.schema.json")
SOURCE_COMMIT = "3c1b233782832b5136db312b8da1ee81b7a88109"
HUB_REVISION = "e" * 40

requires_secure_immutable_json_publication = pytest.mark.skipif(
    not supports_secure_immutable_json_publication(),
    reason="secure immutable publication requires anchored dir_fd operations",
)


class _FakeHub:
    """Test boundary implementing the small Hub surface used by postflight."""

    def __init__(self, *, root: Path, repo_files: list[str], revision: str) -> None:
        self.root = root
        self.repo_files = repo_files
        self.revision = revision
        self.api_calls: list[tuple[str, dict[str, object]]] = []
        self.download_calls: list[dict[str, object]] = []

    def module(self) -> SimpleNamespace:
        boundary = self

        class HfApi:
            def __init__(self, *, token: str | None = None) -> None:
                boundary.api_calls.append(("init", {"token": token}))

            def repo_info(self, **kwargs: object) -> object:
                boundary.api_calls.append(("repo_info", kwargs))
                return SimpleNamespace(sha=boundary.revision)

            def list_repo_files(self, **kwargs: object) -> list[str]:
                boundary.api_calls.append(("list_repo_files", kwargs))
                return boundary.repo_files

        def hf_hub_download(**kwargs: object) -> str:
            boundary.download_calls.append(kwargs)
            return str(boundary.root / str(kwargs["filename"]))

        return SimpleNamespace(HfApi=HfApi, hf_hub_download=hf_hub_download)


@requires_secure_immutable_json_publication
def test_remote_postflight_verifies_one_exact_revision_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    output = tmp_path / "postflight.json"
    hub = _FakeHub(
        root=fixture.root,
        repo_files=[".gitattributes", *fixture.repo_files],
        revision=HUB_REVISION,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        [
            "remote-postflight",
            "--repo-id",
            fixture.repo_id,
            "--revision",
            HUB_REVISION,
            "--namespace",
            fixture.namespace,
            "--expected-source-commit",
            SOURCE_COMMIT,
            "--expected-chromosome",
            "22",
            "--output-json",
            str(output),
        ]
    )

    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["repo_id"] == fixture.repo_id
    assert report["revision"] == HUB_REVISION
    assert report["namespace"] == fixture.namespace
    assert report["source_commit"] == SOURCE_COMMIT
    assert report["chromosome"] == "22"
    assert report["parquet_audit"]["scanned_row_count"] == 2
    assert report["verified_files"] == sorted(fixture.relative_files)
    exact_revision_calls = [
        kwargs for name, kwargs in hub.api_calls if name in {"repo_info", "list_repo_files"}
    ] + hub.download_calls
    assert exact_revision_calls
    assert {call["revision"] for call in exact_revision_calls} == {HUB_REVISION}
    assert {call.get("repo_type") for call in exact_revision_calls} == {"dataset"}
    assert {call.get("force_download") for call in hub.download_calls} == {True}
    cache_directories = {Path(str(call["cache_dir"])) for call in hub.download_calls}
    assert len(cache_directories) == 1
    cache_directory = cache_directories.pop()
    assert cache_directory.name == "hf-cache"
    assert {Path(str(call["local_dir"])) for call in hub.download_calls} == {cache_directory.parent}


def test_remote_postflight_reports_unsupported_publication_without_creating_output_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    hub = _FakeHub(
        root=fixture.root,
        repo_files=[".gitattributes", *fixture.repo_files],
        revision=HUB_REVISION,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    monkeypatch.setattr(
        "tools.data._immutable_json.supports_secure_immutable_json_publication",
        lambda: False,
    )
    output = tmp_path / "publication" / "postflight.json"

    result = main(_postflight_args(fixture, output=output))

    assert result == 2
    assert (
        "requires anchored dir_fd operations; this platform is unsupported"
        in capsys.readouterr().err
    )
    assert not output.parent.exists()


def test_remote_postflight_rejects_mutable_main_before_hub_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        [
            "remote-postflight",
            "--repo-id",
            fixture.repo_id,
            "--revision",
            "main",
            "--namespace",
            fixture.namespace,
            "--expected-source-commit",
            SOURCE_COMMIT,
            "--expected-chromosome",
            "22",
            "--output-json",
            str(tmp_path / "postflight.json"),
        ]
    )

    assert result == 2
    assert "revision must be a full lowercase 40-character commit SHA" in capsys.readouterr().err
    assert hub.api_calls == []
    assert hub.download_calls == []


def test_remote_postflight_rejects_a_missing_namespace_file_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    missing = f"{fixture.namespace}/evidence/receipt.json"
    hub = _FakeHub(
        root=fixture.root,
        repo_files=[path for path in fixture.repo_files if path != missing],
        revision=HUB_REVISION,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(_postflight_args(fixture, output=tmp_path / "postflight.json"))

    assert result == 2
    assert "missing=['evidence/receipt.json']" in capsys.readouterr().err
    assert hub.download_calls == []


def test_remote_postflight_rejects_an_extra_namespace_file_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    hub = _FakeHub(
        root=fixture.root,
        repo_files=[*fixture.repo_files, f"{fixture.namespace}/evidence/unbound.json"],
        revision=HUB_REVISION,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(_postflight_args(fixture, output=tmp_path / "postflight.json"))

    assert result == 2
    assert "unexpected=['evidence/unbound.json']" in capsys.readouterr().err
    assert hub.download_calls == []


def test_remote_postflight_rejects_tampered_parquet_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    parquet = fixture.root / fixture.namespace / "data/gnomad/v4.1/variants.parquet"
    parquet.write_bytes(parquet.read_bytes() + b"tampered")
    hub = _FakeHub(
        root=fixture.root,
        repo_files=fixture.repo_files,
        revision=HUB_REVISION,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(_postflight_args(fixture, output=tmp_path / "postflight.json"))

    assert result == 2
    assert "prepare report.output_parquet.sha256 drifted" in capsys.readouterr().err


@requires_secure_immutable_json_publication
def test_remote_postflight_scans_the_same_gnomad_parquet_bytes_it_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    parquet = fixture.root / fixture.namespace / "data/gnomad/v4.1/variants.parquet"
    original = parquet.read_bytes()
    replacement = parquet.with_name("replacement.parquet")
    _write_gnomad_parquet(replacement, chromosome="21")
    assert replacement.stat().st_size == len(original)
    swapped = _replace_after_first_binary_read(
        monkeypatch,
        target=parquet,
        replacement=replacement,
    )
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    output = tmp_path / "postflight.json"

    result = main(_postflight_args(fixture, output=output))

    assert swapped["done"]
    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    relative = "data/gnomad/v4.1/variants.parquet"
    assert report["file_identities"][relative] == {
        "sha256": hashlib.sha256(original).hexdigest(),
        "size_bytes": len(original),
    }
    assert report["parquet_audit"]["canonical_chromosome"] == "22"


@requires_secure_immutable_json_publication
def test_remote_postflight_reuses_the_captured_selection_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    selection = fixture.root / fixture.namespace / "evidence/selection.json"
    original = selection.read_bytes()
    replacement = original + b" "
    path_read_bytes = Path.read_bytes
    replaced = False

    def replace_after_read(path: Path) -> bytes:
        nonlocal replaced
        payload = path_read_bytes(path)
        if path == selection and not replaced:
            selection.write_bytes(replacement)
            replaced = True
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    output = tmp_path / "postflight.json"

    result = main(_postflight_args(fixture, output=output))

    assert replaced
    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["file_identities"]["evidence/selection.json"] == {
        "sha256": hashlib.sha256(original).hexdigest(),
        "size_bytes": len(original),
    }


@pytest.mark.parametrize(
    "relative_path",
    [
        "evidence/gcs-object-metadata.json",
        "evidence/gcs-metadata-verification.json",
        "evidence/source-stream-identity.json",
        "evidence/prepare-report.json",
        "evidence/receipt.json",
        "evidence/source-lock.json",
        "evidence/source-lock.schema.json",
    ],
)
@requires_secure_immutable_json_publication
def test_remote_postflight_reuses_each_captured_json_evidence_file(
    relative_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    evidence = fixture.root / fixture.namespace / relative_path
    original = evidence.read_bytes()
    path_read_bytes = Path.read_bytes
    replaced = False

    def replace_after_read(path: Path) -> bytes:
        nonlocal replaced
        payload = path_read_bytes(path)
        if path == evidence and not replaced:
            evidence.write_bytes(original + b" ")
            replaced = True
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    output = tmp_path / "postflight.json"

    result = main(_postflight_args(fixture, output=output))

    assert replaced
    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["file_identities"][relative_path] == {
        "sha256": hashlib.sha256(original).hexdigest(),
        "size_bytes": len(original),
    }


def test_remote_postflight_rejects_an_unprefixed_prepare_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    prepare_path = fixture.root / fixture.namespace / "evidence/prepare-report.json"
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    prepare["output_parquet"]["sha256"] = prepare["output_parquet"]["sha256"].removeprefix(
        "sha256:"
    )
    _write_json(prepare_path, prepare)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(_postflight_args(fixture, output=tmp_path / "postflight.json"))

    assert result == 2
    assert (
        "prepare report.output_parquet.sha256 must be a lowercase sha256:<hex> digest"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize(
    ("relative_path", "field", "value", "binding_field", "expected"),
    [
        (
            "evidence/source-stream-identity.json",
            "ok",
            1,
            "source_identity",
            "source identity.ok drifted",
        ),
        (
            "evidence/prepare-report.json",
            "already_exists",
            0,
            "prepare_report",
            "prepare report.already_exists drifted",
        ),
        ("evidence/receipt.json", "ok", 1, None, "receipt.ok drifted"),
    ],
    ids=["source-ok-int", "already-exists-int", "receipt-ok-int"],
)
def test_remote_postflight_rejects_bool_int_equality_aliases(
    relative_path: str,
    field: str,
    value: object,
    binding_field: str | None,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    namespace_root = fixture.root / fixture.namespace
    evidence_path = namespace_root / relative_path
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence[field] = value
    _write_json(evidence_path, evidence)
    if binding_field is not None:
        receipt_path = namespace_root / "evidence/receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["evidence"][binding_field] = _identity(evidence_path)
        _write_json(receipt_path, receipt)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    output = tmp_path / "postflight.json"

    result = main(_postflight_args(fixture, output=output))

    assert result == 2
    assert expected in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    ("relative_path", "needle", "duplicate_key"),
    [
        ("evidence/source-stream-identity.json", '"ok": true', "ok"),
        (
            "evidence/selection.json",
            '"schema_version": "geno-lewm.gnomad-stage-selection.v1"',
            "schema_version",
        ),
        (
            "evidence/source-lock.json",
            '"schema_version": "geno-lewm.gnomad-source-lock.v1"',
            "schema_version",
        ),
    ],
    ids=["source-identity-ok", "selection-schema-version", "source-lock-schema-version"],
)
def test_remote_postflight_rejects_duplicate_json_keys(
    relative_path: str,
    needle: str,
    duplicate_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    evidence_path = fixture.root / fixture.namespace / relative_path
    payload = evidence_path.read_text(encoding="utf-8")
    assert payload.count(needle) == 1
    evidence_path.write_text(
        payload.replace(needle, f'{needle},\n  "{duplicate_key}": null', 1),
        encoding="utf-8",
    )
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    output = tmp_path / "postflight.json"

    result = main(_postflight_args(fixture, output=output))

    assert result == 2
    assert f"duplicate JSON key {duplicate_key!r}" in capsys.readouterr().err
    assert not output.exists()


def test_remote_postflight_rejects_expected_source_commit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    args = _postflight_args(fixture, output=tmp_path / "postflight.json")
    args[args.index("--expected-source-commit") + 1] = "c" * 40

    result = main(args)

    assert result == 2
    assert "cannot read trusted source artifact" in capsys.readouterr().err


def test_remote_postflight_rejects_hub_revision_resolution_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision="f" * 40)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(_postflight_args(fixture, output=tmp_path / "postflight.json"))

    assert result == 2
    assert "Hugging Face resolved revision drifted" in capsys.readouterr().err
    assert hub.download_calls == []


def test_remote_postflight_rejects_a_misdirected_namespace_before_hub_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    args = _postflight_args(fixture, output=tmp_path / "postflight.json")
    args[args.index("--namespace") + 1] = fixture.namespace.replace("/chr22-", "/chr21-")

    result = main(args)

    assert result == 2
    assert "requested namespace drifted from the trusted source lock" in capsys.readouterr().err
    assert hub.api_calls == []
    assert hub.download_calls == []


def test_remote_postflight_rejects_expected_chromosome_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    args = _postflight_args(fixture, output=tmp_path / "postflight.json")
    args[args.index("--expected-chromosome") + 1] = "21"

    result = main(args)

    assert result == 2
    assert "requested namespace drifted from the trusted source lock" in capsys.readouterr().err


def test_remote_postflight_rejects_a_coherently_reauthored_source_lock_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    replacement_dir = tmp_path / "replacement" / "configs/data_v03"
    replacement_dir.mkdir(parents=True)
    replacement_lock = replacement_dir / SOURCE_LOCK.name
    replacement_schema = replacement_dir / SOURCE_LOCK_SCHEMA.name
    replacement_lock.write_bytes(SOURCE_LOCK.read_bytes() + b"\n")
    replacement_schema.write_bytes(SOURCE_LOCK_SCHEMA.read_bytes())
    fixture = _write_remote_fixture(
        tmp_path / "hub",
        source_lock_path=replacement_lock,
        source_lock_schema_path=replacement_schema,
    )
    trusted_selection = select_source(
        SOURCE_LOCK,
        chromosome="22",
        commit_sha=SOURCE_COMMIT,
        container_image=json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["job"][
            "container_image"
        ],
    )
    trusted_namespace = trusted_selection["publication"]["namespace"]
    assert isinstance(trusted_namespace, str)
    old_namespace = fixture.namespace
    (fixture.root / trusted_namespace).parent.mkdir(parents=True, exist_ok=True)
    (fixture.root / old_namespace).rename(fixture.root / trusted_namespace)
    fixture.namespace = trusted_namespace
    fixture.repo_files = [
        path.replace(f"{old_namespace}/", f"{trusted_namespace}/", 1) for path in fixture.repo_files
    ]
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(_postflight_args(fixture, output=tmp_path / "postflight.json"))

    assert result == 2
    assert "remote source lock bytes at source commit drifted" in capsys.readouterr().err


def test_remote_postflight_rejects_semantically_equivalent_selection_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    selection = fixture.root / fixture.namespace / "evidence/selection.json"
    selection.write_bytes(selection.read_bytes() + b"\n")
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(_postflight_args(fixture, output=tmp_path / "postflight.json"))

    assert result == 2
    assert "remote metadata verification drifted" in capsys.readouterr().err


def test_remote_postflight_rejects_receipt_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    receipt_path = fixture.root / fixture.namespace / "evidence/receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["execution"]["commit_sha"] = "c" * 40
    _write_json(receipt_path, receipt)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(_postflight_args(fixture, output=tmp_path / "postflight.json"))

    assert result == 2
    assert "receipt.execution drifted" in capsys.readouterr().err


def test_remote_postflight_reruns_the_canonical_chromosome_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    parquet = fixture.root / fixture.namespace / "data/gnomad/v4.1/variants.parquet"
    _write_gnomad_parquet(parquet, chromosome="21")
    _rebind_parquet_and_prepare_report(fixture)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(_postflight_args(fixture, output=tmp_path / "postflight.json"))

    assert result == 2
    assert "chromosome drifted: expected '22', observed '21'" in capsys.readouterr().err


def test_remote_postflight_reruns_the_exact_schema_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    parquet = fixture.root / fixture.namespace / "data/gnomad/v4.1/variants.parquet"
    _write_gnomad_parquet(parquet, global_af_float64=True)
    _rebind_parquet_and_prepare_report(fixture)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(_postflight_args(fixture, output=tmp_path / "postflight.json"))

    assert result == 2
    assert "Parquet schema drifted" in capsys.readouterr().err


def test_remote_postflight_reruns_the_population_non_null_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    parquet = fixture.root / fixture.namespace / "data/gnomad/v4.1/variants.parquet"
    _write_gnomad_parquet(parquet, population_value=None)
    _rebind_parquet_and_prepare_report(fixture)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(_postflight_args(fixture, output=tmp_path / "postflight.json"))

    assert result == 2
    assert "no values for required gnomAD v4.1 population AF columns" in capsys.readouterr().err


def test_remote_postflight_reruns_the_exact_row_count_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _write_remote_fixture(tmp_path / "hub")
    namespace_root = fixture.root / fixture.namespace
    prepare_path = namespace_root / "evidence/prepare-report.json"
    receipt_path = namespace_root / "evidence/receipt.json"
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    prepare["records_written"] = 3
    prepare["allele_records_seen"] = 5
    _write_json(prepare_path, prepare)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["transform"]["counts"]["records_written"] = 3
    receipt["transform"]["counts"]["allele_records_seen"] = 5
    receipt["evidence"]["prepare_report"] = _identity(prepare_path)
    _write_json(receipt_path, receipt)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(_postflight_args(fixture, output=tmp_path / "postflight.json"))

    assert result == 2
    assert "metadata/preparer row-count mismatch" in capsys.readouterr().err


class _RemoteFixture(SimpleNamespace):
    root: Path
    repo_id: str
    namespace: str
    relative_files: list[str]
    repo_files: list[str]


def _postflight_args(fixture: _RemoteFixture, *, output: Path) -> list[str]:
    return [
        "remote-postflight",
        "--repo-id",
        fixture.repo_id,
        "--revision",
        HUB_REVISION,
        "--namespace",
        fixture.namespace,
        "--expected-source-commit",
        SOURCE_COMMIT,
        "--expected-chromosome",
        "22",
        "--output-json",
        str(output),
    ]


def _write_remote_fixture(
    root: Path,
    *,
    source_lock_path: Path = SOURCE_LOCK,
    source_lock_schema_path: Path = SOURCE_LOCK_SCHEMA,
) -> _RemoteFixture:
    lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    image = lock["job"]["container_image"]
    selection = select_source(
        source_lock_path,
        chromosome="22",
        commit_sha=SOURCE_COMMIT,
        container_image=image,
    )
    selection["source_lock"]["path"] = SOURCE_LOCK.as_posix()
    selection["source_lock"]["schema"]["path"] = SOURCE_LOCK_SCHEMA.as_posix()
    repo_id = selection["publication"]["repo"]
    namespace = selection["publication"]["namespace"]
    assert isinstance(repo_id, str)
    assert isinstance(namespace, str)
    namespace_root = root / namespace
    evidence = namespace_root / "evidence"
    parquet = namespace_root / "data" / "gnomad" / "v4.1" / "variants.parquet"
    evidence.mkdir(parents=True)
    _write_gnomad_parquet(parquet)

    selection_path = evidence / "selection.json"
    _write_json(selection_path, selection)
    source = selection["source"]
    assert isinstance(source, dict)
    gcs_metadata = {
        "bucket": source["bucket"],
        "name": source["object"],
        "generation": source["generation"],
        "size": str(source["size_bytes"]),
        "md5Hash": source["md5_base64"],
    }
    gcs_metadata_path = evidence / "gcs-object-metadata.json"
    _write_json(gcs_metadata_path, gcs_metadata)
    metadata_verification_path = evidence / "gcs-metadata-verification.json"
    _write_json(
        metadata_verification_path,
        verify_gcs_metadata(selection_path, gcs_metadata_path),
    )

    source_path = (
        "/tmp/geno-lewm-v03-stage-gnomad/input/" + str(source["object"]).rsplit("/", 1)[-1]
    )
    source_identity = {
        "schema_version": "geno-lewm.gnomad-stream-identity.v1",
        "ok": True,
        "selection_sha256": _sha256(selection_path),
        "path": source_path,
        "size_bytes": source["size_bytes"],
        "md5_base64": source["md5_base64"],
        "md5_hex": source["md5_hex"],
        "sha256": "a" * 64,
        "hash_method": "single_pass_chunked_file_read",
        "chunk_size_bytes": 1 << 20,
    }
    source_identity_path = evidence / "source-stream-identity.json"
    _write_json(source_identity_path, source_identity)

    dataset_root = "/tmp/geno-lewm-v03-stage-gnomad/publish/data"
    output_path = f"{dataset_root}/gnomad/v4.1/variants.parquet"
    transform = selection["transform"]
    assert isinstance(transform, dict)
    report_argv = [
        transform["command"],
        "--input-vcf",
        source_path,
        "--output",
        dataset_root,
        "--release",
        selection["release"],
        "--min-af",
        str(transform["min_af"]),
        "--max-allele-len",
        str(transform["max_allele_len"]),
    ]
    prepare_report = {
        "output_path": output_path,
        "input_path": source_path,
        "command": shlex.join(report_argv),
        "release": selection["release"],
        "input_sha256": f"sha256:{source_identity['sha256']}",
        "output_sha256": f"sha256:{_sha256(parquet)}",
        "input_size_bytes": source["size_bytes"],
        "size_bytes": parquet.stat().st_size,
        "elapsed_seconds": 3.0,
        "input_vcf": {
            "path": source_path,
            "sha256": f"sha256:{source_identity['sha256']}",
            "size_bytes": source["size_bytes"],
        },
        "output_parquet": {
            "path": output_path,
            "sha256": f"sha256:{_sha256(parquet)}",
            "size_bytes": parquet.stat().st_size,
        },
        "records_read": 4,
        "allele_records_seen": 4,
        "records_written": 2,
        "skipped_filter": 1,
        "skipped_af": 1,
        "skipped_allele": 0,
        "already_exists": False,
        "runtime": {
            "elapsed_seconds": 3.5,
            "process_peak_rss_bytes": 123456,
            "peak_memory_note": "fixture ru_maxrss",
        },
    }
    prepare_report_path = evidence / "prepare-report.json"
    _write_json(prepare_report_path, prepare_report)

    parquet_audit = audit_gnomad_parquet(
        parquet,
        chromosome="22",
        expected_records=2,
        min_af=float(transform["min_af"]),
        max_allele_len=int(transform["max_allele_len"]),
    )
    receipt = {
        "schema_version": "geno-lewm.gnomad-staging-receipt.v1",
        "created_at": "2026-07-13T00:00:00Z",
        "ok": True,
        "dataset_id": selection["dataset_id"],
        "release": selection["release"],
        "reference_genome": selection["reference_genome"],
        "source_lock": selection["source_lock"],
        "source": {
            "chromosome": source["chromosome"],
            "split_role": source["split_role"],
            "bucket": source["bucket"],
            "object": source["object"],
            "generation": source["generation"],
            "size_bytes": source["size_bytes"],
            "upstream_md5_base64": source["md5_base64"],
            "upstream_md5_hex": source["md5_hex"],
            "streamed_sha256": source_identity["sha256"],
        },
        "transform": {
            "command": transform["command"],
            "argv": ["uv", "run", report_argv[0], "--quiet", "--no-banner", *report_argv[1:]],
            "filters": {
                "filter": transform["filter"],
                "min_af": transform["min_af"],
                "max_allele_len": transform["max_allele_len"],
            },
            "runtime": prepare_report["runtime"],
            "counts": {
                key: prepare_report[key]
                for key in (
                    "records_read",
                    "allele_records_seen",
                    "records_written",
                    "skipped_filter",
                    "skipped_af",
                    "skipped_allele",
                )
            },
        },
        "output": {
            "path": output_path,
            "sha256": _sha256(parquet),
            "size_bytes": parquet.stat().st_size,
            "parquet_audit": parquet_audit,
        },
        "execution": selection["execution"],
        "publication": selection["publication"],
        "evidence": {
            "selection": _identity(selection_path),
            "metadata_verification": _identity(metadata_verification_path),
            "source_identity": _identity(source_identity_path),
            "prepare_report": _identity(prepare_report_path),
        },
        "claim_boundary": selection["claim_boundary"],
    }
    _write_json(evidence / "receipt.json", receipt)
    (evidence / "source-lock.json").write_bytes(source_lock_path.read_bytes())
    (evidence / "source-lock.schema.json").write_bytes(source_lock_schema_path.read_bytes())

    relative_files = [
        "data/gnomad/v4.1/variants.parquet",
        "evidence/gcs-metadata-verification.json",
        "evidence/gcs-object-metadata.json",
        "evidence/prepare-report.json",
        "evidence/receipt.json",
        "evidence/selection.json",
        "evidence/source-lock.json",
        "evidence/source-lock.schema.json",
        "evidence/source-stream-identity.json",
    ]
    return _RemoteFixture(
        root=root,
        repo_id=repo_id,
        namespace=namespace,
        relative_files=relative_files,
        repo_files=[f"{namespace}/{relative}" for relative in relative_files],
    )


def _write_gnomad_parquet(
    path: Path,
    *,
    chromosome: str = "22",
    population_value: float | None = 0.1,
    global_af_float64: bool = False,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    schema = pa.schema(
        [
            ("chrom", pa.string()),
            ("pos", pa.int64()),
            ("ref", pa.string()),
            ("alt", pa.string()),
            ("af_global", pa.float64() if global_af_float64 else pa.float32()),
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
    population = dict.fromkeys(
        (
            "af_afr",
            "af_amr",
            "af_asj",
            "af_eas",
            "af_fin",
            "af_mid",
            "af_nfe",
            "af_remaining",
            "af_sas",
        ),
        population_value,
    )
    rows = [
        {
            **population,
            "chrom": chromosome,
            "pos": 101,
            "ref": "A",
            "alt": "C",
            "af_global": 0.02,
            "filter": "PASS",
            "schema_version": "2.0.0",
        },
        {
            **population,
            "chrom": chromosome,
            "pos": 202,
            "ref": "G",
            "alt": "T",
            "af_global": 0.2,
            "filter": "PASS",
            "schema_version": "2.0.0",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _rebind_parquet_and_prepare_report(fixture: _RemoteFixture) -> None:
    namespace_root = fixture.root / fixture.namespace
    parquet = namespace_root / "data/gnomad/v4.1/variants.parquet"
    prepare_path = namespace_root / "evidence/prepare-report.json"
    receipt_path = namespace_root / "evidence/receipt.json"
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    prepare["output_parquet"]["sha256"] = f"sha256:{_sha256(parquet)}"
    prepare["output_parquet"]["size_bytes"] = parquet.stat().st_size
    prepare["output_sha256"] = f"sha256:{_sha256(parquet)}"
    prepare["size_bytes"] = parquet.stat().st_size
    _write_json(prepare_path, prepare)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output"]["sha256"] = _sha256(parquet)
    receipt["output"]["size_bytes"] = parquet.stat().st_size
    receipt["evidence"]["prepare_report"] = _identity(prepare_path)
    _write_json(receipt_path, receipt)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": f"/tmp/geno-lewm-v03-stage-gnomad/publish/evidence/{path.name}",
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _replace_after_first_binary_read(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target: Path,
    replacement: Path,
) -> dict[str, bool]:
    path_open = Path.open
    os_open = os.open
    state = {"done": False}

    class _ReplaceOnClose:
        def __init__(self, stream: Any) -> None:
            self._stream = stream

        def __enter__(self) -> Any:
            self._stream.__enter__()
            return self._stream

        def __exit__(self, *args: object) -> object:
            result = self._stream.__exit__(*args)
            replacement.replace(target)
            state["done"] = True
            return result

    def open_and_replace(path: Path, *args: object, **kwargs: object) -> Any:
        stream = path_open(path, *args, **kwargs)
        mode = str(args[0]) if args else str(kwargs.get("mode", "r"))
        if path == target and mode == "rb" and not state["done"]:
            return _ReplaceOnClose(stream)
        return stream

    def open_fd_and_replace(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = os_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == target and not state["done"]:
            replacement.replace(target)
            state["done"] = True
        return descriptor

    monkeypatch.setattr(Path, "open", open_and_replace)
    monkeypatch.setattr(os, "open", open_fd_and_replace)
    return state
