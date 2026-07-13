# SPDX-License-Identifier: Apache-2.0
"""Remote postflight contracts for the corrected ClinVar staging shard."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from tools.data._immutable_json import supports_secure_immutable_json_publication
from tools.data.v03_clinvar_postflight import main

HUB_REVISION = "e" * 40
RELEASE = "2026-04-15"
REPORT_SCHEMA = (
    Path(__file__).resolve().parents[2] / "configs/data_v03/clinvar-remote-postflight.schema.json"
)

requires_secure_immutable_json_publication = pytest.mark.skipif(
    not supports_secure_immutable_json_publication(),
    reason="secure immutable publication requires anchored dir_fd operations",
)


class _FakeHub:
    """Test boundary implementing the exact Hub calls used by postflight."""

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
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    output = tmp_path / "postflight.json"
    hub = _FakeHub(
        root=fixture.root,
        repo_files=[".gitattributes", *fixture.repo_files],
        revision=HUB_REVISION,
    )
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(_postflight_args(fixture, source_commit=source_commit, output=output))

    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "geno-lewm.clinvar-remote-postflight.v1"
    assert report["ok"] is True
    assert report["repo_id"] == fixture.repo_id
    assert report["revision"] == HUB_REVISION
    assert report["namespace"] == fixture.namespace
    assert report["source_commit"] == source_commit
    assert report["release"] == RELEASE
    assert report["parquet_audit"]["scanned_row_count"] == 3
    assert report["parquet_audit"]["class_balance"] == {"B": 1, "P": 1, "VUS": 1}
    assert report["parquet_audit"]["chromosome_balance"] == {"1": 2, "X": 1}
    assert report["verified_files"] == sorted(fixture.relative_files)
    assert report["source_identity"]["verification_scope"] == [
        "release_reconciled",
        "sha256_reconciled",
        "size_bytes_reconciled",
    ]
    report_schema = _read_json(REPORT_SCHEMA)
    assert set(report) == set(report_schema["required"]) == set(report_schema["properties"])
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
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    hub = _FakeHub(
        root=fixture.root,
        repo_files=[".gitattributes", *fixture.repo_files],
        revision=HUB_REVISION,
    )
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    monkeypatch.setattr(
        "tools.data._immutable_json.supports_secure_immutable_json_publication",
        lambda: False,
    )
    output = tmp_path / "publication" / "postflight.json"

    result = main(_postflight_args(fixture, source_commit=source_commit, output=output))

    assert result == 2
    assert (
        "requires anchored dir_fd operations; this platform is unsupported"
        in capsys.readouterr().err
    )
    assert not output.parent.exists()


def test_remote_postflight_rejects_mutable_revision_before_hub_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    args = _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    args[args.index("--revision") + 1] = "main"

    result = main(args)

    assert result == 2
    assert "revision must be a full lowercase 40-character commit SHA" in capsys.readouterr().err
    assert hub.api_calls == []
    assert hub.download_calls == []


def test_remote_postflight_rejects_misdirected_namespace_before_hub_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    args = _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    args[args.index("--namespace") + 1] = fixture.namespace + "-copy"

    result = main(args)

    assert result == 2
    assert "requested ClinVar namespace drifted" in capsys.readouterr().err
    assert hub.api_calls == []
    assert hub.download_calls == []


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("--repo-id", "not-a-repo-id", "repo_id must be a namespace/name pair"),
        (
            "--expected-source-commit",
            "short",
            "source commit must be a full lowercase 40-character Git SHA",
        ),
        ("--expected-release", "not-a-date", "release must be an ISO YYYY-MM-DD date"),
        ("--expected-release", "20260415", "release must be a canonical ISO YYYY-MM-DD date"),
        ("--namespace", "../unsafe", "namespace contains an unsafe path component"),
    ],
)
def test_remote_postflight_rejects_invalid_request_fields_before_hub_access(
    field: str,
    value: str,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    args = _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    args[args.index(field) + 1] = value

    result = main(args)

    assert result == 2
    assert expected in capsys.readouterr().err
    assert hub.api_calls == []
    assert hub.download_calls == []


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("missing", "missing=['evidence/runtime_report.json']"),
        ("extra", "unexpected=['evidence/unbound.json']"),
    ],
)
def test_remote_postflight_rejects_namespace_file_set_drift_before_download(
    mode: str,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    if mode == "missing":
        changed_files = [
            path
            for path in fixture.repo_files
            if not path.endswith("/evidence/runtime_report.json")
        ]
    else:
        changed_files = [*fixture.repo_files, f"{fixture.namespace}/evidence/unbound.json"]
    hub = _FakeHub(root=fixture.root, repo_files=changed_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert expected in capsys.readouterr().err
    assert hub.download_calls == []


def test_remote_postflight_rejects_hub_revision_resolution_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision="f" * 40)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "Hugging Face resolved revision drifted" in capsys.readouterr().err
    assert hub.download_calls == []


def test_remote_postflight_rejects_unavailable_source_commit_before_hub_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    missing_commit = "f" * 40
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    args = _postflight_args(fixture, source_commit=missing_commit, output=tmp_path / "out.json")
    args[args.index("--namespace") + 1] = (
        f"staging/clinvar-{RELEASE}-archive-{missing_commit[:12]}-r1"
    )

    result = main(args)

    assert result == 2
    assert "cannot read trusted source artifact" in capsys.readouterr().err
    assert hub.api_calls == []
    assert hub.download_calls == []


def test_remote_postflight_rejects_drift_in_exact_source_contract_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_source_repository(tmp_path / "source")
    cli_path = tmp_path / "source/geno_lewm/cli/prepare_clinvar.py"
    cli_path.write_text(
        cli_path.read_text(encoding="utf-8").replace("max_allele_len=16", "max_allele_len=0"),
        encoding="utf-8",
    )
    source_commit = _commit_source_repository(tmp_path / "source", "drift CLI default")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "trusted max_allele_len default must be positive" in capsys.readouterr().err
    assert hub.api_calls == []
    assert hub.download_calls == []


def test_remote_postflight_rejects_audit_source_commit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    audit_path = _fixture_path(fixture, "evidence/audit.json")
    audit = _read_json(audit_path)
    audit["commit_sha"] = "f" * 40
    _write_json(audit_path, audit)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "audit.commit_sha drifted" in capsys.readouterr().err


def test_remote_postflight_rejects_an_unbound_audit_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    audit_path = _fixture_path(fixture, "evidence/audit.json")
    audit = _read_json(audit_path)
    audit["unbound"] = True
    _write_json(audit_path, audit)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    error = capsys.readouterr().err
    assert "audit keys drifted" in error
    assert "unexpected=['unbound']" in error


@pytest.mark.parametrize(
    ("relative_path", "path", "value", "expected"),
    [
        ("evidence/audit.json", ("ok",), 1, "audit.ok drifted"),
        (
            "evidence/runtime_report.json",
            ("returncode",),
            False,
            "audit.runtime.returncode drifted",
        ),
        (
            "evidence/prepare_report.json",
            ("already_exists",),
            0,
            "audit.prepare_report drifted",
        ),
    ],
    ids=["audit-ok-int", "runtime-returncode-bool", "already-exists-int"],
)
def test_remote_postflight_rejects_bool_int_equality_aliases(
    relative_path: str,
    path: tuple[str, ...],
    value: object,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    evidence_path = _fixture_path(fixture, relative_path)
    evidence = _read_json(evidence_path)
    target = evidence
    for component in path[:-1]:
        nested = target[component]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value
    _write_json(evidence_path, evidence)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    output = tmp_path / "out.json"

    result = main(_postflight_args(fixture, source_commit=source_commit, output=output))

    assert result == 2
    assert expected in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    ("relative_path", "needle", "duplicate_key"),
    [
        ("evidence/audit.json", '"ok": true', "ok"),
        ("evidence/runtime_report.json", '"returncode": 0', "returncode"),
    ],
    ids=["audit-ok", "runtime-returncode"],
)
def test_remote_postflight_rejects_duplicate_json_keys(
    relative_path: str,
    needle: str,
    duplicate_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    evidence_path = _fixture_path(fixture, relative_path)
    payload = evidence_path.read_text(encoding="utf-8")
    assert payload.count(needle) == 1
    evidence_path.write_text(
        payload.replace(needle, f'{needle},\n  "{duplicate_key}": null', 1),
        encoding="utf-8",
    )
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    output = tmp_path / "out.json"

    result = main(_postflight_args(fixture, source_commit=source_commit, output=output))

    assert result == 2
    assert f"duplicate JSON key {duplicate_key!r}" in capsys.readouterr().err
    assert not output.exists()


def test_remote_postflight_rejects_weakened_claim_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    audit_path = _fixture_path(fixture, "evidence/audit.json")
    audit = _read_json(audit_path)
    audit["claim_boundary"] = "This receipt establishes model quality."
    _write_json(audit_path, audit)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "claim_boundary lost a required scientific limitation" in capsys.readouterr().err


def test_remote_postflight_rejects_prepare_receipt_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    prepare_path = _fixture_path(fixture, "evidence/prepare_report.json")
    prepare = _read_json(prepare_path)
    prepare["records_read"] = 4
    _write_json(prepare_path, prepare, compact=True)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "audit.prepare_report drifted" in capsys.readouterr().err


def test_remote_postflight_rejects_runtime_receipt_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    runtime_path = _fixture_path(fixture, "evidence/runtime_report.json")
    runtime = _read_json(runtime_path)
    runtime["returncode"] = 1
    _write_json(runtime_path, runtime)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "audit.runtime.returncode drifted" in capsys.readouterr().err


def test_remote_postflight_rejects_source_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    audit_path = _fixture_path(fixture, "evidence/audit.json")
    audit = _read_json(audit_path)
    audit["source"]["sha256"] = f"sha256:{'d' * 64}"
    _write_json(audit_path, audit)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "prepare report.input_vcf drifted" in capsys.readouterr().err


def test_remote_postflight_rejects_invalid_source_md5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    audit_path = _fixture_path(fixture, "evidence/audit.json")
    audit = _read_json(audit_path)
    audit["source"]["md5"] = "not-a-digest"
    _write_json(audit_path, audit)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "source.md5 must be a lowercase MD5" in capsys.readouterr().err


def test_remote_postflight_rejects_coherently_reauthored_prepare_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    prepare_path = _fixture_path(fixture, "evidence/prepare_report.json")
    audit_path = _fixture_path(fixture, "evidence/audit.json")
    prepare = _read_json(prepare_path)
    prepare["command"] = str(prepare["command"]) + " --overwrite"
    _write_json(prepare_path, prepare, compact=True)
    audit = _read_json(audit_path)
    audit["prepare_report"] = prepare
    _write_json(audit_path, audit)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "prepare report.command drifted" in capsys.readouterr().err


def test_remote_postflight_rejects_coherently_reauthored_runtime_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    runtime_path = _fixture_path(fixture, "evidence/runtime_report.json")
    audit_path = _fixture_path(fixture, "evidence/audit.json")
    runtime = _read_json(runtime_path)
    runtime["command"][0] = "/repo/.venv/bin/not-the-clinvar-command"
    _write_json(runtime_path, runtime)
    audit = _read_json(audit_path)
    audit["runtime"]["command"] = runtime["command"]
    _write_json(audit_path, audit)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "runtime command executable drifted" in capsys.readouterr().err


def test_remote_postflight_rejects_unknown_declared_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    audit_path = _fixture_path(fixture, "evidence/audit.json")
    audit = _read_json(audit_path)
    audit["output"]["class_balance"] = {"NOT_A_CLINVAR_CLASS": 3}
    _write_json(audit_path, audit)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "class_balance contains unknown" in capsys.readouterr().err


def test_remote_postflight_rejects_declared_class_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    audit_path = _fixture_path(fixture, "evidence/audit.json")
    audit = _read_json(audit_path)
    audit["output"]["class_balance"] = {"B": 1, "P": 1}
    _write_json(audit_path, audit)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "class balance does not sum to records" in capsys.readouterr().err


def test_remote_postflight_rejects_tampered_parquet_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    parquet = _fixture_path(fixture, f"clinvar/{RELEASE}/variants.parquet")
    parquet.write_bytes(parquet.read_bytes() + b"tampered")
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "Parquet SHA-256 drifted" in capsys.readouterr().err


@requires_secure_immutable_json_publication
def test_remote_postflight_binds_audit_identity_to_validated_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    audit_path = _fixture_path(fixture, "evidence/audit.json")
    original = audit_path.read_bytes()
    replacement = original + b" "
    replaced = False
    read_bytes = Path.read_bytes

    def replace_after_read(path: Path) -> bytes:
        nonlocal replaced
        payload = read_bytes(path)
        if path == audit_path and not replaced:
            audit_path.write_bytes(replacement)
            replaced = True
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    output = tmp_path / "out.json"

    result = main(_postflight_args(fixture, source_commit=source_commit, output=output))

    assert replaced
    assert result == 0
    report = _read_json(output)
    assert report["file_identities"]["evidence/audit.json"] == {
        "sha256": hashlib.sha256(original).hexdigest(),
        "size_bytes": len(original),
    }


@requires_secure_immutable_json_publication
def test_remote_postflight_scans_the_same_parquet_bytes_it_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    parquet = _fixture_path(fixture, f"clinvar/{RELEASE}/variants.parquet")
    original = parquet.read_bytes()
    replacement = parquet.with_name("replacement.parquet")
    rows = pq.read_table(parquet).to_pylist()
    rows[0]["clinical_significance"] = "LP"
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(parquet)), replacement)
    swapped = _replace_after_first_binary_read(
        monkeypatch,
        target=parquet,
        replacement=replacement,
    )
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    output = tmp_path / "out.json"

    result = main(_postflight_args(fixture, source_commit=source_commit, output=output))

    assert swapped["done"]
    assert result == 0
    report = _read_json(output)
    relative = f"clinvar/{RELEASE}/variants.parquet"
    assert report["file_identities"][relative] == {
        "sha256": hashlib.sha256(original).hexdigest(),
        "size_bytes": len(original),
    }
    assert report["parquet_audit"]["class_balance"] == {"B": 1, "P": 1, "VUS": 1}


def test_remote_postflight_rejects_coherently_rebound_schema_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    _rewrite_parquet(fixture, position_as_string=True)
    _rebind_output_identities(fixture)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "ClinVar Parquet schema drifted" in capsys.readouterr().err


def test_remote_postflight_rejects_coherently_rebound_schema_metadata_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    _rewrite_parquet(fixture, schema_metadata={b"unexpected": b"metadata"})
    _rebind_output_identities(fixture)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "ClinVar Parquet schema drifted" in capsys.readouterr().err


def test_remote_postflight_rejects_coherently_rebound_field_metadata_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    _rewrite_parquet(fixture, first_field_metadata={b"unexpected": b"metadata"})
    _rebind_output_identities(fixture)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "ClinVar Parquet schema drifted" in capsys.readouterr().err


def test_remote_postflight_rejects_coherently_rebound_class_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    _rewrite_parquet(fixture, first_class="LP")
    _rebind_output_identities(fixture)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "Parquet class balance drifted" in capsys.readouterr().err


def test_remote_postflight_rejects_empty_chromosome_after_identity_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    _rewrite_parquet(fixture, first_chromosome="")
    _rebind_output_identities(fixture)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "chromosome values must be non-empty" in capsys.readouterr().err


def test_remote_postflight_rejects_invalid_allele_after_identity_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    _rewrite_parquet(fixture, first_ref="N")
    _rebind_output_identities(fixture)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "ref violates the trusted allele contract" in capsys.readouterr().err


def test_remote_postflight_rejects_no_op_allele_after_identity_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    _rewrite_parquet(fixture, first_alt="A")
    _rebind_output_identities(fixture)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "ClinVar Parquet ref and alt must differ" in capsys.readouterr().err


def test_remote_postflight_rejects_required_null_after_identity_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    _rewrite_parquet(fixture, first_review_status=None)
    _rebind_output_identities(fixture)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())

    result = main(
        _postflight_args(fixture, source_commit=source_commit, output=tmp_path / "out.json")
    )

    assert result == 2
    assert "required column 'review_status' has nulls" in capsys.readouterr().err


@requires_secure_immutable_json_publication
def test_remote_postflight_report_is_byte_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    assert main(_postflight_args(fixture, source_commit=source_commit, output=first)) == 0
    assert main(_postflight_args(fixture, source_commit=source_commit, output=second)) == 0

    assert first.read_bytes() == second.read_bytes()


def test_remote_postflight_report_schema_closes_every_object() -> None:
    schema = _read_json(REPORT_SCHEMA)

    for field, value in _walk_json(schema):
        if isinstance(value, dict) and value.get("type") == "object":
            assert value.get("additionalProperties") is False, field


@requires_secure_immutable_json_publication
def test_remote_postflight_schema_validates_real_report_and_exact_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_commit = _write_source_repository(tmp_path / "source")
    fixture = _write_remote_fixture(tmp_path / "hub", source_commit=source_commit)
    hub = _FakeHub(root=fixture.root, repo_files=fixture.repo_files, revision=HUB_REVISION)
    monkeypatch.chdir(tmp_path / "source")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    output = tmp_path / "postflight.json"
    assert main(_postflight_args(fixture, source_commit=source_commit, output=output)) == 0

    schema = _read_json(REPORT_SCHEMA)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    report = _read_json(output)
    assert not list(validator.iter_errors(report))
    for path in (
        ("checks",),
        ("source_identity", "verification_scope"),
        ("trusted_source_contract", "file_identity_fields"),
        ("trusted_source_contract", "nullable_fields"),
        ("trusted_source_contract", "prepare_report_enrichments"),
        ("verified_files",),
    ):
        invalid = copy.deepcopy(report)
        target: Any = invalid
        for component in path[:-1]:
            target = target[component]
        target[path[-1]] = []
        assert list(validator.iter_errors(invalid)), path

    for field, value in _walk_json(schema):
        if isinstance(value, dict) and "prefixItems" in value:
            expected = len(value["prefixItems"])
            assert value.get("minItems") == expected, field
            assert value.get("maxItems") == expected, field


class _RemoteFixture(SimpleNamespace):
    root: Path
    repo_id: str
    namespace: str
    relative_files: list[str]
    repo_files: list[str]


def _postflight_args(fixture: _RemoteFixture, *, source_commit: str, output: Path) -> list[str]:
    return [
        "--repo-id",
        fixture.repo_id,
        "--revision",
        HUB_REVISION,
        "--namespace",
        fixture.namespace,
        "--expected-source-commit",
        source_commit,
        "--expected-release",
        RELEASE,
        "--output-json",
        str(output),
    ]


def _write_source_repository(root: Path) -> str:
    (root / "geno_lewm/data").mkdir(parents=True)
    (root / "geno_lewm/cli").mkdir(parents=True)
    (root / "geno_lewm/data/clinvar.py").write_text(
        """# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from pathlib import Path

CLINVAR_SCHEMA_VERSION = "1.0.0"
CLINVAR_LABELLED_CLASSES = frozenset({"P", "LP", "B", "LB"})

@dataclass
class ClinvarVariant:
    chrom: str
    pos: int
    ref: str
    alt: str
    clinical_significance: str
    review_status: str
    gene_symbol: str | None
    clinvar_id: int
    schema_version: str = CLINVAR_SCHEMA_VERSION

def prepare_clinvar_shard(input_vcf, output_dir, *, release, max_allele_len=16):
    target = Path(output_dir) / "clinvar" / release / "variants.parquet"
    return target

def _clinical_significance(value):
    if value == "vus":
        return "VUS"
    if value == "lp":
        return "LP"
    if value == "p":
        return "P"
    if value == "lb":
        return "LB"
    if value == "b":
        return "B"
    return "OTHER"

def _parquet_schema(pa):
    return pa.schema([
        ("chrom", pa.string()),
        ("pos", pa.int64()),
        ("ref", pa.string()),
        ("alt", pa.string()),
        ("clinical_significance", pa.string()),
        ("review_status", pa.string()),
        ("gene_symbol", pa.string()),
        ("clinvar_id", pa.int64()),
        ("schema_version", pa.string()),
    ])
""",
        encoding="utf-8",
    )
    (root / "geno_lewm/data/_vcf.py").write_text(
        """# SPDX-License-Identifier: Apache-2.0
_ACGT = frozenset({"A", "C", "G", "T"})

def is_supported_allele(value: str, *, max_len: int) -> bool:
    return bool(value) and len(value) <= max_len and set(value) <= _ACGT
""",
        encoding="utf-8",
    )
    (root / "geno_lewm/cli/prepare_clinvar.py").write_text(
        """# SPDX-License-Identifier: Apache-2.0
import typer

app = typer.Typer(name="geno-lewm-prepare-clinvar")

def main(input_vcf=None, output=None, release=None, max_allele_len=16, overwrite=False):
    pass
""",
        encoding="utf-8",
    )
    (root / "geno_lewm/cli/_prepare_report.py").write_text(
        """# SPDX-License-Identifier: Apache-2.0
def augment_prepare_report(payload):
    enriched = dict(payload)
    enriched["command"] = "command"
    enriched["input_vcf"] = _file_identity(None)
    enriched["output_parquet"] = _file_identity(None)
    enriched["runtime"] = {}
    return enriched

def _file_identity(path):
    return {"path": str(path), "sha256": "digest", "size_bytes": 1}
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.test"], cwd=root, check=True)
    return _commit_source_repository(root, "fixture source contract")


def _commit_source_repository(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", message], cwd=root, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_remote_fixture(root: Path, *, source_commit: str) -> _RemoteFixture:
    import pyarrow as pa
    import pyarrow.parquet as pq

    repo_id = "test/geno-lewm-data"
    namespace = f"staging/clinvar-{RELEASE}-archive-{source_commit[:12]}-r1"
    namespace_root = root / namespace
    parquet = namespace_root / f"clinvar/{RELEASE}/variants.parquet"
    parquet.parent.mkdir(parents=True)
    schema = pa.schema(
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
    rows = [
        {
            "chrom": "1",
            "pos": 101,
            "ref": "A",
            "alt": "C",
            "clinical_significance": "P",
            "review_status": "criteria_provided",
            "gene_symbol": "GENE1",
            "clinvar_id": 1,
            "schema_version": "1.0.0",
        },
        {
            "chrom": "1",
            "pos": 202,
            "ref": "G",
            "alt": "T",
            "clinical_significance": "VUS",
            "review_status": "criteria_provided",
            "gene_symbol": None,
            "clinvar_id": 2,
            "schema_version": "1.0.0",
        },
        {
            "chrom": "X",
            "pos": 303,
            "ref": "C",
            "alt": "A",
            "clinical_significance": "B",
            "review_status": "reviewed_by_expert_panel",
            "gene_symbol": "GENE2",
            "clinvar_id": 3,
            "schema_version": "1.0.0",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), parquet)
    output_sha256 = _sha256(parquet)
    input_path = "/tmp/clinvar-corrected/clinvar_20260415.vcf.gz"
    output_root = "/tmp/clinvar-corrected/publish"
    output_path = f"{output_root}/clinvar/{RELEASE}/variants.parquet"
    source_sha256 = "a" * 64
    command = (
        "geno-lewm-prepare-clinvar --input-vcf "
        f"{input_path} --output {output_root} --release {RELEASE} --max-allele-len 16"
    )
    prepare_report = {
        "allele_records_seen": 3,
        "already_exists": False,
        "command": command,
        "elapsed_seconds": 2.0,
        "input_path": input_path,
        "input_sha256": f"sha256:{source_sha256}",
        "input_size_bytes": 123,
        "input_vcf": {
            "path": input_path,
            "sha256": f"sha256:{source_sha256}",
            "size_bytes": 123,
        },
        "output_parquet": {
            "path": output_path,
            "sha256": f"sha256:{output_sha256}",
            "size_bytes": parquet.stat().st_size,
        },
        "output_path": output_path,
        "output_sha256": f"sha256:{output_sha256}",
        "records_read": 3,
        "records_written": 3,
        "release": RELEASE,
        "runtime": {
            "elapsed_seconds": 2.1,
            "peak_memory_note": "fixture process peak",
            "process_peak_rss_bytes": 1024,
        },
        "size_bytes": parquet.stat().st_size,
        "skipped_allele": 0,
    }
    runtime_report = {
        "command": [
            "/repo/.venv/bin/geno-lewm-prepare-clinvar",
            "--no-banner",
            *command.split()[1:],
        ],
        "peak_rss_bytes": 2048,
        "peak_rss_source": "resource.getrusage(RUSAGE_CHILDREN).ru_maxrss on Linux",
        "returncode": 0,
        "wall_time_seconds": 2.5,
    }
    audit = {
        "claim_boundary": (
            "This receipt covers normalization of the pinned ClinVar GRCh38 archive. "
            "It does not define a leakage-safe eval split or establish label correctness, "
            "representativeness, clinical utility, or model quality."
        ),
        "commit_sha": source_commit,
        "container_image": f"ghcr.io/example/image@sha256:{'c' * 64}",
        "generated_at": "2026-07-13T08:36:11.233097Z",
        "generated_by": "hf-job:clinvar-corrected-shard-audit",
        "ok": True,
        "output": {
            "class_balance": {"B": 1, "P": 1, "VUS": 1},
            "path": f"clinvar/{RELEASE}/variants.parquet",
            "records": 3,
            "sha256": f"sha256:{output_sha256}",
            "size_bytes": parquet.stat().st_size,
        },
        "prepare_report": prepare_report,
        "runtime": {
            "command": runtime_report["command"],
            "cpu_count": 8,
            "flavor": "cpu-upgrade",
            "peak_rss_bytes": runtime_report["peak_rss_bytes"],
            "peak_rss_source": runtime_report["peak_rss_source"],
            "platform": "Linux-fixture",
            "python": "3.13.11",
            "ram_gb": 32,
            "returncode": runtime_report["returncode"],
            "wall_time_seconds": runtime_report["wall_time_seconds"],
        },
        "schema_version": "1.0.0",
        "source": {
            "md5": "b" * 32,
            "release": RELEASE,
            "sha256": f"sha256:{source_sha256}",
            "size_bytes": 123,
            "url": (
                "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/"
                "archive_2.0/2026/clinvar_20260415.vcf.gz"
            ),
        },
    }
    evidence = namespace_root / "evidence"
    evidence.mkdir()
    _write_json(evidence / "audit.json", audit)
    _write_json(evidence / "prepare_report.json", prepare_report, compact=True)
    _write_json(evidence / "runtime_report.json", runtime_report)
    relative_files = [
        f"clinvar/{RELEASE}/variants.parquet",
        "evidence/audit.json",
        "evidence/prepare_report.json",
        "evidence/runtime_report.json",
    ]
    return _RemoteFixture(
        root=root,
        repo_id=repo_id,
        namespace=namespace,
        relative_files=relative_files,
        repo_files=[f"{namespace}/{relative}" for relative in relative_files],
    )


def _write_json(path: Path, value: object, *, compact: bool = False) -> None:
    if compact:
        payload = json.dumps(value, sort_keys=True) + "\n"
    else:
        payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_path(fixture: _RemoteFixture, relative_path: str) -> Path:
    return fixture.root / fixture.namespace / relative_path


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rewrite_parquet(
    fixture: _RemoteFixture,
    *,
    position_as_string: bool = False,
    first_class: str = "P",
    first_chromosome: str = "1",
    first_ref: str = "A",
    first_alt: str | None = None,
    first_review_status: str | None = "criteria_provided",
    schema_metadata: dict[bytes, bytes] | None = None,
    first_field_metadata: dict[bytes, bytes] | None = None,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = _fixture_path(fixture, f"clinvar/{RELEASE}/variants.parquet")
    rows = pq.read_table(parquet).to_pylist()
    rows[0]["clinical_significance"] = first_class
    rows[0]["chrom"] = first_chromosome
    rows[0]["ref"] = first_ref
    if first_alt is not None:
        rows[0]["alt"] = first_alt
    rows[0]["review_status"] = first_review_status
    if position_as_string:
        for row in rows:
            row["pos"] = str(row["pos"])
    schema = pa.schema(
        [
            ("chrom", pa.string()),
            ("pos", pa.string() if position_as_string else pa.int64()),
            ("ref", pa.string()),
            ("alt", pa.string()),
            ("clinical_significance", pa.string()),
            ("review_status", pa.string()),
            ("gene_symbol", pa.string()),
            ("clinvar_id", pa.int64()),
            ("schema_version", pa.string()),
        ]
    )
    if schema_metadata is not None:
        schema = schema.with_metadata(schema_metadata)
    if first_field_metadata is not None:
        schema = schema.set(0, schema.field(0).with_metadata(first_field_metadata))
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), parquet)


def _rebind_output_identities(fixture: _RemoteFixture) -> None:
    parquet = _fixture_path(fixture, f"clinvar/{RELEASE}/variants.parquet")
    prepare_path = _fixture_path(fixture, "evidence/prepare_report.json")
    audit_path = _fixture_path(fixture, "evidence/audit.json")
    prepare = _read_json(prepare_path)
    digest = f"sha256:{_sha256(parquet)}"
    size_bytes = parquet.stat().st_size
    prepare["output_sha256"] = digest
    prepare["size_bytes"] = size_bytes
    prepare["output_parquet"]["sha256"] = digest
    prepare["output_parquet"]["size_bytes"] = size_bytes
    _write_json(prepare_path, prepare, compact=True)
    audit = _read_json(audit_path)
    audit["prepare_report"] = prepare
    audit["output"]["sha256"] = digest
    audit["output"]["size_bytes"] = size_bytes
    _write_json(audit_path, audit)


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


def _walk_json(value: object, field: str = "$") -> list[tuple[str, object]]:
    rows = [(field, value)]
    if isinstance(value, dict):
        for key, child in value.items():
            rows.extend(_walk_json(child, f"{field}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_walk_json(child, f"{field}[{index}]"))
    return rows
