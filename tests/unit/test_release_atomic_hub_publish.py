# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for conditional single-commit Hub publication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from geno_lewm.errors import InputError, ResourceError
from tools.release.atomic_hub_publish import (
    HubParent,
    HuggingFaceHubClient,
    PublishFile,
    StaleParentError,
    main,
    publish_success_namespace,
)
from tools.research.training_reproducibility_preflight import (
    EXPECTED_CARBON_RUNTIME_HASH,
)

SOURCE_COMMIT = "b" * 40
RUN_NAME = "run-1"
REPO_ID = "owner/runs"


@dataclass
class FakeHubClient:
    parents: list[HubParent]
    stale_calls: set[int] = field(default_factory=set)
    fail_on_call: Exception | None = None
    tamper_download: bool = False
    read_calls: int = 0
    create_calls: list[tuple[str, tuple[PublishFile, ...]]] = field(default_factory=list)
    committed: dict[str, bytes] = field(default_factory=dict)

    def read_parent(self, *, repo_id: str, repo_type: str) -> HubParent:
        assert repo_id == "owner/runs"
        assert repo_type == "model"
        parent = self.parents[min(self.read_calls, len(self.parents) - 1)]
        self.read_calls += 1
        return parent

    def create_commit(
        self,
        *,
        repo_id: str,
        repo_type: str,
        namespace: str,
        files: tuple[PublishFile, ...],
        parent_commit: str,
        commit_message: str,
    ) -> str:
        assert repo_id == "owner/runs"
        assert repo_type == "model"
        assert commit_message
        self.create_calls.append((parent_commit, files))
        call = len(self.create_calls)
        if self.fail_on_call is not None:
            raise self.fail_on_call
        if call in self.stale_calls:
            raise StaleParentError(parent_commit)
        self.committed = {
            f"{namespace}/{item.relative_path}": item.source_path.read_bytes() for item in files
        }
        return "c" * 40

    def download_namespace(
        self,
        *,
        repo_id: str,
        repo_type: str,
        namespace: str,
        revision: str,
        destination: Path,
    ) -> Path:
        assert repo_id == "owner/runs"
        assert repo_type == "model"
        assert revision == "c" * 40
        destination.mkdir(parents=True)
        for remote_path, content in self.committed.items():
            relative = Path(remote_path).relative_to(namespace)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        if self.tamper_download:
            (destination / "evidence/training_reproducibility_report.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
        return destination


def test_publish_commits_completion_and_evidence_once_then_verifies_download(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    client = FakeHubClient(parents=[_parent("a")])

    result = publish_success_namespace(
        bundle_dir=bundle,
        repo_id="owner/runs",
        run_name="run-1",
        source_commit_sha="b" * 40,
        verification_dir=tmp_path / "verified",
        client=client,
        generated_at="2026-07-13T12:00:00Z",
    )

    assert result.source_commit_sha == SOURCE_COMMIT
    assert result.hub_commit_sha == "c" * 40
    assert result.hub_parent_commit_sha == "a" * 40
    assert result.attempts == 1
    assert result.verified is True
    assert len(client.create_calls) == 1
    parent, files = client.create_calls[0]
    assert parent == "a" * 40
    assert {item.relative_path for item in files} == {
        "SHA256SUMS",
        "completion.json",
        "evidence/job_contract_preflight.json",
        "evidence/runtime_preflight.json",
        "evidence/training_reproducibility_report.json",
    }
    marker = json.loads(client.committed["run-1/success/completion.json"])
    assert marker["commit_sha"] == SOURCE_COMMIT
    assert marker["source_commit_sha"] == SOURCE_COMMIT
    assert marker["checksums"] == {
        "path": "SHA256SUMS",
        "sha256": _sha256_uri(bundle / "SHA256SUMS"),
    }
    assert marker["report"] == {
        "path": "evidence/training_reproducibility_report.json",
        "sha256": _sha256_uri(bundle / "evidence/training_reproducibility_report.json"),
    }
    assert (tmp_path / "verified/completion.json").is_file()


def test_stale_parent_retries_only_after_fresh_absence_proof(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    client = FakeHubClient(
        parents=[_parent("a"), _parent("b")],
        stale_calls={1},
    )

    result = publish_success_namespace(
        bundle_dir=bundle,
        repo_id="owner/runs",
        run_name="run-1",
        source_commit_sha="b" * 40,
        verification_dir=tmp_path / "verified",
        client=client,
        max_attempts=3,
    )

    assert result.attempts == 2
    assert result.hub_parent_commit_sha == "b" * 40
    assert [parent for parent, _files in client.create_calls] == ["a" * 40, "b" * 40]
    assert client.read_calls == 2


def test_concurrent_winner_blocks_retry_before_second_commit(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    client = FakeHubClient(
        parents=[
            _parent("a"),
            _parent("b", files=("run-1/success/completion.json",)),
        ],
        stale_calls={1},
    )

    with pytest.raises(ResourceError, match="already exists"):
        publish_success_namespace(
            bundle_dir=bundle,
            repo_id="owner/runs",
            run_name="run-1",
            source_commit_sha="b" * 40,
            verification_dir=tmp_path / "verified",
            client=client,
            max_attempts=3,
        )

    assert len(client.create_calls) == 1
    assert client.read_calls == 2


def test_non_stale_remote_error_is_not_retried(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    failure = RuntimeError("HTTP 503")
    client = FakeHubClient(parents=[_parent("a")], fail_on_call=failure)

    with pytest.raises(RuntimeError, match="503"):
        publish_success_namespace(
            bundle_dir=bundle,
            repo_id="owner/runs",
            run_name="run-1",
            source_commit_sha="b" * 40,
            verification_dir=tmp_path / "verified",
            client=client,
            max_attempts=3,
        )

    assert len(client.create_calls) == 1
    assert client.read_calls == 1


@pytest.mark.parametrize(("status", "stale"), [(412, True), (409, False), (503, False)])
def test_huggingface_adapter_classifies_only_http_412_as_stale_parent(
    tmp_path: Path,
    status: int,
    stale: bool,
) -> None:
    captured: dict[str, object] = {}

    class FakeHTTPError(Exception):
        def __init__(self) -> None:
            self.response = SimpleNamespace(status_code=status)

    class FakeApi:
        @staticmethod
        def create_commit(**kwargs: object) -> object:
            captured.update(kwargs)
            raise FakeHTTPError

    client = object.__new__(HuggingFaceHubClient)
    client._api = FakeApi()
    client._operation_add = lambda **kwargs: kwargs
    client._http_error_type = FakeHTTPError
    file = tmp_path / "file.txt"
    file.write_text("evidence\n", encoding="utf-8")
    publish_file = PublishFile(
        relative_path="file.txt",
        source_path=file,
        sha256=_sha256_uri(file),
        size_bytes=file.stat().st_size,
    )

    expected_error = StaleParentError if stale else ResourceError
    with pytest.raises(expected_error):
        client.create_commit(
            repo_id="owner/runs",
            repo_type="model",
            namespace="run-1/success",
            files=(publish_file,),
            parent_commit="a" * 40,
            commit_message="atomic",
        )
    assert captured == {
        "repo_id": "owner/runs",
        "repo_type": "model",
        "revision": "main",
        "create_pr": False,
        "operations": [
            {
                "path_in_repo": "run-1/success/file.txt",
                "path_or_fileobj": str(file),
            }
        ],
        "commit_message": "atomic",
        "parent_commit": "a" * 40,
        "token": True,
    }


def test_stale_parent_retry_budget_is_bounded(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    client = FakeHubClient(
        parents=[_parent("a"), _parent("b"), _parent("d")],
        stale_calls={1, 2, 3},
    )

    with pytest.raises(ResourceError, match="retry budget"):
        publish_success_namespace(
            bundle_dir=bundle,
            repo_id="owner/runs",
            run_name="run-1",
            source_commit_sha="b" * 40,
            verification_dir=tmp_path / "verified",
            client=client,
            max_attempts=2,
        )

    assert len(client.create_calls) == 2
    assert client.read_calls == 2


def test_downloaded_commit_must_match_checksums_marker_and_uploaded_bytes(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    client = FakeHubClient(parents=[_parent("a")], tamper_download=True)

    with pytest.raises(InputError, match="immutable downloaded namespace"):
        publish_success_namespace(
            bundle_dir=bundle,
            repo_id="owner/runs",
            run_name="run-1",
            source_commit_sha="b" * 40,
            verification_dir=tmp_path / "verified",
            client=client,
        )


@pytest.mark.parametrize(
    ("relative_path", "field_path"),
    [
        ("evidence/training_reproducibility_report.json", ("runs", 2, "commit_sha")),
        ("evidence/runtime_preflight.json", ("source_commit_sha",)),
        ("evidence/runtime_preflight.json", ("run_name",)),
        ("evidence/job_contract_preflight.json", ("repository", "expected_commit_sha")),
        ("evidence/job_contract_preflight.json", ("repository", "observed_commit_sha")),
        ("evidence/job_contract_preflight.json", ("job", "run_name")),
    ],
)
def test_success_evidence_must_match_requested_source_identity_before_remote_read(
    tmp_path: Path,
    relative_path: str,
    field_path: tuple[str | int, ...],
) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    path = bundle / relative_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    _set_nested(payload, field_path, "mismatched")
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    _write_checksums(bundle)
    client = FakeHubClient(parents=[_parent("a")])

    with pytest.raises(InputError, match="evidence identity"):
        publish_success_namespace(
            bundle_dir=bundle,
            repo_id=REPO_ID,
            run_name=RUN_NAME,
            source_commit_sha=SOURCE_COMMIT,
            verification_dir=tmp_path / "verified",
            client=client,
        )

    assert client.read_calls == 0
    assert client.create_calls == []


def test_cli_runs_complete_protocol_through_hub_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    client = FakeHubClient(parents=[_parent("a")])
    monkeypatch.setattr(
        "tools.release.atomic_hub_publish._load_default_client",
        lambda: client,
    )

    rc = main(
        [
            "--bundle-dir",
            str(bundle),
            "--repo-id",
            "owner/runs",
            "--run-name",
            "run-1",
            "--source-commit-sha",
            "b" * 40,
            "--verification-dir",
            str(tmp_path / "verified"),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] is True
    assert payload["source_commit_sha"] == SOURCE_COMMIT
    assert payload["hub_commit_sha"] == "c" * 40


def _write_bundle(root: Path) -> Path:
    report = root / "evidence/training_reproducibility_report.json"
    job_preflight = root / "evidence/job_contract_preflight.json"
    runtime = root / "evidence/runtime_preflight.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "schema_version": "2.0.0",
                "generated_by": "tools.release.training_reproducibility",
                "ok": True,
                "runs": [
                    {"label": label, "commit_sha": SOURCE_COMMIT}
                    for label in (
                        "baseline_a",
                        "deterministic_a",
                        "deterministic_b",
                        "baseline_b",
                    )
                ],
                "run_contract": {"ok": True},
                "deterministic_pair": {"ok": True},
                "throughput": {"ok": True, "status": "pass"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    job_preflight.write_text(
        json.dumps(
            {
                "schema_version": "2.0.0",
                "generated_by": "tools.research.training_reproducibility_preflight",
                "ok": True,
                "repository": {
                    "expected_commit_sha": SOURCE_COMMIT,
                    "observed_commit_sha": SOURCE_COMMIT,
                    "worktree_clean": True,
                },
                "job": {
                    "run_name": RUN_NAME,
                    "upload_repo": REPO_ID,
                    "expected_carbon_runtime_hash": EXPECTED_CARBON_RUNTIME_HASH,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    runtime.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_by": "tools.jobs.training_reproducibility_run",
                "ok": True,
                "source_commit_sha": SOURCE_COMMIT,
                "run_name": RUN_NAME,
                "accelerator": {"device_name": "NVIDIA H200"},
                "carbon_runtime_hash": EXPECTED_CARBON_RUNTIME_HASH,
                "expected_carbon_runtime_hash": EXPECTED_CARBON_RUNTIME_HASH,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_checksums(root)
    return root


def _write_checksums(root: Path) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    checksum_lines = [f"{_sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def _set_nested(payload: object, field_path: tuple[str | int, ...], value: object) -> None:
    cursor = payload
    for key in field_path[:-1]:
        if isinstance(key, int):
            assert isinstance(cursor, list)
        else:
            assert isinstance(cursor, dict)
        cursor = cursor[key]
    final_key = field_path[-1]
    if isinstance(final_key, int):
        assert isinstance(cursor, list)
    else:
        assert isinstance(cursor, dict)
    cursor[final_key] = value


def _parent(marker: str, *, files: tuple[str, ...] = ()) -> HubParent:
    return HubParent(commit_sha=marker * 40, files=files)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_uri(path: Path) -> str:
    return f"sha256:{_sha256(path)}"
