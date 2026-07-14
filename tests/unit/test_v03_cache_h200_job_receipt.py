# SPDX-License-Identifier: Apache-2.0
"""Contracts for the terminal JobInfo receipt of the v0.3 H200 cache proof."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jsonschema
import pytest

from geno_lewm.errors import InputError, ResourceError
from tools.research import v03_cache_h200_job_receipt as receipt, v03_cache_h200_launch as launch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "configs/data_v03/cache-h200-job-receipt.schema.json"
PROOF_RUNTIME_HASH = f"sha256:{'6' * 64}"


def test_job_receipt_schema_is_versioned_and_closed() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$id"].endswith("/cache-h200-job-receipt.v1.json")
    assert schema["additionalProperties"] is False
    for field in ("source", "job", "trace", "proof", "publication", "claim_boundary"):
        assert schema["properties"][field]["additionalProperties"] is False
    assert schema["properties"]["job"]["properties"]["status"]["properties"]["stage"] == {
        "const": "COMPLETED"
    }
    claim = schema["properties"]["claim_boundary"]["properties"]
    assert claim["jobinfo_terminal_status_attested"] == {"const": True}
    assert claim["timeout_server_echo_attested"] == {"const": False}
    runtime_hash = schema["properties"]["proof"]["properties"]["runtime_hash"]
    assert runtime_hash == {"pattern": "^sha256:[0-9a-f]{64}$", "type": "string"}


def test_author_cli_never_logs_token_or_job_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    secret = "hf_secret_value_that_must_never_be_logged"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setattr(receipt, "HuggingFaceReadClient", lambda **_kwargs: object())
    monkeypatch.setattr(
        receipt,
        "capture_terminal_receipt",
        lambda **_kwargs: {"ok": True, "server_payload": secret},
    )

    exit_code = receipt.main(
        [
            "author",
            "--output-dir",
            str(tmp_path / "receipt"),
            "--proof-download-dir",
            str(tmp_path / "proof"),
            "--job-id",
            "job-secret-regression",
            "--source-commit",
            "a" * 40,
            "--run-attempt",
            "1",
            "--proof-revision",
            "b" * 40,
            "--proof-namespace",
            receipt.expected_proof_namespace("a" * 40, 1),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "GENO_LEWM_V03_CACHE_H200_JOB_RECEIPT_OK\n"
    assert captured.err == ""
    assert secret not in captured.out
    assert secret not in captured.err


@dataclass
class _Volume:
    type: str
    source: str
    mount_path: str
    revision: str | None
    read_only: bool
    path: str | None = None


def _matching_job(*, job_id: str, source_commit: str, run_attempt: int) -> Any:
    spec = launch.build_launch_spec(source_commit=source_commit, run_attempt=run_attempt)
    return SimpleNamespace(
        id=job_id,
        url=f"https://huggingface.co/jobs/abdelstark/{job_id}",
        owner=SimpleNamespace(name="abdelstark"),
        status=SimpleNamespace(stage="COMPLETED", message=None),
        docker_image=spec.image,
        space_id=None,
        command=list(spec.command),
        arguments=[],
        environment=dict(spec.environment),
        flavor=spec.flavor,
        labels=dict(spec.labels),
        secrets={"HF_TOKEN": "server-redacted"},
        volumes=[_Volume(**asdict(spec.volume))],
    )


class _ReadClient:
    def __init__(self, *, job: Any, proof_dir: Path) -> None:
        self.job = job
        self.proof_dir = proof_dir
        self.inspect_calls: list[tuple[str, str]] = []
        self.download_calls: list[dict[str, object]] = []

    def inspect_job(self, *, job_id: str, namespace: str) -> Any:
        self.inspect_calls.append((job_id, namespace))
        return self.job

    def download_exact_namespace(self, **kwargs: object) -> Path:
        self.download_calls.append(dict(kwargs))
        return self.proof_dir


class _SequenceReadClient(_ReadClient):
    def __init__(self, *, downloads: list[Path]) -> None:
        super().__init__(job=None, proof_dir=downloads[0])
        self.downloads = iter(downloads)

    def download_exact_namespace(self, **kwargs: object) -> Path:
        self.download_calls.append(dict(kwargs))
        return next(self.downloads)


class _HubApi:
    def __init__(self, *, resolved_revision: str) -> None:
        self.resolved_revision = resolved_revision
        self.repo_calls: list[dict[str, object]] = []

    def repo_info(self, **kwargs: object) -> Any:
        self.repo_calls.append(dict(kwargs))
        return SimpleNamespace(sha=self.resolved_revision)

    def inspect_job(self, **kwargs: object) -> Any:
        return SimpleNamespace(**kwargs)


class _HubModule:
    __version__ = "1.8.0"

    def __init__(self, *, api: _HubApi) -> None:
        self.api = api
        self.snapshot_calls: list[dict[str, object]] = []

    def HfApi(self, *, token: str | bool) -> _HubApi:  # noqa: N802 - external API name
        assert token == "host-token"
        return self.api

    def snapshot_download(self, **kwargs: object) -> str:
        self.snapshot_calls.append(dict(kwargs))
        root = Path(str(kwargs["local_dir"]))
        namespace = str(kwargs["allow_patterns"][0]).removesuffix("/**")  # type: ignore[index]
        (root / namespace).mkdir(parents=True)
        return str(root)


def _proof_report(
    *,
    source_commit: str,
    runtime_hash: str = PROOF_RUNTIME_HASH,
) -> dict[str, object]:
    return {
        "ok": True,
        "producer": {
            "git_commit": source_commit,
            "origin": "https://github.com/AbdelStark/GenoLeWM.git",
            "declared_container_image": launch.CONTAINER_IMAGE,
            "container_binding": "launcher_environment_declaration",
        },
        "trace": {
            "repository": launch.TRACE_REPOSITORY,
            "revision": launch.TRACE_REVISION,
            "artifact_path": launch.TRACE_ARTIFACT_PATH,
        },
        "runtime": {"runtime_hash": runtime_hash},
        "claim_boundary": {"hf_job_terminal_status_attested": False},
    }


def test_capture_authors_canonical_receipt_after_exact_proof_replay(tmp_path: Path) -> None:
    source_commit = "c" * 40
    run_attempt = 2
    job_id = "6a56055fb1669a49bf071e81"
    proof_revision = "d" * 40
    proof_namespace = receipt.expected_proof_namespace(source_commit, run_attempt)
    proof_dir = tmp_path / "proof"
    (proof_dir / "proof").mkdir(parents=True)
    (proof_dir / "SHA256SUMS").write_text("proof closure\n", encoding="utf-8")
    proof_report = _proof_report(source_commit=source_commit)
    (proof_dir / "proof/cache-h200-proof.json").write_bytes(
        receipt.canonical_json_bytes(proof_report)
    )
    client = _ReadClient(
        job=_matching_job(
            job_id=job_id,
            source_commit=source_commit,
            run_attempt=run_attempt,
        ),
        proof_dir=proof_dir,
    )
    replayed: list[Path] = []

    def verify_proof(path: Path) -> dict[str, object]:
        replayed.append(path)
        return proof_report

    output_dir = tmp_path / "receipt"
    payload = receipt.capture_terminal_receipt(
        output_dir=output_dir,
        proof_download_dir=tmp_path / "proof-download",
        job_id=job_id,
        source_commit=source_commit,
        run_attempt=run_attempt,
        proof_revision=proof_revision,
        proof_namespace=proof_namespace,
        client=client,
        proof_verifier=verify_proof,
    )

    assert client.inspect_calls == [(job_id, "abdelstark")]
    assert client.download_calls == [
        {
            "repository": "abdelstark/geno-lewm-data",
            "repo_type": "dataset",
            "revision": proof_revision,
            "namespace": proof_namespace,
            "destination": tmp_path / "proof-download",
            "token": False,
        }
    ]
    assert replayed == [proof_dir]
    assert payload["job"]["status"] == {"stage": "COMPLETED"}
    assert payload["job"]["secret_names"] == ["HF_TOKEN"]
    assert payload["job"]["volumes"] == [
        {
            "type": "model",
            "source": launch.CARBON_REPOSITORY,
            "mount_path": "/carbon",
            "revision": launch.CARBON_REVISION,
            "read_only": True,
            "path": None,
        }
    ]
    assert payload["proof"]["revision"] == proof_revision
    assert payload["proof"]["runtime_hash"] == PROOF_RUNTIME_HASH
    assert payload["publication"] == {
        "repository": "abdelstark/geno-lewm-data",
        "namespace": receipt.expected_receipt_namespace(proof_namespace),
        "sibling_of_proof_namespace": True,
    }
    claim = payload["claim_boundary"]
    assert claim["requested_submission_timeout"] == "8h"
    assert claim["timeout_server_echo_attested"] is False

    report_bytes = (output_dir / receipt.RECEIPT_NAME).read_bytes()
    assert report_bytes == receipt.canonical_json_bytes(payload)
    assert receipt.verify_existing_receipt(output_dir) == payload
    assert {path.name for path in output_dir.iterdir()} == {
        receipt.RECEIPT_NAME,
        receipt.SCHEMA_NAME,
        receipt.CHECKSUMS_NAME,
    }
    assert b"server-redacted" not in report_bytes

    tampered = json.loads(report_bytes)
    tampered["job"]["url"] = "https://huggingface.co/jobs/abdelstark/different-job"
    (output_dir / receipt.RECEIPT_NAME).write_bytes(receipt.canonical_json_bytes(tampered))
    checksums = "".join(
        f"{hashlib.sha256((output_dir / name).read_bytes()).hexdigest()}  {name}\n"
        for name in sorted((receipt.RECEIPT_NAME, receipt.SCHEMA_NAME))
    )
    (output_dir / receipt.CHECKSUMS_NAME).write_bytes(checksums.encode("ascii"))
    with pytest.raises(InputError, match="JobInfo fields"):
        receipt.verify_existing_receipt(output_dir)


def test_capture_rejects_proof_report_bytes_that_differ_from_replay_result(
    tmp_path: Path,
) -> None:
    source_commit = "e" * 40
    run_attempt = 1
    proof_dir = tmp_path / "proof"
    (proof_dir / "proof").mkdir(parents=True)
    (proof_dir / "SHA256SUMS").write_text("proof closure\n", encoding="utf-8")
    (proof_dir / "proof/cache-h200-proof.json").write_text("{}\n", encoding="utf-8")
    client = _ReadClient(
        job=_matching_job(
            job_id="job-proof-byte-drift",
            source_commit=source_commit,
            run_attempt=run_attempt,
        ),
        proof_dir=proof_dir,
    )

    with pytest.raises(InputError, match="report bytes differ"):
        receipt.capture_terminal_receipt(
            output_dir=tmp_path / "receipt",
            proof_download_dir=tmp_path / "download",
            job_id="job-proof-byte-drift",
            source_commit=source_commit,
            run_attempt=run_attempt,
            proof_revision="f" * 40,
            proof_namespace=receipt.expected_proof_namespace(source_commit, run_attempt),
            client=client,
            proof_verifier=lambda _path: _proof_report(source_commit=source_commit),
        )

    assert not (tmp_path / "receipt").exists()


def test_capture_rejects_nonterminal_job_before_proof_download(tmp_path: Path) -> None:
    source_commit = "a" * 40
    job = _matching_job(
        job_id="job-still-running",
        source_commit=source_commit,
        run_attempt=1,
    )
    job.status.stage = "RUNNING"
    client = _ReadClient(job=job, proof_dir=tmp_path / "unused")

    with pytest.raises(InputError, match="must be COMPLETED"):
        receipt.capture_terminal_receipt(
            output_dir=tmp_path / "receipt",
            proof_download_dir=tmp_path / "proof-download",
            job_id="job-still-running",
            source_commit=source_commit,
            run_attempt=1,
            proof_revision="b" * 40,
            proof_namespace=receipt.expected_proof_namespace(source_commit, 1),
            client=client,
            proof_verifier=lambda _path: {},
        )

    assert client.download_calls == []
    assert not (tmp_path / "receipt").exists()


def test_capture_rejects_unsuccessful_proof_without_writing_receipt(tmp_path: Path) -> None:
    source_commit = "7" * 40
    run_attempt = 1
    proof_dir = tmp_path / "proof"
    (proof_dir / "proof").mkdir(parents=True)
    (proof_dir / "SHA256SUMS").write_text("proof closure\n", encoding="utf-8")
    proof_report = _proof_report(source_commit=source_commit)
    proof_report["ok"] = False
    (proof_dir / "proof/cache-h200-proof.json").write_bytes(
        receipt.canonical_json_bytes(proof_report)
    )
    client = _ReadClient(
        job=_matching_job(
            job_id="job-unsuccessful-proof",
            source_commit=source_commit,
            run_attempt=run_attempt,
        ),
        proof_dir=proof_dir,
    )

    with pytest.raises(InputError, match="does not bind the exact launch"):
        receipt.capture_terminal_receipt(
            output_dir=tmp_path / "receipt",
            proof_download_dir=tmp_path / "proof-download",
            job_id="job-unsuccessful-proof",
            source_commit=source_commit,
            run_attempt=run_attempt,
            proof_revision="8" * 40,
            proof_namespace=receipt.expected_proof_namespace(source_commit, run_attempt),
            client=client,
            proof_verifier=lambda _path: proof_report,
        )

    assert not (tmp_path / "receipt").exists()


def test_remote_verification_redownloads_and_replays_receipt_and_bound_proof(
    tmp_path: Path,
) -> None:
    source_commit = "1" * 40
    run_attempt = 3
    job_id = "job-remote-replay"
    proof_revision = "2" * 40
    proof_namespace = receipt.expected_proof_namespace(source_commit, run_attempt)
    proof_dir = tmp_path / "proof-source"
    (proof_dir / "proof").mkdir(parents=True)
    proof_report = _proof_report(source_commit=source_commit)
    (proof_dir / "SHA256SUMS").write_text("proof closure\n", encoding="utf-8")
    (proof_dir / "proof/cache-h200-proof.json").write_bytes(
        receipt.canonical_json_bytes(proof_report)
    )
    author_client = _ReadClient(
        job=_matching_job(
            job_id=job_id,
            source_commit=source_commit,
            run_attempt=run_attempt,
        ),
        proof_dir=proof_dir,
    )
    receipt_dir = tmp_path / "receipt-source"
    receipt.capture_terminal_receipt(
        output_dir=receipt_dir,
        proof_download_dir=tmp_path / "author-proof-download",
        job_id=job_id,
        source_commit=source_commit,
        run_attempt=run_attempt,
        proof_revision=proof_revision,
        proof_namespace=proof_namespace,
        client=author_client,
        proof_verifier=lambda _path: proof_report,
    )
    remote_client = _SequenceReadClient(downloads=[receipt_dir, proof_dir])
    replayed: list[Path] = []

    payload = receipt.verify_remote_receipt(
        receipt_revision="3" * 40,
        receipt_namespace=receipt.expected_receipt_namespace(proof_namespace),
        download_root=tmp_path / "remote-download",
        client=remote_client,
        proof_verifier=lambda path: replayed.append(path) or proof_report,
    )

    assert payload["proof"]["revision"] == proof_revision
    assert payload["proof"]["runtime_hash"] == PROOF_RUNTIME_HASH
    assert replayed == [proof_dir]
    assert remote_client.download_calls == [
        {
            "repository": receipt.PROOF_REPOSITORY,
            "repo_type": "dataset",
            "revision": "3" * 40,
            "namespace": receipt.expected_receipt_namespace(proof_namespace),
            "destination": tmp_path / "remote-download/receipt",
            "token": False,
        },
        {
            "repository": receipt.PROOF_REPOSITORY,
            "repo_type": "dataset",
            "revision": proof_revision,
            "namespace": proof_namespace,
            "destination": tmp_path / "remote-download/proof",
            "token": False,
        },
    ]

    drifted_report = _proof_report(
        source_commit=source_commit,
        runtime_hash=f"sha256:{'9' * 64}",
    )
    (proof_dir / "proof/cache-h200-proof.json").write_bytes(
        receipt.canonical_json_bytes(drifted_report)
    )
    with pytest.raises(InputError, match="runtime hash differs"):
        receipt.verify_remote_receipt(
            receipt_revision="3" * 40,
            receipt_namespace=receipt.expected_receipt_namespace(proof_namespace),
            download_root=tmp_path / "runtime-drift-download",
            client=_SequenceReadClient(downloads=[receipt_dir, proof_dir]),
            proof_verifier=lambda _path: drifted_report,
        )


def test_hub_adapter_requires_exact_revision_resolution(tmp_path: Path) -> None:
    revision = "4" * 40
    namespace = "some/exact/namespace"
    api = _HubApi(resolved_revision=revision)
    hub = _HubModule(api=api)
    client = receipt.HuggingFaceReadClient(token="host-token", hub_module=hub)

    downloaded = client.download_exact_namespace(
        repository=receipt.PROOF_REPOSITORY,
        repo_type="dataset",
        revision=revision,
        namespace=namespace,
        destination=tmp_path / "download",
        token=False,
    )

    assert downloaded == tmp_path / "download" / namespace
    assert api.repo_calls == [
        {
            "repo_id": receipt.PROOF_REPOSITORY,
            "repo_type": "dataset",
            "revision": revision,
            "files_metadata": False,
            "token": False,
        }
    ]
    assert hub.snapshot_calls == [
        {
            "repo_id": receipt.PROOF_REPOSITORY,
            "repo_type": "dataset",
            "revision": revision,
            "allow_patterns": [f"{namespace}/**"],
            "local_dir": str(tmp_path / "download"),
            "force_download": True,
            "token": False,
        }
    ]

    drifting = receipt.HuggingFaceReadClient(
        token="host-token",
        hub_module=_HubModule(api=_HubApi(resolved_revision="5" * 40)),
    )
    with pytest.raises(ResourceError, match="different Hub revision"):
        drifting.download_exact_namespace(
            repository=receipt.PROOF_REPOSITORY,
            repo_type="dataset",
            revision=revision,
            namespace=namespace,
            destination=tmp_path / "drift",
            token=False,
        )


def test_cli_exposes_capture_and_read_only_replay_without_upload_surface() -> None:
    help_text = receipt._parser().format_help()

    assert "author" in help_text
    assert "verify-existing" in help_text
    assert "verify-remote" in help_text
    assert "publish" not in help_text
