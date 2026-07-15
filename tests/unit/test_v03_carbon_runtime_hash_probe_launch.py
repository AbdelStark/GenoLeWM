# SPDX-License-Identifier: Apache-2.0
"""Contracts for the exact-volume Carbon runtime-hash CPU launcher."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pytest

from tools.research import v03_carbon_runtime_hash_probe_launch as launch

EXPECTED_HASH = "sha256:" + "c" * 64


@dataclass
class _FakeVolume:
    type: str
    source: str
    mount_path: str
    revision: str | None
    read_only: bool
    path: str | None = None


@dataclass
class _FakeJob:
    id: str
    volumes: list[_FakeVolume]
    docker_image: str
    space_id: str | None
    command: list[str]
    arguments: list[str]
    environment: dict[str, str]
    flavor: str
    labels: dict[str, str]
    secrets: list[str]
    owner: Any
    url: str = "https://huggingface.co/jobs/abdelstark/fake-job"


class _FakeApi:
    def __init__(self) -> None:
        self.run_kwargs: dict[str, Any] | None = None
        self.canceled: list[tuple[str, str]] = []

    def run_job(self, **kwargs: Any) -> _FakeJob:
        self.run_kwargs = kwargs
        requested = kwargs["volumes"][0]
        return _FakeJob(
            id="fake-job",
            volumes=[
                _FakeVolume(
                    type=requested.type,
                    source=requested.source,
                    mount_path=requested.mount_path,
                    revision=requested.revision,
                    read_only=requested.read_only,
                    path=requested.path,
                )
            ],
            docker_image=kwargs["image"],
            space_id=None,
            command=kwargs["command"],
            arguments=[],
            environment=kwargs["env"],
            flavor=kwargs["flavor"],
            labels=kwargs["labels"],
            secrets=list(kwargs["secrets"]),
            owner=type("Owner", (), {"name": kwargs["namespace"]})(),
        )

    def cancel_job(self, *, job_id: str, namespace: str) -> None:
        self.canceled.append((job_id, namespace))


class _TamperingApi(_FakeApi):
    def __init__(self, field: str) -> None:
        super().__init__()
        self.field = field

    def run_job(self, **kwargs: Any) -> _FakeJob:
        job = super().run_job(**kwargs)
        if self.field == "image":
            job.docker_image = "unexpected/image:latest"
        elif self.field == "command":
            job.command = ["unexpected"]
        elif self.field == "arguments":
            job.arguments = ["unexpected"]
        elif self.field == "environment":
            job.environment = {**job.environment, "RUN_ATTEMPT": "999"}
        elif self.field == "flavor":
            job.flavor = "cpu-upgrade"
        elif self.field == "labels":
            job.labels = {"purpose": "unexpected"}
        elif self.field == "secret_names":
            job.secrets = ["HF_TOKEN"]
        elif self.field == "namespace":
            job.owner = type("Owner", (), {"name": "someone-else"})()
        elif self.field == "volume":
            job.volumes[0].revision = None
        elif self.field == "space_id":
            job.space_id = "unexpected/space"
        else:  # pragma: no cover - protects the test helper itself.
            raise AssertionError(self.field)
        return job


def test_launch_spec_is_a_cheap_nonpublishing_exact_volume_probe() -> None:
    spec = launch.build_launch_spec(
        source_commit="a" * 40,
        expected_runtime_hash=EXPECTED_HASH,
        run_attempt=2,
    )
    payload = spec.public_payload()

    assert payload == {
        "command": list(spec.command),
        "environment": dict(spec.environment),
        "flavor": "cpu-basic",
        "image": launch.CONTAINER_IMAGE,
        "labels": {"purpose": "geno-lewm-v03-carbon-runtime-hash-probe"},
        "namespace": "abdelstark",
        "timeout": "30m",
        "volumes": [
            {
                "type": "model",
                "source": "HuggingFaceBio/Carbon-500M",
                "mount_path": "/carbon",
                "revision": "5d31d59b3c845b288a13aedb1358934196852eec",
                "read_only": True,
                "path": None,
            }
        ],
    }
    assert spec.secret_names == ()
    assert spec.environment["EXPECTED_RUNTIME_HASH"] == EXPECTED_HASH
    assert spec.environment["JOB_TIMEOUT"] == "30m"
    assert spec.environment["RUN_ATTEMPT"] == "2"
    assert "add3c1a663" not in repr(spec)

    inner_command = spec.command[2]
    workspace_guard = inner_command.index("test ! -L /workspace")
    clone = inner_command.index(
        "git clone --filter=blob:none --no-checkout https://github.com/AbdelStark/GenoLeWM.git"
    )
    checkout = inner_command.index('git checkout --detach "$COMMIT_SHA"')
    clean = inner_command.index('test -z "$(git status --porcelain=v1 --untracked-files=all)"')
    sync = inner_command.index("uv sync --frozen --no-dev")
    execute = inner_command.index(
        "exec uv run --no-sync python -m tools.research.v03_carbon_runtime_hash_probe"
    )
    assert workspace_guard < clone < checkout < clean < sync < execute
    assert 'test "$(cd /carbon && pwd -P)" = /carbon' in inner_command
    assert re.search(r"(?<![A-Za-z0-9_])/work(?![A-Za-z0-9_])", inner_command) is None
    assert "--extra train" not in inner_command
    assert "--extra evidence" not in inner_command
    assert "HF_TOKEN" not in inner_command
    assert "upload" not in inner_command.lower()
    assert "publish" not in inner_command.lower()


def test_submit_uses_exact_volume_and_does_not_inject_host_token_into_job() -> None:
    spec = launch.build_launch_spec(
        source_commit="d" * 40,
        expected_runtime_hash=EXPECTED_HASH,
        run_attempt=3,
    )
    api = _FakeApi()

    job = launch.submit_exact_revision_job(
        spec,
        token="host-api-token",
        api=api,
        volume_class=_FakeVolume,
    )

    assert job.id == "fake-job"
    assert api.run_kwargs is not None
    requested = api.run_kwargs["volumes"][0]
    assert requested == _FakeVolume(
        type="model",
        source="HuggingFaceBio/Carbon-500M",
        mount_path="/carbon",
        revision="5d31d59b3c845b288a13aedb1358934196852eec",
        read_only=True,
        path=None,
    )
    assert api.run_kwargs["secrets"] == {}
    assert "host-api-token" not in repr(api.run_kwargs)
    assert api.canceled == []


@pytest.mark.parametrize(
    "field",
    (
        "image",
        "command",
        "arguments",
        "environment",
        "flavor",
        "labels",
        "secret_names",
        "namespace",
        "volume",
        "space_id",
    ),
)
def test_submit_cancels_every_jobinfo_visible_contract_drift(field: str) -> None:
    spec = launch.build_launch_spec(
        source_commit="e" * 40,
        expected_runtime_hash=EXPECTED_HASH,
        run_attempt=4,
    )
    api = _TamperingApi(field)

    with pytest.raises(RuntimeError, match="HF JobInfo"):
        launch.submit_exact_revision_job(
            spec,
            token="host-api-token",
            api=api,
            volume_class=_FakeVolume,
        )

    assert api.canceled == [("fake-job", "abdelstark")]


def test_dry_run_is_credential_free_and_keeps_expected_hash_explicit(capsys) -> None:
    assert (
        launch.main(
            [
                "--source-commit",
                "f" * 40,
                "--expected-runtime-hash",
                EXPECTED_HASH,
                "--run-attempt",
                "5",
                "--dry-run",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert f'"EXPECTED_RUNTIME_HASH": "{EXPECTED_HASH}"' in captured.out
    assert '"flavor": "cpu-basic"' in captured.out
    assert '"timeout": "30m"' in captured.out
    assert "HF_TOKEN" not in captured.out
    assert "secret_names" not in captured.out
    assert "add3c1a663" not in captured.out
