# SPDX-License-Identifier: Apache-2.0
"""Contracts for the exact-revision host-side H200 proof launcher."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pytest

from tools.research import v03_cache_h200_launch as launch


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
    def __init__(self, *, returned_revision: str | None) -> None:
        self.returned_revision = returned_revision
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
                    revision=self.returned_revision,
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


class _TamperedEnvironmentApi(_FakeApi):
    def run_job(self, **kwargs: Any) -> _FakeJob:
        job = super().run_job(**kwargs)
        job.environment = {**job.environment, "CARBON_REVISION": "0" * 40}
        return job


class _TamperedArgumentsApi(_FakeApi):
    def run_job(self, **kwargs: Any) -> _FakeJob:
        job = super().run_job(**kwargs)
        job.arguments = ["unexpected"]
        return job


class _TamperedVolumePathApi(_FakeApi):
    def run_job(self, **kwargs: Any) -> _FakeJob:
        job = super().run_job(**kwargs)
        job.volumes[0].path = "unexpected/subdir"
        return job


class _TamperedSpaceImageApi(_FakeApi):
    def run_job(self, **kwargs: Any) -> _FakeJob:
        job = super().run_job(**kwargs)
        job.space_id = "unexpected/space"
        return job


class _TamperedLabelsApi(_FakeApi):
    def run_job(self, **kwargs: Any) -> _FakeJob:
        job = super().run_job(**kwargs)
        job.labels = {"unexpected": "metadata"}
        return job


def test_launch_spec_pins_exact_carbon_revision_and_never_writes_work() -> None:
    spec = launch.build_launch_spec(source_commit="a" * 40, run_attempt=2)
    payload = spec.public_payload()

    assert payload["image"] == launch.CONTAINER_IMAGE
    assert payload["flavor"] == "h200"
    assert payload["labels"] == {"purpose": "geno-lewm-v03-cache-h200-proof"}
    assert payload["namespace"] == "abdelstark"
    assert "secret_names" not in payload
    assert payload["volumes"] == [
        {
            "type": "model",
            "source": "HuggingFaceBio/Carbon-500M",
            "mount_path": "/carbon",
            "revision": "5d31d59b3c845b288a13aedb1358934196852eec",
            "read_only": True,
            "path": None,
        }
    ]
    assert spec.environment["CARBON_REPOSITORY"] == spec.volume.source
    assert spec.environment["CARBON_REVISION"] == spec.volume.revision
    assert spec.environment["RUN_ATTEMPT"] == "2"
    inner_command = spec.command[2]
    assert re.search(r"(?<![A-Za-z0-9_])/work(?![A-Za-z0-9_])", inner_command) is None
    assert 'test "$(cd /carbon && pwd -P)" = /carbon' in inner_command
    workspace_guard = inner_command.index("test ! -L /workspace")
    clone = inner_command.index("git clone https://github.com/AbdelStark/GenoLeWM.git")
    clean = inner_command.index('test -z "$(git status --porcelain=v1 --untracked-files=all)"')
    sync = inner_command.index("uv sync --frozen --extra train --extra evidence")
    execute = inner_command.index("exec uv run --no-sync bash tools/jobs/v03_cache_h200_proof.sh")
    assert workspace_guard < clone < clean < sync < execute


def test_launch_sends_revision_bearing_volume_and_accepts_matching_jobinfo() -> None:
    spec = launch.build_launch_spec(source_commit="b" * 40, run_attempt=3)
    api = _FakeApi(returned_revision=launch.CARBON_REVISION)

    job = launch.submit_exact_revision_job(
        spec,
        token="secret-token",
        api=api,
        volume_class=_FakeVolume,
    )

    assert job.id == "fake-job"
    assert api.run_kwargs is not None
    requested = api.run_kwargs["volumes"][0]
    assert requested.revision == launch.CARBON_REVISION
    assert requested.read_only is True
    assert api.run_kwargs["env"]["CARBON_REVISION"] == launch.CARBON_REVISION
    assert api.run_kwargs["labels"] == {"purpose": "geno-lewm-v03-cache-h200-proof"}
    assert api.run_kwargs["secrets"] == {"HF_TOKEN": "secret-token"}
    assert api.canceled == []


def test_launch_cancels_jobinfo_that_drops_exact_volume_revision() -> None:
    spec = launch.build_launch_spec(source_commit="c" * 40, run_attempt=4)
    api = _FakeApi(returned_revision=None)

    with pytest.raises(RuntimeError, match="differs from the exact Carbon revision"):
        launch.submit_exact_revision_job(
            spec,
            token="secret-token",
            api=api,
            volume_class=_FakeVolume,
        )

    assert api.canceled == [("fake-job", "abdelstark")]


def test_launch_cancels_jobinfo_that_changes_public_environment() -> None:
    spec = launch.build_launch_spec(source_commit="e" * 40, run_attempt=5)
    api = _TamperedEnvironmentApi(returned_revision=launch.CARBON_REVISION)

    with pytest.raises(RuntimeError, match="exact observable launch contract"):
        launch.submit_exact_revision_job(
            spec,
            token="secret-token",
            api=api,
            volume_class=_FakeVolume,
        )

    assert api.canceled == [("fake-job", "abdelstark")]


def test_launch_cancels_jobinfo_that_adds_command_arguments() -> None:
    spec = launch.build_launch_spec(source_commit="f" * 40, run_attempt=6)
    api = _TamperedArgumentsApi(returned_revision=launch.CARBON_REVISION)

    with pytest.raises(RuntimeError, match="exact observable launch contract"):
        launch.submit_exact_revision_job(
            spec,
            token="secret-token",
            api=api,
            volume_class=_FakeVolume,
        )

    assert api.canceled == [("fake-job", "abdelstark")]


def test_launch_cancels_jobinfo_that_adds_volume_subpath() -> None:
    spec = launch.build_launch_spec(source_commit="1" * 40, run_attempt=7)
    api = _TamperedVolumePathApi(returned_revision=launch.CARBON_REVISION)

    with pytest.raises(RuntimeError, match="exact Carbon revision"):
        launch.submit_exact_revision_job(
            spec,
            token="secret-token",
            api=api,
            volume_class=_FakeVolume,
        )

    assert api.canceled == [("fake-job", "abdelstark")]


def test_launch_cancels_jobinfo_that_adds_space_image() -> None:
    spec = launch.build_launch_spec(source_commit="2" * 40, run_attempt=8)
    api = _TamperedSpaceImageApi(returned_revision=launch.CARBON_REVISION)

    with pytest.raises(RuntimeError, match="exact observable launch contract"):
        launch.submit_exact_revision_job(
            spec,
            token="secret-token",
            api=api,
            volume_class=_FakeVolume,
        )

    assert api.canceled == [("fake-job", "abdelstark")]


def test_launch_cancels_jobinfo_that_adds_labels() -> None:
    spec = launch.build_launch_spec(source_commit="3" * 40, run_attempt=9)
    api = _TamperedLabelsApi(returned_revision=launch.CARBON_REVISION)

    with pytest.raises(RuntimeError, match="exact observable launch contract"):
        launch.submit_exact_revision_job(
            spec,
            token="secret-token",
            api=api,
            volume_class=_FakeVolume,
        )

    assert api.canceled == [("fake-job", "abdelstark")]


def test_dry_run_is_secret_free_and_does_not_require_hub_runtime(capsys: Any) -> None:
    assert (
        launch.main(
            [
                "--source-commit",
                "d" * 40,
                "--run-attempt",
                "5",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"revision": "5d31d59b3c845b288a13aedb1358934196852eec"' in output
    assert "HF_TOKEN" not in output
    assert "secret-token" not in output
