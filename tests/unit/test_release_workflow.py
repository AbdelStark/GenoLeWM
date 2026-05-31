# SPDX-License-Identifier: Apache-2.0
"""Static checks for the PyPI release workflow."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "release-pypi.yml"


def test_release_pypi_workflow_is_the_trusted_publisher_path() -> None:
    assert WORKFLOW.exists()
    assert not (ROOT / ".github" / "workflows" / "release.yml").exists()


def test_release_pypi_workflow_builds_from_committed_lockfile() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert (ROOT / "uv.lock").exists()
    assert "uv==${UV_VERSION}" in text
    assert "uv lock --check" in text
    assert "uv sync --locked --extra dev" in text
    assert "uv run python -m build" in text
    assert "uv run twine check --strict dist/*" in text


def test_release_pypi_workflow_uses_oidc_publish_and_sigstore_attestation() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "tags:" in text
    assert '"v[0-9]+.[0-9]+.[0-9]+"' in text
    assert "pypa/gh-action-pypi-publish@release/v1" in text
    assert "actions/attest-build-provenance@v4.1.0" in text
    assert "id-token: write" in text
    assert "attestations: write" in text
    assert "dist/SHA256SUMS" in text
    assert "gh attestation verify --repo" in text


def test_release_pypi_workflow_publishes_pypi_only_from_tags() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    publish_job = text.split("  publish-pypi:", maxsplit=1)[1].split(
        "  github-release:", maxsplit=1
    )[0]
    assert "if: startsWith(github.ref, 'refs/tags/v')" in publish_job
    assert "github.event.inputs.target == 'pypi'" not in publish_job
