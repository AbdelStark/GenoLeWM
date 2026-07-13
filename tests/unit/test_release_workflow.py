# SPDX-License-Identifier: Apache-2.0
"""Static checks for release workflows."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT = ROOT / "pyproject.toml"
PYPI_WORKFLOW = ROOT / ".github" / "workflows" / "release-pypi.yml"
HUB_DRY_RUN_WORKFLOW = ROOT / ".github" / "workflows" / "release-hub-dry-run.yml"
HUB_PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "release-hub-publish.yml"


def test_release_pypi_workflow_is_the_trusted_publisher_path() -> None:
    assert PYPI_WORKFLOW.exists()
    assert not (ROOT / ".github" / "workflows" / "release.yml").exists()


def test_release_pypi_workflow_builds_from_committed_lockfile() -> None:
    text = PYPI_WORKFLOW.read_text(encoding="utf-8")

    assert (ROOT / "uv.lock").exists()
    assert "uv==${UV_VERSION}" in text
    assert "uv lock --check" in text
    assert "uv sync --locked --extra dev" in text
    assert "uv run python -m tools.release.dataset_snapshot" in text
    assert "--spec-json configs/first_experiment/dataset-snapshot-snv.json" in text
    assert "--check-spec" in text
    assert "uv run python -m build" in text
    assert "uv run twine check --strict dist/*" in text
    assert "uv run python -m tools.release.check_sdist_assets dist/*.tar.gz" in text


def test_ci_build_workflow_checks_sdist_release_assets() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "python -m build" in text
    assert "twine check dist/*" in text
    assert "python -m tools.release.check_sdist_assets dist/*.tar.gz" in text
    assert "python -m tests.wheel_membership_smoke prepare" in text
    assert 'python -I "$GITHUB_WORKSPACE/tests/wheel_membership_smoke.py"' in text


def test_ci_build_workflow_smokes_console_scripts_outside_the_checkout() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    smoke_step = text.split("      - name: Smoke install", maxsplit=1)[1].split(
        "      - name: Upload dist artifacts", maxsplit=1
    )[0]

    assert 'repo_root="$PWD"' in smoke_step
    assert 'smoke_dir="$(mktemp -d)"' in smoke_step
    assert 'cd "$smoke_dir"' in smoke_step
    assert 'python "$repo_root/tests/wheel_console_scripts_smoke.py"' in smoke_step


def test_ci_type_dependencies_are_bounded_to_validated_versions() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    dev_dependencies = text.split("dev = [", maxsplit=1)[1].split("]", maxsplit=1)[0]

    assert '"mypy>=1.10,<2.3"' in dev_dependencies
    assert '"pyarrow>=15,<25"' in dev_dependencies


def test_ci_workflow_checks_first_experiment_dataset_spec() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "python -m tools.release.dataset_snapshot" in text
    assert "--spec-json configs/first_experiment/dataset-snapshot-snv.json" in text
    assert "--check-spec" in text


def test_ci_workflow_runs_dedicated_ml_smoke_gate() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "  ml-smoke:" in text
    assert "name: ML smoke (fixture-backed)" in text
    assert "timeout-minutes: 10" in text
    assert 'python-version: "3.12"' in text
    assert 'python -m pip install -e ".[dev]"' in text
    assert "pytest tests/ml -q --tb=long --durations=10" in text
    assert "needs: [lint, types, gates, tests, ml-smoke, eval-smoke, build, docs, paper]" in text
    assert "needs.ml-smoke.result != 'success'" in text


def test_ci_windows_coverage_excludes_the_posix_cache_modules() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    pytest_step = text.split("      - name: pytest", maxsplit=1)[1].split(
        "      - name: Upload pytest output", maxsplit=1
    )[0]
    windows_branch, non_windows_branch = pytest_step.split("          else", maxsplit=1)

    assert "--cov-fail-under=0" in windows_branch
    assert "coverage report --show-missing --fail-under=84" in windows_branch
    assert (
        '--omit="*/encoder/cache.py,*/encoder/cache_build.py,*/cli/cache_windows.py"'
        in windows_branch
    )
    assert "coverage report" not in non_windows_branch
    assert "--cov-report=xml" in non_windows_branch
    assert "--omit=" not in non_windows_branch
    assert pytest_step.count('tee "$RUNNER_TEMP/pytest.out"') == 2
    assert "tee pytest.out" not in pytest_step

    changed_files_gate = text.split("      - name: Changed-files coverage gate", maxsplit=1)[
        1
    ].split("  ml-smoke:", maxsplit=1)[0]
    assert "matrix.os == 'ubuntu-latest'" in changed_files_gate
    assert "--threshold 0.84" in changed_files_gate

    upload_step = text.split("      - name: Upload pytest output", maxsplit=1)[1].split(
        "      - name: Upload coverage to Codecov", maxsplit=1
    )[0]
    assert "path: ${{ runner.temp }}/pytest.out" in upload_step


def test_ci_workflow_runs_dedicated_eval_smoke_gate() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "  eval-smoke:" in text
    assert "name: Eval smoke (fixture-backed)" in text
    assert "timeout-minutes: 10" in text
    assert "python -m tools.ci.eval_smoke_gate" in text
    assert "--work-dir .eval-smoke" in text
    assert "--summary-json .eval-smoke/eval_smoke_summary.json" in text
    assert "name: eval-smoke-summary" in text
    assert "needs: [lint, types, gates, tests, ml-smoke, eval-smoke, build, docs, paper]" in text
    assert "needs.eval-smoke.result != 'success'" in text


def test_ci_workflow_builds_checked_paper_pdf_artifact() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "  paper:" in text
    assert "name: Paper PDF (tectonic)" in text
    assert "wtfjoke/setup-tectonic@8a63d072f8390efdff59da7fa08aa49e3c1f5e1b" in text
    assert 'tectonic-version: "0.16.9"' in text
    assert "apt-get install -y tectonic" not in text
    assert "make -C paper" in text
    assert "python -m tools.release.paper_tex" in text
    assert "--output paper/paper_tex_build_report.json" in text
    assert "name: paper-pdf" in text
    assert "paper/main.pdf" in text
    assert "paper/paper_tex_build_report.json" in text
    assert "needs.paper.result != 'success'" in text


def test_release_pypi_workflow_uses_oidc_publish_and_sigstore_build_provenance() -> None:
    text = PYPI_WORKFLOW.read_text(encoding="utf-8")

    assert "tags:" in text
    assert '"v[0-9]+.[0-9]+.[0-9]+"' in text
    assert "pypa/gh-action-pypi-publish@release/v1" in text
    assert "actions/attest-build-provenance@v4.1.1" in text
    assert "id-token: write" in text
    assert "attestations: write" in text
    assert "dist/SHA256SUMS" in text
    assert "GitHub/Sigstore build provenance" in text
    assert "provenance verification command" in text


def test_release_pypi_workflow_publishes_pypi_only_from_tags() -> None:
    text = PYPI_WORKFLOW.read_text(encoding="utf-8")

    publish_job = text.split("  publish-pypi:", maxsplit=1)[1].split(
        "  github-release:", maxsplit=1
    )[0]
    assert "if: startsWith(github.ref, 'refs/tags/v')" in publish_job
    assert "github.event.inputs.target == 'pypi'" not in publish_job


def test_release_hub_dry_run_workflow_is_non_publishing_gate() -> None:
    text = HUB_DRY_RUN_WORKFLOW.read_text(encoding="utf-8")

    assert HUB_DRY_RUN_WORKFLOW.exists()
    assert "name: Release Hub Dry Run" in text
    assert "workflow_dispatch:" in text
    assert "HF_TOKEN" not in text
    assert "huggingface-cli upload" not in text
    assert "tools.release.paper_package" in text
    assert "tools.release.hub_release" in text
    assert "tools.release.release_candidate" in text
    assert "actions/upload-artifact@v4" in text


def test_release_hub_dry_run_workflow_uses_locked_release_environment() -> None:
    text = HUB_DRY_RUN_WORKFLOW.read_text(encoding="utf-8")

    assert "uv==${UV_VERSION}" in text
    assert "uv lock --check" in text
    assert "uv sync --locked --extra dev" in text
    assert 'python-version: "3.12"' in text


def test_release_hub_dry_run_workflow_requires_public_artifact_inputs() -> None:
    text = HUB_DRY_RUN_WORKFLOW.read_text(encoding="utf-8")

    for name in (
        "model_dir:",
        "dataset_dir:",
        "demo_dir:",
        "repo_id:",
        "dataset_url:",
        "demo_url:",
        "commit_sha:",
    ):
        assert name in text
    assert "paper_url:" in text
    assert "allow_fixture_manifest:" in text
    assert "ALLOW_FIXTURE_MANIFEST" in text
    assert "--allow-fixture-manifest" in text
    assert "--skip-public-link-check" in text
    assert "release_candidate_report.json" in text


def test_release_hub_publish_workflow_is_credentialed_manual_gate() -> None:
    text = HUB_PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert HUB_PUBLISH_WORKFLOW.exists()
    assert "name: Release Hub Publish" in text
    assert "workflow_dispatch:" in text
    assert "environment: release" in text
    assert "contents: write" in text
    assert "HF_TOKEN: ${{ secrets.HF_TOKEN }}" in text
    assert "GH_TOKEN: ${{ github.token }}" in text
    assert "tools.release.hub_publish" in text
    assert "tools.release.clean_machine_demo" in text
    assert "tools.release.publication_report" in text
    assert "--skip-public-link-check" not in text
    assert "--allow-fixture-manifest" not in text
    assert "--no-require-native-runtime" not in text
    assert "hub_publish_report.json" in text
    assert "release_candidate_report.json" in text
    assert "clean-machine-public-replay/clean_machine_demo_report.json" in text
    assert "publication_evidence_report.json" in text
    assert "uv sync --locked --extra dev --extra train --extra eval --extra deploy" in text


def test_release_hub_publish_workflow_replays_public_demo_after_upload() -> None:
    text = HUB_PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert "demo_backend:" in text
    assert "demo_batch_size:" in text
    assert "DEMO_BACKEND: ${{ inputs.demo_backend }}" in text
    assert "DEMO_BATCH_SIZE: ${{ inputs.demo_batch_size }}" in text
    replay_step = text.split("Replay public terminal demo from uploaded artifacts", maxsplit=1)[
        1
    ].split("Bind publication evidence", maxsplit=1)[0]
    assert "HF_TOKEN: ${{ secrets.HF_TOKEN }}" in replay_step
    assert "GH_TOKEN: ${{ github.token }}" in replay_step
    publish_position = text.index("tools.release.hub_publish")
    replay_position = text.index("tools.release.clean_machine_demo")
    publication_position = text.index("tools.release.publication_report")
    public_upload_position = text.index("Upload public publication evidence assets")
    artifact_upload_position = text.index("actions/upload-artifact@v4")
    assert publish_position < replay_position < publication_position < public_upload_position
    assert public_upload_position < artifact_upload_position
    assert "--release-candidate-report release_candidate_report.json" in text
    assert "--output-dir clean-machine-public-replay" in text
    assert '--backend "$DEMO_BACKEND"' in text
    assert '--batch-size "$DEMO_BATCH_SIZE"' in text
    assert "clean-machine-public-replay" in text
    assert "--output publication_evidence_report.json" in text


def test_release_hub_publish_workflow_uploads_public_evidence_assets() -> None:
    text = HUB_PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    bind_step = text.split("Bind publication evidence asset manifest", maxsplit=1)[1].split(
        "Upload public publication evidence assets",
        maxsplit=1,
    )[0]
    evidence_step = text.split("Upload public publication evidence assets", maxsplit=1)[1].split(
        "Upload publication reports",
        maxsplit=1,
    )[0]
    assert "DEMO_URL: ${{ inputs.demo_url }}" in bind_step
    assert "tools.release.publication_assets" in bind_step
    assert '--demo-url "$DEMO_URL"' in bind_step
    assert "publication-evidence-target.env" in bind_step
    assert "hub_release_plan.json" in bind_step
    assert "release_candidate_report.json" in bind_step
    assert "hub_publish_report.json" in bind_step
    assert "clean-machine-public-replay/clean_machine_demo_report.json" in bind_step
    assert "clean-machine-public-replay/replay" in bind_step
    assert "publication_evidence_report.json" in bind_step
    assert "publication_evidence_assets.json" in bind_step
    assert "GH_TOKEN: ${{ github.token }}" in evidence_step
    assert 'Path("publication_evidence_assets.json").read_text()' in evidence_step
    assert 'command = payload.get("upload_command")' in evidence_step
    assert 'command[:3] != ["gh", "release", "upload"]' in evidence_step
    assert "subprocess.run(command, check=True)" in evidence_step
