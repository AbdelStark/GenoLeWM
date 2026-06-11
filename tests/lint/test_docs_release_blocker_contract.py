# SPDX-License-Identifier: Apache-2.0
"""Regression tests for public documentation claim boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
ROADMAP = REPO_ROOT / "ROADMAP.md"
AGENT_CONTEXT = REPO_ROOT / "AGENTS.md"
IMPLEMENTATION_TRACKER = REPO_ROOT / "docs" / "roadmap" / "IMPLEMENTATION.md"
DOCS_INDEX = REPO_ROOT / "docs" / "index.md"
DOCS_QUICKSTART = REPO_ROOT / "docs" / "quickstart.md"
DOCS_FAQ = REPO_ROOT / "docs" / "faq.md"
EXAMPLES_README = REPO_ROOT / "examples" / "README.md"
RELEASE_SIGNING_KEYS = REPO_ROOT / "docs" / "release" / "signing-keys.md"
HF_MODEL_CARD = REPO_ROOT / "docs" / "release" / "huggingface-model-card.md"
RELEASE_SPEC = REPO_ROOT / "docs" / "spec" / "09-release-and-versioning.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_INIT = REPO_ROOT / "geno_lewm" / "__init__.py"


def test_readme_is_research_readme_not_release_chronology() -> None:
    text = README.read_text(encoding="utf-8")
    required = (
        "# GenoLeWM",
        "Action-conditioned latent world models for genomic edits.",
        "## Install",
        "python -m pip install geno-lewm",
        "## What Ships",
        "## Public Artifacts",
        "## Results",
        "## Training Pipeline",
        "## Evaluation And Benchmarks",
        "## Demo Pipeline",
        "## Quality Gates",
        "## Limitations",
        "It is not a diagnostic device",
        "checksum provenance",
    )
    forbidden = (
        "## Reader Map",
        "## First Experiment Evidence",
        "### Release Evidence Matrix",
        "## Paper-Ready Checklist",
        "## v0.2 Readiness Work",
        "Completed v0.1 release gates",
        "| Gate | Issue |",
        "first Python package release target",
        "after the `v0.2.1` tag workflow publishes",
    )

    for fragment in required:
        assert fragment in text
    for fragment in forbidden:
        assert fragment not in text
    assert len(text.splitlines()) < 340


def test_readme_preserves_measured_results_and_boundaries() -> None:
    text = README.read_text(encoding="utf-8")
    required = (
        "AUROC `0.734375`",
        "AP `0.8529761904761904`",
        "balanced accuracy `0.75`",
        "Spearman rho `0.14919354838709678`",
        "K=20 speedup `2.4732225135799566`",
        "best_distance=23.656930390534644",
        "mixed or negative",
        "Do not cite it as broad superiority over Carbon",
        "not useful-planning evidence",
        "No clinical utility claim",
    )
    for fragment in required:
        assert fragment in text


def test_public_docs_match_published_package_state() -> None:
    public_docs = (
        README,
        DOCS_INDEX,
        DOCS_FAQ,
        DOCS_QUICKSTART,
        HF_MODEL_CARD,
        RELEASE_SIGNING_KEYS,
        RELEASE_SPEC,
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in public_docs)
    normalized = " ".join(combined.split())
    forbidden = (
        "first PyPI release has not been cut yet",
        "after the `v0.2.1` tag workflow publishes",
        "0.2.1 release target",
        "hosted package-publication completion",
        "no first PyPI package tag has been cut",
        "PyPI | Stable",
        "HuggingFace Hub | Stable",
        "GitHub releases | Stable",
    )

    assert not _package_version().is_dev
    assert "python -m pip install geno-lewm" in normalized
    assert "0.2.1 published" in RELEASE_SPEC.read_text(encoding="utf-8")
    for fragment in forbidden:
        assert fragment not in combined


def test_huggingface_model_card_is_model_documentation_not_artifact_listing() -> None:
    text = HF_MODEL_CARD.read_text(encoding="utf-8")
    required = (
        "# GenoLeWM checkpoint and evidence bundle",
        "## Which Checkpoint Should I Use?",
        "## Scoring Contract",
        "WindowMismatchError",
        "## v0.2.1 Benchmark Evidence",
        "K=20 measured 2.47x against a 5x target",
        "best_distance=23.656930390534644",
        "## Troubleshooting",
        "paper.serious-completion.md",
        "No clinical utility claim",
        "not a standard `transformers.AutoModel.from_pretrained()` checkpoint",
    )
    forbidden = (
        "clinically validated",
        "deployment ready",
        "provides privacy assurance",
        "establishes broad superiority",
        "GenoLeWM broadly outperforms Carbon",
    )

    for fragment in required:
        assert fragment in text
    lower = text.lower()
    for fragment in forbidden:
        assert fragment.lower() not in lower


def test_context_docs_are_current_and_concise() -> None:
    docs = {
        "AGENTS.md": AGENT_CONTEXT.read_text(encoding="utf-8"),
        "ROADMAP.md": ROADMAP.read_text(encoding="utf-8"),
        "docs/roadmap/IMPLEMENTATION.md": IMPLEMENTATION_TRACKER.read_text(encoding="utf-8"),
    }
    for name, text in docs.items():
        assert "geno-lewm==0.2.1" in text or "0.2.1" in text
        assert "K=20" in text
        assert "planning" in text.lower()
        assert "checksum provenance" in text
        assert "uv run pytest" in text
        assert len(text.splitlines()) < 180, name

    combined = "\n".join(docs.values())
    forbidden = (
        "Current Direction",
        "Not Done Yet",
        "High-Priority Work",
        "Issue Anchors",
        "first-publication",
        "first package-release target",
        "protected `0.2.1` Python package tag publication",
    )
    for fragment in forbidden:
        assert fragment not in combined


def test_release_commands_remain_documented() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (README, ROADMAP, AGENT_CONTEXT, IMPLEMENTATION_TRACKER)
    )
    required = (
        "python -m tools.release.dataset_snapshot",
        "geno-lewm-train --carbon-preflight",
        "geno-lewm-train --carbon-train --package-release-run",
        "geno-lewm-eval",
        "geno-lewm-eval-all",
        "python -m tools.release.v02_benchmark_suite",
        "python tools/demo/terminal_inference.py",
        "python -m tools.release.check_sdist_assets dist/*.tar.gz",
        "uv run python -m tools.lint.check_scope_language",
    )
    for fragment in required:
        assert fragment in combined


def test_ci_fixture_gates_keep_claim_boundaries() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (README, AGENT_CONTEXT, ROADMAP, IMPLEMENTATION_TRACKER)
    )
    assert "Fixture outputs are test evidence, not model results" in combined
    assert "fixture smoke" in combined
    assert "not model-quality evidence" in combined or "not model results" in combined
    assert "real model quality" not in combined


def test_examples_readme_names_rollout_and_planning_notebook_blockers() -> None:
    text = EXAMPLES_README.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    required = (
        "Rollout and planning notebooks remain blocked until measured release evidence exists",
        "#97 requires rollout-fidelity rows with documented cosine-similarity targets",
        "#98 requires planner latency and useful-planning boundary evidence",
        "Fixture smoke outputs are test evidence, not model results",
        "`04_multi_edit_rollout.ipynb`",
        "release-backed rollout-state examples",
        "encoder-ground-truth comparisons",
        "`05_planning_minimal_edits.ipynb`",
        "does not prove useful planning behavior",
    )
    forbidden = (
        "planning notebooks remain planned until the planner is implemented",
        "Demonstrates the world-model claim",
        "Demonstrates the planner",
    )

    for fragment in required:
        assert fragment in normalized
    for fragment in forbidden:
        assert fragment not in normalized


def test_package_metadata_preserves_research_boundary() -> None:
    metadata = PYPROJECT.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    assert '"Development Status :: 3 - Alpha"' in metadata
    assert '"Intended Audience :: Science/Research"' in metadata
    assert '"Intended Audience :: Developers"' in metadata
    assert "Healthcare Industry" not in metadata
    assert "Clinical" not in metadata
    assert "It is not a diagnostic device" in readme


def test_sdist_includes_release_contract_assets() -> None:
    metadata = PYPROJECT.read_text(encoding="utf-8")
    release_spec = RELEASE_SPEC.read_text(encoding="utf-8")
    sdist_block = metadata.split("[tool.hatch.build.targets.sdist]", maxsplit=1)[1].split(
        "\n[",
        maxsplit=1,
    )[0]
    required_fragments = (
        '"/bench"',
        '"/configs"',
        '"/geno_lewm"',
        '"/tools"',
        '"/docs"',
        '"/rfcs"',
        '"/examples"',
        '"/AGENTS.md"',
        '"/README.md"',
        '"/ROADMAP.md"',
    )

    for fragment in required_fragments:
        assert fragment in sdist_block
    assert "The source distribution must" in release_spec
    assert "include the release-critical repo assets" in release_spec
    assert "`tools.release.check_sdist_assets`" in release_spec
    assert "python -m tools.release.check_sdist_assets dist/*.tar.gz" in README.read_text(
        encoding="utf-8"
    )


def _package_version() -> _Version:
    """Read the package version without importing optional runtime modules."""
    module = ast.parse(PACKAGE_INIT.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__version__"
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    return _Version(node.value.value)
    raise AssertionError("__version__ assignment not found")


class _Version(str):
    @property
    def is_dev(self) -> bool:
        return ".dev" in self
