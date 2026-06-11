# SPDX-License-Identifier: Apache-2.0
"""Regression tests for public documentation claim boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
ARCHITECTURE = REPO_ROOT / "ARCHITECTURE.md"
CONTRIBUTING = REPO_ROOT / "CONTRIBUTING.md"
DOCS_INDEX = REPO_ROOT / "docs" / "index.md"
DOCS_QUICKSTART = REPO_ROOT / "docs" / "quickstart.md"
DOCS_FAQ = REPO_ROOT / "docs" / "faq.md"
DOCS_GLOSSARY = REPO_ROOT / "docs" / "glossary.md"
DOCS_PUBLIC_API = REPO_ROOT / "docs" / "api" / "public-surface.md"
RELEASE_SIGNING_KEYS = REPO_ROOT / "docs" / "release" / "signing-keys.md"
HF_MODEL_CARD = REPO_ROOT / "docs" / "release" / "huggingface-model-card.md"
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
        DOCS_PUBLIC_API,
        HF_MODEL_CARD,
        RELEASE_SIGNING_KEYS,
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


def test_public_docs_do_not_reintroduce_agent_spec_or_roadmap_scaffolding() -> None:
    obsolete_paths = (
        "AGENTS.md",
        "ROADMAP.md",
        "SPEC.md",
        "SPECIFICATION.md",
        "rfcs",
        "docs/spec",
        "docs/roadmap",
        "docs/design-decisions.md",
    )
    for rel in obsolete_paths:
        assert not (REPO_ROOT / rel).exists(), rel

    public_docs = (
        README,
        ARCHITECTURE,
        DOCS_INDEX,
        DOCS_FAQ,
        DOCS_GLOSSARY,
        DOCS_PUBLIC_API,
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in public_docs)
    forbidden = (
        "Implementation tracker",
        "Project scope and goals",
        "Phase 0",
        "Write an RFC",
        "docs/spec/",
        "rfcs/",
    )
    for fragment in forbidden:
        assert fragment not in combined


def test_release_commands_remain_documented() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in (README, CONTRIBUTING))
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
    combined = "\n".join(path.read_text(encoding="utf-8") for path in (README, DOCS_QUICKSTART))
    assert "Fixture outputs are test evidence, not model results" in combined
    assert "fixture smoke" in combined
    assert "not model results" in combined
    assert "real model quality" not in combined


def test_package_metadata_preserves_research_boundary() -> None:
    metadata = PYPROJECT.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    assert '"Development Status :: 3 - Alpha"' in metadata
    assert '"Intended Audience :: Science/Research"' in metadata
    assert '"Intended Audience :: Developers"' in metadata
    assert "Healthcare Industry" not in metadata
    assert "Clinical" not in metadata
    assert "It is not a diagnostic device" in readme


def test_sdist_includes_public_release_assets_not_agent_scaffolding() -> None:
    metadata = PYPROJECT.read_text(encoding="utf-8")
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
        '"/examples"',
        '"/paper"',
        '"/spaces"',
        '"/ARCHITECTURE.md"',
        '"/README.md"',
    )
    forbidden_fragments = (
        '"/AGENTS.md"',
        '"/ROADMAP.md"',
        '"/SPEC.md"',
        '"/SPECIFICATION.md"',
        '"/rfcs"',
    )

    for fragment in required_fragments:
        assert fragment in sdist_block
    for fragment in forbidden_fragments:
        assert fragment not in sdist_block
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
