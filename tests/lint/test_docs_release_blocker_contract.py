# SPDX-License-Identifier: Apache-2.0
"""Regression tests for first paper/demo release blocker documentation."""

from __future__ import annotations

from pathlib import Path

from tools.release.issue_refs import (
    DATASET_ISSUE,
    DEMO_ISSUE,
    EVAL_ISSUE,
    MODEL_RELEASE_ISSUE,
    PAPER_ISSUE,
    TRAINING_ISSUE,
    issue_ref_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
ROADMAP = REPO_ROOT / "ROADMAP.md"
AGENT_CONTEXT = REPO_ROOT / "AGENTS.md"
IMPLEMENTATION_TRACKER = REPO_ROOT / "docs" / "roadmap" / "IMPLEMENTATION.md"
DOCS_INDEX = REPO_ROOT / "docs" / "index.md"
DOCS_FAQ = REPO_ROOT / "docs" / "faq.md"
DOCS_GLOSSARY = REPO_ROOT / "docs" / "glossary.md"
DOCS_QUICKSTART = REPO_ROOT / "docs" / "quickstart.md"
RELEASE_SIGNING_KEYS = REPO_ROOT / "docs" / "release" / "signing-keys.md"
TESTING_SPEC = REPO_ROOT / "docs" / "spec" / "07-testing-strategy.md"
RELEASE_SPEC = REPO_ROOT / "docs" / "spec" / "09-release-and-versioning.md"
SPEC_OVERVIEW = REPO_ROOT / "docs" / "spec" / "00-overview.md"
SPEC_ARCHITECTURE = REPO_ROOT / "docs" / "spec" / "01-architecture.md"
SPEC_SECURITY = REPO_ROOT / "docs" / "spec" / "06-security.md"
SPEC_PERFORMANCE = REPO_ROOT / "docs" / "spec" / "08-performance-budget.md"
SPEC_GLOSSARY = REPO_ROOT / "docs" / "spec" / "10-glossary.md"
DEPLOYMENT_RFC = REPO_ROOT / "rfcs" / "0010-on-device-personal-genome-deployment.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"

RELEASE_GATE_ISSUES = {
    DATASET_ISSUE: "Dataset snapshot and data card",
    TRAINING_ISSUE: "First Carbon-backed run",
    EVAL_ISSUE: "Paper-ready results report",
    DEMO_ISSUE: "Terminal real-inference showcase",
    PAPER_ISSUE: "First experiment paper package",
    MODEL_RELEASE_ISSUE: "Model checkpoint Hub release",
}


def test_readme_and_roadmap_publish_completed_v0_1_release_map() -> None:
    """Top-level public docs should expose completed v0.1 release evidence."""
    assert tuple(RELEASE_GATE_ISSUES) == (
        DATASET_ISSUE,
        TRAINING_ISSUE,
        EVAL_ISSUE,
        DEMO_ISSUE,
        PAPER_ISSUE,
        MODEL_RELEASE_ISSUE,
    )
    for path in (README, ROADMAP):
        text = path.read_text(encoding="utf-8")
        assert "Completed v0.1" in text
        for number, gate in RELEASE_GATE_ISSUES.items():
            assert gate in text
            assert _issue_link(number) in text
            assert _issue_url(number) in text
        assert "v0.1" in text
        assert "public" in text
        assert "v0.2" in text


def test_agent_context_and_tracker_name_the_same_release_gate_issues() -> None:
    """Agent context and implementation tracker should stay aligned with public docs."""
    agent_context = AGENT_CONTEXT.read_text(encoding="utf-8")
    tracker = IMPLEMENTATION_TRACKER.read_text(encoding="utf-8")

    for number in RELEASE_GATE_ISSUES:
        assert f"#{number}" in agent_context
    assert "#163 through #167 tracked the v0.1 paper/demo chain" in tracker
    assert "#101 tracked the v0.1 model Hub release" in tracker


def test_agent_context_defines_release_evidence_rules() -> None:
    """Agents should not confuse local contracts with future release evidence."""
    text = AGENT_CONTEXT.read_text(encoding="utf-8")
    required_fragments = (
        "## Release Evidence Rules",
        "local release tooling as contracts",
        "v0.1 exercised these gates",
        "| Evidence gate | Local contract that exists | v0.1 status and future boundary |",
        "python -m tools.release.dataset_snapshot",
        "--check-inputs",
        "dataset_input_check_report.json",
        "stale dataset input-check evidence",
        "dataset core-file evidence",
        "python -m tools.release.dataset_package",
        "geno-lewm-train --carbon-preflight",
        "geno-lewm-train --carbon-train --package-release-run",
        "geno-lewm-eval",
        "geno-lewm-carbon-baseline",
        "geno-lewm-eval-all",
        "python -m bench.inference --release-efficiency",
        "python -m tools.release.check_sdist_assets dist/*.tar.gz",
        "python tools/demo/terminal_inference.py",
        "python -m tools.release.paper_draft",
        "publication_report",
        "Completed and published for v0.1",
        "Completed for the narrow v0.1 release",
        "v0.2",
    )

    for fragment in required_fragments:
        assert fragment in text
    for number in RELEASE_GATE_ISSUES:
        assert _issue_link(number) in text


def test_readme_publish_release_evidence_matrix() -> None:
    """README should distinguish local contracts from v0.1/v0.2 evidence."""
    text = README.read_text(encoding="utf-8")
    required_fragments = (
        "### Release Evidence Matrix",
        "| Evidence artifact | Local contract | Paper-release status |",
        "Green local tooling is necessary",
        "not a substitute",
        "future releases",
        "python -m tools.release.dataset_snapshot",
        "--check-inputs",
        "dataset_input_check_report.json",
        "geno-lewm-train --carbon-preflight",
        "geno-lewm-train --carbon-train --package-release-run",
        "geno-lewm-eval",
        "geno-lewm-carbon-baseline",
        "geno-lewm-eval-all",
        "python -m bench.inference --release-efficiency",
        "python -m tools.release.check_sdist_assets dist/*.tar.gz",
        "python tools/demo/terminal_inference.py",
        "python -m tools.release.clean_machine_demo",
        "replayed `terminal_demo_manifest.json`",
        "replay artifact hash/size drift",
        "python -m tools.release.publication_report",
        "Completed for v0.1",
        "the June 8 v0.2 readiness run added broader reproducibly staged benchmark and rollout evidence",
        "the June 9 #203 rerun applied it to the #202 checkpoint lineage",
        "neither run closes the true #42 K20 speed target",
    )

    for fragment in required_fragments:
        assert fragment in text
    for number in RELEASE_GATE_ISSUES:
        assert _issue_link(number) in text


def test_readme_exposes_reader_map_and_local_contract_commands() -> None:
    """README should be navigable and name runnable local contract checks."""
    text = README.read_text(encoding="utf-8")
    required_fragments = (
        "## Reader Map",
        "| If you want to... | Start here |",
        "[What You Can Run Today](#what-you-can-run-today)",
        "[First Experiment Evidence](#first-experiment-evidence)",
        "[v0.2 Readiness Work](#v02-readiness-work)",
        "## What You Can Run Today",
        "These commands exercise local contracts",
        "they do not replace the public v0.1 artifact",
        "terminal-demo-transcript.md",
        "geno-lewm-verify examples/data/verify_receipt/receipt.json",
        "geno-lewm-train --fixture-smoke --run-dir /tmp/geno-lewm-smoke --steps 50",
        "python -m tools.release.dataset_snapshot --spec-json configs/first_experiment/dataset-snapshot-snv.json --check-spec",
        "uv run python tools/api/snapshot.py check",
        "uv run python -m tools.lint.check_scope_language",
        "uv run mkdocs build --strict",
        "├── bench/               # local benchmark and release-efficiency harnesses",
        "├── configs/             # checked first-experiment training/eval configs",
        "| Lockfile | `uv lock --check` |",
        "| Scope language | `python -m tools.lint.check_scope_language` |",
        "| Dataset spec | `python -m tools.release.dataset_snapshot --spec-json configs/first_experiment/dataset-snapshot-snv.json --check-spec` |",
        "| Release docs contract | `pytest tests/lint/test_docs_release_blocker_contract.py -q` |",
    )

    for fragment in required_fragments:
        assert fragment in text


def test_roadmap_defines_release_evidence_gates() -> None:
    """Roadmap should map local release contracts to completed/future evidence."""
    text = ROADMAP.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    required_fragments = (
        "## Release Evidence Gates",
        "Local tools define the reusable contract",
        "v0.1 release crossed the first-publication line",
        "| Gate | Local contract | v0.1 status and v0.2 boundary |",
        "python -m tools.release.dataset_snapshot",
        "--check-inputs",
        "dataset_input_check_report.json",
        "python -m tools.release.dataset_package",
        "geno-lewm-train --carbon-preflight",
        "geno-lewm-train --carbon-train --package-release-run",
        "geno-lewm-eval",
        "geno-lewm-carbon-baseline",
        "geno-lewm-eval-all",
        "python -m bench.inference --release-efficiency",
        "python tools/demo/terminal_inference.py",
        "python -m tools.release.paper_draft",
        "python -m tools.release.clean_machine_demo",
        "replay artifact hash/size drift",
        "python -m tools.release.publication_report",
        "Completed and published for v0.1",
        "Completed for the narrow v0.1 release",
        "final binder has `ok=true`",
    )

    for fragment in required_fragments:
        assert fragment in text or fragment in normalized
    for number in RELEASE_GATE_ISSUES:
        assert _issue_link(number) in text


def test_implementation_tracker_defines_release_evidence_ledger() -> None:
    """Implementation tracker should keep local contracts separate from future evidence."""
    text = IMPLEMENTATION_TRACKER.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    required_fragments = (
        "## Release Evidence Ledger",
        "Do not use local fixture/tooling evidence alone for v0.2 claims",
        "| Issue | Local contract | v0.1 status and future boundary |",
        "#163 dataset snapshot",
        "#164 first Carbon-backed run",
        "#165 results report",
        "#20 release packaging",
        "#166 terminal showcase",
        "#167/#101 paper and publication",
        "python -m tools.release.dataset_snapshot",
        "--check-inputs",
        "dataset_input_check_report.json",
        "python -m tools.release.dataset_package",
        "geno-lewm-train --carbon-preflight",
        "geno-lewm-train --carbon-train --package-release-run",
        "geno-lewm-eval",
        "geno-lewm-carbon-baseline",
        "geno-lewm-eval-all",
        "python -m bench.inference --release-efficiency",
        "python -m tools.release.check_sdist_assets dist/*.tar.gz",
        "python tools/demo/terminal_inference.py",
        "python -m tools.release.clean_machine_demo",
        "replay artifact hash/size drift",
        "python -m tools.release.paper_draft",
        "python -m tools.release.paper_package",
        "python -m tools.release.hub_release",
        "python -m tools.release.hub_publish",
        "python -m tools.release.publication_report",
        "Completed for v0.1",
        "v0.2",
        "public model, dataset, demo, paper, and final binder links",
    )

    for fragment in required_fragments:
        assert fragment in text or fragment in normalized


def test_ml_smoke_gate_is_documented_as_fixture_backed_ci_gate() -> None:
    """ML smoke docs should describe the hosted gate without paper-quality claims."""
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (README, AGENT_CONTEXT, ROADMAP, IMPLEMENTATION_TRACKER, TESTING_SPEC)
    )
    normalized = " ".join(combined.split())
    required_fragments = (
        "pytest tests/ml -q --tb=long --durations=10",
        "fixture-backed",
        "finite fixture training loss",
        "collapse-health signals",
        "deterministic resume identity",
        "optional torch predictor",
        "ml-smoke",
        "must not require private model or data files",
    )
    forbidden_fragments = (
        "paper-quality evidence",
        "model-quality evidence",
        "real model quality",
    )

    for fragment in required_fragments:
        assert fragment in normalized
    for fragment in forbidden_fragments:
        assert fragment not in normalized


def test_eval_smoke_gate_is_documented_as_generated_fixture_gate() -> None:
    """Eval smoke docs should name the generated fixture gate and claim boundary."""
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (README, AGENT_CONTEXT, ROADMAP, IMPLEMENTATION_TRACKER, TESTING_SPEC)
    )
    normalized = " ".join(combined.split())
    required_fragments = (
        "python -m tools.ci.eval_smoke_gate",
        "score/label JSONL fixtures",
        "geno-lewm-eval",
        "geno-lewm-eval-all",
        "AUROC/AP/balanced-accuracy/baseline-delta thresholds",
        "real checkpoint/dataset evaluation is not attempted",
        "eval-smoke",
        "real_model_path.status=not_attempted",
    )
    forbidden_fragments = (
        "hosted gate proves first-experiment results",
        "hosted gate uses released checkpoints",
        "hosted gate uses private data",
    )

    for fragment in required_fragments:
        assert fragment in normalized
    for fragment in forbidden_fragments:
        assert fragment not in normalized


def test_release_candidate_docs_explain_issue_refs() -> None:
    """Release-candidate docs should route failures to live tracker issues."""
    for path in (
        README,
        ROADMAP,
        AGENT_CONTEXT,
        REPO_ROOT / "docs" / "spec" / "09-release-and-versioning.md",
    ):
        text = path.read_text(encoding="utf-8")
        assert "issue_refs" in text
        for number in RELEASE_GATE_ISSUES:
            assert f"#{number}" in text or _issue_link(number) in text


def test_publication_report_docs_explain_issue_refs() -> None:
    """Final publication evidence docs should route failures to live tracker issues."""
    for path in (
        README,
        ROADMAP,
        AGENT_CONTEXT,
        REPO_ROOT / "docs" / "spec" / "09-release-and-versioning.md",
    ):
        text = path.read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        assert "publication_evidence_report.json" in text
        assert "Its `issues` entries" in normalized or "its `issues` entries" in normalized
        assert "final publication failures" in normalized
        assert "issue_refs" in text
        for number in RELEASE_GATE_ISSUES:
            assert f"#{number}" in text or _issue_link(number) in text


def test_release_blocker_docs_keep_claim_boundaries() -> None:
    """Release blocker docs should preserve current evidence boundaries."""
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (README, ROADMAP, AGENT_CONTEXT, IMPLEMENTATION_TRACKER)
    )
    banned_phrases = [
        " ".join(("independently", "certified")),
        "-".join(("inference", "certification")),
        " ".join(("verifiable", "inference")),
    ]

    assert "checksum provenance" in combined
    assert "runtime assurance mechanisms beyond checksum provenance" in combined
    for phrase in banned_phrases:
        assert phrase not in combined.lower()


def test_public_docs_do_not_claim_unreleased_package_distribution() -> None:
    """Public docs should not advertise package-index installs before first release."""
    public_docs = (
        README,
        DOCS_INDEX,
        DOCS_FAQ,
        DOCS_QUICKSTART,
        RELEASE_SIGNING_KEYS,
        RELEASE_SPEC,
        DEPLOYMENT_RFC,
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in public_docs)
    forbidden_fragments = (
        "pypi/v/geno-lewm",
        "pypi/pyversions",
        "PyPI exists",
        "PyPI | Stable",
        "HuggingFace Hub | Stable",
        "GitHub releases | Stable",
        "pip install " + "geno-lewm",
        "uv pip install " + "geno-lewm",
    )

    for fragment in forbidden_fragments:
        assert fragment not in combined

    normalized = " ".join(combined.split())
    assert "first PyPI release has not been cut yet" in normalized
    assert "install from source" in normalized
    assert "Planned first tag" in RELEASE_SPEC.read_text(encoding="utf-8")


def test_faq_keeps_target_numbers_behind_release_evidence() -> None:
    """FAQ should frame target numbers and local-only behavior as release gates."""
    text = DOCS_FAQ.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    required_fragments = (
        "The v0.1 release reports a narrow chr21 ClinVar slice",
        "The June 8 v0.2 readiness run adds measured GenoLeWM-vs-Carbon evidence",
        "The June 9 #203 rerun applied that suite to the #202 checkpoint lineage",
        "not clinical utility claims or a public v0.2 model release",
        "not a measured GenoLeWM dataset result",
        "The v0.1 model package includes a public `calibration.parquet`",
        "The v0.1 release does not establish separate non-coding performance",
        "The v0.1 release includes an `efficiency_report.json`",
        "Apple Silicon and quantized local runtime targets still need dedicated measurement",
        "The public v0.1 checkpoint has a manifest identity",
        "not every personal-genome import workflow",
    )
    forbidden_fragments = (
        "We report on the same metrics",
        "The current calibration is built",
        "No. The inference runs entirely on your device",
        "There is no telemetry. We do not collect usage data",
        "the model fits in ~600 MB of memory and scores single variants in < 200 ms",
        "Published results referencing",
        "Conversion never leaves your machine",
    )

    for fragment in required_fragments:
        assert fragment in normalized
    for fragment in forbidden_fragments:
        assert fragment not in normalized


def test_public_spec_docs_keep_targets_behind_release_evidence() -> None:
    """Spec docs should label target metrics and privacy behavior as release gates."""
    spec_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            SPEC_OVERVIEW,
            SPEC_ARCHITECTURE,
            SPEC_SECURITY,
            SPEC_PERFORMANCE,
        )
    )
    normalized = " ".join(spec_text.split())
    required_fragments = (
        "The v0.1 paper/demo release is published",
        "v0.1 establishes a narrow first baseline, not broad model quality",
        "measured release gates of single-variant scoring < 200 ms",
        "budgets and targets unless explicitly reported from a measured release artifact",
        "v0.2 measurement before docs can describe them as achieved",
        "v0.1 model package and terminal demo are published",
        "does not establish a general privacy assurance",
        "Release-dependent invariants describe required behavior for published inference paths",
        "Published inference paths perform no network call after first-run setup",
        "The released runtime must make no network call after first-run setup",
        "After the first measured release baseline exists",
    )
    forbidden_fragments = (
        "Numbers here are commitments, not aspirations",
        "CI runs the per-PR benchmarks on a GitHub-Actions-hosted GPU runner",
        "The runtime makes no network call after first-run setup",
        "No inference path performs a network call after first-run setup",
        "The desktop app has no cloud sync, no accounts, no telemetry",
        "Released artifacts are built from pinned dependency lockfiles",
    )

    for fragment in required_fragments:
        assert fragment in normalized
    for fragment in forbidden_fragments:
        assert fragment not in normalized


def test_glossaries_keep_alpha_release_status_clear() -> None:
    """Glossaries should describe released artifacts without overclaiming validity."""
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in (DOCS_GLOSSARY, SPEC_GLOSSARY)
    )
    normalized = " ".join(combined.split())
    required_fragments = (
        "The v0.1 model package ships `calibration.parquet`",
        "population-stratified calibration validity is not established",
        "release scoring paths can emit",
        "output commitment",
        "specific released checkpoint artifact set",
        "completed v0.1 paper/demo release",
        "not broader model-quality evidence",
    )
    forbidden_fragments = (
        "distributed with the GenoLeWM checkpoint",
        "emitted by every inference call",
        "Part of every cache key and every receipt",
        "Globally identifies a specific release.",
    )

    for fragment in required_fragments:
        assert fragment in normalized
    for fragment in forbidden_fragments:
        assert fragment not in normalized


def test_package_metadata_preserves_research_boundary() -> None:
    """Package metadata should not imply clinical or healthcare deployment."""
    metadata = PYPROJECT.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    assert '"Development Status :: 3 - Alpha"' in metadata
    assert '"Intended Audience :: Science/Research"' in metadata
    assert '"Intended Audience :: Developers"' in metadata
    assert "Healthcare Industry" not in metadata
    assert "Clinical" not in metadata
    assert "It is not a diagnostic device" in readme


def test_sdist_includes_release_contract_assets() -> None:
    """Source distributions should include assets used by advertised release gates."""
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
    assert "`bench/`, `configs/first_experiment/`, `tools/`, `docs/`" in release_spec
    assert "README, ROADMAP, and AGENTS" in release_spec
    assert "The sdist includes release tooling, benchmark harnesses" in release_spec
    assert "checked dataset snapshot spec" in release_spec
    assert "`tools.release.check_sdist_assets`" in release_spec
    for path in (AGENT_CONTEXT, ROADMAP, IMPLEMENTATION_TRACKER):
        text = path.read_text(encoding="utf-8")
        assert "source-distribution inventory checking" in text
        assert "python -m tools.release.check_sdist_assets dist/*.tar.gz" in text
    assert (
        "| Package build | `python -m build && twine check --strict dist/* && "
        "python -m tools.release.check_sdist_assets dist/*.tar.gz` |"
        in README.read_text(encoding="utf-8")
    )


def _issue_link(number: int) -> str:
    return f"[#{number}]({_issue_url(number)})"


def _issue_url(number: int) -> str:
    refs = issue_ref_payload((number,))
    assert len(refs) == 1
    url = refs[0]["url"]
    assert isinstance(url, str)
    return url
