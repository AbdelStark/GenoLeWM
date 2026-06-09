# SPDX-License-Identifier: Apache-2.0
"""Contract checks for the published Hugging Face Space source."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPACE_DIR = REPO_ROOT / "spaces" / "geno-lewm"
SPACE_README = SPACE_DIR / "README.md"
SPACE_APP = SPACE_DIR / "app.py"
SPACE_REQUIREMENTS = SPACE_DIR / "requirements.txt"


def test_space_files_exist_and_are_gradio_space() -> None:
    readme = SPACE_README.read_text(encoding="utf-8")
    requirements = SPACE_REQUIREMENTS.read_text(encoding="utf-8")

    assert SPACE_APP.is_file()
    assert "sdk: gradio" in readme
    assert "app_file: app.py" in readme
    assert "geno-lewm[train]==0.2.1" in requirements
    assert "gradio" in requirements
    assert "huggingface_hub" in requirements


def test_space_preserves_claim_boundaries_and_artifact_links() -> None:
    combined = (
        SPACE_README.read_text(encoding="utf-8") + "\n" + SPACE_APP.read_text(encoding="utf-8")
    )

    required = (
        "No clinical utility claim",
        "mixed or negative",
        "not as a diagnostic",
        "Carbon-500M",
        "abdelstark/geno-lewm",
        "abdelstark/geno-lewm-runs",
        "geno-lewm-v021-strong-4f36eef-10k-r1",
        "score_single_variant",
    )
    for fragment in required:
        assert fragment in combined

    forbidden = (
        "clinically validated",
        "is deployment ready",
        "provides privacy assurance",
        "broad superiority",
        "geno-lewm beats carbon",
    )
    lower = combined.lower()
    for fragment in forbidden:
        assert fragment.lower() not in lower


def test_space_default_scoring_example_is_sequence_consistent() -> None:
    app = SPACE_APP.read_text(encoding="utf-8")

    assert 'DEFAULT_VARIANT = "chrSynthetic:3073:A:T"' in app
    assert "DEFAULT_WINDOW_START_BP = 0" in app
    assert '"ACGT" * 3072' in app
    assert "reference base mismatch before scoring" in app
    assert "observed_ref" in app


def test_space_is_in_source_distribution_contract() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"/spaces"' in pyproject
