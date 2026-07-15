"""Public console-script coverage for production Carbon resume equivalence."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.test_training_preflight import _write_release_dataset
from tools.research.production_resume_equivalence import (
    run_production_resume_equivalence,
    verify_production_resume_equivalence,
)


@pytest.mark.integration
def test_public_cli_runs_uninterrupted_prefix_and_fresh_resume_processes(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("transformers")
    repo_root, commit_sha, tree_sha = _write_clean_git_fixture(tmp_path / "repo")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    dataset_dir = _write_release_dataset(inputs)
    carbon_model_dir = _write_tiny_carbon_runtime(inputs)
    training_config = _write_tiny_training_config(inputs, carbon_model_dir)
    output_dir = tmp_path / "evidence"
    train_executable = str(Path(sys.executable).with_name("geno-lewm-train"))

    report = run_production_resume_equivalence(
        repo_root=repo_root,
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        carbon_model_dir=carbon_model_dir,
        training_config=training_config,
        expected_source_commit=commit_sha,
        expected_source_tree=tree_sha,
        total_steps=2,
        split_step=1,
        train_executable=train_executable,
    )
    verified = verify_production_resume_equivalence(
        output_dir / "production_resume_equivalence.json",
        repo_root=repo_root,
        expected_source_commit=commit_sha,
        expected_source_tree=tree_sha,
        total_steps=2,
        split_step=1,
    )

    assert verified == report
    assert report["comparison"]["passed"] is True
    assert report["claim_scope"]["software_only"] is True
    assert len({arm["pid"] for arm in report["processes"].values()}) == 3


def _write_clean_git_fixture(root: Path) -> tuple[Path, str, str]:
    root.mkdir()
    source_root = Path(__file__).resolve().parents[2]
    shutil.copytree(
        source_root / "geno_lewm",
        root / "geno_lewm",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "GenoLeWM test")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "add", "geno_lewm")
    _git(root, "commit", "--quiet", "-m", "fixture")
    return root, _git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", "HEAD^{tree}")


def _write_tiny_carbon_runtime(root: Path) -> Path:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    carbon = root / "tiny-carbon"
    torch.manual_seed(3107)
    config = transformers.BertConfig(
        vocab_size=4100,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=1024,
        pad_token_id=0,
    )
    transformers.BertModel(config).save_pretrained(carbon, safe_serialization=True)
    (carbon / "dna_config.json").write_text(
        json.dumps(
            {
                "k": 6,
                "dna_start_id": 1,
                "dna_vocab_size": 4099,
                "dna_special_tokens": ["<dna>", "</dna>", "<oov>"],
                "auto_dna_tags": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (carbon / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "<pad>",
                "added_tokens_decoder": {"0": {"content": "<pad>"}},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (carbon / "tokenizer.py").write_text(
        "# Offline tokenizer identity fixture; GenoLeWM uses CarbonDNATokenizer.\n",
        encoding="utf-8",
    )
    return carbon


def _write_tiny_training_config(root: Path, carbon_model_dir: Path) -> Path:
    path = root / "train.tiny.yaml"
    path.write_text(
        "\n".join(
            [
                "run_id: production-resume-cli-fixture",
                "seed: 104729",
                "phase: phase1",
                "deterministic: true",
                "schema_version: 1.1.0",
                "encoder:",
                f"  model_id: {carbon_model_dir}",
                "  revision: main",
                "  dtype: fp32",
                "  state_layer: -1",
                "  pool_type: centered_mean",
                "  pool_radius: 1",
                "  normalize: true",
                "  state_contract_version: l2_normalized_v2",
                "  trust_remote_code: false",
                "data:",
                "  batch_size: 1",
                "  corpus_id: fixture/carbon-corpus",
                "  corpus_revision: fixture-v1",
                "  num_workers: 0",
                "  shuffle_buffer: 0",
                "predictor:",
                "  architecture: cross_attention",
                "  n_layers: 1",
                "  n_heads: 1",
                "  d_state: 8",
                "  d_action: 4",
                "  dtype: fp32",
                "action:",
                "  d_action: 4",
                "  max_len: 16",
                "  sub_encoders:",
                "    - snv",
                "training:",
                "  max_steps: 2",
                "  collapse_log_every_steps: 1",
                "optimizer:",
                "  name: adamw",
                "  lr: 0.001",
                "  beta1: 0.9",
                "  beta2: 0.95",
                "  weight_decay: 0.01",
                "  grad_clip: 1.0",
                "  warmup_steps: 0",
                "  schedule: constant",
                "runtime:",
                "  backend: torch",
                "  device: cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
