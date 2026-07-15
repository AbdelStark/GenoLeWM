# SPDX-License-Identifier: Apache-2.0
"""Verify that source distributions include release-critical repository assets."""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Final

REQUIRED_SDIST_ASSETS: Final[tuple[str, ...]] = (
    "ARCHITECTURE.md",
    "README.md",
    "bench/inference.py",
    "configs/correction_control/dataset-snapshot-snv-l2-smoke-v1.json",
    "configs/correction_control/train-carbon-500m-snv-l2-smoke-v1.yaml",
    "configs/data_v03/cache-h200-job-receipt.schema.json",
    "configs/data_v03/cache-h200-proof.schema.json",
    "configs/data_v03/carbon-500m-l2-runtime-identity.json",
    "configs/data_v03/carbon-500m-runtime-content-lock.json",
    "configs/data_v03/carbon-500m-runtime-content-lock.schema.json",
    "configs/data_v03/gnomad-v4.1-exomes-autosomes.source-lock.json",
    "configs/data_v03/gnomad-v4.1-exomes-autosomes.source-lock.schema.json",
    "configs/data_v03/train-carbon-500m-snv-l2-epoch-r1.yaml",
    "configs/data_v03/training-trace.schema.json",
    "configs/first_experiment/dataset-snapshot-snv.json",
    "configs/first_experiment/eval-clinvar-snv.yaml",
    "configs/first_experiment/train-carbon-500m-snv.yaml",
    "configs/serious_completion/dataset-snapshot-snv-post-v02.json",
    "configs/serious_completion/train-carbon-500m-snv-post-v02.yaml",
    "docs/carbon-runtime-hash-probe.md",
    "docs/api/public-surface.md",
    "docs/index.md",
    "docs/production-resume-equivalence.md",
    "docs/quickstart.md",
    "docs/release/huggingface-model-card.md",
    "examples/data/verify_receipt/manifest.json",
    "examples/data/verify_receipt/receipt.json",
    "paper/README.md",
    "paper/main.tex",
    "paper/refs.bib",
    "spaces/geno-lewm/app.py",
    "spaces/geno-lewm/README.md",
    "tools/demo/terminal_inference.py",
    "tools/data/v03_gnomad_lock.py",
    "tools/data/v03_training_trace.py",
    "tools/lint/check_scope_language.py",
    "tools/release/atomic_hub_publish.py",
    "tools/release/batch_receipt_report.py",
    "tools/release/check_sdist_assets.py",
    "tools/release/clean_machine_demo.py",
    "tools/release/dataset_integrity.py",
    "tools/release/dataset_package.py",
    "tools/release/dataset_snapshot.py",
    "tools/release/efficiency_report.py",
    "tools/release/eval_report.py",
    "tools/release/hub_publish.py",
    "tools/release/hub_release.py",
    "tools/release/issue_refs.py",
    "tools/release/model_package.py",
    "tools/release/paper_draft.py",
    "tools/release/paper_package.py",
    "tools/release/paper_pdf.py",
    "tools/release/publication_assets.py",
    "tools/release/publication_report.py",
    "tools/release/release_candidate.py",
    "tools/release/runtime_preflight.py",
    "tools/release/serious_completion_paper.py",
    "tools/release/rollout_state_examples.py",
    "tools/release/rollout_state_rows.py",
    "tools/release/rollout_speed_scope.py",
    "tools/release/training_reproducibility.py",
    "tools/release/training_run.py",
    "tools/release/v02_benchmark_inputs.py",
    "tools/release/v02_benchmark_readiness.py",
    "tools/release/v02_benchmark_suite.py",
    "tools/release/v02_rollout_inputs.py",
    "tools/research/correction_control_model_manifest.py",
    "tools/research/correction_control_postflight.py",
    "tools/research/correction_control_preflight.py",
    "tools/research/correction_control_replay.py",
    "tools/research/production_resume_equivalence.py",
    "tools/research/state_contract_audit.py",
    "tools/research/v03_carbon_runtime_hash_probe.py",
    "tools/research/v03_carbon_runtime_hash_probe_launch.py",
    "tools/research/v03_cache_h200_job_receipt.py",
    "tools/research/v03_cache_h200_launch.py",
    "tools/research/v03_cache_h200_proof.py",
    "tools/research/verify_carbon_runtime_lock.py",
    "tools/jobs/demo_run.sh",
    "tools/jobs/eval_run.sh",
    "tools/jobs/planning_demo_run.sh",
    "tools/jobs/proof_run.sh",
    "tools/jobs/publish_run.sh",
    "tools/jobs/v03_cache_h200_proof.sh",
    "tools/jobs/v03_stage_gnomad.sh",
    "tools/jobs/v02_suite_run.sh",
)


def sdist_members(sdist_path: Path) -> frozenset[str]:
    """Return normalized source distribution member paths without the archive root."""
    with tarfile.open(sdist_path, "r:gz") as archive:
        normalized = {
            _strip_archive_root(member.name)
            for member in archive.getmembers()
            if member.isfile() and _strip_archive_root(member.name)
        }
    return frozenset(normalized)


def missing_sdist_assets(sdist_path: Path) -> tuple[str, ...]:
    """Return required release assets missing from ``sdist_path``."""
    members = sdist_members(sdist_path)
    return tuple(asset for asset in REQUIRED_SDIST_ASSETS if asset not in members)


def check_sdist_assets(sdist_path: Path) -> None:
    """Raise ``ValueError`` if ``sdist_path`` misses any required release asset."""
    missing = missing_sdist_assets(sdist_path)
    if missing:
        formatted = "\n".join(f"  - {asset}" for asset in missing)
        raise ValueError(f"{sdist_path} is missing release-critical assets:\n{formatted}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdists", nargs="+", type=Path, help="sdist .tar.gz artifacts to check")
    args = parser.parse_args(argv)

    try:
        for sdist_path in args.sdists:
            check_sdist_assets(sdist_path)
            print(f"ok: {sdist_path}")
    except (tarfile.TarError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _strip_archive_root(member_name: str) -> str:
    parts = PurePosixPath(member_name).parts
    if not parts:
        return ""
    if parts[0] in ("", ".", ".."):
        return ""
    if len(parts) == 1:
        return parts[0]
    return "/".join(parts[1:])


if __name__ == "__main__":
    raise SystemExit(main())
