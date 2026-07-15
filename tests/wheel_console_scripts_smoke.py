# SPDX-License-Identifier: Apache-2.0
"""Smoke every installed GenoLeWM console script from a wheel-only environment.

This file is executed explicitly by the package-build job after changing to a
temporary directory outside the checkout.  It intentionally is not a pytest
module: importing it from the source tree would make the top-level ``tools``
package visible and mask wheel packaging defects.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_SCRIPTS = frozenset(
    {
        "geno-lewm-cache-windows",
        "geno-lewm-calibrate",
        "geno-lewm-carbon-baseline",
        "geno-lewm-eval",
        "geno-lewm-eval-all",
        "geno-lewm-export",
        "geno-lewm-plan",
        "geno-lewm-prepare-clinvar",
        "geno-lewm-prepare-gnomad",
        "geno-lewm-rollout",
        "geno-lewm-score",
        "geno-lewm-train",
        "geno-lewm-update",
        "geno-lewm-verify",
    }
)


def main() -> None:
    if importlib.util.find_spec("tools") is not None:
        raise AssertionError("wheel smoke must run with the source-only tools package absent")

    distribution = importlib.metadata.distribution("geno-lewm")
    scripts = sorted(
        (entry for entry in distribution.entry_points if entry.group == "console_scripts"),
        key=lambda entry: entry.name,
    )
    observed_scripts = {entry.name for entry in scripts}
    if observed_scripts != EXPECTED_SCRIPTS:
        raise AssertionError(
            "installed console-script set drifted: "
            f"missing={sorted(EXPECTED_SCRIPTS - observed_scripts)}, "
            f"unexpected={sorted(observed_scripts - EXPECTED_SCRIPTS)}"
        )

    for entry in scripts:
        target = entry.load()
        if not callable(target):
            raise AssertionError(f"{entry.name} does not resolve to a callable: {entry.value}")
        executable = _find_executable(entry.name)
        if executable is None:
            raise AssertionError(f"installed console script is not on PATH: {entry.name}")
        completed = subprocess.run(
            [executable, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        rendered = completed.stdout + completed.stderr
        if completed.returncode != 0 or "usage" not in rendered.lower():
            raise AssertionError(
                f"{entry.name} --help failed with exit {completed.returncode}:\n{rendered}"
            )

    executables = {entry.name: _find_executable(entry.name) for entry in scripts}
    _exercise_calibration_error_contract(_required_executable(executables, "geno-lewm-calibrate"))
    _exercise_eval_all(_required_executable(executables, "geno-lewm-eval-all"))
    _exercise_training_package()


def _required_executable(executables: dict[str, str | None], name: str) -> str:
    executable = executables.get(name)
    if executable is None:
        raise AssertionError(f"installed console script is not on PATH: {name}")
    return executable


def _find_executable(name: str) -> str | None:
    beside_python = Path(sys.executable).with_name(name)
    if beside_python.is_file():
        return str(beside_python)
    return shutil.which(name)


def _exercise_calibration_error_contract(executable: str) -> None:
    """Reach the installed calibration implementation without optional ML dependencies."""
    with tempfile.TemporaryDirectory(prefix="geno-lewm-calibrate-wheel-") as raw_dir:
        root = Path(raw_dir)
        model_dir = root / "model"
        model_dir.mkdir()
        (model_dir / "manifest.json").write_text("{invalid\n", encoding="utf-8")
        vcf = root / "background.vcf"
        fasta = root / "reference.fa"
        vcf.write_text(
            "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n",
            encoding="utf-8",
        )
        fasta.write_text(">chr1\nACGT\n", encoding="utf-8")
        completed = subprocess.run(
            [
                executable,
                "--model-dir",
                str(model_dir),
                "--vcf",
                str(vcf),
                "--fasta",
                str(fasta),
                "--output",
                str(root / "calibration.parquet"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        rendered = completed.stdout + completed.stderr
        if completed.returncode != 2 or "manifest is not valid JSON" not in rendered:
            raise AssertionError(
                "geno-lewm-calibrate did not preserve the typed invalid-manifest contract: "
                f"exit={completed.returncode}\n{rendered}"
            )


def _exercise_eval_all(executable: str) -> None:
    """Aggregate one valid metrics fixture entirely through the installed wheel."""
    with tempfile.TemporaryDirectory(prefix="geno-lewm-eval-all-wheel-") as raw_dir:
        root = Path(raw_dir)
        metrics_input = root / "metrics.json"
        metrics_output = root / "aggregate.json"
        report_output = root / "eval_report.md"
        metrics_input.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "generated_by": "geno-lewm-eval",
                    "generated_at": "2026-01-01T00:00:00Z",
                    "model_id": "sha256:" + "1" * 64,
                    "model_release": "geno-lewm-v0.2.1-r1",
                    "dataset_snapshot": "geno-lewm-data-v0.2.1-r1",
                    "commit": "abcdef1",
                    "hardware": "CPU",
                    "metrics": [
                        {
                            "name": "accuracy",
                            "value": 0.5,
                            "split": "validation",
                            "unit": "fraction",
                            "higher_is_better": True,
                            "n": 2,
                        }
                    ],
                    "artifacts": {
                        "checkpoint": "model/checkpoint.safetensors",
                        "config": "configs/eval.yaml",
                        "dataset_manifest": "data/manifest.json",
                        "eval_config": "model/eval.yaml",
                        "efficiency_report": "model/efficiency.json",
                    },
                    "limitations": ["Single-row wheel smoke only."],
                    "negative_findings": ["No comparative conclusion is drawn."],
                    "conclusions": ["accuracy on validation is 0.5."],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                executable,
                "--metrics-json",
                str(metrics_input),
                "--output-metrics",
                str(metrics_output),
                "--output-report",
                str(report_output),
                "--quiet",
                "--no-banner",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        rendered = completed.stdout + completed.stderr
        if completed.returncode != 0:
            raise AssertionError(f"geno-lewm-eval-all fixture failed:\n{rendered}")
        aggregate = json.loads(metrics_output.read_text(encoding="utf-8"))
        if aggregate.get("generated_by") != "geno-lewm-eval-all":
            raise AssertionError("eval-all aggregate did not record its installed generator")
        report = report_output.read_text(encoding="utf-8")
        if "# Evaluation Report" not in report or "accuracy" not in report:
            raise AssertionError("eval-all did not render the expected report")


def _exercise_training_package() -> None:
    """Reach the wheel-only release-package path used after Carbon training."""
    from geno_lewm.cli.train import _build_release_training_run_package

    with tempfile.TemporaryDirectory(prefix="geno-lewm-train-package-wheel-") as raw_dir:
        root = Path(raw_dir)
        files = {
            "dataset_manifest.json": '{"snapshot_id":"snapshot-r1"}\n',
            "train_config.yaml": "seed: 0\n",
            "metrics.json": '{"sample_count":1,"metrics":{"loss":0.5}}\n',
            "train.log": "step=1 loss=0.5\n",
        }
        for relative, contents in files.items():
            (root / relative).write_text(contents, encoding="utf-8")
        (root / "predictor.safetensors").write_bytes(b"predictor")
        metadata = root / "training_run.json"
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "run_id": "wheel-training-package-r1",
                    "generated_by": "tools.release.training_run",
                    "generated_at": "2026-01-01T00:00:00Z",
                    "command": "geno-lewm-train --carbon-train --package-release-run",
                    "commit_sha": "abcdef1",
                    "package_version": "0.2.1",
                    "dataset_snapshot_id": "snapshot-r1",
                    "dataset_manifest": "dataset_manifest.json",
                    "training_config": "train_config.yaml",
                    "metrics": "metrics.json",
                    "logs": ["train.log"],
                    "checkpoint_files": ["predictor.safetensors"],
                    "status": "completed",
                    "hardware": ["CPU wheel smoke"],
                    "runtime": ["Installed wheel"],
                    "seeds": {"python": 0},
                    "determinism": "Seeded wheel smoke.",
                    "monitoring": {"collapse_monitoring": True, "nan_monitoring": True},
                    "result_summary": "Completed dependency-closure wheel smoke.",
                    "limitations": ["Packaging mechanics only; no model-quality claim."],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        report = _build_release_training_run_package(root, metadata)
        if report.manifest_path != root / "training_run_manifest.json":
            raise AssertionError("training package did not write the expected manifest")
        for relative in (
            "training_run_manifest.json",
            "training_run_card.md",
            "training_run_SHA256SUMS",
        ):
            if not (root / relative).is_file():
                raise AssertionError(f"training package omitted {relative}")


if __name__ == "__main__":
    main()
