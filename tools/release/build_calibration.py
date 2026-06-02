# SPDX-License-Identifier: Apache-2.0
"""Generate ``calibration.parquet`` by scoring a background variant set.

Runs a trained model (loaded from a deploy ``model_dir`` via
:func:`geno_lewm.deploy.load_scorer_modules`, which does *not* require an
existing calibration artifact) over a background VCF, records raw surprise per
functional bucket, and writes the empirical-CDF calibration table consumed by
the runtime/demo manifest. Run as::

    python -m tools.release.build_calibration \
        --model-dir ARTIFACTS/model \
        --vcf ARTIFACTS/background/common.vcf \
        --fasta ARTIFACTS/reference/GRCh38.fa \
        --output ARTIFACTS/model/calibration.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from geno_lewm.deploy import load_scorer_modules
from geno_lewm.encoder.windowing import DEFAULT_WINDOW_BP
from geno_lewm.errors import GenoLeWMError, exit_code_for
from geno_lewm.surprise.calibration import build_calibration_table, write_calibration_table
from geno_lewm.surprise.score import build_calibration_examples_from_vcf

GENERATED_BY = "tools.release.build_calibration"


def build_calibration(
    *,
    model_dir: Path,
    vcf: Path,
    fasta: Path,
    output: Path,
    window_bp: int = DEFAULT_WINDOW_BP,
    seed: int = 0,
) -> dict[str, Any]:
    """Score the background VCF with the model in ``model_dir`` and write the table."""
    encoder, action_encoder, predictor = load_scorer_modules(model_dir)
    examples = build_calibration_examples_from_vcf(
        vcf,
        encoder,
        action_encoder,
        predictor,
        reference_fasta=fasta,
        window_bp=window_bp,
    )
    table = build_calibration_table(examples, seed=seed)
    written = write_calibration_table(table, output)
    return {
        "generated_by": GENERATED_BY,
        "model_dir": str(model_dir),
        "vcf": str(vcf),
        "output": str(written),
        "examples": len(examples),
        "buckets": len(table.buckets),
        "low_confidence_buckets": sum(1 for bucket in table.buckets if bucket.low_confidence),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--vcf", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window-bp", type=int, default=DEFAULT_WINDOW_BP)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        summary = build_calibration(
            model_dir=args.model_dir,
            vcf=args.vcf,
            fasta=args.fasta,
            output=args.output,
            window_bp=args.window_bp,
            seed=args.seed,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
