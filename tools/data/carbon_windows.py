# SPDX-License-Identifier: Apache-2.0
"""Materialize pinned Carbon pretraining-corpus windows for the dataset snapshot.

Wraps :func:`geno_lewm.data.corpus.load_hf_carbon_records` at a pinned corpus
revision and emits the ``carbon/source-mix-windows.jsonl`` rows consumed by the
dataset-snapshot builder and the training window loader
(:func:`geno_lewm.training.real._load_windows`). Run as::

    python -m tools.data.carbon_windows \
        --revision cb4c13a78102933b3a6ac65734d326f7b431d9b7 \
        --output configs/first_experiment/inputs/carbon/source-mix-windows.jsonl \
        --max-windows 200000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from geno_lewm.data.corpus import (
    DEFAULT_CARBON_DATASET_ID,
    DEFAULT_PHASE1_SUBSET_FRACTION,
    CarbonCorpusConfig,
    iter_record_windows,
    load_hf_carbon_records,
)
from geno_lewm.encoder.windowing import DEFAULT_WINDOW_BP
from geno_lewm.errors import GenoLeWMError, exit_code_for

GENERATED_BY = "tools.data.carbon_windows"


def export_carbon_windows(
    *,
    output: Path,
    dataset_id: str = DEFAULT_CARBON_DATASET_ID,
    dataset_config: str | None = None,
    revision: str | None = None,
    default_source: str | None = None,
    split: str = "train",
    subset_fraction: float = DEFAULT_PHASE1_SUBSET_FRACTION,
    subset_seed: int = 0,
    window_bp: int = DEFAULT_WINDOW_BP,
    max_windows: int | None = None,
) -> dict[str, Any]:
    """Stream the pinned Carbon corpus and write source-mix windows as JSONL."""
    config = CarbonCorpusConfig(
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        revision=revision,
        default_source=default_source,
        split=split,
        subset_fraction=subset_fraction,
        subset_seed=subset_seed,
        window_bp=window_bp,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(subset_seed)
    records = 0
    windows = 0
    sources: dict[str, int] = {}
    with output.open("w", encoding="utf-8") as handle:
        for record in load_hf_carbon_records(config):
            records += 1
            for window in iter_record_windows(
                record,
                window_bp=window_bp,
                margin_bp=config.margin_bp,
                stride_bp=config.stride_bp,
                rng=rng,
            ):
                handle.write(
                    json.dumps(
                        {
                            "record_id": window.record_id,
                            "source": window.source,
                            "start_bp": window.start_bp,
                            "end_bp": window.end_bp,
                            "sequence": window.sequence,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                windows += 1
                sources[window.source] = sources.get(window.source, 0) + 1
                if max_windows is not None and windows >= max_windows:
                    break
            if max_windows is not None and windows >= max_windows:
                break
    return {
        "generated_by": GENERATED_BY,
        "output": str(output),
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "revision": revision,
        "split": split,
        "subset_fraction": subset_fraction,
        "records": records,
        "windows": windows,
        "sources": dict(sorted(sources.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-id", default=DEFAULT_CARBON_DATASET_ID)
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Corpus config/subset name (e.g. eukaryote_generator_10B_subset).",
    )
    parser.add_argument("--revision", default=None, help="Pinned corpus commit/revision.")
    parser.add_argument(
        "--default-source",
        default=None,
        help="Source label for single-source configs (e.g. eukaryotic_genes).",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset-fraction", type=float, default=DEFAULT_PHASE1_SUBSET_FRACTION)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--window-bp", type=int, default=DEFAULT_WINDOW_BP)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        summary = export_carbon_windows(
            output=args.output,
            dataset_id=args.dataset_id,
            dataset_config=args.dataset_config,
            revision=args.revision,
            default_source=args.default_source,
            split=args.split,
            subset_fraction=args.subset_fraction,
            subset_seed=args.subset_seed,
            window_bp=args.window_bp,
            max_windows=args.max_windows,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
