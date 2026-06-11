# SPDX-License-Identifier: Apache-2.0
"""Render the GenoLeWM-FX Borzoi alignment and overlap report."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.research.fx_borzoi_overlap"
DEFAULT_MANIFEST: Final = Path("configs/fx/borzoi_rescue_sources.json")
DEFAULT_OUTPUT_JSON: Final = Path("docs/research/fx-borzoi-overlap-report.json")
DEFAULT_OUTPUT_MD: Final = Path("docs/research/fx-borzoi-overlap-report.md")

JsonDict = dict[str, Any]
VariantKey = tuple[str, int, str, str]


def build_borzoi_overlap_report(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    generated_at: str | None = None,
    traitgym_records: Sequence[Mapping[str, Any]] | None = None,
    score_values: Sequence[Any] | None = None,
    traitgym_slice_receipt: Mapping[str, Any] | None = None,
    score_receipt: Mapping[str, Any] | None = None,
    fipip_metadata: Mapping[str, Any] | None = None,
    fipip_score_table: Path | None = None,
) -> JsonDict:
    """Build the machine-readable Borzoi rescue alignment report."""
    manifest = _load_manifest(manifest_path)
    traitgym_slice = _require_mapping(manifest["traitgym_slice"], "traitgym_slice")
    borzoi_score = _require_mapping(manifest["traitgym_borzoi_score"], "traitgym_borzoi_score")
    fipip_source = _require_mapping(manifest["fipip_borzoi_source"], "fipip_borzoi_source")

    records = (
        list(traitgym_records)
        if traitgym_records is not None
        else _load_traitgym_records(traitgym_slice)
    )
    if score_values is None or score_receipt is None:
        score_path = _download_hf_artifact(borzoi_score)
        loaded_score_values = _read_parquet_column(score_path, str(borzoi_score["score_column"]))
        resolved_score_receipt = _artifact_receipt(borzoi_score, score_path)
    else:
        loaded_score_values = list(score_values)
        resolved_score_receipt = dict(score_receipt)

    if traitgym_slice_receipt is None:
        slice_path = _download_hf_artifact(
            {
                "repo_id": traitgym_slice["dataset"],
                "repo_type": "dataset",
                "artifact_path": traitgym_slice["artifact_path"],
                "revision": traitgym_slice["revision"],
            }
        )
        resolved_slice_receipt = _artifact_receipt(
            {
                "repo_id": traitgym_slice["dataset"],
                "repo_type": "dataset",
                "artifact_path": traitgym_slice["artifact_path"],
                "revision": traitgym_slice["revision"],
            },
            slice_path,
        )
    else:
        resolved_slice_receipt = dict(traitgym_slice_receipt)

    resolved_fipip_metadata = (
        dict(fipip_metadata) if fipip_metadata is not None else _fetch_gcs_metadata(fipip_source)
    )
    alignment = _build_traitgym_alignment(
        records=records,
        score_values=loaded_score_values,
        manifest=manifest,
        traitgym_slice=traitgym_slice,
        borzoi_score=borzoi_score,
    )
    fipip_exact_join = _build_fipip_exact_join_status(
        fipip_source=fipip_source,
        fipip_metadata=resolved_fipip_metadata,
        fipip_score_table=fipip_score_table,
        traitgym_keys=alignment["variant_key_summary"]["keys_for_internal_scan"],
    )
    del alignment["variant_key_summary"]["keys_for_internal_scan"]

    blockers = _decision_blockers(
        manifest=manifest,
        alignment=alignment,
        fipip_exact_join=fipip_exact_join,
    )
    decision = "go_traitgym_native_borzoi" if not blockers else "no_go"
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": generated_at or _utc_now(),
        "decision": decision,
        "ok_to_build_cache": decision == "go_traitgym_native_borzoi",
        "epic_issue": manifest["epic_issue"],
        "contract_issue": manifest["contract_issue"],
        "overlap_issue": manifest["overlap_issue"],
        "cache_issue": manifest["cache_issue"],
        "manifest": {
            "path": _repo_relative(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "minimum_usable_rows": manifest["min_usable_rows"],
        "source_inputs": {
            "traitgym_slice": resolved_slice_receipt,
            "traitgym_borzoi_score": resolved_score_receipt,
            "fipip_borzoi_source": resolved_fipip_metadata,
        },
        "traitgym_native_alignment": alignment,
        "fipip_exact_join": fipip_exact_join,
        "blockers": blockers,
        "recommended_issue_actions": _recommended_issue_actions(decision),
        "claim_boundary": manifest["claim_boundary"],
    }


def write_report(
    *,
    report: Mapping[str, Any],
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
) -> None:
    """Write JSON and Markdown reports."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a human-facing Markdown report for docs."""
    alignment = _require_mapping(report["traitgym_native_alignment"], "traitgym_native_alignment")
    key_summary = _require_mapping(alignment["variant_key_summary"], "variant_key_summary")
    label_summary = _require_mapping(alignment["label_summary"], "label_summary")
    split_summary = _require_mapping(alignment["split_summary"], "split_summary")
    score_summary = _require_mapping(alignment["score_summary"], "score_summary")
    fipip_join = _require_mapping(report["fipip_exact_join"], "fipip_exact_join")

    lines = [
        "# GenoLeWM-FX Borzoi alignment and overlap report",
        "",
        f"Generated by `{report['generated_by']}` at `{report['generated_at']}`.",
        "",
        f"Parent epic: #{report['epic_issue']}. Contract: #{report['contract_issue']}. "
        f"Overlap gate: #{report['overlap_issue']}. Cache gate: #{report['cache_issue']}.",
        "",
        f"Decision: **{report['decision']}**.",
        "",
        str(report["claim_boundary"]),
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python -m tools.research.fx_borzoi_overlap \\",
        "  --manifest configs/fx/borzoi_rescue_sources.json \\",
        "  --output-json docs/research/fx-borzoi-overlap-report.json \\",
        "  --output-md docs/research/fx-borzoi-overlap-report.md",
        "```",
        "",
        "Machine-readable report: [fx-borzoi-overlap-report.json](fx-borzoi-overlap-report.json).",
        "",
        "## Interpretation",
        "",
        "This report validates the compact TraitGym-native, row-aligned Borzoi score "
        "artifact as the first executable rescue substrate. It does not claim exact "
        "overlap against the full fipip table unless that optional table join is run.",
        "",
        "## Artifact Receipts",
        "",
        "| Artifact | Identity | SHA-256 / checksum | Size |",
        "| --- | --- | --- | ---: |",
    ]
    source_inputs = _require_mapping(report["source_inputs"], "source_inputs")
    for name in ("traitgym_slice", "traitgym_borzoi_score", "fipip_borzoi_source"):
        receipt = _require_mapping(source_inputs[name], f"source_inputs.{name}")
        lines.append(_artifact_row(name, receipt))

    lines.extend(
        [
            "",
            "## TraitGym-Native Alignment Gate",
            "",
            "| Check | Value |",
            "| --- | ---: |",
            f"| TraitGym rows | {alignment['observed_traitgym_rows']} |",
            f"| Borzoi score rows | {alignment['observed_score_rows']} |",
            f"| Usable row-aligned rows | {alignment['usable_rows']} |",
            f"| Minimum usable rows | {report['minimum_usable_rows']} |",
            f"| Unique variant keys | {key_summary['unique_keys']} |",
            f"| Duplicate variant keys | {key_summary['duplicate_key_count']} |",
            f"| Positive labels | {label_summary['positive']} |",
            f"| Negative labels | {label_summary['negative']} |",
            f"| Holdout rows | {split_summary['holdout_rows']} |",
            f"| Train rows | {split_summary['train_rows']} |",
            f"| Finite score rows | {score_summary['finite_scores']} |",
            f"| Null score rows | {score_summary['null_scores']} |",
            "",
            "Row-order status: "
            f"`{alignment['row_order_status']}`. The score file has no row ID column, so "
            "row alignment is accepted only because the artifact lives under TraitGym's "
            "matched-slice `preds/all` path, uses the same dataset revision, and has the "
            "same row count as the public slice.",
            "",
            "## Optional fipip Exact Join",
            "",
            f"Status: `{fipip_join['status']}`.",
            "",
        ]
    )
    if fipip_join["status"] == "ran_local_table":
        lines.extend(
            [
                "| Check | Value |",
                "| --- | ---: |",
                f"| Scanned fipip rows | {fipip_join['scanned_rows']} |",
                f"| Exact key matches | {fipip_join['exact_key_matches']} |",
                f"| Reverse-allele key hits | {fipip_join['reverse_allele_key_hits']} |",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The full fipip table was not staged for this run. This report therefore "
                "records public fipip object metadata but makes no exact fipip overlap claim.",
                "",
            ]
        )

    lines.extend(["## Blockers", ""])
    blockers = _require_list(report["blockers"])
    if blockers:
        for blocker in blockers:
            item = _require_mapping(blocker, "blockers[]")
            lines.append(f"- `{item['code']}`: {item['message']}")
    else:
        lines.append("No #268 blockers remain for the TraitGym-native row-aligned path.")
    lines.extend(
        [
            "",
            "## Recommended Issue Actions",
            "",
            "| Issue | Action | Reason |",
            "| ---: | --- | --- |",
        ]
    )
    for raw_action in _require_list(report["recommended_issue_actions"]):
        action = _require_mapping(raw_action, "recommended_issue_actions[]")
        lines.append(f"| #{action['issue']} | `{action['action']}` | {action['reason']} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--generated-at")
    parser.add_argument(
        "--fipip-score-table",
        type=Path,
        help="Optional local fipip score table to exact-join by CHROM,POS,REF,ALT.",
    )
    args = parser.parse_args(argv)
    try:
        report = build_borzoi_overlap_report(
            manifest_path=args.manifest,
            generated_at=args.generated_at,
            fipip_score_table=args.fipip_score_table,
        )
        write_report(report=report, output_json=args.output_json, output_md=args.output_md)
    except GenoLeWMError as exc:
        print(exc.to_json(), file=sys.stderr)
        return exit_code_for(exc)
    return 0


def _load_manifest(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(
            "FX Borzoi rescue manifest not found", details={"path": str(path)}
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("FX Borzoi rescue manifest must be a JSON object")
    manifest = cast(JsonDict, payload)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise InputError(
            "unsupported FX Borzoi rescue manifest schema",
            details={"expected": SCHEMA_VERSION, "actual": manifest.get("schema_version")},
        )
    required = (
        "epic_issue",
        "contract_issue",
        "overlap_issue",
        "cache_issue",
        "min_usable_rows",
        "holdout_chromosomes",
        "traitgym_slice",
        "traitgym_borzoi_score",
        "fipip_borzoi_source",
        "claim_boundary",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise InputError(
            "FX Borzoi rescue manifest is missing required keys", details={"missing": missing}
        )
    return manifest


def _load_traitgym_records(traitgym_slice: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    datasets = cast(Any, importlib.import_module("datasets"))
    dataset = datasets.load_dataset(
        str(traitgym_slice["dataset"]),
        str(traitgym_slice["config"]),
        split=str(traitgym_slice["split"]),
    )
    return [cast(Mapping[str, Any], row) for row in dataset]


def _download_hf_artifact(artifact: Mapping[str, Any]) -> Path:
    hub = cast(Any, importlib.import_module("huggingface_hub"))
    return Path(
        hub.hf_hub_download(
            str(artifact["repo_id"]),
            str(artifact["artifact_path"]),
            repo_type=str(artifact.get("repo_type", "dataset")),
            revision=str(artifact["revision"]),
        )
    )


def _artifact_receipt(artifact: Mapping[str, Any], path: Path) -> JsonDict:
    return {
        "repo_id": artifact["repo_id"],
        "repo_type": artifact.get("repo_type", "dataset"),
        "path": artifact["artifact_path"],
        "revision": artifact["revision"],
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _read_parquet_column(path: Path, column: str) -> list[Any]:
    pq = cast(Any, importlib.import_module("pyarrow.parquet"))
    table = pq.read_table(path, columns=[column])
    return list(table.column(column).to_pylist())


def _fetch_gcs_metadata(fipip_source: Mapping[str, Any]) -> JsonDict:
    bucket = str(fipip_source["bucket"])
    object_name = str(fipip_source["object"])
    encoded_object = urllib.parse.quote(object_name, safe="")
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{encoded_object}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {
        "repository": fipip_source["repository"],
        "genome_build": fipip_source["genome_build"],
        "bucket": payload["bucket"],
        "object": payload["name"],
        "expected_rows": fipip_source["expected_rows"],
        "size_bytes": int(payload["size"]),
        "md5_hash": payload.get("md5Hash"),
        "crc32c": payload.get("crc32c"),
        "generation": payload.get("generation"),
        "updated": payload.get("updated"),
        "score_kind": fipip_source["score_kind"],
    }


def _build_traitgym_alignment(
    *,
    records: Sequence[Mapping[str, Any]],
    score_values: Sequence[Any],
    manifest: Mapping[str, Any],
    traitgym_slice: Mapping[str, Any],
    borzoi_score: Mapping[str, Any],
) -> JsonDict:
    expected_rows = int(traitgym_slice["expected_rows"])
    holdout_chromosomes = {str(chrom) for chrom in _require_list(manifest["holdout_chromosomes"])}
    keys: list[VariantKey] = []
    chrom_counts: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    split_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "holdout": Counter(),
    }
    for record in records:
        key = _variant_key(record)
        keys.append(key)
        chrom = key[0]
        chrom_counts[chrom] += 1
        label = _label(record, str(traitgym_slice["label_column"]))
        label_counts[label] += 1
        split_name = "holdout" if chrom in holdout_chromosomes else "train"
        split_counts[split_name]["rows"] += 1
        split_counts[split_name]["positive"] += int(label == 1)
        split_counts[split_name]["negative"] += int(label == 0)

    score_summary = _score_summary(score_values)
    row_count_match = len(records) == len(score_values)
    usable_rows = (
        min(len(records), int(score_summary["finite_scores"]))
        if row_count_match and _has_both_classes(label_counts)
        else 0
    )
    duplicate_key_count = len(keys) - len(set(keys))
    return {
        "dataset": traitgym_slice["dataset"],
        "config": traitgym_slice["config"],
        "split": traitgym_slice["split"],
        "score_id": borzoi_score["score_id"],
        "score_column": borzoi_score["score_column"],
        "score_semantics": borzoi_score["score_semantics"],
        "expected_traitgym_rows": expected_rows,
        "observed_traitgym_rows": len(records),
        "observed_score_rows": len(score_values),
        "row_count_match": row_count_match,
        "expected_row_count_match": len(records) == expected_rows,
        "usable_rows": usable_rows,
        "minimum_usable_rows_passed": usable_rows >= int(manifest["min_usable_rows"]),
        "row_order_status": "assumed_by_traitgym_matched_slice_artifact_layout",
        "variant_key_summary": {
            "unique_keys": len(set(keys)),
            "duplicate_key_count": duplicate_key_count,
            "duplicate_key_passed": duplicate_key_count == 0,
            "keys_for_internal_scan": keys,
        },
        "label_summary": {
            "positive": label_counts[1],
            "negative": label_counts[0],
            "both_classes_present": _has_both_classes(label_counts),
        },
        "split_summary": {
            "holdout_chromosomes": sorted(holdout_chromosomes),
            "train_rows": split_counts["train"]["rows"],
            "train_positive": split_counts["train"]["positive"],
            "train_negative": split_counts["train"]["negative"],
            "holdout_rows": split_counts["holdout"]["rows"],
            "holdout_positive": split_counts["holdout"]["positive"],
            "holdout_negative": split_counts["holdout"]["negative"],
            "split_has_both_classes": _split_has_both_classes(split_counts),
        },
        "chromosome_counts": dict(
            sorted(chrom_counts.items(), key=lambda item: _chrom_sort_key(item[0]))
        ),
        "score_summary": score_summary,
    }


def _score_summary(score_values: Sequence[Any]) -> JsonDict:
    finite_scores: list[float] = []
    null_scores = 0
    non_numeric_scores = 0
    for raw_value in score_values:
        if raw_value is None:
            null_scores += 1
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            non_numeric_scores += 1
            continue
        if math.isfinite(value):
            finite_scores.append(value)
        else:
            non_numeric_scores += 1
    summary: JsonDict = {
        "finite_scores": len(finite_scores),
        "null_scores": null_scores,
        "non_numeric_or_non_finite_scores": non_numeric_scores,
    }
    if finite_scores:
        sorted_scores = sorted(finite_scores)
        summary.update(
            {
                "min": sorted_scores[0],
                "max": sorted_scores[-1],
                "mean": sum(sorted_scores) / len(sorted_scores),
                "p50": _percentile(sorted_scores, 0.5),
            }
        )
    return summary


def _build_fipip_exact_join_status(
    *,
    fipip_source: Mapping[str, Any],
    fipip_metadata: Mapping[str, Any],
    fipip_score_table: Path | None,
    traitgym_keys: Sequence[VariantKey],
) -> JsonDict:
    if fipip_score_table is None:
        return {
            "status": str(fipip_source["exact_join_default"]),
            "reason": "The full fipip table was not staged locally for this run.",
            "metadata_only": {
                "bucket": fipip_metadata.get("bucket"),
                "object": fipip_metadata.get("object"),
                "size_bytes": fipip_metadata.get("size_bytes"),
                "generation": fipip_metadata.get("generation"),
            },
        }
    if not fipip_score_table.is_file():
        raise InputError("fipip score table not found", details={"path": str(fipip_score_table)})
    return _scan_fipip_score_table(fipip_score_table, traitgym_keys)


def _scan_fipip_score_table(path: Path, traitgym_keys: Sequence[VariantKey]) -> JsonDict:
    pd = cast(Any, importlib.import_module("pandas"))
    traitgym_key_set = set(traitgym_keys)
    reverse_key_set = {(chrom, pos, alt, ref) for chrom, pos, ref, alt in traitgym_key_set}
    exact_matches: set[VariantKey] = set()
    reverse_hits: set[VariantKey] = set()
    scanned_rows = 0
    duplicate_table_keys = 0
    seen_table_keys: set[VariantKey] = set()
    reader = pd.read_csv(
        path,
        sep=r"\s+",
        usecols=["CHROM", "POS", "REF", "ALT"],
        chunksize=100_000,
        compression="infer",
    )
    for chunk in reader:
        for row in chunk.itertuples(index=False):
            scanned_rows += 1
            key = _variant_key(
                {
                    "chrom": row.CHROM,
                    "pos": row.POS,
                    "ref": row.REF,
                    "alt": row.ALT,
                }
            )
            if key in seen_table_keys:
                duplicate_table_keys += 1
            seen_table_keys.add(key)
            if key in traitgym_key_set:
                exact_matches.add(key)
            if key in reverse_key_set:
                reverse_hits.add(key)
    return {
        "status": "ran_local_table",
        "path": _repo_relative(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "scanned_rows": scanned_rows,
        "exact_key_matches": len(exact_matches),
        "reverse_allele_key_hits": len(reverse_hits),
        "duplicate_table_key_count": duplicate_table_keys,
    }


def _decision_blockers(
    *,
    manifest: Mapping[str, Any],
    alignment: Mapping[str, Any],
    fipip_exact_join: Mapping[str, Any],
) -> list[JsonDict]:
    blockers: list[JsonDict] = []
    if alignment["observed_traitgym_rows"] != alignment["expected_traitgym_rows"]:
        blockers.append(
            {
                "code": "traitgym_row_count_changed",
                "message": "The loaded TraitGym slice row count does not match the locked manifest.",
            }
        )
    if not alignment["row_count_match"]:
        blockers.append(
            {
                "code": "traitgym_borzoi_row_count_mismatch",
                "message": "The TraitGym slice and row-aligned Borzoi score vector have different row counts.",
            }
        )
    variant_summary = _require_mapping(alignment["variant_key_summary"], "variant_key_summary")
    if variant_summary["duplicate_key_count"]:
        blockers.append(
            {
                "code": "traitgym_duplicate_variant_keys",
                "message": "Duplicate normalized variant keys would make row-level leakage auditing ambiguous.",
            }
        )
    label_summary = _require_mapping(alignment["label_summary"], "label_summary")
    if not label_summary["both_classes_present"]:
        blockers.append(
            {
                "code": "traitgym_label_class_missing",
                "message": "The matched slice does not contain both positive and negative labels.",
            }
        )
    split_summary = _require_mapping(alignment["split_summary"], "split_summary")
    if not split_summary["split_has_both_classes"]:
        blockers.append(
            {
                "code": "holdout_split_class_missing",
                "message": "The locked train/holdout split does not preserve both classes in each split.",
            }
        )
    score_summary = _require_mapping(alignment["score_summary"], "score_summary")
    if score_summary["finite_scores"] != alignment["observed_score_rows"]:
        blockers.append(
            {
                "code": "invalid_borzoi_scores",
                "message": "The Borzoi score vector contains null, non-numeric, or non-finite values.",
            }
        )
    if alignment["usable_rows"] < int(manifest["min_usable_rows"]):
        blockers.append(
            {
                "code": "below_minimum_usable_rows",
                "message": "The usable row-aligned slice is below the 10,000-row rescue threshold.",
            }
        )
    if fipip_exact_join["status"] == "ran_local_table" and fipip_exact_join.get(
        "exact_key_matches", 0
    ) < int(manifest["min_usable_rows"]):
        blockers.append(
            {
                "code": "fipip_exact_join_below_threshold",
                "message": "A staged fipip exact join did not meet the 10,000-row threshold.",
            }
        )
    return blockers


def _recommended_issue_actions(decision: str) -> list[JsonDict]:
    if decision == "go_traitgym_native_borzoi":
        return [
            {
                "issue": 268,
                "action": "close-completed",
                "reason": "The TraitGym-native row-aligned Borzoi path passes the 10,000-row alignment gate.",
            },
            {
                "issue": 269,
                "action": "open-next",
                "reason": "Build the manifest-backed cache from the compact row-aligned Borzoi score vector.",
            },
        ]
    return [
        {
            "issue": 268,
            "action": "close-no-go",
            "reason": "The alignment or overlap gate failed; publish this report as the no-go artifact.",
        },
        {
            "issue": 269,
            "action": "close-not-planned",
            "reason": "Cache work must not proceed after a failed overlap gate.",
        },
    ]


def _variant_key(record: Mapping[str, Any]) -> VariantKey:
    chrom = str(record["chrom"]).removeprefix("chr").removeprefix("Chr")
    pos = int(record["pos"])
    ref = str(record["ref"]).upper()
    alt = str(record["alt"]).upper()
    if not chrom or pos <= 0 or not ref or not alt:
        raise InputError("invalid variant key", details={"record": dict(record)})
    return chrom, pos, ref, alt


def _label(record: Mapping[str, Any], column: str) -> int:
    raw_label = record[column]
    if isinstance(raw_label, bool):
        return int(raw_label)
    label = int(raw_label)
    if label not in {0, 1}:
        raise InputError("TraitGym labels must be binary", details={"label": raw_label})
    return label


def _has_both_classes(label_counts: Mapping[int, int]) -> bool:
    return label_counts.get(0, 0) > 0 and label_counts.get(1, 0) > 0


def _split_has_both_classes(split_counts: Mapping[str, Mapping[str, int]]) -> bool:
    return all(
        split.get("positive", 0) > 0 and split.get("negative", 0) > 0
        for split in split_counts.values()
    )


def _artifact_row(name: str, receipt: Mapping[str, Any]) -> str:
    if "sha256" in receipt:
        identity = f"`{receipt.get('repo_id')}:{receipt.get('path')}@{receipt.get('revision')}`"
        return f"| `{name}` | {identity} | `{receipt['sha256']}` | {receipt['size_bytes']} |"
    identity = f"`gs://{receipt.get('bucket')}/{receipt.get('object')}`"
    checksum = f"md5 `{receipt.get('md5_hash')}`, crc32c `{receipt.get('crc32c')}`"
    return f"| `{name}` | {identity} | {checksum} | {receipt.get('size_bytes')} |"


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise InputError("cannot compute percentile over empty sequence")
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _chrom_sort_key(chrom: str) -> tuple[int, str]:
    if chrom.isdigit():
        return int(chrom), chrom
    return 10_000, chrom


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _require_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise InputError("expected a JSON array")
    return value


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
