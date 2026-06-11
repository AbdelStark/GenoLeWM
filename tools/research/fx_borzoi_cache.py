# SPDX-License-Identifier: Apache-2.0
"""Build the GenoLeWM-FX row-aligned Borzoi score cache."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.research.fx_borzoi_cache"
DEFAULT_SOURCE_MANIFEST: Final = Path("configs/fx/borzoi_rescue_sources.json")
DEFAULT_OVERLAP_REPORT: Final = Path("docs/research/fx-borzoi-overlap-report.json")
DEFAULT_OUTPUT_CACHE: Final = Path("docs/research/fx-borzoi-score-cache.parquet")
DEFAULT_OUTPUT_MANIFEST: Final = Path("docs/research/fx-borzoi-cache-manifest.json")
DEFAULT_OUTPUT_MD: Final = Path("docs/research/fx-borzoi-cache-report.md")

JsonDict = dict[str, Any]


def build_cache_package(
    *,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    overlap_report_path: Path = DEFAULT_OVERLAP_REPORT,
    output_cache: Path = DEFAULT_OUTPUT_CACHE,
    generated_at: str | None = None,
    traitgym_records: Sequence[Mapping[str, Any]] | None = None,
    score_values: Sequence[Any] | None = None,
) -> tuple[list[JsonDict], JsonDict]:
    """Build cache rows plus their manifest."""
    source_manifest = _load_json(source_manifest_path, label="source manifest")
    overlap_report = _load_json(overlap_report_path, label="overlap report")
    _assert_overlap_gate_passed(overlap_report)
    traitgym_slice = _require_mapping(source_manifest["traitgym_slice"], "traitgym_slice")
    borzoi_score = _require_mapping(
        source_manifest["traitgym_borzoi_score"], "traitgym_borzoi_score"
    )
    records = (
        list(traitgym_records)
        if traitgym_records is not None
        else _load_traitgym_records(traitgym_slice)
    )
    loaded_score_values = (
        list(score_values) if score_values is not None else _load_borzoi_scores(borzoi_score)
    )
    rows = build_cache_rows(
        records=records,
        score_values=loaded_score_values,
        holdout_chromosomes=[
            str(chrom) for chrom in _require_list(source_manifest["holdout_chromosomes"])
        ],
        score_id=str(borzoi_score["score_id"]),
    )
    manifest = _build_cache_manifest(
        rows=rows,
        source_manifest=source_manifest,
        source_manifest_path=source_manifest_path,
        overlap_report=overlap_report,
        overlap_report_path=overlap_report_path,
        output_cache=output_cache,
        generated_at=generated_at or _utc_now(),
    )
    return rows, manifest


def build_cache_rows(
    *,
    records: Sequence[Mapping[str, Any]],
    score_values: Sequence[Any],
    holdout_chromosomes: Sequence[str],
    score_id: str,
) -> list[JsonDict]:
    """Build row dictionaries for the row-aligned score cache."""
    if len(records) != len(score_values):
        raise InputError(
            "TraitGym records and Borzoi scores must have identical row counts",
            details={"records": len(records), "scores": len(score_values)},
        )
    holdout = set(holdout_chromosomes)
    rows: list[JsonDict] = []
    seen_keys: set[tuple[str, int, str, str]] = set()
    for index, (record, raw_score) in enumerate(zip(records, score_values, strict=True)):
        chrom, pos, ref, alt = _variant_key(record)
        key = (chrom, pos, ref, alt)
        if key in seen_keys:
            raise InputError(
                "duplicate normalized variant key in Borzoi cache input",
                details={"chrom": chrom, "pos": pos, "ref": ref, "alt": alt},
            )
        seen_keys.add(key)
        score = _finite_float(raw_score, label="Borzoi score")
        split = "holdout" if chrom in holdout else "train"
        rows.append(
            {
                "row_index": index,
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "label": _label(record),
                "split": split,
                "trait": str(record.get("trait", "")),
                "consequence": str(record.get("consequence", "")),
                "match_group": str(record.get("match_group", "")),
                "maf": _optional_float(record.get("maf")),
                "ld_score": _optional_float(record.get("ld_score")),
                "tss_dist": _optional_float(record.get("tss_dist")),
                "borzoi_score": score,
                "borzoi_score_id": score_id,
                "target_kind": "teacher_derived_traitgym_native_borzoi_score",
            }
        )
    return rows


def write_cache_package(
    *,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    output_cache: Path = DEFAULT_OUTPUT_CACHE,
    output_manifest: Path = DEFAULT_OUTPUT_MANIFEST,
    output_md: Path = DEFAULT_OUTPUT_MD,
) -> None:
    """Write cache rows, manifest, and human-facing report."""
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(rows, output_cache)
    finalized_manifest = dict(manifest)
    finalized_manifest["cache_artifact"] = _cache_artifact_identity(output_cache, rows)
    output_manifest.write_text(
        json.dumps(finalized_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(render_markdown(finalized_manifest), encoding="utf-8")


def load_cache_manifest(path: Path = DEFAULT_OUTPUT_MANIFEST) -> JsonDict:
    """Load a Borzoi cache manifest for downstream tooling."""
    payload = _load_json(path, label="Borzoi cache manifest")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise InputError(
            "unsupported Borzoi cache manifest schema",
            details={"expected": SCHEMA_VERSION, "actual": payload.get("schema_version")},
        )
    return payload


def read_cache_rows(
    manifest_path: Path = DEFAULT_OUTPUT_MANIFEST,
    *,
    columns: Sequence[str] | None = None,
) -> list[JsonDict]:
    """Read cache rows without exposing fipip or Hugging Face internals."""
    manifest = load_cache_manifest(manifest_path)
    artifact = _require_mapping(manifest["cache_artifact"], "cache_artifact")
    path = Path(str(artifact["path"]))
    if not path.is_absolute():
        path = path if path.is_file() else manifest_path.parent / path.name
    if not path.is_file():
        raise InputError("Borzoi cache artifact not found", details={"path": str(path)})
    if sha256_file(path) != artifact["sha256"]:
        raise InputError(
            "Borzoi cache artifact checksum mismatch",
            details={
                "path": str(path),
                "expected": artifact["sha256"],
                "actual": sha256_file(path),
            },
        )
    pq = cast(Any, importlib.import_module("pyarrow.parquet"))
    table = pq.read_table(path, columns=list(columns) if columns is not None else None)
    return [cast(JsonDict, row) for row in table.to_pylist()]


def render_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a human-facing cache report."""
    artifact = _require_mapping(manifest["cache_artifact"], "cache_artifact")
    split_summary = _require_mapping(manifest["split_summary"], "split_summary")
    label_summary = _require_mapping(manifest["label_summary"], "label_summary")
    return "\n".join(
        [
            "# GenoLeWM-FX Borzoi score cache report",
            "",
            f"Generated by `{manifest['generated_by']}` at `{manifest['generated_at']}`.",
            "",
            f"Parent epic: #{manifest['epic_issue']}. Cache gate: #{manifest['cache_issue']}.",
            "",
            str(manifest["claim_boundary"]),
            "",
            "## Reproduce",
            "",
            "```bash",
            "uv run python -m tools.research.fx_borzoi_cache \\",
            "  --source-manifest configs/fx/borzoi_rescue_sources.json \\",
            "  --overlap-report docs/research/fx-borzoi-overlap-report.json \\",
            "  --output-cache docs/research/fx-borzoi-score-cache.parquet \\",
            "  --output-manifest docs/research/fx-borzoi-cache-manifest.json \\",
            "  --output-md docs/research/fx-borzoi-cache-report.md",
            "```",
            "",
            "## Cache Artifact",
            "",
            "| Artifact | Rows | SHA-256 | Size |",
            "| --- | ---: | --- | ---: |",
            f"| `{artifact['path']}` | {artifact['rows']} | `{artifact['sha256']}` | "
            f"{artifact['size_bytes']} |",
            "",
            "## Row Summary",
            "",
            "| Check | Value |",
            "| --- | ---: |",
            f"| Total rows | {manifest['row_count']} |",
            f"| Train rows | {split_summary['train_rows']} |",
            f"| Holdout rows | {split_summary['holdout_rows']} |",
            f"| Positive labels | {label_summary['positive']} |",
            f"| Negative labels | {label_summary['negative']} |",
            f"| Excluded rows | {manifest['excluded_rows']} |",
            f"| Unmatched rows | {manifest['unmatched_rows']} |",
            f"| Duplicate variant keys | {manifest['duplicate_variant_keys']} |",
            "",
            "The cache stores TraitGym-native row-aligned Borzoi scores as teacher-derived "
            "targets. It distinguishes that path from the optional fipip exact-join lane "
            f"with `fipip_exact_join_status={manifest['fipip_exact_join_status']}`.",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--overlap-report", type=Path, default=DEFAULT_OVERLAP_REPORT)
    parser.add_argument("--output-cache", type=Path, default=DEFAULT_OUTPUT_CACHE)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    try:
        rows, manifest = build_cache_package(
            source_manifest_path=args.source_manifest,
            overlap_report_path=args.overlap_report,
            output_cache=args.output_cache,
            generated_at=args.generated_at,
        )
        write_cache_package(
            rows=rows,
            manifest=manifest,
            output_cache=args.output_cache,
            output_manifest=args.output_manifest,
            output_md=args.output_md,
        )
    except GenoLeWMError as exc:
        print(exc.to_json(), file=sys.stderr)
        return exit_code_for(exc)
    return 0


def _build_cache_manifest(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    source_manifest_path: Path,
    overlap_report: Mapping[str, Any],
    overlap_report_path: Path,
    output_cache: Path,
    generated_at: str,
) -> JsonDict:
    split_counter: Counter[str] = Counter(str(row["split"]) for row in rows)
    label_counter: Counter[int] = Counter(int(row["label"]) for row in rows)
    keys = [(row["chrom"], row["pos"], row["ref"], row["alt"]) for row in rows]
    overlap_alignment = _require_mapping(
        overlap_report["traitgym_native_alignment"], "traitgym_native_alignment"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": generated_at,
        "epic_issue": source_manifest["epic_issue"],
        "contract_issue": source_manifest["contract_issue"],
        "overlap_issue": source_manifest["overlap_issue"],
        "cache_issue": source_manifest["cache_issue"],
        "source_manifest": {
            "path": _repo_relative(source_manifest_path),
            "sha256": sha256_file(source_manifest_path),
        },
        "overlap_report": {
            "path": _repo_relative(overlap_report_path),
            "sha256": sha256_file(overlap_report_path),
            "decision": overlap_report["decision"],
        },
        "cache_artifact": {
            "path": _repo_relative(output_cache),
            "rows": len(rows),
            "sha256": "pending",
            "size_bytes": 0,
        },
        "row_count": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "score_column": "borzoi_score",
        "score_id": source_manifest["traitgym_borzoi_score"]["score_id"],
        "target_kind": "teacher_derived_traitgym_native_borzoi_score",
        "row_alignment_method": "traitgym_native_matched_slice_row_order",
        "genome_build_decision": "No liftover or fipip exact join was performed for this cache.",
        "fipip_exact_join_status": overlap_report["fipip_exact_join"]["status"],
        "source_inputs": overlap_report["source_inputs"],
        "split_summary": {
            "train_rows": split_counter["train"],
            "holdout_rows": split_counter["holdout"],
            "holdout_chromosomes": source_manifest["holdout_chromosomes"],
        },
        "label_summary": {
            "positive": label_counter[1],
            "negative": label_counter[0],
        },
        "excluded_rows": 0,
        "unmatched_rows": 0,
        "duplicate_variant_keys": len(keys) - len(set(keys)),
        "overlap_alignment_rows": overlap_alignment["usable_rows"],
        "claim_boundary": source_manifest["claim_boundary"],
    }


def _assert_overlap_gate_passed(overlap_report: Mapping[str, Any]) -> None:
    if overlap_report.get("decision") != "go_traitgym_native_borzoi":
        raise InputError(
            "Borzoi cache requires a passing TraitGym-native overlap report",
            details={"decision": overlap_report.get("decision")},
        )
    if overlap_report.get("ok_to_build_cache") is not True:
        raise InputError("Borzoi overlap report does not allow cache work")


def _load_traitgym_records(traitgym_slice: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    datasets = cast(Any, importlib.import_module("datasets"))
    dataset = datasets.load_dataset(
        str(traitgym_slice["dataset"]),
        str(traitgym_slice["config"]),
        split=str(traitgym_slice["split"]),
    )
    return [cast(Mapping[str, Any], row) for row in dataset]


def _load_borzoi_scores(borzoi_score: Mapping[str, Any]) -> list[Any]:
    hub = cast(Any, importlib.import_module("huggingface_hub"))
    score_path = Path(
        hub.hf_hub_download(
            str(borzoi_score["repo_id"]),
            str(borzoi_score["artifact_path"]),
            repo_type=str(borzoi_score.get("repo_type", "dataset")),
            revision=str(borzoi_score["revision"]),
        )
    )
    pq = cast(Any, importlib.import_module("pyarrow.parquet"))
    table = pq.read_table(score_path, columns=[str(borzoi_score["score_column"])])
    return list(table.column(str(borzoi_score["score_column"])).to_pylist())


def _write_parquet(rows: Sequence[Mapping[str, Any]], output_cache: Path) -> None:
    pa = cast(Any, importlib.import_module("pyarrow"))
    pq = cast(Any, importlib.import_module("pyarrow.parquet"))
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, output_cache)


def _cache_artifact_identity(path: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "path": _repo_relative(path),
        "rows": len(rows),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path, *, label: str) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(f"{label} not found", details={"path": str(path)}) from exc
    if not isinstance(payload, dict):
        raise InputError(f"{label} must be a JSON object", details={"path": str(path)})
    return cast(JsonDict, payload)


def _variant_key(record: Mapping[str, Any]) -> tuple[str, int, str, str]:
    chrom = str(record["chrom"]).removeprefix("chr").removeprefix("Chr")
    pos = int(record["pos"])
    ref = str(record["ref"]).upper()
    alt = str(record["alt"]).upper()
    if not chrom or pos <= 0 or not ref or not alt:
        raise InputError("invalid variant key", details={"record": dict(record)})
    return chrom, pos, ref, alt


def _label(record: Mapping[str, Any]) -> int:
    raw_label = record["label"]
    if isinstance(raw_label, bool):
        return int(raw_label)
    label = int(raw_label)
    if label not in {0, 1}:
        raise InputError("TraitGym labels must be binary", details={"label": raw_label})
    return label


def _finite_float(raw_value: Any, *, label: str) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{label} must be numeric", details={"value": raw_value}) from exc
    if not math.isfinite(value):
        raise InputError(f"{label} must be finite", details={"value": raw_value})
    return value


def _optional_float(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    return _finite_float(raw_value, label="optional numeric field")


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
