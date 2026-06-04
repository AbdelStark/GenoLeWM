# SPDX-License-Identifier: Apache-2.0
"""Generate split-integrity evidence for a release dataset package."""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, TextIO

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file
from geno_lewm.provenance.hashing import looks_like_sha256

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.dataset_integrity"
DEFAULT_REPORT_NAME: Final = "split_integrity.json"
_EVAL_PREFIXES: Final = ("eval", "test", "holdout", "validation", "val")
_LABEL_ALIASES: Final = {
    "P": "P",
    "PATHOGENIC": "P",
    "LP": "LP",
    "LIKELY_PATHOGENIC": "LP",
    "B": "B",
    "BENIGN": "B",
    "LB": "LB",
    "LIKELY_BENIGN": "LB",
    "VUS": "VUS",
    "UNCERTAIN_SIGNIFICANCE": "VUS",
    "UNCERTAIN": "VUS",
}


@dataclass(frozen=True, slots=True)
class IntegrityFile:
    """Observed record-count and key evidence for one dataset file."""

    path: str
    split: str
    records: int
    sha256: str
    size_bytes: int
    comparable_keys: int
    label_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "split": self.split,
            "records": self.records,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "comparable_keys": self.comparable_keys,
            "label_counts": dict(sorted(self.label_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class SplitIntegrityReport:
    """Machine-readable split integrity evidence for a dataset snapshot."""

    schema_version: str
    generated_by: str
    snapshot_id: str
    generated_at: str
    manifest_sha256: str
    files: tuple[IntegrityFile, ...]
    splits: dict[str, dict[str, object]]
    leakage_checks: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "snapshot_id": self.snapshot_id,
            "generated_at": self.generated_at,
            "manifest_sha256": self.manifest_sha256,
            "files": [file.to_dict() for file in self.files],
            "splits": self.splits,
            "leakage_checks": list(self.leakage_checks),
        }


def build_dataset_integrity_report(
    dataset_dir: Path,
    manifest_path: Path | None = None,
    *,
    generated_at: str | None = None,
) -> SplitIntegrityReport:
    """Build split and leakage evidence from ``dataset_manifest.json``."""
    manifest_path = (
        dataset_dir / "dataset_manifest.json" if manifest_path is None else manifest_path
    )
    payload = _load_manifest(manifest_path)
    snapshot_id = _required_text(payload, "snapshot_id")
    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict) or not raw_splits:
        raise InputError("dataset manifest splits must be a non-empty object")
    split_specs = _parse_split_specs(raw_splits)
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise InputError("dataset manifest files must be a non-empty list")

    files: list[IntegrityFile] = []
    split_counts: dict[str, int] = dict.fromkeys(split_specs, 0)
    split_keys: dict[str, set[str]] = {split: set() for split in split_specs}
    split_label_counts: dict[str, dict[str, int]] = {split: {} for split in split_specs}
    for index, item in enumerate(raw_files):
        if not isinstance(item, dict):
            raise InputError(
                "dataset manifest file entries must be objects", details={"index": index}
            )
        file_report, keys = _inspect_file(
            dataset_dir, item, index=index, splits=frozenset(split_specs)
        )
        files.append(file_report)
        split_counts[file_report.split] = (
            split_counts.get(file_report.split, 0) + file_report.records
        )
        split_keys.setdefault(file_report.split, set()).update(keys)
        _merge_counts(
            split_label_counts.setdefault(file_report.split, {}), file_report.label_counts
        )

    splits: dict[str, dict[str, object]] = {}
    for split, spec in split_specs.items():
        observed = split_counts.get(split, 0)
        declared = spec["records"]
        if observed != declared:
            raise InputError(
                "dataset split record count mismatch",
                details={"split": split, "declared": declared, "observed": observed},
            )
        splits[split] = {
            "declared_records": declared,
            "observed_records": observed,
            "files": [file.path for file in files if file.split == split],
            "label_counts": dict(sorted(split_label_counts.get(split, {}).items())),
            "labelled_records": sum(split_label_counts.get(split, {}).values()),
            "unlabelled_records": max(
                0, observed - sum(split_label_counts.get(split, {}).values())
            ),
        }
        description = spec.get("description")
        if isinstance(description, str):
            splits[split]["description"] = description

    leakage_checks = _build_leakage_checks(split_keys)
    for check in leakage_checks:
        if check["status"] != "passed":
            raise InputError(
                "dataset split leakage check failed",
                details={
                    "check": check["name"],
                    "split_a": check["split_a"],
                    "split_b": check["split_b"],
                    "failure_reason": check["failure_reason"],
                    "split_a_comparable_keys": check["split_a_comparable_keys"],
                    "split_b_comparable_keys": check["split_b_comparable_keys"],
                    "overlap_count": check["overlap_count"],
                    "examples": check["examples"],
                },
            )

    return SplitIntegrityReport(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        snapshot_id=snapshot_id,
        generated_at=_utc_now() if generated_at is None else generated_at,
        manifest_sha256=sha256_file(manifest_path),
        files=tuple(files),
        splits=splits,
        leakage_checks=tuple(leakage_checks),
    )


def write_dataset_integrity_report(
    dataset_dir: Path,
    *,
    manifest_path: Path | None = None,
    output_path: Path | None = None,
    generated_at: str | None = None,
) -> SplitIntegrityReport:
    """Write ``split_integrity.json`` and return the generated report."""
    report = build_dataset_integrity_report(
        dataset_dir,
        manifest_path,
        generated_at=generated_at,
    )
    output_path = dataset_dir / DEFAULT_REPORT_NAME if output_path is None else output_path
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = write_dataset_integrity_report(
            args.dataset_dir,
            manifest_path=args.manifest,
            output_path=args.output,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build split-integrity evidence for a release dataset package.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError("failed to read dataset manifest", details={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "dataset manifest JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("dataset manifest must be a JSON object")
    schema_version = _required_text(payload, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise InputError(
            "unsupported dataset manifest schema version",
            details={"expected": SCHEMA_VERSION, "observed": schema_version},
        )
    return payload


def _parse_split_specs(raw_splits: dict[str, Any]) -> dict[str, dict[str, object]]:
    splits: dict[str, dict[str, object]] = {}
    for split, raw in raw_splits.items():
        if not isinstance(split, str) or not split:
            raise InputError("dataset split names must be non-empty strings")
        if not isinstance(raw, dict):
            raise InputError("dataset split entries must be objects", details={"split": split})
        records = raw.get("records")
        if isinstance(records, bool) or not isinstance(records, int) or records < 0:
            raise InputError(
                "dataset split records must be a non-negative integer",
                details={"split": split},
            )
        entry: dict[str, object] = {"records": records}
        description = raw.get("description")
        if isinstance(description, str) and description:
            entry["description"] = description
        splits[split] = entry
    return splits


def _inspect_file(
    dataset_dir: Path,
    raw_file: dict[str, Any],
    *,
    index: int,
    splits: frozenset[str],
) -> tuple[IntegrityFile, set[str]]:
    relative = _required_text(raw_file, "path", prefix=f"files[{index}].")
    split = _required_text(raw_file, "split", prefix=f"files[{index}].")
    if split not in splits:
        raise InputError(
            "dataset file split is not declared", details={"path": relative, "split": split}
        )
    path = _safe_relative(dataset_dir, relative)
    if not path.is_file():
        raise InputError("dataset file is missing", details={"path": str(path)})
    expected_hash = _required_text(raw_file, "sha256", prefix=f"files[{index}].")
    if not looks_like_sha256(expected_hash):
        raise InputError("dataset file sha256 is invalid", details={"path": relative})
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise InputError(
            "dataset file hash mismatch",
            details={"path": relative, "expected": expected_hash, "observed": observed_hash},
        )
    size_bytes = raw_file.get("size_bytes")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise InputError(
            "dataset file size_bytes must be a non-negative integer", details={"path": relative}
        )
    observed_size = path.stat().st_size
    if observed_size != size_bytes:
        raise InputError(
            "dataset file size mismatch",
            details={"path": relative, "expected": size_bytes, "observed": observed_size},
        )
    observed_records, keys, label_counts = _read_records(path)
    declared_records = raw_file.get("records")
    if declared_records is not None:
        if (
            isinstance(declared_records, bool)
            or not isinstance(declared_records, int)
            or declared_records < 0
        ):
            raise InputError(
                "dataset file records must be a non-negative integer", details={"path": relative}
            )
        if observed_records is not None and observed_records != declared_records:
            raise InputError(
                "dataset file record count mismatch",
                details={
                    "path": relative,
                    "declared": declared_records,
                    "observed": observed_records,
                },
            )
        records = declared_records
    elif observed_records is not None:
        records = observed_records
    else:
        raise InputError(
            "dataset file format is not record-countable; supply files[].records",
            details={"path": relative},
        )
    return (
        IntegrityFile(
            path=relative,
            split=split,
            records=records,
            sha256=observed_hash,
            size_bytes=observed_size,
            comparable_keys=len(keys),
            label_counts=label_counts,
        ),
        keys,
    )


def _read_records(path: Path) -> tuple[int | None, set[str], dict[str, int]]:
    name = path.name.lower()
    if name.endswith(".jsonl"):
        return _read_jsonl_records(path)
    if name.endswith((".vcf", ".vcf.gz")):
        return _read_vcf_records(path)
    if name.endswith(".parquet"):
        return _read_parquet_records(path)
    if name.endswith((".txt", ".tsv", ".csv")):
        return _read_line_records(path)
    return None, set(), {}


def _read_parquet_records(path: Path) -> tuple[int, set[str], dict[str, int]]:
    pq = _require_pyarrow_parquet()
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:
        raise InputError("dataset Parquet file is invalid", details={"path": str(path)}) from exc
    records = int(parquet_file.metadata.num_rows)
    schema = parquet_file.schema_arrow
    names = set(schema.names)
    variant_columns = ("chrom", "pos", "ref", "alt")
    has_variant_columns = set(variant_columns) <= names
    label_columns = tuple(
        column for column in ("clinical_significance", "label", "clnsig") if column in names
    )
    if not has_variant_columns and not label_columns:
        return records, set(), {}
    keys: set[str] = set()
    label_counts: dict[str, int] = {}
    columns = [*variant_columns] if has_variant_columns else []
    columns.extend(column for column in label_columns if column not in columns)
    try:
        batches = parquet_file.iter_batches(columns=columns, batch_size=100_000)
        for batch in batches:
            values = batch.to_pydict()
            for index in range(batch.num_rows):
                if has_variant_columns:
                    chrom = values["chrom"][index]
                    pos = values["pos"][index]
                    ref = values["ref"][index]
                    alt = values["alt"][index]
                    if None not in (chrom, pos, ref, alt):
                        keys.add(_variant_key(str(chrom), str(pos), str(ref), str(alt)))
                _add_label_count(label_counts, _first_label_value(values, label_columns, index))
    except Exception as exc:
        raise InputError(
            "failed to inspect dataset Parquet variant keys", details={"path": str(path)}
        ) from exc
    return records, keys, label_counts


def _read_jsonl_records(path: Path) -> tuple[int, set[str], dict[str, int]]:
    keys: set[str] = set()
    label_counts: dict[str, int] = {}
    records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            records += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(
                    "dataset JSONL row is invalid",
                    details={"path": str(path), "line": line_no},
                ) from exc
            if isinstance(row, dict):
                key = _key_from_mapping(row)
                if key is not None:
                    keys.add(key)
                _add_label_count(label_counts, _label_from_mapping(row))
    return records, keys, label_counts


def _read_vcf_records(path: Path) -> tuple[int, set[str], dict[str, int]]:
    keys: set[str] = set()
    label_counts: dict[str, int] = {}
    records = 0
    with _open_text(path) as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                raise InputError(
                    "VCF row must have at least 5 columns",
                    details={"path": str(path), "line": line_no},
                )
            chrom, pos, _id, ref, alts = parts[:5]
            info = _parse_vcf_info(parts[7] if len(parts) > 7 else "")
            alt_values = [alt for alt in alts.split(",") if alt]
            records += max(1, len(alt_values))
            for alt_index, alt in enumerate(alt_values):
                keys.add(_variant_key(chrom, pos, ref, alt))
                _add_label_count(
                    label_counts,
                    _info_value_for_alt(info, ("CLNSIG", "CLNSIGCONF", "label"), alt_index),
                )
    return records, keys, label_counts


def _read_line_records(path: Path) -> tuple[int, set[str], dict[str, int]]:
    records = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line and not line.startswith("#"):
                records += 1
    return records, set(), {}


def _open_text(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _key_from_mapping(row: dict[str, Any]) -> str | None:
    for key in ("locus_key", "variant_key"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    chrom = row.get("chrom")
    pos = row.get("pos")
    ref = row.get("ref")
    alt = row.get("alt")
    if all(isinstance(value, str | int) for value in (chrom, pos, ref, alt)):
        return _variant_key(str(chrom), str(pos), str(ref), str(alt))
    record_id = row.get("record_id")
    if isinstance(record_id, str) and record_id:
        return "record:" + record_id
    return None


def _label_from_mapping(row: dict[str, Any]) -> object | None:
    for key in ("clinical_significance", "label", "clnsig"):
        value: object | None = row.get(key)
        if value is not None:
            return value
    return None


def _first_label_value(
    values: dict[str, list[Any]],
    label_columns: tuple[str, ...],
    index: int,
) -> object | None:
    for column in label_columns:
        value: object | None = values[column][index]
        if value is not None:
            return value
    return None


def _add_label_count(counts: dict[str, int], raw: object | None) -> None:
    label = _canonical_label(raw)
    if label is not None:
        counts[label] = counts.get(label, 0) + 1


def _canonical_label(raw: object | None) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return "positive" if raw else "negative"
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        if raw == 1:
            return "positive"
        if raw == 0:
            return "negative"
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or text == ".":
        return None
    normalized = (
        text.upper().replace("%2F", "/").replace("%20", "_").replace(" ", "_").replace("-", "_")
    )
    tokens = {
        token
        for token in re.split(r"[/|,;&]+", normalized)
        if token and token not in {".", "NA", "N/A"}
    }
    mapped = {_LABEL_ALIASES.get(token, token) for token in tokens}
    if "VUS" in mapped:
        return "VUS"
    if "LP" in mapped:
        return "LP"
    if "P" in mapped:
        return "P"
    if "LB" in mapped:
        return "LB"
    if "B" in mapped:
        return "B"
    return "OTHER"


def _parse_vcf_info(raw: str) -> dict[str, str | bool]:
    if raw in {"", "."}:
        return {}
    info: dict[str, str | bool] = {}
    for item in raw.split(";"):
        if not item:
            continue
        if "=" not in item:
            info[item] = True
            continue
        key, value = item.split("=", maxsplit=1)
        info[key] = value
    return info


def _info_value_for_alt(
    info: dict[str, str | bool],
    keys: tuple[str, ...],
    alt_index: int,
) -> str | None:
    for key in keys:
        raw = info.get(key)
        if raw is None or isinstance(raw, bool):
            continue
        values = raw.split(",")
        if len(values) == 1:
            return values[0]
        if alt_index < len(values):
            return values[alt_index]
    return None


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for label, count in source.items():
        target[label] = target.get(label, 0) + count


def _build_leakage_checks(split_keys: dict[str, set[str]]) -> tuple[dict[str, object], ...]:
    train_splits = [split for split in split_keys if split.lower().startswith("train")]
    eval_splits = [
        split
        for split in split_keys
        if not split.lower().startswith("train") and split.lower().startswith(_EVAL_PREFIXES)
    ]
    checks: list[dict[str, object]] = []
    for train_split in train_splits:
        for eval_split in eval_splits:
            train_keys = split_keys[train_split]
            eval_keys = split_keys[eval_split]
            overlap = sorted(train_keys & eval_keys)
            missing_comparable_keys = not train_keys or not eval_keys
            status = "failed" if missing_comparable_keys or overlap else "passed"
            checks.append(
                {
                    "name": "no_shared_comparable_keys_between_train_and_eval",
                    "split_a": train_split,
                    "split_b": eval_split,
                    "status": status,
                    "failure_reason": (
                        "missing_comparable_keys" if missing_comparable_keys else ""
                    ),
                    "split_a_comparable_keys": len(train_keys),
                    "split_b_comparable_keys": len(eval_keys),
                    "overlap_count": len(overlap),
                    "examples": overlap[:20],
                }
            )
    if not checks:
        checks.append(
            {
                "name": "no_shared_comparable_keys_between_train_and_eval",
                "split_a": "",
                "split_b": "",
                "status": "failed",
                "failure_reason": "missing_train_or_eval_split",
                "split_a_comparable_keys": 0,
                "split_b_comparable_keys": 0,
                "overlap_count": 0,
                "examples": [],
            }
        )
    return tuple(checks)


def _variant_key(chrom: str, pos: str, ref: str, alt: str) -> str:
    return f"{chrom}:{pos}:{ref.upper()}>{alt.upper()}"


def _require_pyarrow_parquet() -> Any:
    try:
        return importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise InputError(
            "dataset Parquet integrity checks require pyarrow",
            remediation="install geno-lewm[dev], geno-lewm[train], or pyarrow",
        ) from exc


def _safe_relative(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise InputError(
            "dataset paths must be relative and stay inside dataset_dir",
            details={"path": relative},
        )
    return root / candidate


def _required_text(payload: dict[str, Any], key: str, *, prefix: str = "") -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{prefix}{key} must be a non-empty string")
    return value.strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
