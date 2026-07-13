# SPDX-License-Identifier: Apache-2.0
"""ClinVar local VCF preparation and shard loading."""

from __future__ import annotations

import importlib
import time
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from geno_lewm.data._vcf import (
    info_value_for_alt,
    is_supported_allele,
    iter_vcf_rows,
    parse_int,
)
from geno_lewm.errors import InputError, RuntimeSetupError
from geno_lewm.provenance import sha256_file

__all__ = [
    "CLINVAR_LABELLED_CLASSES",
    "CLINVAR_SCHEMA_VERSION",
    "ClinvarPrepareReport",
    "ClinvarVariant",
    "iter_clinvar_shard",
    "iter_clinvar_vcf_variants",
    "label_set",
    "prepare_clinvar_shard",
]

CLINVAR_SCHEMA_VERSION = "1.0.0"
CLINVAR_LABELLED_CLASSES = frozenset({"P", "LP", "B", "LB"})
_PARQUET_BATCH_ROWS = 100_000


@dataclass(frozen=True, slots=True)
class ClinvarVariant:
    """One normalized ClinVar row."""

    chrom: str
    pos: int
    ref: str
    alt: str
    clinical_significance: str
    review_status: str
    gene_symbol: str | None
    clinvar_id: int
    schema_version: str = CLINVAR_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "chrom": self.chrom,
            "pos": self.pos,
            "ref": self.ref,
            "alt": self.alt,
            "clinical_significance": self.clinical_significance,
            "review_status": self.review_status,
            "gene_symbol": self.gene_symbol,
            "clinvar_id": self.clinvar_id,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class ClinvarPrepareReport:
    """Summary emitted by ``geno-lewm-prepare-clinvar``."""

    output_path: Path
    release: str
    records_read: int
    allele_records_seen: int
    records_written: int
    skipped_allele: int
    size_bytes: int
    already_exists: bool = False
    input_path: Path | None = field(default=None, init=False)
    input_sha256: str | None = field(default=None, init=False)
    output_sha256: str | None = field(default=None, init=False)
    input_size_bytes: int | None = field(default=None, init=False)
    elapsed_seconds: float = field(default=0.0, init=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "output_path": str(self.output_path),
            "input_path": None if self.input_path is None else str(self.input_path),
            "release": self.release,
            "records_read": self.records_read,
            "allele_records_seen": self.allele_records_seen,
            "records_written": self.records_written,
            "skipped_allele": self.skipped_allele,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "input_size_bytes": self.input_size_bytes,
            "size_bytes": self.size_bytes,
            "elapsed_seconds": self.elapsed_seconds,
            "already_exists": self.already_exists,
        }


def prepare_clinvar_shard(
    input_vcf: str | Path,
    output_dir: str | Path,
    *,
    release: str,
    max_allele_len: int = 16,
    overwrite: bool = False,
) -> ClinvarPrepareReport:
    """Normalize a local ClinVar VCF/VCF.gz into the release shard schema."""
    _require_release(release)
    _require_positive_int("max_allele_len", max_allele_len)
    started_at = time.perf_counter()
    input_path, input_sha256, input_size_bytes = _input_file_identity(input_vcf)
    target = Path(output_dir) / "clinvar" / release / "variants.parquet"
    if target.exists() and not overwrite:
        return _with_prepare_identity(
            ClinvarPrepareReport(
                output_path=target,
                release=release,
                records_read=0,
                allele_records_seen=0,
                records_written=_parquet_num_rows(target),
                skipped_allele=0,
                size_bytes=target.stat().st_size,
                already_exists=True,
            ),
            input_path=input_path,
            input_sha256=input_sha256,
            output_sha256=sha256_file(target),
            input_size_bytes=input_size_bytes,
            elapsed_seconds=max(time.perf_counter() - started_at, 0.0),
        )

    records_read = 0
    allele_records_seen = 0
    skipped_allele = 0

    def _selected_rows() -> Iterator[ClinvarVariant]:
        nonlocal records_read, allele_records_seen, skipped_allele
        for row in iter_vcf_rows(input_vcf):
            records_read += 1
            for alt_index, alt in enumerate(row.alts):
                allele_records_seen += 1
                if (
                    not is_supported_allele(row.ref, max_len=max_allele_len)
                    or not is_supported_allele(alt, max_len=max_allele_len)
                    or row.ref == alt
                ):
                    skipped_allele += 1
                    continue
                yield ClinvarVariant(
                    chrom=row.chrom,
                    pos=row.pos,
                    ref=row.ref,
                    alt=alt,
                    clinical_significance=_clinical_significance(row.info, alt_index),
                    review_status=_review_status(row.info),
                    gene_symbol=_gene_symbol(row.info),
                    clinvar_id=_clinvar_id(row.info, row.variant_id, alt_index),
                )

    records_written = _write_parquet(_selected_rows(), target)
    return _with_prepare_identity(
        ClinvarPrepareReport(
            output_path=target,
            release=release,
            records_read=records_read,
            allele_records_seen=allele_records_seen,
            records_written=records_written,
            skipped_allele=skipped_allele,
            size_bytes=target.stat().st_size,
        ),
        input_path=input_path,
        input_sha256=input_sha256,
        output_sha256=sha256_file(target),
        input_size_bytes=input_size_bytes,
        elapsed_seconds=max(time.perf_counter() - started_at, 1e-9),
    )


def iter_clinvar_vcf_variants(
    input_vcf: str | Path,
    *,
    max_allele_len: int = 16,
) -> Iterator[ClinvarVariant]:
    """Yield normalized ClinVar rows from a local VCF without writing a shard."""
    _require_positive_int("max_allele_len", max_allele_len)
    for row in iter_vcf_rows(input_vcf):
        for alt_index, alt in enumerate(row.alts):
            if (
                not is_supported_allele(row.ref, max_len=max_allele_len)
                or not is_supported_allele(alt, max_len=max_allele_len)
                or row.ref == alt
            ):
                continue
            yield ClinvarVariant(
                chrom=row.chrom,
                pos=row.pos,
                ref=row.ref,
                alt=alt,
                clinical_significance=_clinical_significance(row.info, alt_index),
                review_status=_review_status(row.info),
                gene_symbol=_gene_symbol(row.info),
                clinvar_id=_clinvar_id(row.info, row.variant_id, alt_index),
            )


def iter_clinvar_shard(path: str | Path) -> Iterator[ClinvarVariant]:
    """Yield normalized ClinVar rows from a Parquet shard."""
    _pa, pq = _require_pyarrow()
    table = pq.read_table(Path(path))
    for row in table.to_pylist():
        yield ClinvarVariant(
            chrom=str(row["chrom"]),
            pos=int(row["pos"]),
            ref=str(row["ref"]),
            alt=str(row["alt"]),
            clinical_significance=str(row["clinical_significance"]),
            review_status=str(row["review_status"]),
            gene_symbol=None if row.get("gene_symbol") is None else str(row["gene_symbol"]),
            clinvar_id=int(row["clinvar_id"]),
            schema_version=str(row["schema_version"]),
        )


def label_set(variants: Iterable[ClinvarVariant]) -> tuple[ClinvarVariant, ...]:
    """Return ClinVar rows usable for labelled eval, excluding VUS/OTHER."""
    return tuple(row for row in variants if row.clinical_significance in CLINVAR_LABELLED_CLASSES)


def _clinical_significance(info: dict[str, str | bool], alt_index: int) -> str:
    raw = info_value_for_alt(info, ("CLNSIG", "CLNSIGCONF"), alt_index) or ""
    normalized = raw.lower().replace("%2f", "/").replace("%20", "_")
    tokens = {token for chunk in normalized.split("|") for token in chunk.split("/")}
    if "uncertain_significance" in tokens or "vus" in tokens:
        return "VUS"
    if "likely_pathogenic" in tokens:
        return "LP"
    if "pathogenic" in tokens:
        return "P"
    if "likely_benign" in tokens:
        return "LB"
    if "benign" in tokens:
        return "B"
    return "OTHER"


def _review_status(info: dict[str, str | bool]) -> str:
    raw = info.get("CLNREVSTAT")
    if raw is None or isinstance(raw, bool) or not raw:
        return "unknown"
    return raw.replace("%20", "_")


def _gene_symbol(info: dict[str, str | bool]) -> str | None:
    raw = info.get("GENEINFO")
    if raw is None or isinstance(raw, bool) or not raw or raw == ".":
        return None
    first = raw.split("|", maxsplit=1)[0]
    symbol = first.split(":", maxsplit=1)[0]
    return symbol or None


def _clinvar_id(info: dict[str, str | bool], row_id: str, alt_index: int) -> int:
    for key in ("CLNVID", "ALLELEID"):
        parsed = parse_int(info_value_for_alt(info, (key,), alt_index))
        if parsed is not None:
            return parsed
    parsed_id = parse_int(row_id)
    if parsed_id is not None:
        return parsed_id
    raise InputError(
        "ClinVar row must contain CLNVID, ALLELEID, or a numeric VCF ID",
        details={"id": row_id},
    )


def _write_parquet(rows: Iterator[ClinvarVariant], target: Path) -> int:
    pa, pq = _require_pyarrow()
    target.parent.mkdir(parents=True, exist_ok=True)
    schema = _parquet_schema(pa)
    tmp = target.with_name(target.name + ".tmp")
    with suppress(OSError):
        tmp.unlink()
    writer: Any | None = None
    batch: list[ClinvarVariant] = []
    written = 0
    try:
        for row in rows:
            batch.append(row)
            if len(batch) >= _PARQUET_BATCH_ROWS:
                writer = _write_batch(batch, tmp=tmp, schema=schema, pa=pa, pq=pq, writer=writer)
                written += len(batch)
                batch.clear()
        if batch:
            writer = _write_batch(batch, tmp=tmp, schema=schema, pa=pa, pq=pq, writer=writer)
            written += len(batch)
        if writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=schema), tmp)
        else:
            writer.close()
            writer = None
        tmp.replace(target)
    except Exception:
        if writer is not None:
            with suppress(Exception):
                writer.close()
        with suppress(OSError):
            tmp.unlink()
        raise
    return written


def _write_batch(
    rows: list[ClinvarVariant],
    *,
    tmp: Path,
    schema: Any,
    pa: Any,
    pq: Any,
    writer: Any | None,
) -> Any:
    if writer is None:
        writer = pq.ParquetWriter(tmp, schema)
    writer.write_table(pa.Table.from_pylist([row.to_dict() for row in rows], schema=schema))
    return writer


def _parquet_schema(pa: Any) -> Any:
    return pa.schema(
        [
            ("chrom", pa.string()),
            ("pos", pa.int64()),
            ("ref", pa.string()),
            ("alt", pa.string()),
            ("clinical_significance", pa.string()),
            ("review_status", pa.string()),
            ("gene_symbol", pa.string()),
            ("clinvar_id", pa.int64()),
            ("schema_version", pa.string()),
        ]
    )


def _parquet_num_rows(path: Path) -> int:
    _pa, pq = _require_pyarrow()
    return int(pq.ParquetFile(path).metadata.num_rows)


def _with_prepare_identity(
    report: ClinvarPrepareReport,
    *,
    input_path: Path,
    input_sha256: str,
    output_sha256: str,
    input_size_bytes: int,
    elapsed_seconds: float,
) -> ClinvarPrepareReport:
    object.__setattr__(report, "input_path", input_path)
    object.__setattr__(report, "input_sha256", input_sha256)
    object.__setattr__(report, "output_sha256", output_sha256)
    object.__setattr__(report, "input_size_bytes", input_size_bytes)
    object.__setattr__(report, "elapsed_seconds", elapsed_seconds)
    return report


def _input_file_identity(path: str | Path) -> tuple[Path, str, int]:
    input_path = Path(path)
    try:
        stat = input_path.stat()
        digest = sha256_file(input_path)
    except OSError as exc:
        raise InputError(
            "failed to read input VCF identity",
            details={"path": str(input_path)},
        ) from exc
    if not input_path.is_file():
        raise InputError("input VCF must be a file", details={"path": str(input_path)})
    return input_path, digest, stat.st_size


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        pa = importlib.import_module("pyarrow")
        pq = importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise RuntimeSetupError(
            "ClinVar shard preparation requires pyarrow",
            remediation="install geno-lewm[dev], geno-lewm[train], or pyarrow",
        ) from exc
    return pa, pq


def _require_release(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise InputError("release must be a non-empty string")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputError(
            f"{name} must be a positive integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )
