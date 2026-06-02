# SPDX-License-Identifier: Apache-2.0
"""gnomAD local VCF preparation and shard loading."""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geno_lewm.data._vcf import (
    info_value_for_alt,
    is_supported_allele,
    iter_vcf_rows,
    parse_float,
)
from geno_lewm.errors import InputError, RuntimeSetupError

__all__ = [
    "GNOMAD_SCHEMA_VERSION",
    "GnomadPrepareReport",
    "GnomadVariant",
    "iter_gnomad_shard",
    "iter_gnomad_vcf_variants",
    "prepare_gnomad_shard",
]

GNOMAD_SCHEMA_VERSION = "1.0.0"
GNOMAD_POPULATIONS: tuple[str, ...] = (
    "afr",
    "ami",
    "amr",
    "asj",
    "eas",
    "fin",
    "nfe",
    "oth",
    "sas",
)
_PARQUET_BATCH_ROWS = 100_000


@dataclass(frozen=True, slots=True)
class GnomadVariant:
    """One normalized common-variant row for the gnomAD shard."""

    chrom: str
    pos: int
    ref: str
    alt: str
    af_global: float
    af_afr: float | None
    af_ami: float | None
    af_amr: float | None
    af_asj: float | None
    af_eas: float | None
    af_fin: float | None
    af_nfe: float | None
    af_oth: float | None
    af_sas: float | None
    filter: str
    schema_version: str = GNOMAD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "chrom": self.chrom,
            "pos": self.pos,
            "ref": self.ref,
            "alt": self.alt,
            "af_global": self.af_global,
            "af_afr": self.af_afr,
            "af_ami": self.af_ami,
            "af_amr": self.af_amr,
            "af_asj": self.af_asj,
            "af_eas": self.af_eas,
            "af_fin": self.af_fin,
            "af_nfe": self.af_nfe,
            "af_oth": self.af_oth,
            "af_sas": self.af_sas,
            "filter": self.filter,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class GnomadPrepareReport:
    """Summary emitted by ``geno-lewm-prepare-gnomad``."""

    output_path: Path
    release: str
    records_read: int
    allele_records_seen: int
    records_written: int
    skipped_filter: int
    skipped_af: int
    skipped_allele: int
    size_bytes: int
    already_exists: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "output_path": str(self.output_path),
            "release": self.release,
            "records_read": self.records_read,
            "allele_records_seen": self.allele_records_seen,
            "records_written": self.records_written,
            "skipped_filter": self.skipped_filter,
            "skipped_af": self.skipped_af,
            "skipped_allele": self.skipped_allele,
            "size_bytes": self.size_bytes,
            "already_exists": self.already_exists,
        }


def prepare_gnomad_shard(
    input_vcf: str | Path,
    output_dir: str | Path,
    *,
    release: str = "v4.1",
    min_af: float = 0.01,
    max_allele_len: int = 16,
    overwrite: bool = False,
) -> GnomadPrepareReport:
    """Filter a local gnomAD VCF/VCF.gz into the release shard schema."""
    _require_release(release)
    _require_probability("min_af", min_af)
    _require_positive_int("max_allele_len", max_allele_len)

    target = Path(output_dir) / "gnomad" / release / "variants.parquet"
    if target.exists() and not overwrite:
        return GnomadPrepareReport(
            output_path=target,
            release=release,
            records_read=0,
            allele_records_seen=0,
            records_written=_parquet_num_rows(target),
            skipped_filter=0,
            skipped_af=0,
            skipped_allele=0,
            size_bytes=target.stat().st_size,
            already_exists=True,
        )

    records_read = 0
    allele_records_seen = 0
    skipped_filter = 0
    skipped_af = 0
    skipped_allele = 0

    def _selected_rows() -> Iterator[GnomadVariant]:
        nonlocal records_read, allele_records_seen, skipped_filter, skipped_af, skipped_allele
        for row in iter_vcf_rows(input_vcf):
            records_read += 1
            for alt_index, alt in enumerate(row.alts):
                allele_records_seen += 1
                if row.filter != "PASS":
                    skipped_filter += 1
                    continue
                if not is_supported_allele(
                    row.ref, max_len=max_allele_len
                ) or not is_supported_allele(alt, max_len=max_allele_len):
                    skipped_allele += 1
                    continue
                af_global = _af_for(row.info, ("AF", "AF_global", "AF_GLOBAL"), alt_index)
                if af_global is None or af_global < min_af:
                    skipped_af += 1
                    continue
                yield GnomadVariant(
                    chrom=row.chrom,
                    pos=row.pos,
                    ref=row.ref,
                    alt=alt,
                    af_global=af_global,
                    af_afr=_af_for(row.info, ("AF_afr", "AF_AFR"), alt_index),
                    af_ami=_af_for(row.info, ("AF_ami", "AF_AMI"), alt_index),
                    af_amr=_af_for(row.info, ("AF_amr", "AF_AMR"), alt_index),
                    af_asj=_af_for(row.info, ("AF_asj", "AF_ASJ"), alt_index),
                    af_eas=_af_for(row.info, ("AF_eas", "AF_EAS"), alt_index),
                    af_fin=_af_for(row.info, ("AF_fin", "AF_FIN"), alt_index),
                    af_nfe=_af_for(row.info, ("AF_nfe", "AF_NFE"), alt_index),
                    af_oth=_af_for(row.info, ("AF_oth", "AF_OTH"), alt_index),
                    af_sas=_af_for(row.info, ("AF_sas", "AF_SAS"), alt_index),
                    filter=row.filter,
                )

    records_written = _write_parquet(_selected_rows(), target)
    return GnomadPrepareReport(
        output_path=target,
        release=release,
        records_read=records_read,
        allele_records_seen=allele_records_seen,
        records_written=records_written,
        skipped_filter=skipped_filter,
        skipped_af=skipped_af,
        skipped_allele=skipped_allele,
        size_bytes=target.stat().st_size,
    )


def iter_gnomad_vcf_variants(
    input_vcf: str | Path,
    *,
    min_af: float = 0.01,
    max_allele_len: int = 16,
) -> Iterator[GnomadVariant]:
    """Yield normalized rows from a local gnomAD VCF without writing a shard."""
    report = prepare_gnomad_shard
    del report
    _require_probability("min_af", min_af)
    _require_positive_int("max_allele_len", max_allele_len)
    for row in iter_vcf_rows(input_vcf):
        for alt_index, alt in enumerate(row.alts):
            if row.filter != "PASS":
                continue
            if not is_supported_allele(row.ref, max_len=max_allele_len) or not is_supported_allele(
                alt, max_len=max_allele_len
            ):
                continue
            af_global = _af_for(row.info, ("AF", "AF_global", "AF_GLOBAL"), alt_index)
            if af_global is None or af_global < min_af:
                continue
            yield GnomadVariant(
                chrom=row.chrom,
                pos=row.pos,
                ref=row.ref,
                alt=alt,
                af_global=af_global,
                af_afr=_af_for(row.info, ("AF_afr", "AF_AFR"), alt_index),
                af_ami=_af_for(row.info, ("AF_ami", "AF_AMI"), alt_index),
                af_amr=_af_for(row.info, ("AF_amr", "AF_AMR"), alt_index),
                af_asj=_af_for(row.info, ("AF_asj", "AF_ASJ"), alt_index),
                af_eas=_af_for(row.info, ("AF_eas", "AF_EAS"), alt_index),
                af_fin=_af_for(row.info, ("AF_fin", "AF_FIN"), alt_index),
                af_nfe=_af_for(row.info, ("AF_nfe", "AF_NFE"), alt_index),
                af_oth=_af_for(row.info, ("AF_oth", "AF_OTH"), alt_index),
                af_sas=_af_for(row.info, ("AF_sas", "AF_SAS"), alt_index),
                filter=row.filter,
            )


def iter_gnomad_shard(path: str | Path) -> Iterator[GnomadVariant]:
    """Yield normalized gnomAD rows from a Parquet shard."""
    _pa, pq = _require_pyarrow()
    table = pq.read_table(Path(path))
    for row in table.to_pylist():
        yield GnomadVariant(
            chrom=str(row["chrom"]),
            pos=int(row["pos"]),
            ref=str(row["ref"]),
            alt=str(row["alt"]),
            af_global=float(row["af_global"]),
            af_afr=_optional_float(row.get("af_afr")),
            af_ami=_optional_float(row.get("af_ami")),
            af_amr=_optional_float(row.get("af_amr")),
            af_asj=_optional_float(row.get("af_asj")),
            af_eas=_optional_float(row.get("af_eas")),
            af_fin=_optional_float(row.get("af_fin")),
            af_nfe=_optional_float(row.get("af_nfe")),
            af_oth=_optional_float(row.get("af_oth")),
            af_sas=_optional_float(row.get("af_sas")),
            filter=str(row["filter"]),
            schema_version=str(row["schema_version"]),
        )


def _af_for(info: Mapping[str, str | bool], keys: tuple[str, ...], alt_index: int) -> float | None:
    return parse_float(info_value_for_alt(dict(info), keys, alt_index))


def _write_parquet(rows: Iterator[GnomadVariant], target: Path) -> int:
    pa, pq = _require_pyarrow()
    target.parent.mkdir(parents=True, exist_ok=True)
    schema = _parquet_schema(pa)
    tmp = target.with_name(target.name + ".tmp")
    with suppress(OSError):
        tmp.unlink()
    writer: Any | None = None
    batch: list[GnomadVariant] = []
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
    rows: list[GnomadVariant],
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
            ("af_global", pa.float32()),
            ("af_afr", pa.float32()),
            ("af_ami", pa.float32()),
            ("af_amr", pa.float32()),
            ("af_asj", pa.float32()),
            ("af_eas", pa.float32()),
            ("af_fin", pa.float32()),
            ("af_nfe", pa.float32()),
            ("af_oth", pa.float32()),
            ("af_sas", pa.float32()),
            ("filter", pa.string()),
            ("schema_version", pa.string()),
        ]
    )


def _parquet_num_rows(path: Path) -> int:
    _pa, pq = _require_pyarrow()
    return int(pq.ParquetFile(path).metadata.num_rows)


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        pa = importlib.import_module("pyarrow")
        pq = importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise RuntimeSetupError(
            "gnomAD shard preparation requires pyarrow",
            remediation="install geno-lewm[dev], geno-lewm[train], or pyarrow",
        ) from exc
    return pa, pq


def _require_release(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise InputError("release must be a non-empty string")


def _require_probability(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise InputError(
            f"{name} must be between 0 and 1",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputError(
            f"{name} must be a positive integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float | str):
        return float(value)
    raise InputError(
        "gnomAD shard contains a non-float allele frequency",
        details={"type": type(value).__name__},
    )
