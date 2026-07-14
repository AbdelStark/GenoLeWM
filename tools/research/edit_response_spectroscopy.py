# SPDX-License-Identifier: Apache-2.0
"""Measure the edit-response geometry of a frozen Carbon state encoder.

For each genomic variant this tool encodes the reference window and the
single-base-edited window with Carbon, then studies the displacement
``delta = s(alt) - s(ref)`` in Carbon's raw state space across a pooling
radius grid. A single Carbon forward per window feeds every radius:
:meth:`geno_lewm.encoder.carbon.CarbonStateEncoder.encode_token_states`
returns the per-token states once, and
:func:`geno_lewm.encoder.pooling.pool_hidden_states` re-pools them at each
radius. Reference windows are extracted with the exact scorer path
(:func:`geno_lewm.surprise.score._reference_window_for_variant`) so the
geometry is measured over windows identical to the surprise scorer's.

The tool is deliberately model-object agnostic: the core
:func:`run_spectroscopy` accepts any object exposing ``encode_token_states``
so tests inject a deterministic fake and never load Carbon. The CLI builds a
frozen :class:`CarbonStateEncoder` and writes a per-``(variant, pool_radius)``
Parquet table plus a provenance/aggregate summary JSON.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, cast

from geno_lewm.action import EditSpec, apply_edit
from geno_lewm.encoder.carbon import EncodedTokenStates
from geno_lewm.encoder.pooling import (
    POOL_CENTERED_MEAN,
    POOL_GLOBAL_MEAN,
    pool_hidden_states,
)
from geno_lewm.encoder.windowing import DEFAULT_WINDOW_BP, SUPPORTED_WINDOW_BP
from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.surprise.score import _load_reference_fasta, _reference_window_for_variant

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_POOL_RADII",
    "DEFAULT_REVISION",
    "DEFAULT_STATE_LAYER",
    "EMBEDDING_COLUMNS",
    "GENERATED_BY",
    "SCHEMA_VERSION",
    "EditGeometry",
    "PreparedVariant",
    "SpectroscopyRun",
    "TokenStateEncoder",
    "VariantRecord",
    "build_summary",
    "edit_geometry",
    "load_variants",
    "main",
    "prepare_variant",
    "read_done_variant_ids",
    "run_spectroscopy",
    "write_embeddings_parquet",
]

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.research.edit_response_spectroscopy"

#: Radius grid swept for every window. ``0`` selects a global mean over the
#: window; positive radii select the tokenizer-centred content span.
DEFAULT_POOL_RADII: Final[tuple[int, ...]] = (0, 8, 64, 256)
DEFAULT_STATE_LAYER: Final = 20
DEFAULT_BATCH_SIZE: Final = 64
DEFAULT_DTYPE: Final = "bf16"
#: Pinned frozen Carbon-500M checkpoint (matches the training/eval configs).
DEFAULT_MODEL_ID: Final = "HuggingFaceBio/Carbon-500M"
DEFAULT_REVISION: Final = "5d31d59b3c845b288a13aedb1358934196852eec"

#: Ordered Parquet columns for the per-``(variant, pool_radius)`` embedding
#: table. Scalar geometry is written as ``float64``; the raw pooled state
#: vectors ``s_ref`` / ``s_alt`` are written as ``list<float32>``.
EMBEDDING_COLUMNS: Final[tuple[str, ...]] = (
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "label_group",
    "continuous_score",
    "region",
    "gene",
    "pool_radius",
    "pool_type",
    "d_state",
    "norm_s_ref",
    "norm_s_alt",
    "cos_ref_alt",
    "l2_delta",
    "rel_delta",
    "s_ref",
    "s_alt",
)

_REQUIRED_VARIANT_KEYS: Final = ("chrom", "pos", "ref", "alt")
_OPTIONAL_STR_KEYS: Final = ("label", "label_group", "region", "gene", "variant_id")


class TokenStateEncoder(Protocol):
    """Structural type for encoders that expose one-forward token states."""

    def encode_token_states(
        self,
        windows: Sequence[str],
        edit_loci: Sequence[int | None],
    ) -> Sequence[EncodedTokenStates]: ...


@dataclass(frozen=True, slots=True)
class VariantRecord:
    """One parsed variant row plus its optional study annotations."""

    variant_id: str
    chrom: str
    pos: int
    ref: str
    alt: str
    label: str | None = None
    label_group: str | None = None
    continuous_score: float | None = None
    region: str | None = None
    gene: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedVariant:
    """A variant whose reference/edited windows resolved successfully."""

    record: VariantRecord
    reference_window: str
    edited_window: str
    rel_pos: int
    window_start_bp: int


@dataclass(frozen=True, slots=True)
class EditGeometry:
    """Scalar edit-response geometry for one pooled ``(s_ref, s_alt)`` pair."""

    d_state: int
    norm_s_ref: float
    norm_s_alt: float
    cos_ref_alt: float
    l2_delta: float
    rel_delta: float


@dataclass(frozen=True, slots=True)
class SkipRecord:
    """A per-variant skip with its typed error code and message."""

    variant_id: str
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        """Return the JSON-native skip payload."""
        return {"variant_id": self.variant_id, "code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class SpectroscopyRun:
    """The outcome of one spectroscopy pass over a variant set."""

    rows: tuple[dict[str, Any], ...]
    skips: tuple[SkipRecord, ...]
    processed_variant_ids: tuple[str, ...]
    n_input: int


# ---------------------------------------------------------------------------
# Geometry


def edit_geometry(s_ref: Sequence[float], s_alt: Sequence[float]) -> EditGeometry:
    """Return the edit-response geometry of a pooled reference/edited pair.

    ``cos_ref_alt`` is the cosine similarity, ``l2_delta`` is
    ``||s_alt - s_ref||`` and ``rel_delta`` is ``l2_delta / ||s_ref||``.
    Zero-norm pooled states are rejected because the geometry is undefined.
    """
    if len(s_ref) != len(s_alt):
        raise InputError(
            "pooled reference and edited states must share a width",
            details={"ref": len(s_ref), "alt": len(s_alt)},
        )
    if not s_ref:
        raise InputError("pooled states must be non-empty")

    norm_ref = math.sqrt(math.fsum(value * value for value in s_ref))
    norm_alt = math.sqrt(math.fsum(value * value for value in s_alt))
    if norm_ref == 0.0 or norm_alt == 0.0:
        raise InputError(
            "pooled state has zero L2 norm; edit-response geometry is undefined",
            details={"norm_s_ref": norm_ref, "norm_s_alt": norm_alt},
        )
    dot = math.fsum(a * b for a, b in zip(s_ref, s_alt, strict=True))
    cosine = dot / (norm_ref * norm_alt)
    cosine = max(-1.0, min(1.0, cosine))
    l2_delta = math.sqrt(math.fsum((b - a) * (b - a) for a, b in zip(s_ref, s_alt, strict=True)))
    return EditGeometry(
        d_state=len(s_ref),
        norm_s_ref=norm_ref,
        norm_s_alt=norm_alt,
        cos_ref_alt=cosine,
        l2_delta=l2_delta,
        rel_delta=l2_delta / norm_ref,
    )


def _pool_state(
    states: EncodedTokenStates,
    *,
    edit_locus: int,
    pool_radius: int,
) -> tuple[float, ...]:
    """Pool ``states`` at ``pool_radius`` (radius ``0`` uses a global mean)."""
    if pool_radius == 0:
        return pool_hidden_states(
            states.rows,
            pool_type=POOL_GLOBAL_MEAN,
            pool_radius=0,
        ).vector
    return pool_hidden_states(
        states.rows,
        edit_locus=edit_locus,
        center_token=states.center_token,
        content_token_bounds=states.content_token_bounds,
        pool_type=POOL_CENTERED_MEAN,
        pool_radius=pool_radius,
    ).vector


def _pool_type_for_radius(pool_radius: int) -> str:
    return POOL_GLOBAL_MEAN if pool_radius == 0 else POOL_CENTERED_MEAN


# ---------------------------------------------------------------------------
# Input parsing


def load_variants(path: str | Path) -> tuple[VariantRecord, ...]:
    """Parse a JSONL variant file into :class:`VariantRecord` rows.

    Each line is a JSON object with required ``chrom, pos, ref, alt`` and
    optional ``label, label_group, continuous_score, region, gene,
    variant_id``. A malformed line (bad JSON, missing field, wrong type)
    fails closed with :class:`InputError` — structural input violations are
    not silently skipped; only per-variant scientific failures are (see
    :func:`run_spectroscopy`).
    """
    src = Path(path)
    try:
        text = src.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(
            "could not read variants JSONL",
            details={"path": str(src), "error": str(exc)},
        ) from exc

    records: list[VariantRecord] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        records.append(_parse_variant_line(line, path=src, line_no=line_no))
    if not records:
        raise InputError("variants JSONL contains no variant rows", details={"path": str(src)})
    return tuple(records)


def _parse_variant_line(line: str, *, path: Path, line_no: int) -> VariantRecord:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise InputError(
            "variants JSONL line is not valid JSON",
            details={"path": str(path), "line": line_no, "error": str(exc)},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError(
            "variants JSONL line must be a JSON object",
            details={"path": str(path), "line": line_no, "type": type(payload).__name__},
        )
    for key in _REQUIRED_VARIANT_KEYS:
        if key not in payload:
            raise InputError(
                "variants JSONL line is missing a required field",
                details={"path": str(path), "line": line_no, "field": key},
            )
    chrom = _require_str(payload["chrom"], field="chrom", path=path, line_no=line_no)
    pos = _require_int(payload["pos"], field="pos", path=path, line_no=line_no)
    ref = _require_str(payload["ref"], field="ref", path=path, line_no=line_no).upper()
    alt = _require_str(payload["alt"], field="alt", path=path, line_no=line_no).upper()
    optional = {
        key: _optional_str(payload.get(key), field=key, path=path, line_no=line_no)
        for key in _OPTIONAL_STR_KEYS
    }
    variant_id = optional["variant_id"] or f"{chrom}:{pos}:{ref}:{alt}"
    return VariantRecord(
        variant_id=variant_id,
        chrom=chrom,
        pos=pos,
        ref=ref,
        alt=alt,
        label=optional["label"],
        label_group=optional["label_group"],
        continuous_score=_optional_float(
            payload.get("continuous_score"),
            field="continuous_score",
            path=path,
            line_no=line_no,
        ),
        region=optional["region"],
        gene=optional["gene"],
    )


def _require_str(value: object, *, field: str, path: Path, line_no: int) -> str:
    if not isinstance(value, str) or not value:
        raise InputError(
            "variants JSONL field must be a non-empty string",
            details={"path": str(path), "line": line_no, "field": field},
        )
    return value


def _require_int(value: object, *, field: str, path: Path, line_no: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(
            "variants JSONL field must be an integer",
            details={"path": str(path), "line": line_no, "field": field},
        )
    return value


def _optional_str(value: object, *, field: str, path: Path, line_no: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise InputError(
            "variants JSONL optional field must be a non-empty string when present",
            details={"path": str(path), "line": line_no, "field": field},
        )
    return value


def _optional_float(value: object, *, field: str, path: Path, line_no: int) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputError(
            "variants JSONL optional field must be a number when present",
            details={"path": str(path), "line": line_no, "field": field},
        )
    return float(value)


# ---------------------------------------------------------------------------
# Window preparation


def prepare_variant(
    record: VariantRecord,
    reference_sequences: Mapping[str, str],
    *,
    window_bp: int,
) -> PreparedVariant:
    """Resolve the reference and single-edit windows for one variant.

    Reuses the surprise scorer's FASTA window path so the reference window
    is byte-identical to the scorer's, then applies the edit with
    length preservation to build the edited window. Raises the typed error
    hierarchy on any per-variant failure; :func:`run_spectroscopy` converts
    those into skips.
    """
    variant = EditSpec(chrom=record.chrom, pos=record.pos, ref=record.ref, alt=record.alt)
    reference = _reference_window_for_variant(
        variant,
        None,
        reference_sequences,
        window_start_bp=0,
        window_bp=window_bp,
    )
    window = reference.sequence
    start_bp = reference.start_bp
    rel_edit = variant.relative_to(start_bp, start_bp + len(window) - 1)
    edited_window = apply_edit(window, rel_edit, preserve_length=True)
    return PreparedVariant(
        record=record,
        reference_window=window,
        edited_window=edited_window,
        rel_pos=rel_edit.rel_pos,
        window_start_bp=start_bp,
    )


# ---------------------------------------------------------------------------
# Core measurement


def run_spectroscopy(
    variants: Sequence[VariantRecord],
    reference_sequences: Mapping[str, str],
    encoder: TokenStateEncoder,
    *,
    window_bp: int = DEFAULT_WINDOW_BP,
    pool_radii: Sequence[int] = DEFAULT_POOL_RADII,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume_variant_ids: Iterable[str] = (),
) -> SpectroscopyRun:
    """Measure edit-response geometry over ``variants`` for a radius grid.

    One Carbon forward per window feeds every radius. Per-variant failures
    (window extraction, out-of-window edit, unsupported edit, degenerate
    pooled state) are recorded as skips instead of aborting the run.
    """
    radii = _validate_pool_radii(pool_radii)
    _validate_window_bp(window_bp)
    _require_positive_int("batch_size", batch_size)
    already_done = frozenset(resume_variant_ids)

    prepared: list[PreparedVariant] = []
    skips: list[SkipRecord] = []
    n_input = len(variants)
    for record in variants:
        if record.variant_id in already_done:
            continue
        try:
            prepared.append(prepare_variant(record, reference_sequences, window_bp=window_bp))
        except GenoLeWMError as exc:
            skips.append(_skip_from_error(record.variant_id, exc))

    rows: list[dict[str, Any]] = []
    processed: list[str] = []
    for group in _chunk(prepared, batch_size):
        encoded, group_skips = _encode_group(encoder, group)
        skips.extend(group_skips)
        for item, ref_states, alt_states in encoded:
            try:
                variant_rows = _rows_for_variant(item, ref_states, alt_states, radii=radii)
            except GenoLeWMError as exc:
                skips.append(_skip_from_error(item.record.variant_id, exc))
                continue
            rows.extend(variant_rows)
            processed.append(item.record.variant_id)

    return SpectroscopyRun(
        rows=tuple(rows),
        skips=tuple(skips),
        processed_variant_ids=tuple(processed),
        n_input=n_input,
    )


def _encode_group(
    encoder: TokenStateEncoder,
    group: Sequence[PreparedVariant],
) -> tuple[
    list[tuple[PreparedVariant, EncodedTokenStates, EncodedTokenStates]],
    list[SkipRecord],
]:
    """Encode a batch of prepared variants with one forward, isolating failures.

    The whole group is encoded in a single forward pass; if that raises, the
    group is retried one variant at a time so a single bad window cannot abort
    the batch.
    """
    if not group:
        return [], []
    try:
        states = _encode_windows(encoder, group)
    except GenoLeWMError:
        return _encode_group_individually(encoder, group)
    return list(zip(group, states[0::2], states[1::2], strict=True)), []


def _encode_group_individually(
    encoder: TokenStateEncoder,
    group: Sequence[PreparedVariant],
) -> tuple[
    list[tuple[PreparedVariant, EncodedTokenStates, EncodedTokenStates]],
    list[SkipRecord],
]:
    encoded: list[tuple[PreparedVariant, EncodedTokenStates, EncodedTokenStates]] = []
    skips: list[SkipRecord] = []
    for item in group:
        try:
            single = _encode_windows(encoder, [item])
        except GenoLeWMError as exc:
            skips.append(_skip_from_error(item.record.variant_id, exc))
            continue
        encoded.append((item, single[0], single[1]))
    return encoded, skips


def _encode_windows(
    encoder: TokenStateEncoder,
    group: Sequence[PreparedVariant],
) -> tuple[EncodedTokenStates, ...]:
    windows: list[str] = []
    edit_loci: list[int | None] = []
    for item in group:
        windows.append(item.reference_window)
        windows.append(item.edited_window)
        edit_loci.append(item.rel_pos)
        edit_loci.append(item.rel_pos)
    states = tuple(encoder.encode_token_states(windows, edit_loci))
    if len(states) != len(windows):
        raise InputError(
            "encoder returned an unexpected number of token-state items",
            details={"expected": len(windows), "observed": len(states)},
        )
    return states


def _rows_for_variant(
    item: PreparedVariant,
    ref_states: EncodedTokenStates,
    alt_states: EncodedTokenStates,
    *,
    radii: Sequence[int],
) -> list[dict[str, Any]]:
    record = item.record
    rows: list[dict[str, Any]] = []
    for pool_radius in radii:
        s_ref = _pool_state(ref_states, edit_locus=item.rel_pos, pool_radius=pool_radius)
        s_alt = _pool_state(alt_states, edit_locus=item.rel_pos, pool_radius=pool_radius)
        geometry = edit_geometry(s_ref, s_alt)
        rows.append(
            {
                "variant_id": record.variant_id,
                "chrom": record.chrom,
                "pos": record.pos,
                "ref": record.ref,
                "alt": record.alt,
                "label": record.label,
                "label_group": record.label_group,
                "continuous_score": record.continuous_score,
                "region": record.region,
                "gene": record.gene,
                "pool_radius": pool_radius,
                "pool_type": _pool_type_for_radius(pool_radius),
                "d_state": geometry.d_state,
                "norm_s_ref": geometry.norm_s_ref,
                "norm_s_alt": geometry.norm_s_alt,
                "cos_ref_alt": geometry.cos_ref_alt,
                "l2_delta": geometry.l2_delta,
                "rel_delta": geometry.rel_delta,
                "s_ref": [float(value) for value in s_ref],
                "s_alt": [float(value) for value in s_alt],
            }
        )
    return rows


def _skip_from_error(variant_id: str, exc: GenoLeWMError) -> SkipRecord:
    return SkipRecord(variant_id=variant_id, code=exc.code, message=exc.message or str(exc))


# ---------------------------------------------------------------------------
# Parquet I/O


def _parquet_schema(pa: Any) -> Any:
    return pa.schema(
        [
            ("variant_id", pa.string()),
            ("chrom", pa.string()),
            ("pos", pa.int64()),
            ("ref", pa.string()),
            ("alt", pa.string()),
            ("label", pa.string()),
            ("label_group", pa.string()),
            ("continuous_score", pa.float64()),
            ("region", pa.string()),
            ("gene", pa.string()),
            ("pool_radius", pa.int64()),
            ("pool_type", pa.string()),
            ("d_state", pa.int64()),
            ("norm_s_ref", pa.float64()),
            ("norm_s_alt", pa.float64()),
            ("cos_ref_alt", pa.float64()),
            ("l2_delta", pa.float64()),
            ("rel_delta", pa.float64()),
            ("s_ref", pa.list_(pa.float32())),
            ("s_alt", pa.list_(pa.float32())),
        ]
    )


def _rows_to_table(pa: Any, rows: Sequence[Mapping[str, Any]]) -> Any:
    columns = {name: [row[name] for row in rows] for name in EMBEDDING_COLUMNS}
    return pa.table(columns, schema=_parquet_schema(pa))


def write_embeddings_parquet(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    resume: bool = False,
) -> Path:
    """Write ``rows`` to a Parquet table (appending when ``resume`` is set).

    On resume the existing table is read back and the new rows are appended,
    so re-running with the same output path grows the table deterministically.
    """
    pa = cast(Any, importlib.import_module("pyarrow"))
    pq = cast(Any, importlib.import_module("pyarrow.parquet"))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_table = _rows_to_table(pa, rows)
    if resume and out.is_file():
        existing = pq.read_table(out).cast(_parquet_schema(pa))
        table = pa.concat_tables([existing, new_table])
    else:
        table = new_table
    pq.write_table(table, out)
    return out


def read_done_variant_ids(path: str | Path) -> frozenset[str]:
    """Return the ``variant_id`` set already present in a Parquet table."""
    out = Path(path)
    if not out.is_file():
        return frozenset()
    pq = cast(Any, importlib.import_module("pyarrow.parquet"))
    table = pq.read_table(out, columns=["variant_id"])
    return frozenset(str(value) for value in table.column("variant_id").to_pylist())


def _scalar_rows_from_parquet(path: str | Path) -> tuple[dict[str, Any], ...]:
    pq = cast(Any, importlib.import_module("pyarrow.parquet"))
    columns = ["variant_id", "pool_radius", "pool_type", "label_group", *_AGG_FIELDS]
    table = pq.read_table(Path(path), columns=columns)
    return tuple(cast("list[dict[str, Any]]", table.to_pylist()))


# ---------------------------------------------------------------------------
# Summary


_AGG_FIELDS: Final = ("cos_ref_alt", "rel_delta", "norm_s_ref")


def build_summary(
    scalar_rows: Sequence[Mapping[str, Any]],
    skips: Sequence[SkipRecord],
    *,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    n_input: int,
) -> dict[str, Any]:
    """Build the provenance + per-radius aggregate summary payload."""
    variant_ids = {str(row["variant_id"]) for row in scalar_rows}
    by_radius: dict[int, list[Mapping[str, Any]]] = {}
    for row in scalar_rows:
        by_radius.setdefault(int(row["pool_radius"]), []).append(row)

    per_pool_radius: list[dict[str, Any]] = []
    for pool_radius in sorted(by_radius):
        radius_rows = by_radius[pool_radius]
        per_pool_radius.append(
            {
                "pool_radius": pool_radius,
                "pool_type": _pool_type_for_radius(pool_radius),
                "n_variants": len(radius_rows),
                **{field: _describe(radius_rows, field) for field in _AGG_FIELDS},
                "n_by_label_group": _count_label_groups(radius_rows),
            }
        )

    skip_by_code = Counter(skip.code for skip in skips)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "config": dict(config),
        "provenance": dict(provenance),
        "counts": {
            "n_input": n_input,
            "n_variants": len(variant_ids),
            "n_rows": len(scalar_rows),
            "n_skipped": len(skips),
        },
        "skips": {
            "by_code": dict(sorted(skip_by_code.items())),
            "records": [skip.as_dict() for skip in skips],
        },
        "per_pool_radius": per_pool_radius,
    }


def _describe(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float | None]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return {"mean": None, "median": None, "std": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
    }


def _count_label_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        group = row.get("label_group")
        counts[str(group) if group is not None else "__unlabeled__"] += 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Validation helpers


def _validate_pool_radii(pool_radii: Sequence[int]) -> tuple[int, ...]:
    if not pool_radii:
        raise InputError("pool_radii must contain at least one radius")
    radii: list[int] = []
    for value in pool_radii:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InputError(
                "pool_radii must be non-negative integers",
                details={"pool_radii": list(pool_radii)},
            )
        radii.append(value)
    ordered = sorted(dict.fromkeys(radii))
    return tuple(ordered)


def _validate_window_bp(window_bp: int) -> int:
    if isinstance(window_bp, bool) or not isinstance(window_bp, int):
        raise InputError("window_bp must be an integer", details={"window_bp": window_bp})
    if window_bp not in SUPPORTED_WINDOW_BP:
        raise InputError(
            "unsupported window length; reuse a scorer-supported width",
            details={"window_bp": window_bp, "supported": list(SUPPORTED_WINDOW_BP)},
            remediation="pass one of the supported window widths so windows match the scorer",
        )
    return window_bp


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(f"{name} must be a positive integer", details={name: value})
    return value


def _chunk(items: Sequence[PreparedVariant], size: int) -> Iterable[Sequence[PreparedVariant]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


# ---------------------------------------------------------------------------
# CLI


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _parse_radii_csv(raw: str) -> tuple[int, ...]:
    parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not parts:
        raise InputError("--pool-radii must list at least one integer")
    try:
        radii = [int(part) for part in parts]
    except ValueError as exc:
        raise InputError(
            "--pool-radii must be comma-separated integers", details={"raw": raw}
        ) from exc
    return _validate_pool_radii(radii)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--reference-fasta", type=Path, required=True)
    parser.add_argument("--out-embeddings", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--window-bp", type=int, default=DEFAULT_WINDOW_BP)
    parser.add_argument(
        "--pool-radii", type=str, default=",".join(str(r) for r in DEFAULT_POOL_RADII)
    )
    parser.add_argument("--state-layer", type=int, default=DEFAULT_STATE_LAYER)
    parser.add_argument(
        "--dtype", type=str, default=DEFAULT_DTYPE, choices=["bf16", "fp16", "fp32"]
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", type=str, default=DEFAULT_REVISION)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def _build_encoder(args: argparse.Namespace) -> TokenStateEncoder:
    from geno_lewm.encoder.carbon import CarbonStateEncoder

    return CarbonStateEncoder(
        args.model_id,
        args.revision,
        dtype=args.dtype,
        state_layer=args.state_layer,
        pool_type=POOL_CENTERED_MEAN,
        pool_radius=0,
        normalize=False,
        device=args.device,
    )


def _encoder_identity(args: argparse.Namespace, encoder: TokenStateEncoder) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "model_id": args.model_id,
        "revision": args.revision,
        "dtype": args.dtype,
        "state_layer": args.state_layer,
        "normalize": False,
        "pool_family": "raw_token_states",
    }
    d_state = getattr(encoder, "d_state", None)
    if isinstance(d_state, int):
        identity["d_state"] = d_state
    return identity


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run spectroscopy and write the Parquet + summary."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        radii = _parse_radii_csv(args.pool_radii)
        _validate_window_bp(args.window_bp)
        _require_positive_int("batch_size", args.batch_size)
        variants = load_variants(args.variants)
        if args.limit is not None:
            _require_positive_int("limit", args.limit)
            variants = variants[: args.limit]
        reference_sequences = _load_reference_fasta(args.reference_fasta)

        done = read_done_variant_ids(args.out_embeddings) if args.resume else frozenset()
        encoder = _build_encoder(args)
        run = run_spectroscopy(
            variants,
            reference_sequences,
            encoder,
            window_bp=args.window_bp,
            pool_radii=radii,
            batch_size=args.batch_size,
            resume_variant_ids=done,
        )
        write_embeddings_parquet(args.out_embeddings, run.rows, resume=args.resume)

        scalar_rows = _scalar_rows_from_parquet(args.out_embeddings)
        config = {
            "window_bp": args.window_bp,
            "pool_radii": list(radii),
            "state_layer": args.state_layer,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "limit": args.limit,
            "resume": bool(args.resume),
            "reference_fasta": args.reference_fasta.name,
            "variants": args.variants.name,
        }
        provenance = {
            "git_commit": _git_commit(),
            "encoder": _encoder_identity(args, encoder),
        }
        summary = build_summary(
            scalar_rows,
            run.skips,
            config=config,
            provenance=provenance,
            n_input=run.n_input,
        )
        _write_summary(args.out_summary, summary)
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        if exc.details:
            sys.stderr.write(json.dumps(exc.details, sort_keys=True) + "\n")
        return exit_code_for(exc)
    sys.stdout.write(
        json.dumps(
            {
                "out_embeddings": str(args.out_embeddings),
                "out_summary": str(args.out_summary),
                "n_rows": len(run.rows),
                "n_skipped": len(run.skips),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def _write_summary(path: str | Path, summary: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
