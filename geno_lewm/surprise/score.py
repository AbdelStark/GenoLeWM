# SPDX-License-Identifier: Apache-2.0
"""Surprise scoring for action-conditioned DNA world-model predictions."""

from __future__ import annotations

import bisect
import gzip
import json
import math
import statistics
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from geno_lewm._artifact_sources import SCORE_JSONL_GENERATED_BY, SCORE_JSONL_SCHEMA_VERSION
from geno_lewm.action import EditSpec, RelEdit, apply_edit
from geno_lewm.encoder.windowing import DEFAULT_WINDOW_BP, canonicalize_dna, extract_window
from geno_lewm.errors import InputError, VcfParseError
from geno_lewm.surprise.calibration import CalibrationBucket, CalibrationTable
from geno_lewm.surprise.context import DEFAULT_MIN_BUCKET_SIZE, classify_context

__all__ = [
    "Aggregation",
    "SurpriseResult",
    "score_variant",
    "score_vcf",
]


Aggregation: TypeAlias = Literal["mean", "max", "median"]
"""Supported aggregation modes for multi-step predictor outputs."""


@dataclass(frozen=True, slots=True)
class SurpriseResult:
    """Calibrated surprise score for one edit."""

    sigma_raw: float
    sigma_calibrated: float
    bucket_id: str
    confidence: float
    low_confidence: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sigma_raw", _require_finite_non_negative("sigma_raw", self.sigma_raw)
        )
        object.__setattr__(
            self,
            "sigma_calibrated",
            _require_probability("sigma_calibrated", self.sigma_calibrated),
        )
        if not isinstance(self.bucket_id, str) or not self.bucket_id:
            raise InputError(
                "bucket_id must be a non-empty string",
                details={"bucket_id": self.bucket_id},
            )
        object.__setattr__(self, "confidence", _require_probability("confidence", self.confidence))
        if not isinstance(self.low_confidence, bool):
            raise InputError(
                "low_confidence must be a bool",
                details={"type": type(self.low_confidence).__name__},
            )

    def to_dict(self) -> dict[str, float | str | bool]:
        """Return a JSON-native payload for CLI and JSONL outputs."""
        return {
            "sigma_raw": self.sigma_raw,
            "sigma_calibrated": self.sigma_calibrated,
            "bucket_id": self.bucket_id,
            "confidence": self.confidence,
            "low_confidence": self.low_confidence,
        }


@dataclass(frozen=True, slots=True)
class _ReferenceWindow:
    sequence: str
    start_bp: int


@dataclass(frozen=True, slots=True)
class _VcfScoreRecord:
    variant: EditSpec
    reference_window: str
    window_start_bp: int
    result: SurpriseResult


def score_variant(
    variant: EditSpec,
    encoder: object,
    action_encoder: object,
    predictor: object,
    calibration: CalibrationTable,
    *,
    reference_window: str,
    window_start_bp: int = 0,
    region: str | Sequence[str] | None = None,
    repeat: str | Sequence[str] | None = None,
    aggregation: str = "mean",
    min_bucket_size: int = DEFAULT_MIN_BUCKET_SIZE,
) -> SurpriseResult:
    """Score one edit against a caller-supplied reference window.

    The scorer is intentionally model-object agnostic: callers can pass
    the concrete training-time modules or small deterministic fakes.
    FASTA-backed window extraction is available through :func:`score_vcf`;
    checkpoint loading is owned by higher runtime layers.
    """
    if not isinstance(variant, EditSpec):
        raise InputError(
            "variant must be an EditSpec",
            details={"type": type(variant).__name__},
        )
    _require_calibration_table(calibration)
    normalized_aggregation = _normalize_aggregation(aggregation)
    start_bp = _require_window_start(window_start_bp)
    min_size = _require_positive_int("min_bucket_size", min_bucket_size)
    window = canonicalize_dna(reference_window)
    if not window:
        raise InputError("reference_window must be non-empty")

    rel_edit = variant.relative_to(start_bp, start_bp + len(window) - 1)
    edited_window = apply_edit(window, rel_edit, preserve_length=True)

    state = _encode_window(encoder, window, edit_locus=rel_edit.rel_pos)
    target = _encode_window(encoder, edited_window, edit_locus=rel_edit.rel_pos)
    action = _encode_action(action_encoder, rel_edit)
    prediction = _predict_next_state(predictor, state=state, action=action)

    target_vector = _as_float_vector(target, name="target encoder output")
    prediction_vector = _as_float_vector(prediction, name="predictor output")
    sigma_raw = _aggregate_distances(
        prediction_vector,
        target_vector,
        aggregation=normalized_aggregation,
    )

    label = classify_context(region=region, gc_window=window, repeat=repeat)
    bucket = calibration.resolve(label.bucket_id, min_bucket_size=min_size)
    return SurpriseResult(
        sigma_raw=sigma_raw,
        sigma_calibrated=_cdf_percentile(bucket, sigma_raw),
        bucket_id=bucket.bucket_id,
        confidence=bucket.confidence,
        low_confidence=bucket.low_confidence,
    )


def score_vcf(
    vcf_path: str | Path,
    encoder: object,
    action_encoder: object,
    predictor: object,
    calibration: CalibrationTable,
    output_path: str | Path,
    *,
    reference_windows: Mapping[str, str] | None = None,
    reference_fasta: str | Path | None = None,
    window_bp: int = DEFAULT_WINDOW_BP,
    window_start_bp: int = 0,
    region: str | Sequence[str] | None = None,
    repeat: str | Sequence[str] | None = None,
    aggregation: str = "mean",
    show_progress: bool = True,
    batch_size: int = 64,
    min_bucket_size: int = DEFAULT_MIN_BUCKET_SIZE,
) -> Path:
    """Score VCF rows and write one JSON object per scored alternate.

    Pass ``reference_fasta`` for local FASTA-backed window extraction.
    ``reference_windows`` remains useful for tests and already-extracted
    windows. Mapping keys are tried in this order:
    ``chrom:pos:ref:alt``, ``chrom:pos``, then ``chrom``.
    """
    if not isinstance(show_progress, bool):
        raise InputError(
            "show_progress must be a bool",
            details={"type": type(show_progress).__name__},
        )
    del show_progress

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in _iter_vcf_scores(
            vcf_path,
            encoder,
            action_encoder,
            predictor,
            calibration,
            reference_windows=reference_windows,
            reference_fasta=reference_fasta,
            window_bp=window_bp,
            window_start_bp=window_start_bp,
            region=region,
            repeat=repeat,
            aggregation=aggregation,
            batch_size=batch_size,
            min_bucket_size=min_bucket_size,
        ):
            variant = record.variant
            handle.write(
                json.dumps(
                    {
                        "schema_version": SCORE_JSONL_SCHEMA_VERSION,
                        "generated_by": SCORE_JSONL_GENERATED_BY,
                        "chrom": variant.chrom,
                        "pos": variant.pos,
                        "ref": variant.ref,
                        "alt": variant.alt,
                        **record.result.to_dict(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return output


def _iter_vcf_scores(
    vcf_path: str | Path,
    encoder: object,
    action_encoder: object,
    predictor: object,
    calibration: CalibrationTable,
    *,
    reference_windows: Mapping[str, str] | None = None,
    reference_fasta: str | Path | None = None,
    window_bp: int = DEFAULT_WINDOW_BP,
    window_start_bp: int = 0,
    region: str | Sequence[str] | None = None,
    repeat: str | Sequence[str] | None = None,
    aggregation: str = "mean",
    batch_size: int = 64,
    min_bucket_size: int = DEFAULT_MIN_BUCKET_SIZE,
) -> Iterator[_VcfScoreRecord]:
    if reference_windows is None and reference_fasta is None:
        raise InputError(
            "score_vcf requires reference_windows or reference_fasta",
            remediation=(
                "pass a local FASTA path, or pre-extracted windows keyed by "
                "chrom:pos:ref:alt, chrom:pos, or chrom"
            ),
        )
    _require_positive_int("batch_size", batch_size)
    reference_sequences = (
        None if reference_fasta is None else _load_reference_fasta(reference_fasta)
    )

    count = 0
    for variant in _iter_vcf_variants(vcf_path):
        ref_window = _reference_window_for_variant(
            variant,
            reference_windows,
            reference_sequences,
            window_start_bp=window_start_bp,
            window_bp=window_bp,
        )
        result = score_variant(
            variant,
            encoder,
            action_encoder,
            predictor,
            calibration,
            reference_window=ref_window.sequence,
            window_start_bp=ref_window.start_bp,
            region=region,
            repeat=repeat,
            aggregation=aggregation,
            min_bucket_size=min_bucket_size,
        )
        yield _VcfScoreRecord(
            variant=variant,
            reference_window=ref_window.sequence,
            window_start_bp=ref_window.start_bp,
            result=result,
        )
        count += 1
    if count == 0:
        raise VcfParseError(
            "VCF contains no scoreable variant rows", details={"path": str(vcf_path)}
        )


def _encode_window(encoder: object, window: str, *, edit_locus: int) -> object:
    method = getattr(encoder, "encode", None)
    if callable(method):
        encode = cast(Callable[..., object], method)
        try:
            return encode(window, edit_locus=edit_locus)
        except TypeError:
            return encode(window)
    if callable(encoder):
        encode = cast(Callable[..., object], encoder)
        try:
            return encode(window, edit_locus=edit_locus)
        except TypeError:
            return encode(window)
    raise InputError(
        "encoder must be callable or expose encode(window)",
        details={"type": type(encoder).__name__},
    )


def _encode_action(action_encoder: object, edit: RelEdit) -> object:
    method = getattr(action_encoder, "encode", None)
    if callable(method):
        return cast(Callable[..., object], method)([edit])
    if callable(action_encoder):
        return cast(Callable[..., object], action_encoder)([edit])
    raise InputError(
        "action_encoder must be callable or expose encode(edits)",
        details={"type": type(action_encoder).__name__},
    )


def _predict_next_state(predictor: object, *, state: object, action: object) -> object:
    method = getattr(predictor, "predict", None)
    if callable(method):
        predict = cast(Callable[..., object], method)
    elif callable(predictor):
        predict = cast(Callable[..., object], predictor)
    else:
        raise InputError(
            "predictor must be callable or expose predict(state, action)",
            details={"type": type(predictor).__name__},
        )

    try:
        return predict(state, action)
    except TypeError as two_arg_error:
        three_arg_result = _try_predict_with_action_mask(predict, state=state, action=action)
        if three_arg_result is not _PREDICT_FAILED:
            return three_arg_result
        raise InputError(
            "predictor must accept (state, action) or (state, actions, action_mask)",
            details={"type": type(predictor).__name__},
        ) from two_arg_error


_PREDICT_FAILED = object()


def _try_predict_with_action_mask(
    predict: Callable[..., object], *, state: object, action: object
) -> object:
    state_batched = _unsqueeze_if_vector(state)
    action_batched = _unsqueeze_action_if_needed(action)
    if state_batched is _PREDICT_FAILED or action_batched is _PREDICT_FAILED:
        return _PREDICT_FAILED
    mask_factory = getattr(action_batched, "new_ones", None)
    if not callable(mask_factory):
        return _PREDICT_FAILED
    shape = getattr(action_batched, "shape", None)
    if not isinstance(shape, Sequence) or len(shape) < 2:
        return _PREDICT_FAILED
    mask = cast(Callable[[object], object], mask_factory)(shape[:-1])
    bool_method = getattr(mask, "bool", None)
    if callable(bool_method):
        mask = cast(Callable[[], object], bool_method)()
    return predict(state_batched, action_batched, mask)


def _unsqueeze_if_vector(value: object) -> object:
    ndim = getattr(value, "ndim", None)
    if ndim == 2:
        return value
    if ndim != 1:
        return _PREDICT_FAILED
    unsqueeze = getattr(value, "unsqueeze", None)
    if not callable(unsqueeze):
        return _PREDICT_FAILED
    return cast(Callable[[int], object], unsqueeze)(0)


def _unsqueeze_action_if_needed(value: object) -> object:
    ndim = getattr(value, "ndim", None)
    if ndim == 3:
        return value
    if ndim != 2:
        return _PREDICT_FAILED
    unsqueeze = getattr(value, "unsqueeze", None)
    if not callable(unsqueeze):
        return _PREDICT_FAILED
    return cast(Callable[[int], object], unsqueeze)(0)


def _as_float_vector(value: object, *, name: str) -> tuple[float, ...]:
    materialized = _materialize_numeric_container(value)
    values: list[float] = []
    _collect_floats(materialized, values, name=name)
    if not values:
        raise InputError(f"{name} must contain at least one numeric value")
    return tuple(values)


def _materialize_numeric_container(value: object) -> object:
    out = value
    for attr in ("detach", "cpu", "flatten"):
        method = getattr(out, attr, None)
        if callable(method):
            out = cast(Callable[[], object], method)()
    tolist = getattr(out, "tolist", None)
    if callable(tolist):
        out = cast(Callable[[], object], tolist)()
    return out


def _collect_floats(value: object, values: list[float], *, name: str) -> None:
    item = getattr(value, "item", None)
    if callable(item):
        _collect_floats(cast(Callable[[], object], item)(), values, name=name)
        return
    if isinstance(value, bool):
        raise InputError(f"{name} must contain numeric values, not bools")
    if isinstance(value, int | float):
        values.append(_require_finite_float(name, value))
        return
    if isinstance(value, str | bytes):
        raise InputError(
            f"{name} must contain numeric values",
            details={"type": type(value).__name__},
        )
    if isinstance(value, Sequence):
        for item_value in cast(Sequence[object], value):
            _collect_floats(item_value, values, name=name)
        return
    raise InputError(
        f"{name} must be a numeric scalar or sequence",
        details={"type": type(value).__name__},
    )


def _aggregate_distances(
    prediction: tuple[float, ...],
    target: tuple[float, ...],
    *,
    aggregation: Aggregation,
) -> float:
    target_len = len(target)
    if len(prediction) % target_len != 0:
        raise InputError(
            "predictor output length must equal or be a multiple of target encoder length",
            details={"prediction_len": len(prediction), "target_len": target_len},
        )
    distances = [
        _l2_distance(prediction[offset : offset + target_len], target)
        for offset in range(0, len(prediction), target_len)
    ]
    if aggregation == "mean":
        return statistics.fmean(distances)
    if aggregation == "max":
        return max(distances)
    return statistics.median(distances)


def _l2_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) * (a - b) for a, b in zip(left, right, strict=True)))


def _cdf_percentile(bucket: CalibrationBucket, sigma_raw: float) -> float:
    if len(bucket.sigma_grid) == 1:
        return _require_probability("cdf[0]", bucket.cdf[0])
    if sigma_raw <= bucket.sigma_grid[0]:
        return _require_probability("cdf[0]", bucket.cdf[0])
    if sigma_raw >= bucket.sigma_grid[-1]:
        return _require_probability("cdf[-1]", bucket.cdf[-1])
    idx = bisect.bisect_left(bucket.sigma_grid, sigma_raw)
    left_grid = bucket.sigma_grid[idx - 1]
    right_grid = bucket.sigma_grid[idx]
    left_cdf = bucket.cdf[idx - 1]
    right_cdf = bucket.cdf[idx]
    if right_grid == left_grid:
        return _require_probability("sigma_calibrated", max(left_cdf, right_cdf))
    ratio = (sigma_raw - left_grid) / (right_grid - left_grid)
    return _require_probability("sigma_calibrated", left_cdf + (ratio * (right_cdf - left_cdf)))


def _iter_vcf_variants(vcf_path: str | Path) -> tuple[EditSpec, ...]:
    path = Path(vcf_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VcfParseError(
            "could not read VCF input",
            details={"path": str(path), "error": str(exc)},
        ) from exc

    variants: list[EditSpec] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 5:
            raise VcfParseError(
                "VCF record must contain at least CHROM, POS, ID, REF, ALT",
                details={"path": str(path), "line": line_no},
            )
        chrom, pos_text, _variant_id, ref, alt_field = fields[:5]
        try:
            pos = int(pos_text)
        except ValueError as exc:
            raise VcfParseError(
                "VCF POS must be an integer",
                details={"path": str(path), "line": line_no, "pos": pos_text},
            ) from exc
        alts = tuple(alt for alt in alt_field.split(",") if alt and alt != ".")
        if not alts:
            raise VcfParseError(
                "VCF ALT must contain at least one explicit allele",
                details={"path": str(path), "line": line_no},
            )
        for alt in alts:
            try:
                variants.append(EditSpec(chrom=chrom, pos=pos, ref=ref.upper(), alt=alt.upper()))
            except InputError as exc:
                raise VcfParseError(
                    "VCF record cannot be converted to EditSpec",
                    details={
                        "path": str(path),
                        "line": line_no,
                        "chrom": chrom,
                        "pos": pos,
                        "ref": ref,
                        "alt": alt,
                    },
                ) from exc
    return tuple(variants)


def _reference_window_for_variant(
    variant: EditSpec,
    windows: Mapping[str, str] | None,
    reference_sequences: Mapping[str, str] | None,
    *,
    window_start_bp: int,
    window_bp: int,
) -> _ReferenceWindow:
    keys = (
        f"{variant.chrom}:{variant.pos}:{variant.ref}:{variant.alt}",
        f"{variant.chrom}:{variant.pos}",
        variant.chrom,
    )
    if windows is not None:
        start_bp = _require_window_start(window_start_bp)
        for key in keys:
            if key in windows:
                return _ReferenceWindow(
                    sequence=canonicalize_dna(windows[key]),
                    start_bp=start_bp,
                )
    if reference_sequences is not None:
        return _extract_reference_window(
            variant,
            reference_sequences,
            window_bp=window_bp,
        )
    raise InputError(
        "reference_windows is missing a VCF variant window",
        details={"keys_tried": list(keys)},
        remediation=(
            "provide a reference window keyed by chrom:pos:ref:alt, chrom:pos, or chrom, "
            "or pass reference_fasta"
        ),
    )


def _load_reference_fasta(path: str | Path) -> dict[str, str]:
    src = Path(path)
    if not src.is_file():
        raise VcfParseError(
            "reference FASTA path must be a file",
            details={"path": str(src)},
        )
    try:
        text = _read_reference_text(src)
    except (OSError, UnicodeDecodeError) as exc:
        raise VcfParseError(
            "could not read reference FASTA",
            details={"path": str(src), "error": str(exc)},
        ) from exc

    chunks_by_contig: dict[str, list[str]] = {}
    current: str | None = None
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            contig = _parse_fasta_header(line, path=src, line_no=line_no)
            if contig in chunks_by_contig:
                raise VcfParseError(
                    "reference FASTA contains a duplicate contig",
                    details={"path": str(src), "line": line_no, "contig": contig},
                )
            chunks_by_contig[contig] = []
            current = contig
            continue
        if current is None:
            raise VcfParseError(
                "reference FASTA sequence appears before any header",
                details={"path": str(src), "line": line_no},
            )
        chunks_by_contig[current].append(line)

    if not chunks_by_contig:
        raise VcfParseError(
            "reference FASTA contains no contigs",
            details={"path": str(src)},
        )

    sequences: dict[str, str] = {}
    for contig, chunks in chunks_by_contig.items():
        if not chunks:
            raise VcfParseError(
                "reference FASTA contig contains no sequence",
                details={"path": str(src), "contig": contig},
            )
        try:
            sequences[contig] = canonicalize_dna("".join(chunks))
        except InputError as exc:
            raise VcfParseError(
                "reference FASTA contig contains unsupported base(s)",
                details={"path": str(src), "contig": contig},
            ) from exc
    return sequences


def _read_reference_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


def _parse_fasta_header(line: str, *, path: Path, line_no: int) -> str:
    parts = line[1:].split()
    if not parts:
        raise VcfParseError(
            "reference FASTA header must name a contig",
            details={"path": str(path), "line": line_no},
        )
    return parts[0]


def _extract_reference_window(
    variant: EditSpec,
    reference_sequences: Mapping[str, str],
    *,
    window_bp: int,
) -> _ReferenceWindow:
    contig, sequence = _reference_sequence_for_variant(variant, reference_sequences)
    ref_start = variant.pos - 1
    ref_end = ref_start + len(variant.ref)
    if ref_end > len(sequence):
        raise VcfParseError(
            "VCF REF extends past reference FASTA contig",
            details={
                "chrom": variant.chrom,
                "contig": contig,
                "pos": variant.pos,
                "ref": variant.ref,
                "contig_len": len(sequence),
            },
        )
    observed = sequence[ref_start:ref_end]
    if observed != variant.ref:
        raise VcfParseError(
            "reference FASTA bases do not match VCF REF",
            details={
                "chrom": variant.chrom,
                "contig": contig,
                "pos": variant.pos,
                "expected_ref": variant.ref,
                "observed_ref": observed,
            },
        )
    try:
        extracted = extract_window(sequence, edit_locus=ref_start, window_bp=window_bp)
    except InputError as exc:
        raise VcfParseError(
            "could not extract reference window from FASTA",
            details={"chrom": variant.chrom, "contig": contig, "pos": variant.pos},
        ) from exc
    return _ReferenceWindow(sequence=extracted.sequence, start_bp=extracted.start_bp)


def _reference_sequence_for_variant(
    variant: EditSpec, reference_sequences: Mapping[str, str]
) -> tuple[str, str]:
    for candidate in _contig_candidates(variant.chrom):
        sequence = reference_sequences.get(candidate)
        if sequence is not None:
            return candidate, sequence
    raise VcfParseError(
        "reference FASTA is missing VCF contig",
        details={
            "chrom": variant.chrom,
            "candidates": list(_contig_candidates(variant.chrom)),
        },
    )


def _contig_candidates(chrom: str) -> tuple[str, ...]:
    candidates = [chrom]
    if chrom.startswith("chr"):
        candidates.append(chrom[3:])
    else:
        candidates.append(f"chr{chrom}")
    upper = chrom.upper().removeprefix("CHR")
    if upper == "M":
        candidates.extend(("MT", "chrM", "chrMT"))
    elif upper == "MT":
        candidates.extend(("M", "chrM", "chrMT"))
    return tuple(dict.fromkeys(candidates))


def _normalize_aggregation(value: str) -> Aggregation:
    if value in {"mean", "max", "median"}:
        return cast(Aggregation, value)
    raise InputError(
        "aggregation must be one of mean, max, or median",
        details={"aggregation": value},
    )


def _require_calibration_table(value: CalibrationTable) -> None:
    if not isinstance(value, CalibrationTable):
        raise InputError(
            "calibration must be a CalibrationTable",
            details={"type": type(value).__name__},
        )


def _require_window_start(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputError(
            "window_start_bp must be a non-negative integer",
            details={"window_start_bp": value, "type": type(value).__name__},
        )
    return value


def _require_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputError(
            f"{name} must be a positive integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )
    return value


def _require_finite_float(name: str, value: int | float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise InputError(f"{name} must contain finite numeric values")
    return out


def _require_finite_non_negative(name: str, value: float) -> float:
    out = _require_finite_float(name, value)
    if out < 0.0:
        raise InputError(f"{name} must be non-negative", details={name: value})
    return out


def _require_probability(name: str, value: float) -> float:
    out = _require_finite_float(name, value)
    if out < 0.0 or out > 1.0:
        raise InputError(
            f"{name} must be in [0, 1]",
            details={name: value},
        )
    return out
