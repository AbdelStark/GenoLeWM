# SPDX-License-Identifier: Apache-2.0
"""Generate Carbon-backed v0.2 rollout input specs and cache rows.

The rollout evaluator already consumes cache-keyed measured latent
examples. This helper builds those upstream examples from released
gnomAD windows, released gnomAD SNVs, and deterministic synthetic edit
chains, then writes the documented window-embedding cache plus spec
JSONL files consumed by ``tools.release.rollout_state_examples``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from geno_lewm.action import EditType, RelEdit
from geno_lewm.action.apply import apply_edits
from geno_lewm.encoder import (
    POOL_CENTERED_MEAN,
    CarbonStateEncoder,
    WindowCacheKey,
    WindowCacheRecord,
    window_sha256,
    write_shard,
)
from geno_lewm.errors import GenoLeWMError, InputError, RuntimeSetupError, exit_code_for
from geno_lewm.provenance import load_manifest, sha256_file

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.v02_rollout_inputs"
SPEC_GENERATED_BY: Final = "tools.release.rollout_state_example_specs"
ISSUE_REFS: Final = ("#57", "#197")
DEFAULT_SYNTHETIC_HORIZON: Final = 16

PHASED_SPLIT: Final = "rollout_phased_haplotypes"
SYNTHETIC_SPLIT: Final = "rollout_synthetic_edit_chains"
_BASES: Final = "ACGT"


@dataclass(frozen=True, slots=True)
class SourceWindow:
    """A released placed-window row."""

    chrom: str
    start_bp: int
    end_bp: int
    sequence: str
    record_id: str


@dataclass(frozen=True, slots=True)
class AbsoluteSNV:
    """A genomic SNV used to construct rollout edits."""

    chrom: str
    pos: int
    ref: str
    alt: str


@dataclass(frozen=True, slots=True)
class CandidateSequence:
    """One target-rank candidate sequence."""

    candidate_id: str
    sequence: str
    edit_locus: int


@dataclass(frozen=True, slots=True)
class RolloutInputExample:
    """One rollout-state spec before Carbon encoding."""

    row_id: str
    split: str
    source: SourceWindow
    edits: tuple[RelEdit, ...]
    target_sequence: str
    candidates: tuple[CandidateSequence, ...]
    target_candidate_id: str

    @property
    def edit_locus(self) -> int:
        return self.edits[0].rel_pos


@dataclass(frozen=True, slots=True)
class EncodedState:
    """Encoded cache record metadata."""

    sequence: str
    key: WindowCacheKey
    record: WindowCacheRecord


def write_v02_rollout_inputs(
    *,
    artifact_root: Path,
    placed_windows_jsonl: Path,
    gnomad_variants_parquet: Path,
    model_manifest: Path,
    carbon_model_dir: Path,
    cache_dir: Path,
    phased_spec_jsonl: Path,
    synthetic_spec_jsonl: Path,
    output_report: Path,
    examples_per_split: int = 16,
    phased_horizon: int = 5,
    synthetic_horizon: int = DEFAULT_SYNTHETIC_HORIZON,
    candidate_count: int = 8,
    batch_size: int = 4,
    state_layer: int = -1,
    pool_type: str = POOL_CENTERED_MEAN,
    pool_radius: int = 256,
    dtype: str = "bf16",
    device: str | None = None,
    trust_remote_code: bool = False,
    allow_network_download: bool = False,
    encoder: object | None = None,
) -> dict[str, object]:
    """Write rollout spec JSONLs, cache rows, and an input provenance report."""
    _require_positive("examples_per_split", examples_per_split)
    _require_positive("phased_horizon", phased_horizon)
    _require_positive("synthetic_horizon", synthetic_horizon)
    _require_positive("candidate_count", candidate_count)
    _require_positive("batch_size", batch_size)
    if candidate_count < 2:
        raise InputError("candidate_count must be at least 2")

    artifact_root = artifact_root.resolve()
    manifest = load_manifest(model_manifest)
    _require_horizons_within_model_action_limit(
        model_manifest=model_manifest,
        manifest=manifest,
        phased_horizon=phased_horizon,
        synthetic_horizon=synthetic_horizon,
    )
    windows = load_source_windows(placed_windows_jsonl)
    variants_by_chrom = load_gnomad_snvs(gnomad_variants_parquet)
    phased_examples, phased_skipped = build_phased_examples(
        windows,
        variants_by_chrom=variants_by_chrom,
        limit=examples_per_split,
        horizon=phased_horizon,
        candidate_count=candidate_count,
    )
    synthetic_examples, synthetic_skipped = build_synthetic_examples(
        windows,
        limit=examples_per_split,
        horizon=synthetic_horizon,
        candidate_count=candidate_count,
        exclude_record_ids={example.source.record_id for example in phased_examples},
    )
    if encoder is None:
        encoder = CarbonStateEncoder(
            str(carbon_model_dir),
            manifest.encoder.revision,
            dtype=dtype,
            state_layer=state_layer,
            pool_type=pool_type,
            pool_radius=pool_radius,
            encoder_hash=manifest.encoder.hash,
            local_files_only=not allow_network_download,
            trust_remote_code=trust_remote_code,
            device=device,
        )
    encoded = encode_example_states(
        (*phased_examples, *synthetic_examples),
        encoder=encoder,
        encoder_hash=_hash_bytes(manifest.encoder.hash),
        state_layer=state_layer,
        pool_type=pool_type,
        pool_radius=pool_radius,
        dtype=dtype,
        batch_size=batch_size,
    )
    _write_cache_records(
        cache_dir=cache_dir,
        encoder_id="carbon-500m-v02-rollout",
        encoded=encoded,
    )
    _write_spec_jsonl(
        phased_spec_jsonl,
        phased_examples,
        encoded=encoded,
    )
    _write_spec_jsonl(
        synthetic_spec_jsonl,
        synthetic_examples,
        encoded=encoded,
    )
    report = _build_report(
        artifact_root=artifact_root,
        output_report=output_report,
        sources={
            "placed_windows_jsonl": placed_windows_jsonl,
            "gnomad_variants_parquet": gnomad_variants_parquet,
            "model_manifest": model_manifest,
            "carbon_model_dir": carbon_model_dir,
        },
        outputs={
            "cache_dir": cache_dir,
            "phased_spec_jsonl": phased_spec_jsonl,
            "synthetic_spec_jsonl": synthetic_spec_jsonl,
        },
        splits={
            PHASED_SPLIT: _example_summary(phased_examples),
            SYNTHETIC_SPLIT: _example_summary(synthetic_examples),
        },
        skipped={
            PHASED_SPLIT: phased_skipped,
            SYNTHETIC_SPLIT: synthetic_skipped,
        },
        settings={
            "examples_per_split": examples_per_split,
            "phased_horizon": phased_horizon,
            "synthetic_horizon": synthetic_horizon,
            "candidate_count": candidate_count,
            "state_layer": state_layer,
            "pool_type": pool_type,
            "pool_radius": pool_radius,
            "dtype": dtype,
            "device": device,
            "trust_remote_code": trust_remote_code,
            "allow_network_download": allow_network_download,
        },
    )
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def load_source_windows(path: Path) -> tuple[SourceWindow, ...]:
    """Load released placed windows from JSONL."""
    windows: list[SourceWindow] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError("placed window JSONL row is invalid", details={"line": line_no}) from exc
        windows.append(
            SourceWindow(
                chrom=_required_text(payload, "chrom", line_no=line_no),
                start_bp=_required_int(payload, "start_bp", line_no=line_no),
                end_bp=_required_int(payload, "end_bp", line_no=line_no),
                sequence=_required_text(payload, "sequence", line_no=line_no).upper(),
                record_id=_required_text(payload, "record_id", line_no=line_no),
            )
        )
    if not windows:
        raise InputError("placed window JSONL must contain at least one row")
    return tuple(windows)


def load_gnomad_snvs(path: Path) -> dict[str, tuple[AbsoluteSNV, ...]]:
    """Load released gnomAD SNVs grouped by chromosome."""
    rows = _read_parquet_rows(path, columns=("chrom", "pos", "ref", "alt", "filter"))
    grouped: dict[str, list[AbsoluteSNV]] = defaultdict(list)
    for row in rows:
        if row.get("filter") not in (None, "PASS"):
            continue
        snv = _absolute_snv(row.get("chrom"), row.get("pos"), row.get("ref"), row.get("alt"))
        if snv is not None:
            grouped[snv.chrom].append(snv)
    return {chrom: tuple(sorted(values, key=lambda snv: snv.pos)) for chrom, values in grouped.items()}


def build_phased_examples(
    windows: Sequence[SourceWindow],
    *,
    variants_by_chrom: Mapping[str, Sequence[AbsoluteSNV]],
    limit: int,
    horizon: int,
    candidate_count: int,
) -> tuple[tuple[RolloutInputExample, ...], dict[str, int]]:
    """Build examples from real gnomAD SNVs falling inside released windows."""
    skipped: Counter[str] = Counter()
    examples: list[RolloutInputExample] = []
    for window in windows:
        edits = _edits_from_window_variants(window, variants_by_chrom.get(window.chrom, ()))
        if len(edits) < horizon:
            skipped["insufficient_matching_gnomad_snvs"] += 1
            continue
        selected = tuple(edits[:horizon])
        target = apply_edits(window.sequence, selected, preserve_length=True)
        candidates = _candidate_sequences(
            source=window.sequence,
            target_sequence=target,
            target_edits=selected,
            horizon=horizon,
            candidate_count=candidate_count,
            seed=len(examples) + 17,
        )
        examples.append(
            RolloutInputExample(
                row_id=f"phased-{len(examples):04d}",
                split=PHASED_SPLIT,
                source=window,
                edits=selected,
                target_sequence=target,
                candidates=candidates,
                target_candidate_id="target",
            )
        )
        if len(examples) >= limit:
            break
    if len(examples) < limit:
        raise InputError(
            "not enough gnomAD-backed rollout examples",
            details={"observed": len(examples), "required": limit, "skipped": dict(skipped)},
        )
    return tuple(examples), dict(skipped)


def build_synthetic_examples(
    windows: Sequence[SourceWindow],
    *,
    limit: int,
    horizon: int,
    candidate_count: int,
    exclude_record_ids: set[str] | None = None,
) -> tuple[tuple[RolloutInputExample, ...], dict[str, int]]:
    """Build deterministic synthetic edit-chain examples from released windows."""
    skipped: Counter[str] = Counter()
    examples: list[RolloutInputExample] = []
    excluded = exclude_record_ids or set()
    for window in windows:
        if window.record_id in excluded:
            skipped["reserved_for_phased_split"] += 1
            continue
        try:
            edits = _synthetic_edits(window.sequence, horizon=horizon, seed=len(examples) + 101)
        except InputError:
            skipped["insufficient_editable_bases"] += 1
            continue
        target = apply_edits(window.sequence, edits, preserve_length=True)
        candidates = _candidate_sequences(
            source=window.sequence,
            target_sequence=target,
            target_edits=edits,
            horizon=horizon,
            candidate_count=candidate_count,
            seed=len(examples) + 211,
        )
        examples.append(
            RolloutInputExample(
                row_id=f"synthetic-{len(examples):04d}",
                split=SYNTHETIC_SPLIT,
                source=window,
                edits=edits,
                target_sequence=target,
                candidates=candidates,
                target_candidate_id="target",
            )
        )
        if len(examples) >= limit:
            break
    if len(examples) < limit:
        raise InputError(
            "not enough synthetic rollout examples",
            details={"observed": len(examples), "required": limit, "skipped": dict(skipped)},
        )
    return tuple(examples), dict(skipped)


def encode_example_states(
    examples: Sequence[RolloutInputExample],
    *,
    encoder: object,
    encoder_hash: bytes,
    state_layer: int,
    pool_type: str,
    pool_radius: int,
    dtype: str,
    batch_size: int,
) -> dict[str, EncodedState]:
    """Encode all unique example states with the supplied Carbon encoder."""
    states: dict[str, tuple[SourceWindow, str, int]] = {}
    for example in examples:
        states[f"{example.row_id}:source"] = (example.source, example.source.sequence, example.edit_locus)
        states[f"{example.row_id}:target"] = (example.source, example.target_sequence, example.edit_locus)
        for candidate in example.candidates:
            states[f"{example.row_id}:candidate:{candidate.candidate_id}"] = (
                example.source,
                candidate.sequence,
                candidate.edit_locus,
            )

    encode_batch = getattr(encoder, "encode_batch", None)
    if not callable(encode_batch):
        raise RuntimeSetupError("rollout input encoder must expose encode_batch(windows, edit_loci)")

    output: dict[str, EncodedState] = {}
    items = list(states.items())
    for offset in range(0, len(items), batch_size):
        batch = items[offset : offset + batch_size]
        sequences = [item[1][1] for item in batch]
        edit_loci = [item[1][2] for item in batch]
        vectors = encode_batch(sequences, edit_loci)
        if len(vectors) != len(batch):
            raise InputError("encoder returned a batch with the wrong length")
        for (state_id, (window, sequence, _edit_locus)), vector in zip(batch, vectors, strict=True):
            embedding = _state_vector(vector, state_id=state_id)
            window_hash = window_sha256(sequence)
            key = WindowCacheKey(
                window_hash=window_hash,
                encoder_hash=encoder_hash,
                state_layer=state_layer,
                pool_type=pool_type,
                pool_radius=pool_radius,
                dtype=dtype,
            )
            output[state_id] = EncodedState(
                sequence=sequence,
                key=key,
                record=WindowCacheRecord(
                    chrom=window.chrom,
                    start_bp=window.start_bp,
                    end_bp=window.end_bp,
                    window_hash=window_hash,
                    encoder_hash=encoder_hash,
                    state_layer=state_layer,
                    pool_type=pool_type,
                    pool_radius=pool_radius,
                    dtype=dtype,
                    embedding=embedding,
                    untargeted=False,
                ),
            )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        write_v02_rollout_inputs(
            artifact_root=args.artifact_root,
            placed_windows_jsonl=args.placed_windows_jsonl,
            gnomad_variants_parquet=args.gnomad_variants_parquet,
            model_manifest=args.model_manifest,
            carbon_model_dir=args.carbon_model_dir,
            cache_dir=args.cache_dir,
            phased_spec_jsonl=args.phased_spec_jsonl,
            synthetic_spec_jsonl=args.synthetic_spec_jsonl,
            output_report=args.output_report,
            examples_per_split=args.examples_per_split,
            phased_horizon=args.phased_horizon,
            synthetic_horizon=args.synthetic_horizon,
            candidate_count=args.candidate_count,
            batch_size=args.batch_size,
            state_layer=args.state_layer,
            pool_type=args.pool_type,
            pool_radius=args.pool_radius,
            dtype=args.dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            allow_network_download=args.allow_network_download,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(f"wrote {args.output_report}\n")
    return 0


def _edits_from_window_variants(
    window: SourceWindow,
    variants: Sequence[AbsoluteSNV],
) -> tuple[RelEdit, ...]:
    edits: list[RelEdit] = []
    for variant in variants:
        rel_pos = variant.pos - 1 - window.start_bp
        if rel_pos < 0:
            continue
        if rel_pos >= len(window.sequence):
            break
        observed = window.sequence[rel_pos].upper()
        if observed != variant.ref:
            continue
        edits.append(
            RelEdit(
                rel_pos=rel_pos,
                edit_type=EditType.SNV,
                ref_bases=variant.ref,
                alt_bases=variant.alt,
            )
        )
    return tuple(edits)


def _synthetic_edits(sequence: str, *, horizon: int, seed: int) -> tuple[RelEdit, ...]:
    editable = [idx for idx, base in enumerate(sequence) if base in _BASES]
    if len(editable) < horizon:
        raise InputError("not enough editable bases for synthetic rollout chain")
    rng = random.Random(seed)
    step = max(1, len(editable) // (horizon + 1))
    positions = [editable[min(len(editable) - 1, (idx + 1) * step)] for idx in range(horizon)]
    rng.shuffle(positions)
    selected = sorted(positions[:horizon])
    return tuple(_synthetic_edit(sequence, pos, salt=seed + index) for index, pos in enumerate(selected))


def _synthetic_edit(sequence: str, pos: int, *, salt: int) -> RelEdit:
    ref = sequence[pos].upper()
    choices = [base for base in _BASES if base != ref]
    alt = choices[salt % len(choices)]
    return RelEdit(rel_pos=pos, edit_type=EditType.SNV, ref_bases=ref, alt_bases=alt)


def _candidate_sequences(
    *,
    source: str,
    target_sequence: str,
    target_edits: Sequence[RelEdit],
    horizon: int,
    candidate_count: int,
    seed: int,
) -> tuple[CandidateSequence, ...]:
    candidates = [
        CandidateSequence(
            candidate_id="target",
            sequence=target_sequence,
            edit_locus=target_edits[0].rel_pos,
        )
    ]
    rng = random.Random(seed)
    attempts = 0
    seen = {target_sequence}
    while len(candidates) < candidate_count and attempts < candidate_count * 25:
        attempts += 1
        edits = list(_synthetic_edits(source, horizon=horizon, seed=seed + attempts))
        if edits and rng.random() < 0.5:
            drop_index = rng.randrange(len(edits))
            edits[drop_index] = _synthetic_edit(source, edits[drop_index].rel_pos, salt=seed + attempts + 1)
        sequence = apply_edits(source, tuple(edits), preserve_length=True)
        if sequence in seen:
            continue
        seen.add(sequence)
        candidates.append(
            CandidateSequence(
                candidate_id=f"distractor-{len(candidates):02d}",
                sequence=sequence,
                edit_locus=edits[0].rel_pos,
            )
        )
    if len(candidates) < candidate_count:
        raise InputError(
            "could not construct enough distinct rollout candidates",
            details={"observed": len(candidates), "required": candidate_count},
        )
    return tuple(candidates)


def _write_cache_records(
    *,
    cache_dir: Path,
    encoder_id: str,
    encoded: Mapping[str, EncodedState],
) -> None:
    by_key: dict[WindowCacheKey, WindowCacheRecord] = {}
    for state in encoded.values():
        existing = by_key.get(state.key)
        if existing is not None and existing.embedding != state.record.embedding:
            raise InputError(
                "duplicate rollout cache key has conflicting embeddings",
                details={"window_hash": state.key.window_hash.hex()},
            )
        by_key[state.key] = state.record
    by_chrom: dict[str, list[WindowCacheRecord]] = defaultdict(list)
    for record in by_key.values():
        by_chrom[record.chrom].append(record)
    for chrom, records in sorted(by_chrom.items()):
        write_shard(
            cache_dir,
            encoder_id=encoder_id,
            contig=chrom,
            stride_block=0,
            records=tuple(records),
        )


def _write_spec_jsonl(
    path: Path,
    examples: Sequence[RolloutInputExample],
    *,
    encoded: Mapping[str, EncodedState],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_spec_row(example, encoded=encoded) for example in examples]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _spec_row(
    example: RolloutInputExample,
    *,
    encoded: Mapping[str, EncodedState],
) -> dict[str, object]:
    source_key = encoded[f"{example.row_id}:source"].key
    target_key = encoded[f"{example.row_id}:target"].key
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": SPEC_GENERATED_BY,
        "id": example.row_id,
        "split": example.split,
        "source_state_key": _key_dict(source_key),
        "target_state_key": _key_dict(target_key),
        "target_candidate_id": example.target_candidate_id,
        "edits": [
            {
                "rel_pos": edit.rel_pos,
                "edit_type": int(edit.edit_type),
                "ref_bases": edit.ref_bases,
                "alt_bases": edit.alt_bases,
            }
            for edit in example.edits
        ],
        "candidates": [
            {
                "id": candidate.candidate_id,
                "state_key": _key_dict(
                    encoded[f"{example.row_id}:candidate:{candidate.candidate_id}"].key
                ),
            }
            for candidate in example.candidates
        ],
        "source_window": {
            "record_id": example.source.record_id,
            "chrom": example.source.chrom,
            "start_bp": example.source.start_bp,
            "end_bp": example.source.end_bp,
        },
    }


def _key_dict(key: WindowCacheKey) -> dict[str, object]:
    return {
        "window_hash": key.window_hash.hex(),
        "encoder_hash": key.encoder_hash.hex(),
        "state_layer": key.state_layer,
        "pool_type": key.pool_type,
        "pool_radius": key.pool_radius,
        "dtype": key.dtype,
    }


def _read_parquet_rows(path: Path, *, columns: Sequence[str]) -> tuple[dict[str, object], ...]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeSetupError(
            "v0.2 rollout input generation requires pyarrow",
            remediation="install the project development or release dependencies",
        ) from exc
    try:
        table = pq.read_table(path, columns=list(columns))
    except Exception as exc:
        raise InputError("failed to read gnomAD parquet input", details={"path": str(path)}) from exc
    return tuple(dict(row) for row in table.to_pylist())


def _absolute_snv(chrom: object, pos: object, ref: object, alt: object) -> AbsoluteSNV | None:
    if isinstance(pos, bool):
        return None
    try:
        pos_int = int(pos)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not isinstance(chrom, str) or not isinstance(ref, str) or not isinstance(alt, str):
        return None
    ref_text = ref.upper()
    alt_text = alt.upper()
    if len(ref_text) != 1 or len(alt_text) != 1:
        return None
    if ref_text not in _BASES or alt_text not in _BASES or ref_text == alt_text:
        return None
    return AbsoluteSNV(chrom=_normalize_chrom(chrom), pos=pos_int, ref=ref_text, alt=alt_text)


def _normalize_chrom(chrom: str) -> str:
    value = chrom.strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    return value


def _state_vector(value: object, *, state_id: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise InputError("encoder state vector is not numeric", details={"state_id": state_id}) from exc
    if not values:
        raise InputError("encoder state vector is empty", details={"state_id": state_id})
    if any(not math.isfinite(item) for item in values):
        raise InputError("encoder state vector contains non-finite values", details={"state_id": state_id})
    return values


def _hash_bytes(value: str) -> bytes:
    prefix = "sha256:"
    if not value.startswith(prefix):
        raise InputError("manifest encoder hash must be sha256-prefixed")
    return bytes.fromhex(value[len(prefix) :])


def _build_report(
    *,
    artifact_root: Path,
    output_report: Path,
    sources: Mapping[str, Path],
    outputs: Mapping[str, Path],
    splits: Mapping[str, Mapping[str, object]],
    skipped: Mapping[str, Mapping[str, int]],
    settings: Mapping[str, object],
) -> dict[str, object]:
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "issue_refs": list(ISSUE_REFS),
        "report_path": _display_path(output_report, root=artifact_root),
        "source_artifacts": {
            label: _path_identity(path, root=artifact_root, label=label)
            for label, path in sorted(sources.items())
        },
        "outputs": {
            label: _path_identity(path, root=artifact_root, label=label)
            for label, path in sorted(outputs.items())
        },
        "settings": dict(settings),
        "splits": dict(splits),
        "skipped": {label: dict(values) for label, values in skipped.items()},
        "limitations": [
            (
                "The phased split uses released gnomAD common SNVs observed in released "
                "placed windows; the current dataset does not expose individual sample "
                "phase blocks."
            ),
            "The synthetic split uses deterministic SNV edit chains on released windows.",
            "All source, target, and candidate states are encoded by Carbon before rollout evaluation.",
        ],
    }


def _path_identity(path: Path, *, root: Path, label: str) -> dict[str, object]:
    if path.is_dir():
        if label == "cache_dir":
            index = path / "embeddings" / "index.sqlite"
            if not index.exists():
                raise InputError("cache directory is missing index.sqlite", details={"label": label})
            return {
                "path": _display_path(path, root=root),
                "index_sha256": sha256_file(index),
                "index_size_bytes": index.stat().st_size,
            }
        return {
            "path": _display_path(path, root=root),
            "directory": True,
            "top_level_files": [
                _file_identity(child, root=root)
                for child in sorted(path.iterdir())
                if child.is_file()
            ][:16],
        }
    if not path.exists():
        raise InputError("rollout input artifact does not exist", details={"label": label, "path": str(path)})
    return {
        "path": _display_path(path, root=root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _file_identity(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": _display_path(path, root=root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _display_path(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _example_summary(examples: Sequence[RolloutInputExample]) -> dict[str, object]:
    horizons = Counter(len(example.edits) for example in examples)
    return {
        "rows": len(examples),
        "horizons": dict(sorted(horizons.items())),
        "candidate_count": len(examples[0].candidates) if examples else 0,
        "source_windows": len({example.source.record_id for example in examples}),
    }


def _required_text(payload: Mapping[str, object], key: str, *, line_no: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise InputError(f"{key} must be a non-empty string", details={"line": line_no})
    return value


def _required_int(payload: Mapping[str, object], key: str, *, line_no: int) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{key} must be an integer", details={"line": line_no})
    return value


def _require_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(f"{name} must be a positive integer")


def _require_horizons_within_model_action_limit(
    *,
    model_manifest: Path,
    manifest: object,
    phased_horizon: int,
    synthetic_horizon: int,
) -> None:
    max_len = _model_action_max_len(model_manifest=model_manifest, manifest=manifest)
    requested = {
        "phased_horizon": phased_horizon,
        "synthetic_horizon": synthetic_horizon,
    }
    offending = {name: value for name, value in requested.items() if value > max_len}
    if offending:
        raise InputError(
            "rollout horizon exceeds model action max_len",
            details={"action_max_len": max_len, **offending},
        )


def _model_action_max_len(*, model_manifest: Path, manifest: object) -> int:
    from geno_lewm.config import load_config

    training = getattr(manifest, "training", None)
    config_file = getattr(training, "config_file", None)
    if not isinstance(config_file, str) or not config_file.strip():
        raise InputError("model manifest must name training.config_file")
    config_path = (model_manifest.parent / config_file).resolve()
    if not config_path.is_file():
        raise InputError(
            "model training config is missing",
            details={"path": str(config_path), "manifest": str(model_manifest)},
        )
    cfg = load_config(config_path)
    max_len = getattr(cfg.action, "max_len", None)
    if isinstance(max_len, bool) or not isinstance(max_len, int) or max_len <= 0:
        raise InputError("model action.max_len must be a positive integer")
    return max_len


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--placed-windows-jsonl", type=Path, required=True)
    parser.add_argument("--gnomad-variants-parquet", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--carbon-model-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--phased-spec-jsonl", type=Path, required=True)
    parser.add_argument("--synthetic-spec-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--examples-per-split", type=int, default=16)
    parser.add_argument("--phased-horizon", type=int, default=5)
    parser.add_argument("--synthetic-horizon", type=int, default=DEFAULT_SYNTHETIC_HORIZON)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--state-layer", type=int, default=-1)
    parser.add_argument("--pool-type", default=POOL_CENTERED_MEAN)
    parser.add_argument("--pool-radius", type=int, default=256)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-network-download", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
