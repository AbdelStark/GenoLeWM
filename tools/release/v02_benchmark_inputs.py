# SPDX-License-Identifier: Apache-2.0
"""Generate public-source v0.2 VEP benchmark inputs.

The v0.2 benchmark suite intentionally keeps scoring and evaluation in
separate tools. This helper only normalizes upstream public benchmark
rows into VCF plus label JSONL inputs consumed by ``geno-lewm-score`` and
``geno-lewm-eval``. It records source identities and row-count
limitations so the benchmark report can distinguish measured evidence
from upstream selection choices.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, TextIO
from urllib.parse import unquote

from geno_lewm.errors import GenoLeWMError, InputError, RuntimeSetupError, exit_code_for
from geno_lewm.provenance import sha256_file

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.v02_benchmark_inputs"
ISSUE_REFS: Final = ("#53", "#55", "#56", "#197")

CLINVAR_CODING_SPLIT: Final = "clinvar_coding"
CLINVAR_NONCODING_SPLIT: Final = "clinvar_noncoding"
BRCA2_SPLIT: Final = "brca2_saturation"
TRAITGYM_SPLIT: Final = "traitgym_mendelian"

_PATHOGENIC = frozenset({"P", "LP", "pathogenic", "likely_pathogenic"})
_BENIGN = frozenset({"B", "LB", "benign", "likely_benign"})
_CODING_CONSEQUENCES = frozenset(
    {
        "coding_sequence_variant",
        "frameshift_variant",
        "inframe_deletion",
        "inframe_insertion",
        "initiator_codon_variant",
        "missense_variant",
        "nonsense",
        "protein_altering_variant",
        "splice_acceptor_variant",
        "splice_donor_variant",
        "start_lost",
        "stop_gained",
        "stop_lost",
        "stop_retained_variant",
        "synonymous_variant",
    }
)
_NCBI_CHROM_RE = re.compile(r"^NC_0*(?P<num>\d+)\.\d+$")
_NCBI_SPECIAL = {
    "NC_000023": "X",
    "NC_000024": "Y",
    "NC_012920": "MT",
}


@dataclass(frozen=True, slots=True)
class VariantKey:
    """A normalized SNV key."""

    chrom: str
    pos: int
    ref: str
    alt: str

    def vcf_tuple(self) -> tuple[str, int, str, str]:
        return (self.chrom, self.pos, self.ref, self.alt)

    def identity(self) -> str:
        return f"{self.chrom}:{self.pos}:{self.ref}>{self.alt}"


@dataclass(frozen=True, slots=True)
class ClinVarRow:
    """One ClinVar classification row after consequence joining."""

    key: VariantKey
    clinical_significance: str
    label: int
    consequences: tuple[str, ...]
    gene_symbol: str | None = None
    clinvar_id: str | None = None


@dataclass(frozen=True, slots=True)
class ContinuousRow:
    """One continuous benchmark-label row."""

    key: VariantKey
    functional_score: float
    source_id: str
    extra: Mapping[str, object]


def write_v02_benchmark_inputs(
    *,
    artifact_root: Path,
    clinvar_parquet: Path,
    clinvar_source_vcf: Path,
    traitgym_parquet: Path,
    brca2_scores_csv: Path,
    brca2_mapped_variants_json: Path,
    output_report: Path,
    max_clinvar_per_split: int = 3_000,
    max_traitgym_rows: int | None = None,
    max_brca2_rows: int | None = None,
    seed: int = 20260608,
    chromosomes: tuple[str, ...] = (),
) -> dict[str, object]:
    """Write VCF/label artifacts for all v0.2 VEP benchmark splits."""
    _require_positive("max_clinvar_per_split", max_clinvar_per_split)
    if max_traitgym_rows is not None:
        _require_positive("max_traitgym_rows", max_traitgym_rows)
    if max_brca2_rows is not None:
        _require_positive("max_brca2_rows", max_brca2_rows)

    artifact_root = artifact_root.resolve()
    consequences = load_clinvar_consequence_index(clinvar_source_vcf)
    clinvar_rows, clinvar_skipped = load_clinvar_rows(
        clinvar_parquet,
        consequences=consequences,
    )
    chromosome_filter = frozenset(_normalize_chrom(chrom) for chrom in chromosomes)
    if chromosome_filter:
        clinvar_rows = tuple(row for row in clinvar_rows if row.key.chrom in chromosome_filter)
    coding, noncoding = split_clinvar_rows(clinvar_rows)
    coding = select_balanced_clinvar_rows(coding, limit=max_clinvar_per_split, seed=seed)
    noncoding = select_balanced_clinvar_rows(
        noncoding,
        limit=max_clinvar_per_split,
        seed=seed + 1,
    )
    brca2_rows, brca2_skipped = load_brca2_rows(
        scores_csv=brca2_scores_csv,
        mapped_variants_json=brca2_mapped_variants_json,
        limit=max_brca2_rows,
        seed=seed + 2,
    )
    traitgym_rows, traitgym_skipped = load_traitgym_rows(
        traitgym_parquet,
        limit=max_traitgym_rows,
        seed=seed + 3,
    )
    if chromosome_filter:
        brca2_rows = tuple(row for row in brca2_rows if row.key.chrom in chromosome_filter)
        traitgym_rows = tuple(row for row in traitgym_rows if row.key.chrom in chromosome_filter)

    output_dir = artifact_root / "benchmark_inputs"
    eval_dir = artifact_root / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, tuple[Path, Path]] = {
        CLINVAR_CODING_SPLIT: (
            output_dir / "clinvar_coding.vcf",
            eval_dir / "clinvar_coding.labels.jsonl",
        ),
        CLINVAR_NONCODING_SPLIT: (
            output_dir / "clinvar_noncoding.vcf",
            eval_dir / "clinvar_noncoding.labels.jsonl",
        ),
        BRCA2_SPLIT: (
            output_dir / "brca2_saturation.vcf",
            eval_dir / "brca2_saturation.labels.jsonl",
        ),
        TRAITGYM_SPLIT: (
            output_dir / "traitgym_mendelian.vcf",
            eval_dir / "traitgym_mendelian.labels.jsonl",
        ),
    }
    write_clinvar_artifacts(
        coding,
        split=CLINVAR_CODING_SPLIT,
        paths=outputs[CLINVAR_CODING_SPLIT],
    )
    write_clinvar_artifacts(
        noncoding,
        split=CLINVAR_NONCODING_SPLIT,
        paths=outputs[CLINVAR_NONCODING_SPLIT],
    )
    write_continuous_artifacts(brca2_rows, split=BRCA2_SPLIT, paths=outputs[BRCA2_SPLIT])
    write_continuous_artifacts(
        traitgym_rows,
        split=TRAITGYM_SPLIT,
        paths=outputs[TRAITGYM_SPLIT],
    )

    report = _build_report(
        artifact_root=artifact_root,
        output_report=output_report,
        sources={
            "clinvar_parquet": clinvar_parquet,
            "clinvar_source_vcf": clinvar_source_vcf,
            "traitgym_parquet": traitgym_parquet,
            "brca2_scores_csv": brca2_scores_csv,
            "brca2_mapped_variants_json": brca2_mapped_variants_json,
        },
        splits={
            CLINVAR_CODING_SPLIT: _clinvar_split_summary(coding),
            CLINVAR_NONCODING_SPLIT: _clinvar_split_summary(noncoding),
            BRCA2_SPLIT: _continuous_split_summary(brca2_rows),
            TRAITGYM_SPLIT: _continuous_split_summary(traitgym_rows),
        },
        outputs=outputs,
        skipped={
            "clinvar": clinvar_skipped,
            "brca2_saturation": brca2_skipped,
            "traitgym_mendelian": traitgym_skipped,
        },
        selection={
            "seed": seed,
            "max_clinvar_per_split": max_clinvar_per_split,
            "max_traitgym_rows": max_traitgym_rows,
            "max_brca2_rows": max_brca2_rows,
            "chromosomes": sorted(chromosome_filter),
        },
    )
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def load_clinvar_consequence_index(path: Path) -> dict[VariantKey, tuple[str, ...]]:
    """Return ``VariantKey -> Sequence Ontology consequence names`` from ClinVar VCF."""
    index: dict[VariantKey, set[str]] = defaultdict(set)
    with _open_text(path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom, pos_text, _var_id, ref, alts, *_rest, info = fields[:8]
            try:
                pos = int(pos_text)
            except ValueError:
                continue
            consequences = _consequences_from_info(info)
            if not consequences:
                continue
            for alt in alts.split(","):
                key = _variant_key(chrom, pos, ref, alt)
                if key is not None:
                    index[key].update(consequences)
    return {key: tuple(sorted(values)) for key, values in index.items()}


def load_clinvar_rows(
    path: Path,
    *,
    consequences: Mapping[VariantKey, tuple[str, ...]],
) -> tuple[tuple[ClinVarRow, ...], dict[str, int]]:
    """Load released ClinVar parquet rows and attach source VCF consequences."""
    rows: list[ClinVarRow] = []
    skipped: Counter[str] = Counter()
    seen: dict[VariantKey, int] = {}
    for payload in _read_parquet_rows(
        path,
        columns=(
            "chrom",
            "pos",
            "ref",
            "alt",
            "clinical_significance",
            "gene_symbol",
            "clinvar_id",
        ),
    ):
        key = _variant_key(payload.get("chrom"), payload.get("pos"), payload.get("ref"), payload.get("alt"))
        if key is None:
            skipped["not_snv"] += 1
            continue
        label = _clinvar_label(payload.get("clinical_significance"))
        if label is None:
            skipped["unsupported_clinical_significance"] += 1
            continue
        observed_consequences = consequences.get(key)
        if not observed_consequences:
            skipped["missing_consequence"] += 1
            continue
        previous = seen.get(key)
        if previous is not None:
            if previous != label:
                skipped["conflicting_duplicate"] += 1
            else:
                skipped["duplicate"] += 1
            continue
        seen[key] = label
        rows.append(
            ClinVarRow(
                key=key,
                clinical_significance=str(payload.get("clinical_significance")),
                label=label,
                consequences=observed_consequences,
                gene_symbol=_optional_text(payload.get("gene_symbol")),
                clinvar_id=_optional_text(payload.get("clinvar_id")),
            )
        )
    return tuple(rows), dict(skipped)


def split_clinvar_rows(rows: Sequence[ClinVarRow]) -> tuple[tuple[ClinVarRow, ...], tuple[ClinVarRow, ...]]:
    """Split ClinVar rows into coding and noncoding SNV benchmark sets."""
    coding: list[ClinVarRow] = []
    noncoding: list[ClinVarRow] = []
    for row in rows:
        if any(value in _CODING_CONSEQUENCES for value in row.consequences):
            coding.append(row)
        else:
            noncoding.append(row)
    return tuple(coding), tuple(noncoding)


def select_balanced_clinvar_rows(
    rows: Sequence[ClinVarRow],
    *,
    limit: int,
    seed: int,
) -> tuple[ClinVarRow, ...]:
    """Select a deterministic, approximately label-balanced ClinVar subset."""
    positives = [row for row in rows if row.label == 1]
    negatives = [row for row in rows if row.label == 0]
    if not positives or not negatives:
        raise InputError(
            "ClinVar benchmark split needs both pathogenic and benign labels",
            details={"positives": len(positives), "negatives": len(negatives)},
        )
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    half = max(1, limit // 2)
    selected = positives[:half] + negatives[: max(1, limit - min(half, len(positives)))]
    selected = selected[:limit]
    if len({row.label for row in selected}) != 2:
        raise InputError("balanced ClinVar selection lost one class")
    return tuple(sorted(selected, key=lambda row: row.key.vcf_tuple()))


def load_traitgym_rows(
    path: Path,
    *,
    limit: int | None,
    seed: int,
) -> tuple[tuple[ContinuousRow, ...], dict[str, int]]:
    """Load TraitGym Mendelian rows into continuous-label benchmark records."""
    skipped: Counter[str] = Counter()
    rows: list[ContinuousRow] = []
    for payload in _read_parquet_rows(
        path,
        columns=("chrom", "pos", "ref", "alt", "label", "consequence", "OMIM"),
    ):
        key = _variant_key(payload.get("chrom"), payload.get("pos"), payload.get("ref"), payload.get("alt"))
        if key is None:
            skipped["not_snv"] += 1
            continue
        label = payload.get("label")
        if not isinstance(label, bool):
            skipped["missing_boolean_label"] += 1
            continue
        rows.append(
            ContinuousRow(
                key=key,
                functional_score=1.0 if label else 0.0,
                source_id=str(payload.get("OMIM") or key.identity()),
                extra={
                    "traitgym_label": label,
                    "consequence": _optional_text(payload.get("consequence")),
                    "label_direction": "1.0 denotes TraitGym positive/causal row",
                },
            )
        )
    return _select_continuous_rows(rows, limit=limit, seed=seed), dict(skipped)


def load_brca2_rows(
    *,
    scores_csv: Path,
    mapped_variants_json: Path,
    limit: int | None,
    seed: int,
) -> tuple[tuple[ContinuousRow, ...], dict[str, int]]:
    """Load BRCA2 MaveDB scores with current successful VRS genomic mappings."""
    scores = _load_mavedb_scores(scores_csv)
    mapped = json.loads(mapped_variants_json.read_text(encoding="utf-8"))
    if not isinstance(mapped, list):
        raise InputError("MaveDB mapped variants JSON must be a list")
    skipped: Counter[str] = Counter()
    rows: list[ContinuousRow] = []
    seen: set[VariantKey] = set()
    for payload in mapped:
        if not isinstance(payload, dict):
            skipped["invalid_mapping_row"] += 1
            continue
        if payload.get("current") is not True:
            skipped["not_current"] += 1
            continue
        accession = payload.get("variantUrn")
        if not isinstance(accession, str) or accession not in scores:
            skipped["missing_score"] += 1
            continue
        mapped_key = _key_from_mavedb_mapping(payload.get("postMapped"))
        if mapped_key is None:
            skipped["unmapped_or_non_snv"] += 1
            continue
        if mapped_key in seen:
            skipped["duplicate_variant_key"] += 1
            continue
        seen.add(mapped_key)
        score, source_score = scores[accession]
        rows.append(
            ContinuousRow(
                key=mapped_key,
                functional_score=score,
                source_id=accession,
                extra={
                    "source_score": source_score,
                    "score_direction": "MaveDB function score direction is preserved.",
                },
            )
        )
    return _select_continuous_rows(rows, limit=limit, seed=seed), dict(skipped)


def write_clinvar_artifacts(
    rows: Sequence[ClinVarRow],
    *,
    split: str,
    paths: tuple[Path, Path],
) -> None:
    """Write ClinVar split VCF and binary labels JSONL."""
    vcf_path, labels_path = paths
    _write_vcf(vcf_path, (row.key for row in rows), source=f"{GENERATED_BY}:{split}")
    payloads = [
        {
            "chrom": row.key.chrom,
            "pos": row.key.pos,
            "ref": row.key.ref,
            "alt": row.key.alt,
            "clinical_significance": row.clinical_significance,
            "label": row.label,
            "split": split,
            "consequence": list(row.consequences),
            "gene_symbol": row.gene_symbol,
            "clinvar_id": row.clinvar_id,
        }
        for row in rows
    ]
    _write_jsonl(labels_path, payloads)


def write_continuous_artifacts(
    rows: Sequence[ContinuousRow],
    *,
    split: str,
    paths: tuple[Path, Path],
) -> None:
    """Write continuous benchmark VCF and label JSONL."""
    vcf_path, labels_path = paths
    _write_vcf(vcf_path, (row.key for row in rows), source=f"{GENERATED_BY}:{split}")
    payloads = [
        {
            "chrom": row.key.chrom,
            "pos": row.key.pos,
            "ref": row.key.ref,
            "alt": row.key.alt,
            "functional_score": row.functional_score,
            "source_id": row.source_id,
            "split": split,
            **{key: value for key, value in row.extra.items() if value is not None},
        }
        for row in rows
    ]
    _write_jsonl(labels_path, payloads)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        write_v02_benchmark_inputs(
            artifact_root=args.artifact_root,
            clinvar_parquet=args.clinvar_parquet,
            clinvar_source_vcf=args.clinvar_source_vcf,
            traitgym_parquet=args.traitgym_parquet,
            brca2_scores_csv=args.brca2_scores_csv,
            brca2_mapped_variants_json=args.brca2_mapped_variants_json,
            output_report=args.output_report,
            max_clinvar_per_split=args.max_clinvar_per_split,
            max_traitgym_rows=args.max_traitgym_rows,
            max_brca2_rows=args.max_brca2_rows,
            seed=args.seed,
            chromosomes=tuple(args.chromosome or ()),
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(f"wrote {args.output_report}\n")
    return 0


def _read_parquet_rows(path: Path, *, columns: Sequence[str]) -> tuple[dict[str, object], ...]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeSetupError(
            "v0.2 benchmark input generation requires pyarrow",
            remediation="install the project development or release dependencies",
        ) from exc
    try:
        table = pq.read_table(path, columns=list(columns))
    except Exception as exc:
        raise InputError(
            "failed to read benchmark parquet input",
            details={"path": str(path)},
        ) from exc
    return tuple(dict(row) for row in table.to_pylist())


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _consequences_from_info(info: str) -> tuple[str, ...]:
    values: set[str] = set()
    for field in info.split(";"):
        if not field.startswith("MC="):
            continue
        for raw_item in field[3:].split(","):
            item = unquote(raw_item)
            consequence = item.rsplit("|", 1)[-1].strip()
            if consequence:
                values.add(consequence)
    return tuple(sorted(values))


def _variant_key(
    chrom: object,
    pos: object,
    ref: object,
    alt: object,
) -> VariantKey | None:
    if isinstance(pos, bool):
        return None
    try:
        pos_int = int(pos)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not isinstance(chrom, str) or not isinstance(ref, str) or not isinstance(alt, str):
        return None
    chrom_text = _normalize_chrom(chrom)
    ref_text = ref.upper()
    alt_text = alt.upper()
    if len(ref_text) != 1 or len(alt_text) != 1:
        return None
    if ref_text not in "ACGT" or alt_text not in "ACGT" or ref_text == alt_text:
        return None
    if pos_int < 1:
        return None
    return VariantKey(chrom=chrom_text, pos=pos_int, ref=ref_text, alt=alt_text)


def _normalize_chrom(chrom: str) -> str:
    value = chrom.strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    prefix = value.split(".", 1)[0]
    if prefix in _NCBI_SPECIAL:
        return _NCBI_SPECIAL[prefix]
    match = _NCBI_CHROM_RE.match(value)
    if match:
        number = int(match.group("num"))
        if number == 23:
            return "X"
        if number == 24:
            return "Y"
        return str(number)
    return value


def _clinvar_label(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    normalized = text.lower().replace(" ", "_").replace("-", "_")
    if text in _PATHOGENIC or normalized in _PATHOGENIC:
        return 1
    if text in _BENIGN or normalized in _BENIGN:
        return 0
    return None


def _load_mavedb_scores(path: Path) -> dict[str, tuple[float, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        scores: dict[str, tuple[float, float]] = {}
        for row in reader:
            accession = row.get("accession")
            score_text = row.get("score")
            if not accession or score_text is None:
                continue
            try:
                score = float(score_text)
            except ValueError:
                continue
            if math.isfinite(score):
                scores[accession] = (score, score)
    return scores


def _key_from_mavedb_mapping(payload: object) -> VariantKey | None:
    if not isinstance(payload, dict):
        return None
    location = payload.get("location")
    state = payload.get("state")
    if not isinstance(location, dict) or not isinstance(state, dict):
        return None
    ref = _extension_value(payload.get("extensions"), "vrs_ref_allele_seq")
    alt = state.get("sequence")
    sequence_reference = location.get("sequenceReference")
    if not isinstance(sequence_reference, dict):
        return None
    label = sequence_reference.get("label")
    end = location.get("end")
    return _variant_key(label, end, ref, alt)


def _extension_value(raw: object, name: str) -> str | None:
    if not isinstance(raw, list):
        return None
    for item in raw:
        if isinstance(item, dict) and item.get("name") == name:
            value = item.get("value")
            return value if isinstance(value, str) else None
    return None


def _select_continuous_rows(
    rows: Sequence[ContinuousRow],
    *,
    limit: int | None,
    seed: int,
) -> tuple[ContinuousRow, ...]:
    if not rows:
        raise InputError("continuous benchmark split has no scoreable rows")
    selected = list(rows)
    if limit is not None and len(selected) > limit:
        rng = random.Random(seed)
        rng.shuffle(selected)
        selected = selected[:limit]
    return tuple(sorted(selected, key=lambda row: row.key.vcf_tuple()))


def _write_vcf(path: Path, keys: Iterable[VariantKey], *, source: str) -> None:
    rows = sorted(keys, key=lambda key: key.vcf_tuple())
    if not rows:
        raise InputError("cannot write an empty VCF", details={"path": str(path)})
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "##fileformat=VCFv4.2",
        f"##source={source}",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    ]
    lines.extend(
        f"{row.chrom}\t{row.pos}\t.\t{row.ref}\t{row.alt}\t.\tPASS\t."
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise InputError("cannot write an empty label JSONL", details={"path": str(path)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(dict(row), sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _build_report(
    *,
    artifact_root: Path,
    output_report: Path,
    sources: Mapping[str, Path],
    splits: Mapping[str, Mapping[str, object]],
    outputs: Mapping[str, tuple[Path, Path]],
    skipped: Mapping[str, Mapping[str, int]],
    selection: Mapping[str, object],
) -> dict[str, object]:
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "issue_refs": list(ISSUE_REFS),
        "report_path": _display_path(output_report, root=artifact_root),
        "source_artifacts": {
            label: _file_identity(path, root=artifact_root, label=label)
            for label, path in sorted(sources.items())
        },
        "selection": dict(selection),
        "splits": dict(splits),
        "skipped": {label: dict(values) for label, values in skipped.items()},
        "outputs": {
            split: {
                "vcf": _file_identity(vcf_path, root=artifact_root, label=f"{split}.vcf"),
                "labels_jsonl": _file_identity(
                    labels_path,
                    root=artifact_root,
                    label=f"{split}.labels_jsonl",
                ),
            }
            for split, (vcf_path, labels_path) in sorted(outputs.items())
        },
        "limitations": [
            "ClinVar coding/noncoding splits are derived from the source VCF MC consequence field.",
            "TraitGym labels are encoded as 1.0 for positive rows and 0.0 for matched controls.",
            "BRCA2 MaveDB function score direction is preserved for Spearman evaluation.",
        ],
    }


def _clinvar_split_summary(rows: Sequence[ClinVarRow]) -> dict[str, object]:
    labels = Counter(row.label for row in rows)
    consequences = Counter(consequence for row in rows for consequence in row.consequences)
    return {
        "rows": len(rows),
        "label_counts": {"benign": labels.get(0, 0), "pathogenic": labels.get(1, 0)},
        "top_consequences": dict(consequences.most_common(10)),
    }


def _continuous_split_summary(rows: Sequence[ContinuousRow]) -> dict[str, object]:
    values = [row.functional_score for row in rows]
    return {
        "rows": len(rows),
        "functional_score_min": min(values),
        "functional_score_max": max(values),
        "functional_score_mean": sum(values) / len(values),
    }


def _file_identity(path: Path, *, root: Path, label: str) -> dict[str, object]:
    if not path.exists():
        raise InputError("benchmark artifact does not exist", details={"label": label, "path": str(path)})
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


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(f"{name} must be a positive integer")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--clinvar-parquet", type=Path, required=True)
    parser.add_argument("--clinvar-source-vcf", type=Path, required=True)
    parser.add_argument("--traitgym-parquet", type=Path, required=True)
    parser.add_argument("--brca2-scores-csv", type=Path, required=True)
    parser.add_argument("--brca2-mapped-variants-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-clinvar-per-split", type=int, default=3_000)
    parser.add_argument("--max-traitgym-rows", type=int)
    parser.add_argument("--max-brca2-rows", type=int)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument(
        "--chromosome",
        action="append",
        help="Restrict benchmark inputs to a normalized chromosome label; repeatable.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
