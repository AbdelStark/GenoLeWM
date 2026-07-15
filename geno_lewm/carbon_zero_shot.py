# SPDX-License-Identifier: Apache-2.0
"""Carbon zero-shot baseline scoring artifacts for release evaluation."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

from geno_lewm._artifact_sources import (
    CARBON_ZERO_SHOT_GENERATED_BY,
    CARBON_ZERO_SHOT_SCHEMA_VERSION,
)
from geno_lewm.action import EditSpec, apply_edit
from geno_lewm.encoder.windowing import DEFAULT_WINDOW_BP, window_sha256, wrap_dna_for_tokenizer
from geno_lewm.errors import InputError, RuntimeSetupError
from geno_lewm.surprise.score import (
    _iter_vcf_variants,
    _load_reference_fasta,
    _reference_window_for_variant,
)

__all__ = [
    "CARBON_ZERO_SHOT_SCORE_FIELD",
    "CarbonLogLikelihoodScorer",
    "CarbonZeroShotRecord",
    "CarbonZeroShotSummary",
    "load_carbon_logp_scorer",
    "write_carbon_zero_shot_scores",
]

CARBON_ZERO_SHOT_SCORE_FIELD: Final = "carbon_zero_shot_score"


@dataclass(frozen=True, slots=True)
class CarbonZeroShotRecord:
    """One Carbon zero-shot score row for ``geno-lewm-eval`` baseline input."""

    chrom: str
    pos: int
    ref: str
    alt: str
    carbon_ref_log_likelihood: float
    carbon_alt_log_likelihood: float
    carbon_alt_minus_ref_log_likelihood: float
    carbon_zero_shot_score: float
    window_start_bp: int
    window_bp: int
    reference_window_sha256: str
    alternate_window_sha256: str

    def to_json_dict(self) -> dict[str, str | int | float]:
        """Return the JSONL row consumed by ``geno-lewm-eval``."""
        return {
            "schema_version": CARBON_ZERO_SHOT_SCHEMA_VERSION,
            "generated_by": CARBON_ZERO_SHOT_GENERATED_BY,
            "chrom": self.chrom,
            "pos": self.pos,
            "ref": self.ref,
            "alt": self.alt,
            "carbon_ref_log_likelihood": self.carbon_ref_log_likelihood,
            "carbon_alt_log_likelihood": self.carbon_alt_log_likelihood,
            "carbon_alt_minus_ref_log_likelihood": self.carbon_alt_minus_ref_log_likelihood,
            CARBON_ZERO_SHOT_SCORE_FIELD: self.carbon_zero_shot_score,
            "window_start_bp": self.window_start_bp,
            "window_bp": self.window_bp,
            "reference_window_sha256": self.reference_window_sha256,
            "alternate_window_sha256": self.alternate_window_sha256,
        }


@dataclass(frozen=True, slots=True)
class CarbonZeroShotSummary:
    """Machine-readable summary for a generated baseline score artifact."""

    generated_by: str
    generated_at: str
    carbon_model: str
    carbon_revision: str
    vcf: str
    fasta: str
    output_scores: str
    score_field: str
    records: int
    window_bp: int
    logp_cache: str | None
    logp_cache_entries: int
    new_logp_evaluations: int
    local_files_only: bool

    def to_json_dict(self) -> dict[str, str | int | bool | None]:
        """Return the JSON-native summary payload."""
        return {
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "carbon_model": self.carbon_model,
            "carbon_revision": self.carbon_revision,
            "vcf": self.vcf,
            "fasta": self.fasta,
            "output_scores": self.output_scores,
            "score_field": self.score_field,
            "records": self.records,
            "window_bp": self.window_bp,
            "logp_cache": self.logp_cache,
            "logp_cache_entries": self.logp_cache_entries,
            "new_logp_evaluations": self.new_logp_evaluations,
            "local_files_only": self.local_files_only,
        }


class CarbonLogLikelihoodScorer:
    """Compute autoregressive Carbon log-likelihood for one DNA window."""

    def __init__(
        self,
        model: object,
        tokenizer: object,
        *,
        torch: object,
        device: str | None = None,
    ) -> None:
        if not callable(tokenizer):
            raise InputError(
                "tokenizer must be callable",
                details={"type": type(tokenizer).__name__},
            )
        if not callable(model):
            raise InputError("model must be callable", details={"type": type(model).__name__})
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        self.device = device
        eval_method = getattr(model, "eval", None)
        if callable(eval_method):
            eval_method()
        if device is not None:
            to_method = getattr(model, "to", None)
            if callable(to_method):
                to_method(device)

    def __call__(self, sequence: str) -> float:
        """Return summed next-token log-likelihood for a Carbon DNA window."""
        tokenizer = cast(Callable[..., Mapping[str, Any]], self.tokenizer)
        tokenized = tokenizer(wrap_dna_for_tokenizer(sequence), return_tensors="pt")
        if not isinstance(tokenized, Mapping):
            raise InputError(
                "Carbon tokenizer must return a mapping",
                details={"type": type(tokenized).__name__},
            )
        batch = _move_mapping(tokenized, self.device)
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise InputError("Carbon tokenizer output must include input_ids")
        shape = getattr(input_ids, "shape", None)
        if shape is None or len(shape) != 2 or shape[1] < 2:
            raise InputError("Carbon tokenizer input_ids must have shape [batch, seq>=2]")
        with _no_grad(self.torch):
            output = cast(Callable[..., object], self.model)(**dict(batch))
        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, tuple) and output:
            logits = output[0]
        if logits is None:
            raise RuntimeSetupError(
                "Carbon model output does not expose logits",
                remediation="load an autoregressive Carbon language-model head",
            )
        return _autoregressive_log_likelihood(
            torch=self.torch,
            logits=logits,
            input_ids=input_ids,
            attention_mask=batch.get("attention_mask"),
        )


def load_carbon_logp_scorer(
    model_dir: str | Path,
    *,
    revision: str = "main",
    dtype: str = "bf16",
    device: str | None = None,
    trust_remote_code: bool = False,
    local_files_only: bool = True,
) -> CarbonLogLikelihoodScorer:
    """Load a local Carbon language-model scorer through Transformers."""
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeSetupError(
            "Carbon zero-shot scoring requires Hugging Face Transformers",
            remediation="install geno-lewm[eval] plus transformers/torch, or run in train extra",
        ) from exc
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeSetupError(
            "Carbon zero-shot scoring requires PyTorch",
            remediation="install geno-lewm[train] in the scoring environment",
        ) from exc

    tokenizer_cls = getattr(transformers, "AutoTokenizer", None)
    model_cls = getattr(transformers, "AutoModelForCausalLM", None)
    if tokenizer_cls is None or model_cls is None:
        raise RuntimeSetupError("transformers must expose AutoTokenizer and AutoModelForCausalLM")
    model_source = str(Path(model_dir).expanduser())
    common_kwargs = {
        "revision": revision,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    tokenizer = tokenizer_cls.from_pretrained(model_source, **common_kwargs)
    model_kwargs = dict(common_kwargs)
    torch_dtype = _torch_dtype(torch, dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = model_cls.from_pretrained(model_source, **model_kwargs)
    return CarbonLogLikelihoodScorer(model, tokenizer, torch=torch, device=device)


def write_carbon_zero_shot_scores(
    *,
    vcf_path: str | Path,
    fasta_path: str | Path,
    output_scores: str | Path,
    scorer: Callable[[str], float],
    carbon_model: str,
    carbon_revision: str,
    window_bp: int = DEFAULT_WINDOW_BP,
    logp_cache_jsonl: str | Path | None = None,
    metadata_output: str | Path | None = None,
    generated_at: str | None = None,
    local_files_only: bool = True,
) -> CarbonZeroShotSummary:
    """Write Carbon zero-shot baseline scores for all VCF alternate alleles."""
    output = Path(output_scores)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = None if logp_cache_jsonl is None else Path(logp_cache_jsonl)
    carbon_model_text = _required_text("carbon_model", carbon_model)
    carbon_revision_text = _required_text("carbon_revision", carbon_revision)
    logp_cache = _load_logp_cache(
        cache_path,
        carbon_model=carbon_model_text,
        carbon_revision=carbon_revision_text,
    )
    initial_cache_keys = frozenset(logp_cache)
    reference_sequences = _load_reference_fasta(fasta_path)
    generated = generated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    records = 0
    with output.open("w", encoding="utf-8") as handle:
        for variant in _iter_vcf_variants(vcf_path):
            record = _score_variant_window(
                variant,
                reference_sequences=reference_sequences,
                window_bp=window_bp,
                scorer=scorer,
                logp_cache=logp_cache,
            )
            handle.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")
            records += 1
    if records == 0:
        raise InputError("VCF produced no Carbon zero-shot baseline rows")
    if cache_path is not None:
        _write_logp_cache(
            cache_path,
            logp_cache,
            carbon_model=carbon_model_text,
            carbon_revision=carbon_revision_text,
        )

    summary = CarbonZeroShotSummary(
        generated_by=CARBON_ZERO_SHOT_GENERATED_BY,
        generated_at=generated,
        carbon_model=carbon_model_text,
        carbon_revision=carbon_revision_text,
        vcf=str(vcf_path),
        fasta=str(fasta_path),
        output_scores=str(output),
        score_field=CARBON_ZERO_SHOT_SCORE_FIELD,
        records=records,
        window_bp=window_bp,
        logp_cache=None if cache_path is None else str(cache_path),
        logp_cache_entries=len(logp_cache),
        new_logp_evaluations=len(set(logp_cache) - initial_cache_keys),
        local_files_only=local_files_only,
    )
    if metadata_output is not None:
        metadata_path = Path(metadata_output)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(summary.to_json_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _score_variant_window(
    variant: EditSpec,
    *,
    reference_sequences: Mapping[str, str],
    window_bp: int,
    scorer: Callable[[str], float],
    logp_cache: dict[str, float],
) -> CarbonZeroShotRecord:
    ref_window = _reference_window_for_variant(
        variant,
        None,
        reference_sequences,
        window_start_bp=0,
        window_bp=window_bp,
    )
    rel_edit = variant.relative_to(
        ref_window.start_bp,
        ref_window.start_bp + len(ref_window.sequence) - 1,
    )
    alternate_window = apply_edit(ref_window.sequence, rel_edit, preserve_length=True)
    ref_logp = _cached_logp(ref_window.sequence, scorer=scorer, cache=logp_cache)
    alt_logp = _cached_logp(alternate_window, scorer=scorer, cache=logp_cache)
    alt_minus_ref = alt_logp - ref_logp
    score = -alt_minus_ref
    return CarbonZeroShotRecord(
        chrom=variant.chrom,
        pos=variant.pos,
        ref=variant.ref,
        alt=variant.alt,
        carbon_ref_log_likelihood=ref_logp,
        carbon_alt_log_likelihood=alt_logp,
        carbon_alt_minus_ref_log_likelihood=alt_minus_ref,
        carbon_zero_shot_score=score,
        window_start_bp=ref_window.start_bp,
        window_bp=len(ref_window.sequence),
        reference_window_sha256=window_sha256(ref_window.sequence).hex(),
        alternate_window_sha256=window_sha256(alternate_window).hex(),
    )


def _cached_logp(
    sequence: str,
    *,
    scorer: Callable[[str], float],
    cache: dict[str, float],
) -> float:
    key = window_sha256(sequence).hex()
    observed = cache.get(key)
    if observed is not None:
        return observed
    value = _require_finite_float("log_likelihood", scorer(sequence))
    cache[key] = value
    return value


def _load_logp_cache(
    path: Path | None,
    *,
    carbon_model: str,
    carbon_revision: str,
) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    cache: dict[str, float] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InputError("logp cache row JSON is invalid", details={"line": line_no}) from exc
        if not isinstance(payload, dict):
            raise InputError("logp cache rows must be JSON objects", details={"line": line_no})
        key = _required_text("sequence_sha256", payload.get("sequence_sha256"))
        if not _looks_like_sha256_hex(key):
            raise InputError("logp cache sequence_sha256 is invalid", details={"line": line_no})
        if not _is_compatible_cache_row(
            payload,
            carbon_model=carbon_model,
            carbon_revision=carbon_revision,
        ):
            continue
        if key in cache:
            raise InputError(
                "logp cache sequence_sha256 values must be unique per Carbon model/revision",
                details={"line": line_no, "sequence_sha256": key},
            )
        cache[key] = _require_finite_float("log_likelihood", payload.get("log_likelihood"))
    return cache


def _write_logp_cache(
    path: Path,
    cache: Mapping[str, float],
    *,
    carbon_model: str,
    carbon_revision: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key in sorted(cache):
            handle.write(
                json.dumps(
                    {
                        "schema_version": CARBON_ZERO_SHOT_SCHEMA_VERSION,
                        "generated_by": CARBON_ZERO_SHOT_GENERATED_BY,
                        "carbon_model": carbon_model,
                        "carbon_revision": carbon_revision,
                        "sequence_sha256": key,
                        "log_likelihood": cache[key],
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _is_compatible_cache_row(
    payload: Mapping[str, Any],
    *,
    carbon_model: str,
    carbon_revision: str,
) -> bool:
    return (
        payload.get("schema_version") == CARBON_ZERO_SHOT_SCHEMA_VERSION
        and payload.get("generated_by") == CARBON_ZERO_SHOT_GENERATED_BY
        and payload.get("carbon_model") == carbon_model
        and payload.get("carbon_revision") == carbon_revision
    )


def _looks_like_sha256_hex(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _autoregressive_log_likelihood(
    *,
    torch: object,
    logits: Any,
    input_ids: Any,
    attention_mask: Any,
) -> float:
    functional = getattr(getattr(torch, "nn", None), "functional", None)
    log_softmax = getattr(functional, "log_softmax", None)
    if not callable(log_softmax):
        raise RuntimeSetupError("torch.nn.functional.log_softmax is unavailable")
    # Accumulate in fp32 regardless of the model's compute dtype. A window
    # log-likelihood is a sum over ~10^3 tokens and lands near -5000, while the
    # quantity of interest -- logP(alt) - logP(ref) for a one-base edit -- is of
    # order 1. In bf16 (8-bit mantissa) the spacing between representable values
    # at that magnitude is 16, so the difference is quantized onto a coarse grid
    # and almost always cancels to exactly zero: scoring 12,993 real SNVs in
    # bf16 produced 90.7% exact zeros across just 18 distinct values, every one a
    # multiple of 16. fp32 resolves ~5e-4 there, which is ample. The cast is on
    # the logits rather than the sum because log_softmax itself must not be
    # evaluated at reduced precision.
    shifted_logits = _to_float32(logits[:, :-1, :])
    shifted_labels = input_ids[:, 1:]
    log_probs = log_softmax(shifted_logits, dim=-1)
    token_logp = log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)
    if attention_mask is not None:
        token_logp = token_logp * attention_mask[:, 1:].to(token_logp.dtype)
    value = token_logp.sum().item()
    return _require_finite_float("sequence log-likelihood", value)


def _to_float32(tensor: Any) -> Any:
    """Return ``tensor`` in fp32, leaving objects without a ``float`` method alone.

    Guarded rather than a bare ``.float()`` because this module resolves every
    torch entry point through ``getattr`` so it can run against stubs.
    """
    caster = getattr(tensor, "float", None)
    if callable(caster):
        return caster()
    return tensor


def _move_mapping(payload: Mapping[str, Any], device: str | None) -> dict[str, Any]:
    if device is None:
        return dict(payload)
    moved: dict[str, Any] = {}
    for key, value in payload.items():
        to_method = getattr(value, "to", None)
        moved[key] = to_method(device) if callable(to_method) else value
    return moved


def _no_grad(torch: object) -> Any:
    no_grad = getattr(torch, "no_grad", None)
    if not callable(no_grad):
        raise RuntimeSetupError("torch.no_grad is unavailable")
    return no_grad()


def _torch_dtype(torch: object, dtype: str) -> object | None:
    if dtype == "bf16":
        return getattr(torch, "bfloat16", None)
    if dtype == "fp16":
        return getattr(torch, "float16", None)
    if dtype == "fp32":
        return getattr(torch, "float32", None)
    raise InputError(
        "unsupported Carbon dtype",
        details={"dtype": dtype, "supported": ["bf16", "fp16", "fp32"]},
    )


def _required_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{name} must be a non-empty string")
    return value.strip()


def _require_finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputError(
            f"{name} must be numeric",
            details={"type": type(value).__name__},
        )
    out = float(value)
    if not math.isfinite(out):
        raise InputError(f"{name} must be finite")
    return out
