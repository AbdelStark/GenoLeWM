# SPDX-License-Identifier: Apache-2.0
"""Typed configuration schema (RFC-0017 §3.2).

Frozen dataclasses for every subsystem listed in RFC-0017 §3.3. Each
field has a default that matches the RFC's documented default; the
docstring explains the field's purpose so the ``--explain`` flag
(PR #29) can render it.

The top-level :class:`GenoLeWMConfig` aggregates the subsystem schemas
and is the single object that every CLI command resolves to (RFC-0018
§3.2). The loader at :mod:`geno_lewm.config.loader` is responsible for
constructing it from YAML; the schema itself is pure data.

Phase 1 stays in lock-step with the RFC field names so the trainer
(#44) can drop in without reshaping the config. Unimplemented
subsystems (predictor, action encoder, planner) still carry default
values here so the schema validates before those modules land.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from typing import Literal

__all__ = [
    "TOP_LEVEL_KEYS",
    "ActionEncoderConfig",
    "DataConfig",
    "EncoderConfig",
    "EvalConfig",
    "GenoLeWMConfig",
    "ObservabilityConfig",
    "OptimizerConfig",
    "PredictorConfig",
    "RuntimeConfig",
    "iter_top_level_field_names",
]

# ---------------------------------------------------------------------------
# Subsystem schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """State encoder configuration (RFC-0002 §3.1, §3.8).

    The Phase 1 default is Carbon-500M with bf16 weights, pinned to a
    specific revision so the encoder hash committed to the manifest is
    reproducible.
    """

    model_id: str = "HuggingFaceBio/Carbon-500M"
    revision: str = "main@deadbeef"
    dtype: str = "bf16"
    state_layer: int = 20
    pool_type: str = "centered_mean"
    pool_radius: int = 8
    normalize: bool = True


@dataclass(frozen=True, slots=True)
class PredictorConfig:
    """Action-conditioned predictor (RFC-0004 §3.1)."""

    architecture: str = "cross_attention"
    n_layers: int = 6
    n_heads: int = 8
    d_state: int = 512
    d_action: int = 64
    dtype: str = "bf16"


@dataclass(frozen=True, slots=True)
class ActionEncoderConfig:
    """Action encoder configuration (RFC-0003 §3.4)."""

    d_action: int = 64
    max_len: int = 16
    sub_encoders: tuple[str, ...] = ("snv", "ins", "del", "mnv")


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Optimizer + learning-rate schedule (RFC-0005)."""

    name: Literal["adamw", "sgd-momentum"] = "adamw"
    lr: float = 3.0e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    schedule: Literal["wsd", "cosine", "constant"] = "wsd"


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Data pipeline configuration (RFC-0006)."""

    corpus_id: str = "HuggingFaceBio/carbon-pretraining-corpus"
    corpus_revision: str = "main@cafef00d"
    batch_size: int = 64
    num_workers: int = 4
    shuffle_buffer: int = 4096


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Evaluation harness (RFC-0007)."""

    benchmarks: tuple[str, ...] = (
        "clinvar_coding",
        "clinvar_noncoding",
        "rollout",
    )
    smoke_variants: int = 1000


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    """Observability sinks (RFC-0013)."""

    log_level: Literal["debug", "info", "warn", "error"] = "info"
    redaction_strict: bool = True
    wandb_project: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Runtime / deployment target (RFC-0010)."""

    backend: Literal["onnx", "coreml", "gguf", "torch"] = "torch"
    device: Literal["cpu", "cuda", "mps"] = "cpu"


# ---------------------------------------------------------------------------
# Top-level schema (RFC-0017 §3.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenoLeWMConfig:
    """Top-level configuration object.

    Every CLI command resolves to one of these. ``run_id`` is the
    primary key for run artifacts (``${run_id}/config.resolved.yaml``,
    ``${run_id}/checkpoints/*``); the trainer auto-generates one if the
    caller does not provide it.

    The :data:`schema_version` field tracks the on-disk shape of
    ``config.resolved.yaml`` — bumps follow RFC-0014's MAJOR/MINOR
    rules on the config-resolution layer.
    """

    run_id: str = "default"
    seed: int = 0
    phase: Literal["phase1", "phase2"] = "phase1"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    action: ActionEncoderConfig = field(default_factory=ActionEncoderConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    deterministic: bool = False
    schema_version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def iter_top_level_field_names() -> Iterator[str]:
    """Yield the canonical top-level keys accepted by the loader.

    The loader rejects any payload key not in this set via
    :class:`geno_lewm.errors.UnknownTopLevelKeyError`. Tests use this
    helper to assert the schema and the AC list stay in sync.
    """
    for f in fields(GenoLeWMConfig):
        yield f.name


#: Set of the same names; cached because the loader checks it on every
#: load and ``fields()`` walks the dataclass each call.
TOP_LEVEL_KEYS: frozenset[str] = frozenset(iter_top_level_field_names())
