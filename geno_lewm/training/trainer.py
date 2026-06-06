# SPDX-License-Identifier: Apache-2.0
"""Torch trainer core for Carbon-backed GenoLeWM runs.

The public CLI still gates real training behind explicit preflight and
fixture modes. This module is the optional-runtime trainer boundary:
it is importable without PyTorch, but constructing batches, optimizers,
or train steps requires a ``geno-lewm[train]`` environment.
"""

from __future__ import annotations

import importlib
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from geno_lewm.action import RelEdit
from geno_lewm.config import GenoLeWMConfig
from geno_lewm.data import TrainingTuple
from geno_lewm.encoder.cache import INDEX_DB_NAME, WindowCacheKey, default_cache_dir, read_embedding
from geno_lewm.encoder.pooling import POOL_GLOBAL_MEAN
from geno_lewm.encoder.windowing import window_sha256
from geno_lewm.errors import InputError, RuntimeSetupError
from geno_lewm.predictor.losses import predictor_loss
from geno_lewm.training.collapse import CollapseMonitor

__all__ = [
    "TorchDeterminismReport",
    "TorchTrainer",
    "TorchTrainerBatch",
    "TorchTrainerStepResult",
    "TrainerSeeds",
    "build_adamw_optimizer",
    "configure_torch_reproducibility",
    "encode_training_batch",
    "make_action_mask",
    "set_optimizer_lr",
    "wsd_lr_multiplier",
]

if TYPE_CHECKING:
    torch: Any = None
    Tensor = Any
else:
    try:  # pragma: no cover - optional runtime exercised only when torch is installed.
        import torch
        from torch import Tensor
    except ImportError:  # pragma: no cover - covered by missing-runtime tests.
        torch = None
        Tensor = Any

ScheduleName = Literal["wsd", "constant", "cosine"]
_SOURCE_STATE_MEMORY_CACHE_ATTR = "_geno_lewm_source_state_cache"


@dataclass(frozen=True, slots=True)
class TrainerSeeds:
    """Distinct RNG seeds consumed by the real training stack."""

    data: int
    predictor: int
    lora: int

    @classmethod
    def from_base_seed(cls, seed: int) -> TrainerSeeds:
        _require_nonnegative_int("seed", seed)
        return cls(data=seed, predictor=seed + 1, lora=seed + 2)

    def to_dict(self) -> dict[str, int]:
        return {"data": self.data, "predictor": self.predictor, "lora": self.lora}


@dataclass(frozen=True, slots=True)
class TorchDeterminismReport:
    """Runtime settings applied before a torch training run."""

    seed: int
    deterministic: bool
    cublas_workspace_config: str | None
    torch_deterministic_algorithms: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "deterministic": self.deterministic,
            "cublas_workspace_config": self.cublas_workspace_config,
            "torch_deterministic_algorithms": self.torch_deterministic_algorithms,
        }


@dataclass(frozen=True, slots=True)
class TorchTrainerBatch:
    """One encoded minibatch consumed by :class:`TorchTrainer`."""

    state: Tensor
    target: Tensor
    rel_edits: tuple[tuple[RelEdit, ...], ...]
    action_mask: Tensor
    window_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TorchTrainerStepResult:
    """Scalar outputs from one optimizer step."""

    step: int
    lr_multiplier: float
    loss: float
    pred_loss: float
    kl_reg: float
    action_count: int
    pred_var_per_dim: float

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "lr_multiplier": self.lr_multiplier,
            "loss": self.loss,
            "pred_loss": self.pred_loss,
            "kl_reg": self.kl_reg,
            "action_count": self.action_count,
            "pred_var_per_dim": self.pred_var_per_dim,
        }


class TorchTrainer:
    """Minimal optimizer loop for Carbon-state predictor training."""

    def __init__(
        self,
        *,
        predictor: object,
        action_encoder: object,
        optimizer: object,
        config: GenoLeWMConfig,
        total_steps: int,
    ) -> None:
        _require_torch("TorchTrainer")
        _require_positive_int("total_steps", total_steps)
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.optimizer = optimizer
        self.config = config
        self.total_steps = total_steps
        self.collapse_monitor = CollapseMonitor(
            log_every_steps=config.training.collapse_log_every_steps,
        )
        self.last_collapse_alerts: tuple[dict[str, object], ...] = ()

    def train_step(self, batch: TorchTrainerBatch, *, step: int) -> TorchTrainerStepResult:
        """Run one optimizer step over an encoded Carbon-state batch."""
        _require_positive_int("step", step)
        if step > self.total_steps:
            raise InputError(
                "step cannot exceed total_steps",
                details={"step": step, "total_steps": self.total_steps},
            )
        lr_multiplier = set_optimizer_lr(
            self.optimizer,
            step=step,
            total_steps=self.total_steps,
            warmup_steps=self.config.optimizer.warmup_steps,
            schedule=self.config.optimizer.schedule,
        )
        _zero_grad(self.optimizer)
        action_embeddings = _call_action_encoder(self.action_encoder, batch.rel_edits)
        prediction = _call_predictor(
            self.predictor,
            state=batch.state,
            actions=action_embeddings,
            action_mask=batch.action_mask,
        )
        loss_result = predictor_loss(
            prediction,
            batch.target,
            phase=self.config.phase,
            mask=batch.action_mask,
        )
        action_count = int(batch.action_mask.sum().item())
        self.last_collapse_alerts = ()
        if action_count > 0:
            collapse_check = self.collapse_monitor.observe(
                _masked_training_rows(prediction, batch.action_mask),
                _masked_training_rows(batch.target, batch.action_mask),
                kl_reg=_scalar(loss_result.kl_reg),
                step=step,
            )
            if collapse_check is not None:
                self.last_collapse_alerts = tuple(
                    {
                        "criterion": alert.criterion,
                        "value": alert.value,
                        "threshold": alert.threshold,
                    }
                    for alert in collapse_check.alerts
                )
        loss_result.loss.backward()
        if self.config.optimizer.grad_clip > 0:
            parameters = _trainable_parameters((self.predictor, self.action_encoder))
            torch.nn.utils.clip_grad_norm_(parameters, self.config.optimizer.grad_clip)
        _optimizer_step(self.optimizer)
        return TorchTrainerStepResult(
            step=step,
            lr_multiplier=lr_multiplier,
            loss=_scalar(loss_result.loss),
            pred_loss=_scalar(loss_result.pred_loss),
            kl_reg=_scalar(loss_result.kl_reg),
            action_count=action_count,
            pred_var_per_dim=_pred_var_per_dim(prediction),
        )


def configure_torch_reproducibility(
    *,
    seed: int,
    deterministic: bool,
) -> TorchDeterminismReport:
    """Seed Python/NumPy/PyTorch and optionally enable deterministic torch kernels."""
    _require_torch("configure_torch_reproducibility")
    _require_nonnegative_int("seed", seed)
    random.seed(seed)
    _seed_numpy(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on host accelerator.
        torch.cuda.manual_seed_all(seed)
    cublas_config: str | None = None
    if deterministic:
        cublas_config = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is not None:
            cudnn.benchmark = False
    else:
        use_deterministic = getattr(torch, "use_deterministic_algorithms", None)
        if callable(use_deterministic):
            use_deterministic(False)
    return TorchDeterminismReport(
        seed=seed,
        deterministic=deterministic,
        cublas_workspace_config=cublas_config,
        torch_deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
    )


def encode_training_batch(
    *,
    encoder: object,
    tuples: Sequence[TrainingTuple],
    source_windows: Mapping[str, str],
    device: str | object | None = None,
    dtype: object | None = None,
) -> TorchTrainerBatch:
    """Encode source/target windows for a real predictor-training minibatch."""
    _require_torch("encode_training_batch")
    if not tuples:
        raise InputError("training batch must contain at least one tuple")
    source_sequences: list[str] = []
    target_sequences: list[str] = []
    rel_edits: list[tuple[RelEdit, ...]] = []
    window_ids: list[str] = []
    for item in tuples:
        if not isinstance(item, TrainingTuple):
            raise InputError(
                "tuples must contain TrainingTuple values",
                details={"type": type(item).__name__},
            )
        try:
            source_sequences.append(source_windows[item.window_id])
        except KeyError as exc:
            raise InputError(
                "source window sequence missing for training tuple",
                details={"window_id": item.window_id},
            ) from exc
        target_sequences.append(item.target_window)
        rel_edits.append(tuple(item.rel_edits))
        window_ids.append(item.window_id)
    target_loci = [edits[0].rel_pos if edits else None for edits in rel_edits]
    encoder_runtime = _encoder_runtime(encoder)
    source_states = _source_states(encoder_runtime, source_sequences)
    target_states = encoder_runtime.encode_batch(target_sequences, target_loci)
    state = torch.tensor(source_states, dtype=dtype or torch.float32, device=device)
    target_single = torch.tensor(target_states, dtype=state.dtype, device=state.device)
    mask = make_action_mask(rel_edits, device=state.device)
    target = target_single.unsqueeze(1).expand(-1, mask.shape[1], -1).clone()
    target = target.masked_fill(~mask.unsqueeze(-1), 0.0)
    return TorchTrainerBatch(
        state=state,
        target=target,
        rel_edits=tuple(rel_edits),
        action_mask=mask,
        window_ids=tuple(window_ids),
    )


def make_action_mask(
    rel_edits: Sequence[Sequence[object]],
    *,
    device: object | None = None,
) -> Tensor:
    """Return a boolean action mask for a ragged batch of relative edits."""
    _require_torch("make_action_mask")
    if not rel_edits:
        raise InputError("rel_edits must contain at least one batch item")
    lengths = []
    for edits in rel_edits:
        if isinstance(edits, str | bytes) or not isinstance(edits, Sequence):
            raise InputError(
                "rel_edits entries must be sequences",
                details={"type": type(edits).__name__},
            )
        if not edits:
            raise InputError("each training batch item must include at least one edit")
        lengths.append(len(edits))
    max_len = max(lengths)
    mask = torch.zeros((len(lengths), max_len), dtype=torch.bool, device=device)
    for row, length in enumerate(lengths):
        mask[row, :length] = True
    return mask


def build_adamw_optimizer(
    *,
    predictor: object,
    action_encoder: object,
    config: GenoLeWMConfig,
) -> object:
    """Build AdamW groups for predictor/action-encoder trainable parameters."""
    _require_torch("build_adamw_optimizer")
    if config.optimizer.name != "adamw":
        raise InputError(
            "real trainer currently supports AdamW only",
            details={"optimizer": config.optimizer.name},
        )
    groups = _adamw_param_groups(
        (("predictor", predictor), ("action_encoder", action_encoder)),
        lr=float(config.optimizer.lr),
        weight_decay=float(config.optimizer.weight_decay),
    )
    if not groups:
        raise InputError("no trainable predictor/action-encoder parameters found")
    return torch.optim.AdamW(
        groups,
        betas=(float(config.optimizer.beta1), float(config.optimizer.beta2)),
        eps=1.0e-8,
    )


def wsd_lr_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    schedule: ScheduleName = "wsd",
) -> float:
    """Return the RFC-0005 WSD learning-rate multiplier for a 1-indexed step."""
    _require_positive_int("step", step)
    _require_positive_int("total_steps", total_steps)
    _require_nonnegative_int("warmup_steps", warmup_steps)
    if step > total_steps:
        raise InputError(
            "step cannot exceed total_steps",
            details={"step": step, "total_steps": total_steps},
        )
    if schedule == "constant":
        return 1.0
    if schedule == "cosine":
        if total_steps == 1:
            return 1.0
        progress = (step - 1) / (total_steps - 1)
        return 0.5 * (1.0 + float(torchless_cos(progress)))
    if schedule != "wsd":
        raise InputError("unsupported learning-rate schedule", details={"schedule": schedule})
    if warmup_steps > 0 and step <= warmup_steps:
        return step / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    post_warmup = total_steps - warmup_steps
    decay_start = warmup_steps + max(1, int(post_warmup * 0.80))
    final_taper_start = warmup_steps + max(1, int(post_warmup * 0.98))
    if step <= decay_start:
        return 1.0
    if step <= final_taper_start:
        span = max(1, final_taper_start - decay_start)
        progress = (step - decay_start) / span
        return 1.0 - 0.9 * progress
    span = max(1, total_steps - final_taper_start)
    progress = (step - final_taper_start) / span
    return max(0.01, 0.1 - 0.09 * progress)


def set_optimizer_lr(
    optimizer: object,
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    schedule: ScheduleName = "wsd",
) -> float:
    """Set optimizer group LRs from each group's ``initial_lr`` and return multiplier."""
    multiplier = wsd_lr_multiplier(
        step,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        schedule=schedule,
    )
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        raise InputError("optimizer must expose non-empty param_groups")
    for group in groups:
        if not isinstance(group, dict):
            raise InputError("optimizer param_groups must contain dictionaries")
        base_lr = group.setdefault("initial_lr", group.get("lr"))
        if isinstance(base_lr, bool) or not isinstance(base_lr, int | float):
            raise InputError("optimizer param group lr must be numeric")
        group["lr"] = float(base_lr) * multiplier
    return multiplier


def _adamw_param_groups(
    modules: Sequence[tuple[str, object]],
    *,
    lr: float,
    weight_decay: float,
) -> list[dict[str, object]]:
    decay: list[object] = []
    no_decay: list[object] = []
    for prefix, module in modules:
        named_parameters = getattr(module, "named_parameters", None)
        if not callable(named_parameters):
            raise InputError(
                "trainable modules must expose named_parameters()",
                details={"module": prefix, "type": type(module).__name__},
            )
        for name, param in named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            full_name = f"{prefix}.{name}"
            if _is_no_decay_parameter(full_name, param):
                no_decay.append(param)
            else:
                decay.append(param)
    groups: list[dict[str, object]] = []
    if decay:
        groups.append(
            {
                "params": decay,
                "lr": lr,
                "initial_lr": lr,
                "weight_decay": weight_decay,
                "name": "decay",
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "lr": lr,
                "initial_lr": lr,
                "weight_decay": 0.0,
                "name": "no_decay",
            }
        )
    return groups


def _is_no_decay_parameter(name: str, param: object) -> bool:
    lowered = name.lower()
    if lowered.endswith(".bias") or "norm" in lowered or "embedding" in lowered:
        return True
    ndim = getattr(param, "ndim", None)
    return isinstance(ndim, int) and ndim < 2


def _trainable_parameters(modules: Sequence[object]) -> list[object]:
    params: list[object] = []
    for module in modules:
        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            continue
        params.extend(param for param in parameters() if getattr(param, "requires_grad", False))
    return params


def _call_action_encoder(action_encoder: object, rel_edits: object) -> Tensor:
    if not callable(action_encoder):
        raise InputError("action_encoder must be callable")
    return action_encoder(rel_edits)


def _call_predictor(
    predictor: object,
    *,
    state: Tensor,
    actions: Tensor,
    action_mask: Tensor,
) -> Tensor:
    if not callable(predictor):
        raise InputError("predictor must be callable")
    return predictor(state, actions, action_mask)


def _encoder_runtime(encoder: object) -> Any:
    encode_batch = getattr(encoder, "encode_batch", None)
    if not callable(encode_batch):
        raise InputError("encoder must expose encode_batch()")
    return encoder


def _source_states(
    encoder: object,
    source_sequences: Sequence[str],
) -> tuple[tuple[float, ...], ...]:
    runtime = _encoder_runtime(encoder)
    memory_cache = _source_state_memory_cache(runtime)
    cache_dir = default_cache_dir()
    disk_cache_exists = _source_cache_index_exists(cache_dir)

    states: list[tuple[float, ...] | None] = []
    missing_indices: list[int] = []
    for index, sequence in enumerate(source_sequences):
        cached = memory_cache.get(sequence)
        if cached is not None:
            states.append(cached)
            continue
        if disk_cache_exists:
            key = _source_cache_key(encoder, sequence)
            cached = read_embedding(cache_dir, key)
            if cached is not None:
                state = _require_state_vector(cached, index=index)
                memory_cache[sequence] = state
                states.append(state)
                continue
        states.append(None)
        missing_indices.append(index)

    if missing_indices:
        missing_by_sequence: dict[str, list[int]] = {}
        for index in missing_indices:
            missing_by_sequence.setdefault(source_sequences[index], []).append(index)
        missing_sequences = list(missing_by_sequence)
        encoded = tuple(runtime.encode_batch(missing_sequences, [None] * len(missing_sequences)))
        for sequence, vector in zip(missing_sequences, encoded, strict=True):
            state = _require_state_vector(vector, index=missing_by_sequence[sequence][0])
            memory_cache[sequence] = state
            for index in missing_by_sequence[sequence]:
                states[index] = state

    return tuple(_require_state_vector(state, index=index) for index, state in enumerate(states))


def _source_state_memory_cache(encoder: object) -> dict[str, tuple[float, ...]]:
    raw_cache = getattr(encoder, _SOURCE_STATE_MEMORY_CACHE_ATTR, None)
    if isinstance(raw_cache, dict):
        return cast(dict[str, tuple[float, ...]], raw_cache)
    cache: dict[str, tuple[float, ...]] = {}
    try:
        setattr(encoder, _SOURCE_STATE_MEMORY_CACHE_ATTR, cache)
    except Exception:
        return {}
    return cache


def _source_cache_index_exists(cache_dir: Path) -> bool:
    return (cache_dir / "embeddings" / INDEX_DB_NAME).is_file()


def _source_cache_key(encoder: object, sequence: str) -> WindowCacheKey:
    return WindowCacheKey(
        window_hash=window_sha256(sequence),
        encoder_hash=_encoder_hash(encoder),
        state_layer=_encoder_int_attr(encoder, "state_layer"),
        pool_type=POOL_GLOBAL_MEAN,
        pool_radius=0,
        dtype=_encoder_text_attr(encoder, "dtype"),
    )


def _encoder_hash(encoder: object) -> bytes:
    try:
        value = encoder.encoder_hash  # type: ignore[attr-defined]
    except RuntimeSetupError:
        raise
    except Exception as exc:
        raise RuntimeSetupError(
            "source-state cache lookup requires encoder.encoder_hash",
            remediation="construct CarbonStateEncoder with the local Carbon weights hash",
        ) from exc
    if isinstance(value, bytes):
        return value
    raise RuntimeSetupError(
        "source-state cache lookup requires encoder.encoder_hash bytes",
        details={"type": type(value).__name__},
        remediation="construct CarbonStateEncoder with the local Carbon weights hash",
    )


def _encoder_int_attr(encoder: object, name: str) -> int:
    value = getattr(encoder, name, None)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeSetupError(
            f"source-state cache lookup requires encoder.{name}",
            details={"type": type(value).__name__},
        )
    return value


def _encoder_text_attr(encoder: object, name: str) -> str:
    value = getattr(encoder, name, None)
    if not isinstance(value, str) or not value:
        raise RuntimeSetupError(
            f"source-state cache lookup requires encoder.{name}",
            details={"type": type(value).__name__},
        )
    return value


def _require_state_vector(value: tuple[float, ...] | None, *, index: int) -> tuple[float, ...]:
    if value is None:
        raise InputError("source state was not produced", details={"index": index})
    return value


def _zero_grad(optimizer: object) -> None:
    zero_grad = getattr(optimizer, "zero_grad", None)
    if not callable(zero_grad):
        raise InputError("optimizer must expose zero_grad()")
    zero_grad(set_to_none=True)


def _optimizer_step(optimizer: object) -> None:
    step = getattr(optimizer, "step", None)
    if not callable(step):
        raise InputError("optimizer must expose step()")
    step()


def _scalar(value: object) -> float:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    item = getattr(value, "item", None)
    if callable(item):
        return float(item())
    return float(value)  # type: ignore[arg-type]


def _pred_var_per_dim(prediction: Tensor) -> float:
    """Mean population variance across prediction latent dims (collapse signal).

    Low values flag representation collapse (RFC-0005 §3.6). Matches the
    ``_mean_variance_per_dim`` semantics in :mod:`geno_lewm.training.collapse`.
    """
    detached = prediction.detach()
    flat = detached.reshape(-1, detached.shape[-1])
    if flat.shape[0] < 1:
        return 0.0
    return float(flat.var(dim=0, unbiased=False).mean().item())


def _masked_training_rows(value: Tensor, mask: Tensor) -> Tensor:
    """Flatten valid action-token rows before collapse diagnostics."""
    if len(value.shape) != 3:
        raise InputError(
            "training collapse monitor expects [batch, actions, dim] tensors",
            details={"shape": tuple(value.shape)},
        )
    if len(mask.shape) != 2:
        raise InputError(
            "training collapse monitor expects [batch, actions] masks",
            details={"shape": tuple(mask.shape)},
        )
    return value[mask]


def _seed_numpy(seed: int) -> None:
    try:
        np = importlib.import_module("numpy")
    except ImportError:
        return
    np.random.seed(seed)


def _require_torch(context: str) -> None:
    if torch is None:
        raise RuntimeSetupError(
            f"{context} requires PyTorch",
            remediation="install geno-lewm[train] or install torch",
        )


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(
            f"{name} must be a positive integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InputError(
            f"{name} must be a non-negative integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def torchless_cos(progress: float) -> float:
    import math

    return math.cos(math.pi * progress)
