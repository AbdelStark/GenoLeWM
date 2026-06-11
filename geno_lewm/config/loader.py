# SPDX-License-Identifier: Apache-2.0
"""Config loader, validator, and resolved-config writer (configuration contract).

This module is the bridge between the YAML files under
:data:`DEFAULTS_DIR` and the dataclass schema in
:mod:`geno_lewm.config.schema`. Three responsibilities:

1. :func:`load_config` — read a YAML payload, type-check it, reject
   unknown top-level keys (configuration contract), and return a frozen
   :class:`GenoLeWMConfig`.
2. :func:`write_resolved_config` — emit a config object as canonical
   YAML (sorted keys, no anchors) so ``${run_id}/config.resolved.yaml``
   is reproducible (configuration contract).
3. :func:`describe_field` — schema introspection used by the
   ``--explain`` CLI flag (PR #29).

We deliberately do **not** load Hydra at runtime. Hydra-style
composition (``defaults:`` blocks, multi-run sweeps) lands with
PR #29 + a future loader change; for Phase 1 a single YAML file plus
optional CLI ``--set key=value`` overrides covers the scoring /
training / eval / planning paths.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

from geno_lewm.config.schema import (
    TOP_LEVEL_KEYS,
    ActionEncoderConfig,
    DataConfig,
    EncoderConfig,
    EvalConfig,
    GenoLeWMConfig,
    ObservabilityConfig,
    OptimizerConfig,
    PredictorConfig,
    RuntimeConfig,
    TrainingConfig,
)
from geno_lewm.errors import (
    ConfigError,
    InputError,
    MissingConfigError,
    UnknownTopLevelKeyError,
)

__all__ = [
    "DEFAULTS_DIR",
    "config_to_dict",
    "describe_field",
    "iter_subsystem_names",
    "load_config",
    "load_default",
    "write_resolved_config",
]

#: Filesystem directory holding the canonical YAML templates per CLI
#: command. Each file declares the same schema; the differences are in
#: which defaults are tuned for that command (e.g., ``score.yaml``
#: pins ``optimizer.lr`` at 0 because scoring does not optimise).
DEFAULTS_DIR: Path = Path(__file__).resolve().parent / "defaults"


#: Map of subsystem dataclass → its key in the top-level payload. The
#: loader uses this so the per-subsystem construction is dispatched
#: from one place; adding a new subsystem means extending the schema
#: and adding one entry here.
_SUBSYSTEM_MAP: tuple[tuple[str, type], ...] = (
    ("encoder", EncoderConfig),
    ("predictor", PredictorConfig),
    ("action", ActionEncoderConfig),
    ("training", TrainingConfig),
    ("optimizer", OptimizerConfig),
    ("data", DataConfig),
    ("eval", EvalConfig),
    ("observability", ObservabilityConfig),
    ("runtime", RuntimeConfig),
)


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_config(source: Path | str | Mapping[str, Any]) -> GenoLeWMConfig:
    """Load + validate a config payload; return a frozen :class:`GenoLeWMConfig`.

    ``source`` may be:

    * A :class:`Path` (or ``str``) — read the file as YAML.
    * A :class:`Mapping` — treat it as the already-parsed payload (used
      by ``--set`` override merging in PR #29 and by the unit tests).

    Validation:

    * Unknown top-level keys → :class:`UnknownTopLevelKeyError`.
    * Missing required subsystem keys → :class:`MissingConfigError`.
    * Wrong value type on any field → :class:`ConfigError`.
    """
    if isinstance(source, Mapping):
        payload: Any = source
    elif isinstance(source, str | Path):
        payload = _resolve_payload(source)
    else:
        raise InputError(
            "config payload must be a mapping at the top level",
            details={"got": type(source).__name__},
        )
    if not isinstance(payload, Mapping):
        raise InputError(
            "config payload must be a mapping at the top level",
            details={"got": type(payload).__name__},
        )
    return _build_top_level(dict(payload))


def load_default(name: str) -> GenoLeWMConfig:
    """Shorthand for ``load_config(DEFAULTS_DIR / f"{name}.yaml")``.

    Accepts the documented command names (``train`` / ``score`` /
    ``eval`` / ``plan``). Raises :class:`MissingConfigError` if the
    YAML template is missing.
    """
    target = DEFAULTS_DIR / f"{name}.yaml"
    if not target.is_file():
        raise MissingConfigError(
            f"no default config for command {name!r}",
            details={"path": str(target), "known": sorted(_known_defaults())},
        )
    return load_config(target)


def _known_defaults() -> list[str]:
    if not DEFAULTS_DIR.is_dir():
        return []
    return [p.stem for p in DEFAULTS_DIR.glob("*.yaml")]


def _resolve_payload(source: Path | str | Mapping[str, Any]) -> Any:
    if isinstance(source, Mapping):
        return source
    path = Path(source)
    if not path.is_file():
        raise MissingConfigError(
            f"config file not found: {path}",
            details={"path": str(path)},
        )
    text = path.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(
            "config file is not valid YAML",
            details={"path": str(path), "error": str(exc)},
        ) from exc


# ---------------------------------------------------------------------------
# Top-level + subsystem construction
# ---------------------------------------------------------------------------


def _build_top_level(payload: dict[str, Any]) -> GenoLeWMConfig:
    unknown = set(payload) - TOP_LEVEL_KEYS
    if unknown:
        raise UnknownTopLevelKeyError(
            "config contains top-level key(s) not in the schema",
            details={"unknown": sorted(unknown), "known": sorted(TOP_LEVEL_KEYS)},
        )

    kwargs: dict[str, Any] = {}
    for name, cls in _SUBSYSTEM_MAP:
        sub_payload = payload.get(name)
        kwargs[name] = _build_dataclass(cls, sub_payload, path=name)

    # Top-level scalars share the same coercion as sub-fields. Pull
    # the type hints from the dataclass so the loader does not
    # duplicate the schema.
    top_hints = get_type_hints(GenoLeWMConfig)
    for scalar in ("run_id", "seed", "phase", "deterministic", "schema_version"):
        if scalar in payload:
            kwargs[scalar] = _coerce(payload[scalar], top_hints[scalar], path=scalar)

    try:
        return GenoLeWMConfig(**kwargs)
    except TypeError as exc:
        raise ConfigError(
            "could not construct GenoLeWMConfig from payload",
            details={"error": str(exc)},
        ) from exc


def _build_dataclass(cls: type, payload: Any, *, path: str) -> Any:
    """Construct a frozen dataclass instance from ``payload``.

    ``payload`` may be ``None`` (use defaults), a mapping (override
    specific fields), or anything else (rejected). Unknown sub-keys
    are rejected because the schema is intentionally closed.
    """
    if payload is None:
        return cls()
    if not isinstance(payload, Mapping):
        raise ConfigError(
            f"{path}: subsystem block must be a mapping",
            details={"path": path, "got": type(payload).__name__},
        )

    declared = {f.name for f in fields(cls)}
    unknown = set(payload) - declared
    if unknown:
        raise ConfigError(
            f"{path}: unknown sub-field(s)",
            details={"path": path, "unknown": sorted(unknown), "known": sorted(declared)},
        )

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in payload:
            continue
        value = payload[f.name]
        kwargs[f.name] = _coerce(value, hints[f.name], path=f"{path}.{f.name}")
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(
            f"{path}: could not construct {cls.__name__}",
            details={"path": path, "error": str(exc)},
        ) from exc


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


def _coerce(value: Any, annotation: Any, *, path: str) -> Any:
    """Coerce a YAML-loaded value to ``annotation`` or raise.

    The schema uses only stdlib / dataclass types (``str``, ``int``,
    ``float``, ``bool``, ``tuple[str, ...]``, ``Literal[…]``, optional
    ``str | None``). YAML returns native Python equivalents already;
    we only need a few targeted coercions (list → tuple).
    """
    origin = get_origin(annotation)

    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(
            f"{path}: expected bool, got {type(value).__name__}",
            details={"path": path, "value": value},
        )
    if annotation is int:
        if isinstance(value, bool):
            raise ConfigError(
                f"{path}: expected int, got bool",
                details={"path": path, "value": value},
            )
        if isinstance(value, int):
            return value
        raise ConfigError(
            f"{path}: expected int, got {type(value).__name__}",
            details={"path": path, "value": value},
        )
    if annotation is float:
        if isinstance(value, bool):
            raise ConfigError(
                f"{path}: expected float, got bool",
                details={"path": path, "value": value},
            )
        if isinstance(value, int | float):
            return float(value)
        raise ConfigError(
            f"{path}: expected float, got {type(value).__name__}",
            details={"path": path, "value": value},
        )
    if annotation is str:
        if isinstance(value, str):
            return value
        raise ConfigError(
            f"{path}: expected str, got {type(value).__name__}",
            details={"path": path, "value": value},
        )

    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ConfigError(
                f"{path}: expected list/tuple, got {type(value).__name__}",
                details={"path": path, "value": value},
            )
        (item_type, _ellipsis) = get_args(annotation)
        return tuple(_coerce(v, item_type, path=f"{path}[{i}]") for i, v in enumerate(value))

    if origin is Literal:
        allowed = get_args(annotation)
        if value not in allowed:
            raise ConfigError(
                f"{path}: value not in allowed set",
                details={"path": path, "value": value, "allowed": list(allowed)},
            )
        return value

    if origin in (Union, UnionType):
        # Typed as ``X | None`` etc. Try each arm and accept the first match.
        for arm in get_args(annotation):
            if arm is type(None) and value is None:
                return None
            try:
                return _coerce(value, arm, path=path)
            except ConfigError:
                continue
        raise ConfigError(
            f"{path}: value does not match any arm of the union",
            details={"path": path, "value": value, "allowed": list(get_args(annotation))},
        )

    # Fallback: trust the YAML value; the dataclass constructor will
    # raise TypeError if it is unworkable, which we surface upstream.
    return value


# ---------------------------------------------------------------------------
# Resolved-config persistence
# ---------------------------------------------------------------------------


def config_to_dict(cfg: GenoLeWMConfig) -> dict[str, Any]:
    """Return a plain dict view of ``cfg`` for serialization."""
    result = _asdict_with_tuples(cfg)
    assert isinstance(result, dict)
    return result


def _asdict_with_tuples(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _asdict_with_tuples(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return [_asdict_with_tuples(v) for v in obj]
    if isinstance(obj, list):
        return [_asdict_with_tuples(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict_with_tuples(v) for k, v in obj.items()}
    return obj


def write_resolved_config(cfg: GenoLeWMConfig, path: Path | str) -> Path:
    """Write ``cfg`` as canonical YAML to ``path``; return the absolute path.

    Canonical = ``sort_keys=True``, ``default_flow_style=False``, no
    anchors. The result hashes byte-stably so the manifest's
    ``training.config_file`` hash matches between machines.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(
        config_to_dict(cfg),
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
    )
    target.write_text(text, encoding="utf-8")
    return target.resolve()


# ---------------------------------------------------------------------------
# --explain (PR #29 surface)
# ---------------------------------------------------------------------------


def describe_field(dotted_key: str) -> dict[str, Any]:
    """Return ``{type, default, doc}`` for ``dotted_key`` (e.g. ``encoder.dtype``).

    Raises :class:`MissingConfigError` if the key is not in the schema.
    """
    parts = dotted_key.split(".")
    if not parts or not parts[0]:
        raise InputError("--explain key must not be empty", details={"key": dotted_key})

    cls: type = GenoLeWMConfig
    field_obj: dataclasses.Field[Any] | None = None
    parent_doc = cls.__doc__ or ""

    for i, part in enumerate(parts):
        if not is_dataclass(cls):
            raise MissingConfigError(
                "--explain: path leaves the schema before resolving",
                details={"key": dotted_key, "where": ".".join(parts[: i + 1])},
            )
        try:
            field_obj = next(f for f in fields(cls) if f.name == part)
        except StopIteration as exc:
            known = [f.name for f in fields(cls)]
            raise MissingConfigError(
                "--explain: key not found in schema",
                details={"key": dotted_key, "where": part, "known": sorted(known)},
            ) from exc
        next_type = get_type_hints(cls).get(part)
        if is_dataclass(field_obj.type) and isinstance(field_obj.type, type):
            cls = field_obj.type
            parent_doc = cls.__doc__ or ""
            continue
        if next_type is not None and isinstance(next_type, type) and is_dataclass(next_type):
            cls = next_type
            parent_doc = cls.__doc__ or ""
            continue
        # Leaf field reached.
        return _format_field_info(field_obj, parent_doc=parent_doc, type_hint=next_type)

    if field_obj is None:  # pragma: no cover - guarded above
        raise InputError("--explain key did not resolve to a field", details={"key": dotted_key})
    return _format_field_info(field_obj, parent_doc=parent_doc, type_hint=None)


def _format_field_info(
    field_obj: dataclasses.Field[Any],
    *,
    parent_doc: str,
    type_hint: Any,
) -> dict[str, Any]:
    if field_obj.default is not dataclasses.MISSING:
        default: Any = field_obj.default
    elif field_obj.default_factory is not dataclasses.MISSING:
        default = field_obj.default_factory()
    else:
        default = None
    return {
        "name": field_obj.name,
        "type": str(type_hint) if type_hint is not None else str(field_obj.type),
        "default": default,
        "doc": _first_paragraph(parent_doc),
    }


def _first_paragraph(text: str) -> str:
    """Return the first non-empty paragraph of a docstring."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    for chunk in cleaned.split("\n\n"):
        s = chunk.strip()
        if s:
            return s
    return cleaned


# ---------------------------------------------------------------------------
# Utility re-exports
# ---------------------------------------------------------------------------


def iter_subsystem_names() -> Iterable[str]:
    """Yield the subsystem keys recognised by the loader."""
    return (name for name, _ in _SUBSYSTEM_MAP)
