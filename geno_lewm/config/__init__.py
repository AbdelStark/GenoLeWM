# SPDX-License-Identifier: Apache-2.0
"""Typed configuration surface (RFC-0017).

The schema is declared as nested frozen dataclasses (RFC-0017 §3.2 left
the choice between Pydantic v2 and dataclasses open; we chose
dataclasses to keep base runtime deps minimal — the package's only new
runtime dep introduced by this module is :mod:`yaml`).

Public surface:

- :class:`GenoLeWMConfig` — top-level dataclass.
- :class:`EncoderConfig`, :class:`PredictorConfig`,
  :class:`ActionEncoderConfig`, :class:`OptimizerConfig`,
  :class:`DataConfig`, :class:`EvalConfig`,
  :class:`ObservabilityConfig`, :class:`RuntimeConfig` —
  per-subsystem schemas.
- :func:`load_config` — load YAML + validate; raises
  :class:`UnknownTopLevelKeyError` on unknown top-level keys
  (RFC-0017 §3.3).
- :func:`write_resolved_config` — emit the resolved config as canonical
  YAML so the run directory is auditable (RFC-0017 §3.5).
- :func:`config_to_dict` — pure dict view of a config tree (used by
  the manifest writer and the ``--print-config`` flag in PR #29).
- :func:`describe_field` — schema introspection for the ``--explain``
  flag (PR #29). Returns the field's docstring snippet, type, and
  default value.
- :data:`DEFAULTS_DIR` — directory containing the canonical YAML
  templates for ``train``, ``score``, ``eval``, and ``plan`` commands.
"""

from __future__ import annotations

from geno_lewm.config.loader import (
    DEFAULTS_DIR,
    config_to_dict,
    describe_field,
    load_config,
    load_default,
    write_resolved_config,
)
from geno_lewm.config.schema import (
    ActionEncoderConfig,
    DataConfig,
    EncoderConfig,
    EvalConfig,
    GenoLeWMConfig,
    ObservabilityConfig,
    OptimizerConfig,
    PredictorConfig,
    RuntimeConfig,
    iter_top_level_field_names,
)

__all__ = [
    "DEFAULTS_DIR",
    "ActionEncoderConfig",
    "DataConfig",
    "EncoderConfig",
    "EvalConfig",
    "GenoLeWMConfig",
    "ObservabilityConfig",
    "OptimizerConfig",
    "PredictorConfig",
    "RuntimeConfig",
    "config_to_dict",
    "describe_field",
    "iter_top_level_field_names",
    "load_config",
    "load_default",
    "write_resolved_config",
]
