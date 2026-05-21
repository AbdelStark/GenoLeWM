# SPDX-License-Identifier: Apache-2.0
"""Coverage-focused tests for ``geno_lewm.config.loader``.

These tests intentionally exercise the error / edge-case branches that
the happy-path suite in ``test_config_schema.py`` does not cover. They
keep ``geno_lewm/config/loader.py`` above the project-wide 95 % branch
coverage gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geno_lewm.config import (
    GenoLeWMConfig,
    config_to_dict,
    describe_field,
    load_config,
    write_resolved_config,
)
from geno_lewm.config.loader import (
    _first_paragraph,
    iter_subsystem_names,
)
from geno_lewm.errors import (
    ConfigError,
    InputError,
    MissingConfigError,
)

# ---------------------------------------------------------------------------
# Loader error paths
# ---------------------------------------------------------------------------


def test_load_config_subsystem_payload_not_mapping() -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config({"encoder": [1, 2, 3]})


def test_load_config_yaml_top_level_list_is_rejected(tmp_path: Path) -> None:
    """A YAML file whose top-level node is a list is rejected."""
    target = tmp_path / "list.yaml"
    target.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(InputError, match="mapping"):
        load_config(target)


def test_load_default_unknown_known_list_includes_existing(tmp_path: Path) -> None:
    """The ``known`` detail enumerates the YAML files present on disk."""
    with pytest.raises(MissingConfigError) as excinfo:
        from geno_lewm.config import load_default

        load_default("doesntexist")
    assert "train" in excinfo.value.details["known"]


def test_load_config_str_path_routes_to_file_loader(tmp_path: Path) -> None:
    """Passing a ``str`` (not Path) reaches the same file-loading branch."""
    target = tmp_path / "c.yaml"
    target.write_text("run_id: from-str\n", encoding="utf-8")
    cfg = load_config(str(target))
    assert cfg.run_id == "from-str"


# ---------------------------------------------------------------------------
# Type coercion edge cases
# ---------------------------------------------------------------------------


def test_int_field_rejects_bool() -> None:
    with pytest.raises(ConfigError, match="expected int, got bool"):
        load_config({"seed": True})


def test_float_field_rejects_bool() -> None:
    with pytest.raises(ConfigError, match="expected float, got bool"):
        load_config({"optimizer": {"lr": True}})


def test_float_field_rejects_string() -> None:
    with pytest.raises(ConfigError, match="expected float"):
        load_config({"optimizer": {"lr": "fast"}})


def test_str_field_rejects_int() -> None:
    with pytest.raises(ConfigError, match="expected str"):
        load_config({"run_id": 123})


def test_tuple_field_rejects_scalar() -> None:
    with pytest.raises(ConfigError, match="expected list/tuple"):
        load_config({"action": {"sub_encoders": "snv"}})


def test_union_field_accepts_first_matching_arm() -> None:
    """Union coercion picks the first arm that succeeds."""
    cfg = load_config({"observability": {"wandb_project": "my-project"}})
    assert cfg.observability.wandb_project == "my-project"


# ---------------------------------------------------------------------------
# config_to_dict edge cases
# ---------------------------------------------------------------------------


def test_config_to_dict_handles_nested_lists() -> None:
    cfg = GenoLeWMConfig()
    d = config_to_dict(cfg)
    # action.sub_encoders is a tuple → list; benchmarks is a tuple → list.
    assert isinstance(d["action"]["sub_encoders"], list)
    assert isinstance(d["eval"]["benchmarks"], list)


def test_config_to_dict_handles_explicit_dict_branch() -> None:
    from geno_lewm.config.loader import _asdict_with_tuples

    assert _asdict_with_tuples({"a": (1, 2)}) == {"a": [1, 2]}


def test_config_to_dict_handles_list_branch() -> None:
    from geno_lewm.config.loader import _asdict_with_tuples

    assert _asdict_with_tuples([1, 2, 3]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# describe_field paths
# ---------------------------------------------------------------------------


def test_describe_field_subsystem_root_is_dataclass() -> None:
    """Describing a subsystem-level key returns its dataclass metadata."""
    # ``encoder`` resolves to a dataclass (not a leaf) — describe_field
    # treats the *next* part as the leaf, so this falls through and
    # the final ``field_obj`` describes the encoder field on the
    # top-level GenoLeWMConfig.
    info = describe_field("encoder")
    assert info["name"] == "encoder"


def test_describe_field_returns_meta_on_subsystem_root() -> None:
    """A subsystem-level field describes the parent's typed slot."""
    info = describe_field("optimizer")
    assert info["name"] == "optimizer"


def test_describe_field_no_doc_returns_empty() -> None:
    """``_first_paragraph`` returns '' on empty input."""
    assert _first_paragraph("") == ""
    assert _first_paragraph("   ") == ""


def test_describe_field_doc_picks_first_paragraph() -> None:
    """Multi-paragraph docstrings collapse to the first non-empty paragraph."""
    assert _first_paragraph("first\n\nsecond") == "first"


def test_describe_field_doc_handles_single_paragraph() -> None:
    assert _first_paragraph("only paragraph") == "only paragraph"


# ---------------------------------------------------------------------------
# write_resolved_config edge cases
# ---------------------------------------------------------------------------


def test_write_resolved_config_creates_parent(tmp_path: Path) -> None:
    """Missing parent directories are created."""
    target = tmp_path / "nested" / "deep" / "r.yaml"
    write_resolved_config(GenoLeWMConfig(), target)
    assert target.is_file()


# ---------------------------------------------------------------------------
# Subsystem iterator
# ---------------------------------------------------------------------------


def test_iter_subsystem_names() -> None:
    names = list(iter_subsystem_names())
    assert "encoder" in names
    assert "runtime" in names
    assert "action" in names
