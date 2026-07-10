# SPDX-License-Identifier: Apache-2.0
"""Tests for the config schema + loader (configuration contract; issue #28)."""

from __future__ import annotations

from pathlib import Path

import pytest

from geno_lewm.config import (
    DEFAULTS_DIR,
    ActionEncoderConfig,
    DataConfig,
    EncoderConfig,
    GenoLeWMConfig,
    ObservabilityConfig,
    OptimizerConfig,
    RuntimeConfig,
    TrainingConfig,
    config_to_dict,
    describe_field,
    load_config,
    load_default,
    write_resolved_config,
)
from geno_lewm.config._state_contract import encoder_uses_normalized_states
from geno_lewm.config.schema import TOP_LEVEL_KEYS
from geno_lewm.errors import (
    ConfigError,
    InputError,
    MissingConfigError,
    UnknownTopLevelKeyError,
)

# ---------------------------------------------------------------------------
# Schema defaults
# ---------------------------------------------------------------------------


def test_default_top_level_constructs() -> None:
    cfg = GenoLeWMConfig()
    assert cfg.run_id == "default"
    assert cfg.phase == "phase1"
    assert isinstance(cfg.encoder, EncoderConfig)
    assert isinstance(cfg.runtime, RuntimeConfig)
    assert isinstance(cfg.training, TrainingConfig)


def test_top_level_keys_match_dataclass_fields() -> None:
    assert isinstance(TOP_LEVEL_KEYS, frozenset)
    assert "encoder" in TOP_LEVEL_KEYS
    assert "deterministic" in TOP_LEVEL_KEYS


@pytest.mark.parametrize("name", ["train", "score", "eval", "plan"])
def test_default_yaml_files_load(name: str) -> None:
    cfg = load_default(name)
    assert cfg.schema_version == "1.1.0"
    assert cfg.encoder.model_id.startswith("HuggingFaceBio/")
    assert cfg.encoder.state_contract_version == "l2_normalized_v2"
    assert encoder_uses_normalized_states(cfg.encoder) is True


def test_load_default_unknown_name_raises() -> None:
    with pytest.raises(MissingConfigError, match="default config"):
        load_default("nonexistent")


# ---------------------------------------------------------------------------
# Unknown-key rejection (configuration contract)
# ---------------------------------------------------------------------------


def test_load_config_rejects_unknown_top_level_key() -> None:
    with pytest.raises(UnknownTopLevelKeyError) as excinfo:
        load_config({"run_id": "x", "foo": 1})
    assert "foo" in excinfo.value.details["unknown"]


def test_load_config_rejects_unknown_sub_field() -> None:
    with pytest.raises(ConfigError, match="unknown sub-field"):
        load_config({"encoder": {"unrelated": 1}})


def test_load_config_payload_not_mapping() -> None:
    with pytest.raises(InputError, match="mapping"):
        load_config([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


def test_int_field_rejects_string() -> None:
    with pytest.raises(ConfigError, match="expected int"):
        load_config({"seed": "not-an-int"})


def test_bool_field_rejects_int() -> None:
    with pytest.raises(ConfigError, match="expected bool"):
        load_config({"deterministic": 1})


def test_float_field_accepts_int() -> None:
    cfg = load_config({"optimizer": {"lr": 1}})
    assert cfg.optimizer.lr == 1.0


def test_literal_field_rejects_off_menu_value() -> None:
    with pytest.raises(ConfigError, match="not in allowed set"):
        load_config({"phase": "phase99"})


def test_tuple_field_accepts_yaml_list() -> None:
    cfg = load_config({"action": {"sub_encoders": ["snv", "ins"]}})
    assert cfg.action.sub_encoders == ("snv", "ins")


def test_optional_field_accepts_none() -> None:
    cfg = load_config({"observability": {"wandb_project": None}})
    assert cfg.observability.wandb_project is None


# ---------------------------------------------------------------------------
# File-based loader
# ---------------------------------------------------------------------------


def test_load_config_from_path(tmp_path: Path) -> None:
    target = tmp_path / "c.yaml"
    target.write_text("run_id: from-disk\nseed: 7\n", encoding="utf-8")
    cfg = load_config(target)
    assert cfg.run_id == "from-disk"
    assert cfg.seed == 7


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MissingConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_invalid_yaml(tmp_path: Path) -> None:
    target = tmp_path / "broken.yaml"
    target.write_text("foo: : bar", encoding="utf-8")
    with pytest.raises(ConfigError, match="valid YAML"):
        load_config(target)


# ---------------------------------------------------------------------------
# Resolved-config writer
# ---------------------------------------------------------------------------


def test_write_resolved_config_round_trips(tmp_path: Path) -> None:
    cfg = load_default("train")
    target = tmp_path / "resolved.yaml"
    written = write_resolved_config(cfg, target)
    assert written == target.resolve()
    reloaded = load_config(target)
    assert reloaded.run_id == cfg.run_id
    assert reloaded.encoder.dtype == cfg.encoder.dtype
    assert reloaded.action.sub_encoders == cfg.action.sub_encoders


def test_write_resolved_config_canonical_yaml(tmp_path: Path) -> None:
    cfg = GenoLeWMConfig(run_id="x")
    target = tmp_path / "r.yaml"
    write_resolved_config(cfg, target)
    text = target.read_text(encoding="utf-8")
    # sort_keys=True → top-level keys appear alphabetically.
    keys = [
        line.split(":", 1)[0] for line in text.splitlines() if line and not line.startswith(" ")
    ]
    assert keys == sorted(keys)


def test_config_to_dict_handles_tuples() -> None:
    cfg = GenoLeWMConfig(
        action=ActionEncoderConfig(sub_encoders=("a", "b")),
    )
    d = config_to_dict(cfg)
    assert d["action"]["sub_encoders"] == ["a", "b"]


# ---------------------------------------------------------------------------
# describe_field (PR #29 surface)
# ---------------------------------------------------------------------------


def test_describe_field_top_level_scalar() -> None:
    info = describe_field("seed")
    assert info["name"] == "seed"
    assert info["default"] == 0
    assert "Top-level" in info["doc"]


def test_describe_field_nested_leaf() -> None:
    info = describe_field("encoder.dtype")
    assert info["name"] == "dtype"
    assert info["default"] == "bf16"
    assert "encoder" in info["doc"].lower() or "encoder" in info["doc"]


def test_describe_field_unknown_key() -> None:
    with pytest.raises(MissingConfigError, match="not found"):
        describe_field("encoder.nope")


def test_describe_field_empty_key() -> None:
    with pytest.raises(InputError, match="must not be empty"):
        describe_field("")


# ---------------------------------------------------------------------------
# Per-subsystem schemas
# ---------------------------------------------------------------------------


def test_encoder_config_defaults() -> None:
    e = EncoderConfig()
    assert e.normalize is True
    assert e.dtype == "bf16"
    assert e.state_contract_version == "l2_normalized_v2"
    assert encoder_uses_normalized_states(e) is True


def test_encoder_state_contract_can_enable_or_disable_normalized_view() -> None:
    normalized = EncoderConfig(state_contract_version="l2_normalized_v2")
    disabled = EncoderConfig(
        normalize=False,
        state_contract_version="l2_normalized_v2",
    )

    assert encoder_uses_normalized_states(normalized) is True
    with pytest.raises(InputError, match=r"requires encoder\.normalize=true"):
        encoder_uses_normalized_states(disabled)


def test_config_rejects_global_pooling_with_nonzero_radius() -> None:
    with pytest.raises(ConfigError, match=r"global_mean requires encoder\.pool_radius=0"):
        load_config(
            {
                "encoder": {
                    "pool_type": "global_mean",
                    "pool_radius": 8,
                }
            }
        )


def test_schema_v1_missing_state_contract_migrates_to_legacy_raw() -> None:
    cfg = load_config({"schema_version": "1.0.0", "encoder": {"normalize": True}})

    assert cfg.encoder.state_contract_version == "legacy_raw_v1"
    assert encoder_uses_normalized_states(cfg.encoder) is False


def test_schema_v1_rejects_normalized_v2_contract() -> None:
    with pytest.raises(ConfigError, match="supports only the legacy_raw_v1"):
        load_config(
            {
                "schema_version": "1.0.0",
                "encoder": {
                    "normalize": True,
                    "state_contract_version": "l2_normalized_v2",
                },
            }
        )


def test_missing_schema_version_uses_current_normalized_contract() -> None:
    cfg = load_config({"encoder": {"normalize": True}})

    assert cfg.schema_version == "1.1.0"
    assert cfg.encoder.state_contract_version == "l2_normalized_v2"
    assert encoder_uses_normalized_states(cfg.encoder) is True


@pytest.mark.parametrize("schema_version", ["1.0", "9.9.9", "invalid"])
def test_loader_rejects_unsupported_schema_versions(schema_version: str) -> None:
    with pytest.raises(ConfigError, match="not in allowed set"):
        load_config({"schema_version": schema_version})


def test_loader_rejects_incoherent_normalized_contract() -> None:
    with pytest.raises(ConfigError, match=r"requires encoder\.normalize=true"):
        load_config(
            {
                "schema_version": "1.1.0",
                "encoder": {
                    "normalize": False,
                    "state_contract_version": "l2_normalized_v2",
                },
            }
        )


def test_optimizer_config_defaults() -> None:
    o = OptimizerConfig()
    assert o.name == "adamw"
    assert o.schedule == "wsd"


def test_observability_optional_wandb() -> None:
    o = ObservabilityConfig()
    assert o.wandb_project is None


def test_data_config_defaults() -> None:
    d = DataConfig()
    assert d.batch_size > 0
    assert d.num_workers >= 0


def test_runtime_config_defaults() -> None:
    r = RuntimeConfig()
    assert r.backend in ("onnx", "coreml", "gguf", "torch")
    assert r.device in ("cpu", "cuda", "mps")


def test_training_config_defaults() -> None:
    t = TrainingConfig()
    assert t.max_steps > 0
    assert t.collapse_log_every_steps > 0


# ---------------------------------------------------------------------------
# DEFAULTS_DIR layout
# ---------------------------------------------------------------------------


def test_defaults_dir_has_all_four_commands() -> None:
    expected = {"train", "score", "eval", "plan"}
    found = {p.stem for p in DEFAULTS_DIR.glob("*.yaml")}
    assert expected <= found
