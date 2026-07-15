"""Tests for the optional torch trainer core."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

import geno_lewm.training.trainer as trainer_module
from geno_lewm.action import EditType, RelEdit
from geno_lewm.config import load_config, load_default
from geno_lewm.data import TrainingTuple, WindowContext
from geno_lewm.encoder._normalization import l2_normalize_state
from geno_lewm.encoder.windowing import window_sha256
from geno_lewm.errors import InputError, RuntimeSetupError
from geno_lewm.training.trainer import (
    TorchTrainer,
    TorchTrainerBatch,
    TrainerSeeds,
    configure_torch_reproducibility,
    encode_training_batch,
    make_action_mask,
    set_optimizer_lr,
    wsd_lr_multiplier,
)


def test_trainer_seeds_are_distinct_and_stable() -> None:
    seeds = TrainerSeeds.from_base_seed(17)

    assert seeds.to_dict() == {"data": 17, "predictor": 18, "lora": 19}


def test_nondeterministic_report_preserves_preexisting_cublas_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def manual_seed(_seed: int) -> None:
            return None

        @staticmethod
        def use_deterministic_algorithms(enabled: bool) -> None:
            FakeTorch.enabled = enabled

        @staticmethod
        def are_deterministic_algorithms_enabled() -> bool:
            return FakeTorch.enabled

    FakeTorch.enabled = False
    monkeypatch.setattr(trainer_module, "torch", FakeTorch())
    monkeypatch.setattr(trainer_module, "_seed_numpy", lambda _seed: None)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    report = configure_torch_reproducibility(seed=17, deterministic=False)

    assert report.cublas_workspace_config == ":16:8"
    assert report.torch_deterministic_algorithms is False


def test_wsd_lr_multiplier_matches_warmup_stable_decay_taper() -> None:
    assert wsd_lr_multiplier(1, total_steps=100, warmup_steps=10) == pytest.approx(0.1)
    assert wsd_lr_multiplier(10, total_steps=100, warmup_steps=10) == pytest.approx(1.0)
    assert wsd_lr_multiplier(82, total_steps=100, warmup_steps=10) == pytest.approx(1.0)
    assert wsd_lr_multiplier(98, total_steps=100, warmup_steps=10) == pytest.approx(0.1)
    assert wsd_lr_multiplier(100, total_steps=100, warmup_steps=10) == pytest.approx(0.01)


def test_set_optimizer_lr_updates_groups_from_initial_lr() -> None:
    optimizer = FakeOptimizer()

    multiplier = set_optimizer_lr(
        optimizer,
        step=5,
        total_steps=10,
        warmup_steps=10,
    )

    assert multiplier == pytest.approx(0.5)
    assert optimizer.param_groups == [
        {"lr": 0.05, "initial_lr": 0.1},
        {"lr": 0.005, "initial_lr": 0.01},
    ]


def test_action_mask_reports_missing_torch_runtime() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed in this environment")

    with pytest.raises(RuntimeSetupError):
        make_action_mask([[_edit()]])


def test_encode_training_batch_uses_carbon_encoder_and_masks_ragged_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(trainer_module, "default_cache_dir", lambda: tmp_path / "empty-cache")
    source = WindowContext(record_id="w1", source="fixture", sequence="ACGTAC", chrom="1")
    first = TrainingTuple(
        window_id=source.window_id,
        source_record_id=source.record_id,
        edit_source="synthetic_snv",
        rel_edits=(_edit(rel_pos=1),),
        target_window="ATGTAC",
        window_start_bp=0,
        window_end_bp=6,
    )
    second = TrainingTuple(
        window_id=source.window_id,
        source_record_id=source.record_id,
        edit_source="synthetic_snv",
        rel_edits=(_edit(rel_pos=2), _edit(rel_pos=3)),
        target_window="ACCTAC",
        window_start_bp=0,
        window_end_bp=6,
    )

    encoder = FakeCarbonEncoder()
    batch = encode_training_batch(
        encoder=encoder,
        tuples=[first, second],
        source_windows={source.window_id: source.sequence},
    )

    assert batch.window_ids == (source.window_id, source.window_id)
    assert batch.state.shape == (2, 2)
    assert batch.target.shape == (2, 2, 2)
    assert batch.action_mask.tolist() == [[True, False], [True, True]]
    assert encoder.calls == [
        (("ACGTAC",), (1,)),
        (("ATGTAC", "ACCTAC"), (1, 2)),
    ]
    torch.testing.assert_close(batch.target[0, 1], torch.zeros(2))


def test_encode_training_batch_reuses_cached_source_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    cache_dir = tmp_path / "cache"
    (cache_dir / "embeddings").mkdir(parents=True)
    (cache_dir / "embeddings" / "index.sqlite").write_text("", encoding="utf-8")
    source = WindowContext(record_id="w1", source="fixture", sequence="ACGTAC", chrom="1")
    item = TrainingTuple(
        window_id=source.window_id,
        source_record_id=source.record_id,
        edit_source="synthetic_snv",
        rel_edits=(_edit(rel_pos=1),),
        target_window="ATGTAC",
        window_start_bp=0,
        window_end_bp=6,
    )
    observed_keys = []

    def fake_read_embedding(cache_root: Path, key: object) -> tuple[float, float]:
        assert cache_root == cache_dir
        observed_keys.append(key)
        return (99.0, 100.0)

    monkeypatch.setattr(trainer_module, "default_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(trainer_module, "read_embedding", fake_read_embedding)
    encoder = FakeCarbonEncoder()

    batch = encode_training_batch(
        encoder=encoder,
        tuples=[item],
        source_windows={source.window_id: source.sequence},
    )

    assert len(observed_keys) == 1
    assert observed_keys[0].pool_type == "centered_mean"
    assert observed_keys[0].pool_radius == 8
    assert observed_keys[0].center_token == 1
    assert encoder.calls == [(("ATGTAC",), (1,))]
    torch.testing.assert_close(batch.state, torch.tensor([[99.0, 100.0]]))


def test_source_states_reuses_cache_without_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    (cache_dir / "embeddings").mkdir(parents=True)
    (cache_dir / "embeddings" / "index.sqlite").write_text("", encoding="utf-8")
    observed_keys = []

    def fake_read_embedding(cache_root: Path, key: object) -> tuple[float, float]:
        assert cache_root == cache_dir
        observed_keys.append(key)
        return (3.0, 4.0)

    monkeypatch.setattr(trainer_module, "default_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(trainer_module, "read_embedding", fake_read_embedding)
    encoder = FakeCarbonEncoder()

    states = trainer_module._source_states(encoder, ["ACGTAC"], [1])

    assert states == ((3.0, 4.0),)
    assert encoder.calls == []
    assert observed_keys[0].pool_type == "centered_mean"
    assert observed_keys[0].pool_radius == 8
    assert observed_keys[0].center_token == 1


def test_source_states_applies_encoder_normalization_to_raw_cache_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    (cache_dir / "embeddings").mkdir(parents=True)
    (cache_dir / "embeddings" / "index.sqlite").write_text("", encoding="utf-8")

    monkeypatch.setattr(trainer_module, "default_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(
        trainer_module,
        "read_embedding",
        lambda _cache_root, _key: (3.0, 4.0),
    )
    encoder = FakeCarbonEncoder(normalize=True)

    states = trainer_module._source_states(encoder, ["ACGTAC"], [1])

    assert states[0] == pytest.approx((0.6, 0.8))
    assert encoder.calls == []


def test_source_states_reuses_in_memory_cache_without_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_module, "default_cache_dir", lambda: tmp_path / "empty-cache")
    encoder = FakeCarbonEncoder()

    first = trainer_module._source_states(encoder, ["ACGTAC", "ACGTAC"], [1, 1])
    second = trainer_module._source_states(encoder, ["ACGTAC"], [1])

    assert first == ((6.0, 1.0), (6.0, 1.0))
    assert second == ((6.0, 1.0),)
    assert encoder.calls == [(("ACGTAC",), (1,))]


def test_encode_training_batch_reuses_in_memory_source_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(trainer_module, "default_cache_dir", lambda: tmp_path / "empty-cache")
    source = WindowContext(record_id="w1", source="fixture", sequence="ACGTAC", chrom="1")
    item = TrainingTuple(
        window_id=source.window_id,
        source_record_id=source.record_id,
        edit_source="synthetic_snv",
        rel_edits=(_edit(rel_pos=1),),
        target_window="ATGTAC",
        window_start_bp=0,
        window_end_bp=6,
    )
    encoder = FakeCarbonEncoder()

    encode_training_batch(
        encoder=encoder,
        tuples=[item],
        source_windows={source.window_id: source.sequence},
    )
    encode_training_batch(
        encoder=encoder,
        tuples=[item],
        source_windows={source.window_id: source.sequence},
    )

    assert encoder.calls == [
        (("ACGTAC",), (1,)),
        (("ATGTAC",), (1,)),
        (("ATGTAC",), (1,)),
    ]


def test_encode_training_batch_rejects_missing_source_window() -> None:
    pytest.importorskip("torch")
    item = TrainingTuple(
        window_id=window_sha256("ACGT").hex(),
        source_record_id="missing",
        edit_source="synthetic_snv",
        rel_edits=(_edit(),),
        target_window="TCGT",
        window_start_bp=0,
        window_end_bp=4,
    )

    with pytest.raises(InputError, match="source window sequence missing"):
        encode_training_batch(encoder=FakeCarbonEncoder(), tuples=[item], source_windows={})


def test_source_pooling_identity_requires_encoder_resolver() -> None:
    with pytest.raises(RuntimeSetupError, match=r"requires encoder\.pooling_identity"):
        trainer_module._source_pooling_identity(object(), "ACGT", 1)


@pytest.mark.parametrize(
    ("resolved", "message"),
    [
        ("centered_mean", "returned invalid cache metadata"),
        (("attention", 8, 1), "supported encoder pool_type"),
        (("global_mean", 1, None), "radius zero and no center"),
        (("centered_mean", 8, None), "requires an exact center_token"),
        (("centered_mean", 7, 1), "disagrees with configured pooling"),
    ],
)
def test_source_pooling_identity_rejects_invalid_resolver_metadata(
    resolved: object,
    message: str,
) -> None:
    encoder = PoolingIdentityEncoder(resolved)

    with pytest.raises(RuntimeSetupError, match=message):
        trainer_module._source_pooling_identity(encoder, "ACGT", 1)


def test_source_pooling_identity_accepts_canonical_global_metadata() -> None:
    encoder = PoolingIdentityEncoder(("global_mean", 0, None))

    assert trainer_module._source_pooling_identity(encoder, "ACGT", 1) == (
        "global_mean",
        0,
        None,
    )


def test_source_pooling_identity_rejects_centered_metadata_without_edit_locus() -> None:
    encoder = PoolingIdentityEncoder(("centered_mean", 8, 1))

    with pytest.raises(RuntimeSetupError, match="requires an edit locus"):
        trainer_module._source_pooling_identity(encoder, "ACGT", None)


def test_encoder_normalization_flag_must_be_boolean() -> None:
    class InvalidNormalizationEncoder:
        normalize = "true"

    with pytest.raises(RuntimeSetupError, match="normalize to be boolean"):
        trainer_module._encoder_normalizes(InvalidNormalizationEncoder())


def test_build_adamw_optimizer_accepts_real_small_modules() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.training.trainer import build_adamw_optimizer

    cfg = load_default("train")
    predictor = torch.nn.Linear(2, 2)
    action_encoder = torch.nn.Sequential(torch.nn.Embedding(4, 2), torch.nn.Linear(2, 2))

    optimizer = build_adamw_optimizer(
        predictor=predictor,
        action_encoder=action_encoder,
        config=cfg,
    )

    assert len(optimizer.param_groups) == 2
    assert {group["weight_decay"] for group in optimizer.param_groups} == {
        cfg.optimizer.weight_decay,
        0.0,
    }


def test_phase2_trainer_and_optimizer_reject_missing_encoder_adapter() -> None:
    torch = pytest.importorskip("torch")
    phase2_cfg = replace(load_default("train"), phase="phase2")
    predictor = torch.nn.Linear(2, 2)
    action_encoder = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(
        list(predictor.parameters()) + list(action_encoder.parameters()),
        lr=0.01,
    )

    with pytest.raises(RuntimeSetupError, match="graph-preserving trainable encoder-adapter"):
        TorchTrainer(
            predictor=predictor,
            action_encoder=action_encoder,
            optimizer=optimizer,
            config=phase2_cfg,
            total_steps=1,
        )
    with pytest.raises(RuntimeSetupError, match="graph-preserving trainable encoder-adapter"):
        trainer_module.build_adamw_optimizer(
            predictor=predictor,
            action_encoder=action_encoder,
            config=phase2_cfg,
        )


def test_pred_var_per_dim_matches_population_variance() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.training.trainer import _pred_var_per_dim

    # Per-dim population variance of columns [0,2] and [0,4] is [1.0, 4.0]; mean 2.5.
    prediction = torch.tensor([[0.0, 0.0], [2.0, 4.0]])
    assert _pred_var_per_dim(prediction) == pytest.approx(2.5)

    # Higher-rank predictions are flattened to (rows, dim) before reduction.
    assert _pred_var_per_dim(torch.zeros(1, 3, 5)) == pytest.approx(0.0)


def test_masked_training_rows_accepts_only_shape_matched_binary_masks() -> None:
    torch = pytest.importorskip("torch")
    values = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])

    selected = trainer_module._masked_training_rows(
        values,
        torch.tensor([[1, 0], [0, 1]]),
    )

    torch.testing.assert_close(selected, torch.tensor([[1.0], [4.0]]))
    with pytest.raises(InputError, match="leading dimensions"):
        trainer_module._masked_training_rows(values, torch.ones(2, 1, dtype=torch.bool))
    with pytest.raises(InputError, match="boolean or 0/1"):
        trainer_module._masked_training_rows(values, torch.tensor([[2, 0], [0, 1]]))


def test_torch_trainer_records_live_collapse_alerts() -> None:
    torch = pytest.importorskip("torch")
    cfg = load_config(
        {"training": {"collapse_log_every_steps": 1}, "optimizer": {"warmup_steps": 0}}
    )
    predictor = CollapsedPredictor(torch)
    action_encoder = DummyActionEncoder(torch)
    optimizer = torch.optim.SGD(
        list(predictor.parameters()) + list(action_encoder.parameters()),
        lr=0.01,
    )
    trainer = TorchTrainer(
        predictor=predictor,
        action_encoder=action_encoder,
        optimizer=optimizer,
        config=cfg,
        total_steps=1,
    )
    batch = TorchTrainerBatch(
        state=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        target=torch.tensor([[[0.0, 0.0]], [[2.0, 2.0]]]),
        rel_edits=((_edit(),), (_edit(),)),
        action_mask=torch.tensor([[True], [True]]),
        window_ids=("a", "b"),
    )

    result = trainer.train_step(batch, step=1)

    assert result.action_count == 2
    assert any(alert["criterion"] == "pred_var_per_dim" for alert in trainer.last_collapse_alerts)


def test_torch_trainer_state_round_trips_collapse_monitor_across_processes() -> None:
    torch = pytest.importorskip("torch")
    cfg = load_config(
        {"training": {"collapse_log_every_steps": 1}, "optimizer": {"warmup_steps": 0}}
    )

    def build() -> TorchTrainer:
        predictor = CollapsedPredictor(torch)
        action_encoder = DummyActionEncoder(torch)
        optimizer = torch.optim.SGD(
            list(predictor.parameters()) + list(action_encoder.parameters()),
            lr=0.01,
        )
        return TorchTrainer(
            predictor=predictor,
            action_encoder=action_encoder,
            optimizer=optimizer,
            config=cfg,
            total_steps=4,
        )

    source = build()
    batch = TorchTrainerBatch(
        state=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        target=torch.tensor([[[0.0, 0.0]], [[2.0, 2.0]]]),
        rel_edits=((_edit(),), (_edit(),)),
        action_mask=torch.tensor([[True], [True]]),
        window_ids=("a", "b"),
    )
    source.train_step(batch, step=1)

    restored = build()
    restored.load_state_dict(source.state_dict())

    assert restored.state_dict() == source.state_dict()


@pytest.mark.parametrize(
    ("contract_drift", "message"),
    [
        ("field-set", "trainer state fields do not match"),
        ("schema", "schema version is unsupported"),
        ("horizon", "horizon does not match"),
        ("monitor-field-set", "monitor state fields do not match"),
        ("monitor-contract", "monitor contract does not match"),
        ("alerts-container", "collapse alerts must be a list"),
        ("empty-criterion", "non-empty criterion"),
        ("negative-threshold", "non-negative thresholds"),
    ],
)
def test_torch_trainer_rejects_open_or_drifted_resume_contracts(
    contract_drift: str,
    message: str,
) -> None:
    torch = pytest.importorskip("torch")
    trainer = _resume_state_trainer(torch)
    state = trainer.state_dict()
    monitor = state["collapse_monitor"]
    assert isinstance(monitor, dict)

    if contract_drift == "field-set":
        del state["schema_version"]
    elif contract_drift == "schema":
        state["schema_version"] = "geno-lewm.torch-trainer-state.v0"
    elif contract_drift == "horizon":
        state["total_steps"] = trainer.total_steps + 1
    elif contract_drift == "monitor-field-set":
        del monitor["thresholds"]
    elif contract_drift == "monitor-contract":
        monitor["log_every_steps"] = 999
    elif contract_drift == "alerts-container":
        state["last_collapse_alerts"] = {"not": "a list"}
    elif contract_drift == "empty-criterion":
        state["last_collapse_alerts"] = [{"criterion": "", "value": 0.1, "threshold": 0.2}]
    else:
        state["last_collapse_alerts"] = [
            {"criterion": "pred_var_per_dim", "value": 0.1, "threshold": -0.2}
        ]

    with pytest.raises(InputError, match=message):
        trainer.load_state_dict(state)


@pytest.mark.parametrize("baseline", [-1.0, float("nan"), float("inf")])
def test_torch_trainer_rejects_invalid_restored_collapse_baseline(baseline: float) -> None:
    torch = pytest.importorskip("torch")
    trainer = _resume_state_trainer(torch)
    state = trainer.state_dict()
    monitor = state["collapse_monitor"]
    assert isinstance(monitor, dict)
    monitor["initial_pairwise_pred_dist_mean"] = baseline

    with pytest.raises(InputError, match="baseline"):
        trainer.load_state_dict(state)


@pytest.mark.parametrize(
    "alert",
    [
        {"criterion": "pred_var_per_dim", "value": 0.1},
        {
            "criterion": "pred_var_per_dim",
            "value": 0.1,
            "threshold": 0.2,
            "unexpected": True,
        },
        {
            "criterion": "pred_var_per_dim",
            "value": float("nan"),
            "threshold": 0.2,
        },
    ],
)
def test_torch_trainer_rejects_open_or_nonfinite_restored_alerts(
    alert: dict[str, object],
) -> None:
    torch = pytest.importorskip("torch")
    trainer = _resume_state_trainer(torch)
    state = trainer.state_dict()
    state["last_collapse_alerts"] = [alert]

    with pytest.raises(InputError, match="collapse alerts"):
        trainer.load_state_dict(state)


def _resume_state_trainer(torch) -> TorchTrainer:
    cfg = load_config(
        {"training": {"collapse_log_every_steps": 1}, "optimizer": {"warmup_steps": 0}}
    )
    predictor = CollapsedPredictor(torch)
    action_encoder = DummyActionEncoder(torch)
    optimizer = torch.optim.SGD(
        list(predictor.parameters()) + list(action_encoder.parameters()),
        lr=0.01,
    )
    return TorchTrainer(
        predictor=predictor,
        action_encoder=action_encoder,
        optimizer=optimizer,
        config=cfg,
        total_steps=4,
    )


def test_normalized_phase1_trainer_does_not_treat_kl_as_collapse_alert() -> None:
    torch = pytest.importorskip("torch")
    cfg = load_config(
        {
            "encoder": {
                "normalize": True,
                "state_contract_version": "l2_normalized_v2",
            },
            "predictor": {"d_state": 64},
            "training": {"collapse_log_every_steps": 1},
            "optimizer": {"warmup_steps": 0, "grad_clip": 0.0},
        }
    )
    predictor = IdentityPredictor(torch)
    action_encoder = DummyActionEncoder(torch)
    optimizer = torch.optim.SGD(
        list(predictor.parameters()) + list(action_encoder.parameters()),
        lr=0.01,
    )
    trainer = TorchTrainer(
        predictor=predictor,
        action_encoder=action_encoder,
        optimizer=optimizer,
        config=cfg,
        total_steps=1,
    )
    states = torch.eye(8, 64)
    batch = TorchTrainerBatch(
        state=states,
        target=states.unsqueeze(1),
        rel_edits=tuple((_edit(),) for _ in range(8)),
        action_mask=torch.ones(8, 1, dtype=torch.bool),
        window_ids=tuple(f"w{index}" for index in range(8)),
    )

    result = trainer.train_step(batch, step=1)

    assert result.kl_reg > 10.0
    assert not any(alert["criterion"] == "kl_reg" for alert in trainer.last_collapse_alerts)


def _edit(rel_pos: int = 0) -> RelEdit:
    return RelEdit(rel_pos=rel_pos, edit_type=EditType.SNV, ref_bases="A", alt_bases="T")


class FakeOptimizer:
    def __init__(self) -> None:
        self.param_groups = [{"lr": 0.1}, {"lr": 0.01, "initial_lr": 0.01}]


class FakeCarbonEncoder:
    encoder_hash = bytes.fromhex("a" * 64)
    state_layer = 20
    pool_type = "centered_mean"
    pool_radius = 8
    dtype = "bf16"

    def __init__(self, *, normalize: bool = False) -> None:
        self.normalize = normalize
        self.calls: list[tuple[tuple[str, ...], tuple[int | None, ...]]] = []

    def encode_batch(
        self,
        windows: list[str],
        edit_loci: list[int | None],
    ) -> tuple[tuple[float, float], ...]:
        assert len(windows) == len(edit_loci)
        self.calls.append((tuple(windows), tuple(edit_loci)))
        states = tuple(
            (float(len(window)), -1.0 if locus is None else float(locus))
            for window, locus in zip(windows, edit_loci, strict=True)
        )
        if self.normalize:
            return tuple(l2_normalize_state(state) for state in states)
        return states

    def pooling_identity(
        self,
        window: str,
        edit_locus: int | None,
    ) -> tuple[str, int, int | None]:
        del window
        if edit_locus is None:
            return "global_mean", 0, None
        return self.pool_type, self.pool_radius, 1 + (edit_locus // 6)


class PoolingIdentityEncoder:
    pool_type = "centered_mean"
    pool_radius = 8

    def __init__(self, resolved: object) -> None:
        self.resolved = resolved

    def pooling_identity(self, _window: str, _edit_locus: int | None) -> object:
        return self.resolved


class DummyActionEncoder:
    def __init__(self, torch_module) -> None:
        self._torch = torch_module
        self.bias = torch_module.nn.Parameter(torch_module.zeros(2))

    def parameters(self):
        return (self.bias,)

    def __call__(self, rel_edits):
        batch = len(rel_edits)
        width = max(len(row) for row in rel_edits)
        return self.bias.view(1, 1, 2).expand(batch, width, 2)


class CollapsedPredictor:
    def __init__(self, torch_module) -> None:
        self._torch = torch_module
        self.bias = torch_module.nn.Parameter(torch_module.zeros(2))

    def parameters(self):
        return (self.bias,)

    def __call__(self, state, actions, action_mask):
        del actions
        return self.bias.view(1, 1, 2).expand(state.shape[0], action_mask.shape[1], 2)


class IdentityPredictor:
    def __init__(self, torch_module) -> None:
        self.scale = torch_module.nn.Parameter(torch_module.ones(()))

    def parameters(self):
        return (self.scale,)

    def __call__(self, state, actions, action_mask):
        del actions
        return (state * self.scale).unsqueeze(1).expand(-1, action_mask.shape[1], -1)
