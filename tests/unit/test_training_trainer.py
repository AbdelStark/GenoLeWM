"""Tests for the optional torch trainer core."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import geno_lewm.training.trainer as trainer_module
from geno_lewm.action import EditType, RelEdit
from geno_lewm.config import load_default
from geno_lewm.data import TrainingTuple, WindowContext
from geno_lewm.encoder.windowing import window_sha256
from geno_lewm.errors import InputError, RuntimeSetupError
from geno_lewm.training.trainer import (
    TrainerSeeds,
    encode_training_batch,
    make_action_mask,
    set_optimizer_lr,
    wsd_lr_multiplier,
)


def test_trainer_seeds_are_distinct_and_stable() -> None:
    seeds = TrainerSeeds.from_base_seed(17)

    assert seeds.to_dict() == {"data": 17, "predictor": 18, "lora": 19}


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
        (("ACGTAC", "ACGTAC"), (None, None)),
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
    assert observed_keys[0].pool_type == "global_mean"
    assert observed_keys[0].pool_radius == 0
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

    states = trainer_module._source_states(encoder, ["ACGTAC"])

    assert states == ((3.0, 4.0),)
    assert encoder.calls == []
    assert observed_keys[0].pool_type == "global_mean"
    assert observed_keys[0].pool_radius == 0


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


def test_pred_var_per_dim_matches_population_variance() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.training.trainer import _pred_var_per_dim

    # Per-dim population variance of columns [0,2] and [0,4] is [1.0, 4.0]; mean 2.5.
    prediction = torch.tensor([[0.0, 0.0], [2.0, 4.0]])
    assert _pred_var_per_dim(prediction) == pytest.approx(2.5)

    # Higher-rank predictions are flattened to (rows, dim) before reduction.
    assert _pred_var_per_dim(torch.zeros(1, 3, 5)) == pytest.approx(0.0)


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

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], tuple[int | None, ...]]] = []

    def encode_batch(
        self,
        windows: list[str],
        edit_loci: list[int | None],
    ) -> tuple[tuple[float, float], ...]:
        assert len(windows) == len(edit_loci)
        self.calls.append((tuple(windows), tuple(edit_loci)))
        return tuple(
            (float(len(window)), -1.0 if locus is None else float(locus))
            for window, locus in zip(windows, edit_loci, strict=True)
        )
