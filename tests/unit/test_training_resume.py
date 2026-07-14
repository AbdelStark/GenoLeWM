"""Behavioral tests for production Carbon checkpoint continuation."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from geno_lewm.training import resume as resume_module
from geno_lewm.training.resume import (
    CHECKPOINT_SCHEMA_VERSION,
    capture_rng_state,
    load_resume_checkpoint,
    restore_rng_state,
    write_resume_checkpoint,
)


def test_rng_state_round_trips_every_training_domain() -> None:
    numpy = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    random.seed(17)
    numpy.random.seed(18)
    torch.manual_seed(19)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20)

    state = capture_rng_state()
    expected = (
        random.random(),
        float(numpy.random.random()),
        torch.rand(3),
    )
    random.random()
    numpy.random.random()
    torch.rand(7)

    restore_rng_state(state)
    observed = (
        random.random(),
        float(numpy.random.random()),
        torch.rand(3),
    )

    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    torch.testing.assert_close(observed[2], expected[2], rtol=0, atol=0)
    assert set(state) == {"python", "numpy", "torch_cpu", "torch_cuda"}


def test_production_checkpoint_round_trips_as_one_closed_atomic_payload(tmp_path) -> None:
    path = tmp_path / "predictor_checkpoint.pt"
    payload = _write_fixture_checkpoint(path)

    loaded = load_resume_checkpoint(path)

    assert loaded["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert loaded["payload_digest"] == payload["payload_digest"]
    assert loaded["progress"]["steps_completed"] == 3
    assert not path.with_name(f".{path.name}.tmp").exists()


def test_checkpoint_loader_rejects_raw_tensor_tampering(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "predictor_checkpoint.pt"
    _write_fixture_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["states"]["predictor"]["weight"][0] = 99.0
    torch.save(payload, path)

    with pytest.raises(InputError, match="payload digest"):
        load_resume_checkpoint(path)


def test_checkpoint_loader_rejects_invalid_closed_progress_even_with_fresh_digest(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "predictor_checkpoint.pt"
    _write_fixture_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["progress"]["collapse_alert_count"] = -1
    payload["payload_digest"] = resume_module._payload_digest(payload)
    torch.save(payload, path)

    with pytest.raises(InputError, match="collapse-alert count"):
        load_resume_checkpoint(path)


@pytest.mark.parametrize("field", ["trainer_state", "rng_state"])
def test_checkpoint_loader_rejects_missing_closed_top_level_field(
    tmp_path: Path,
    field: str,
) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "predictor_checkpoint.pt"
    _write_fixture_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    del payload[field]
    torch.save(payload, path)

    with pytest.raises(InputError, match="fields do not match the closed contract"):
        load_resume_checkpoint(path)


def test_interrupted_atomic_checkpoint_write_preserves_previous_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "predictor_checkpoint.pt"
    original = _write_fixture_checkpoint(path)
    original_bytes = path.read_bytes()

    def fail_after_partial_write(_payload, stream) -> None:
        stream.write(b"partial replacement")
        stream.flush()
        raise RuntimeError("injected checkpoint write failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="injected checkpoint write failure"):
        _write_fixture_checkpoint(path)

    assert path.read_bytes() == original_bytes
    assert load_resume_checkpoint(path)["payload_digest"] == original["payload_digest"]
    assert not path.with_name(f".{path.name}.tmp").exists()


def _write_fixture_checkpoint(path: Path) -> dict[str, object]:
    torch = pytest.importorskip("torch")
    return write_resume_checkpoint(
        path,
        source={"commit_sha": "a" * 40, "tree_sha": "b" * 40},
        training_contract={"target_steps": 8, "batch_size": 2, "config": {"seed": 7}},
        identities={"dataset": "sha256:" + ("c" * 64), "encoder": "sha256:" + ("d" * 64)},
        progress={
            "steps_completed": 3,
            "samples_consumed": 6,
            "consumed_window_ids": [f"w{index}" for index in range(6)],
            "collapse_alert_count": 1,
        },
        states={
            "predictor": {"weight": torch.tensor([1.0])},
            "action_encoder": {"weight": torch.tensor([2.0])},
            "optimizer": {"state": {}, "param_groups": []},
        },
        trainer_state={"schema_version": "fixture", "total_steps": 8},
        rng_state=capture_rng_state(),
        metric_history=[{"step": 1, "lr_multiplier": 0.5}],
    )
