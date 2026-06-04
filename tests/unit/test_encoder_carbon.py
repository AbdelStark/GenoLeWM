"""Unit tests for the lazy Carbon state encoder wrapper."""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

import geno_lewm.encoder.carbon as carbon_mod
from geno_lewm.encoder import CarbonStateEncoder
from geno_lewm.errors import InputError, RuntimeSetupError


def test_carbon_state_encoder_reports_missing_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> Any:
        if name == "transformers":
            raise ImportError("no transformers")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(RuntimeSetupError, match="Transformers"):
        CarbonStateEncoder("HuggingFaceBio/Carbon-500M", "main")


def test_carbon_state_encoder_encodes_with_injected_components() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=model,
        tokenizer=tokenizer,
        encoder_hash="sha256:" + ("a" * 64),
        pool_radius=0,
    )

    state = encoder.encode("acgtac", edit_locus=0)

    assert state == (1.0, 0.0)
    assert tokenizer.calls == [["<dna>ACGTAC</dna>"]]
    assert model.eval_called is True
    assert encoder.d_state == 2
    assert encoder.encoder_hash == bytes.fromhex("a" * 64)


def test_carbon_state_encoder_batch_uses_per_item_loci() -> None:
    encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        pool_radius=0,
    )

    states = encoder.encode_batch(["ACGTAC", "CCCCCC"], [0, None])

    assert states == ((1.0, 0.0), (0.0, 4.0))


def test_carbon_state_encoder_calls_model_in_inference_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_active = False

    @contextlib.contextmanager
    def fake_inference_context() -> Iterator[None]:
        nonlocal context_active
        context_active = True
        try:
            yield
        finally:
            context_active = False

    class AssertingModel(FakeModel):
        def __call__(self, *, input_ids: list[list[int]], output_hidden_states: bool) -> object:
            assert context_active is True
            return super().__call__(
                input_ids=input_ids,
                output_hidden_states=output_hidden_states,
            )

    monkeypatch.setattr(carbon_mod, "torch_inference_context", fake_inference_context)
    encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=AssertingModel(),
        tokenizer=FakeTokenizer(),
        pool_radius=0,
    )

    assert encoder.encode("ACGTAC", edit_locus=0) == (1.0, 0.0)


def test_carbon_state_encoder_validates_component_pairing() -> None:
    with pytest.raises(InputError, match="supplied together"):
        CarbonStateEncoder("HuggingFaceBio/Carbon-500M", "main", model=FakeModel())


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self,
        texts: list[str],
        *,
        return_tensors: str,
        padding: bool,
    ) -> dict[str, list[list[int]]]:
        assert return_tensors == "pt"
        assert padding is True
        self.calls.append(texts)
        return {"input_ids": [[idx] for idx, _text in enumerate(texts)]}


class FakeModel:
    config = SimpleNamespace(hidden_size=2)

    def __init__(self) -> None:
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, *, input_ids: list[list[int]], output_hidden_states: bool) -> object:
        assert output_hidden_states is True
        rows_by_item = []
        for idx, _row in enumerate(input_ids):
            if idx == 0:
                rows_by_item.append(((1.0, 0.0), (3.0, 0.0), (5.0, 0.0)))
            else:
                rows_by_item.append(((0.0, 2.0), (0.0, 4.0), (0.0, 6.0)))
        return SimpleNamespace(hidden_states=(rows_by_item,))


class _DeviceFakeModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.moved_to: str | None = None

    def to(self, device: str) -> _DeviceFakeModel:
        self.moved_to = device
        return self


def test_resolve_device_explicit_and_default() -> None:
    from geno_lewm.encoder.carbon import _resolve_device

    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda:0") == "cuda:0"
    # None / "auto" resolve to cuda when a GPU is present, else cpu.
    assert _resolve_device(None) in {"cpu", "cuda"}
    assert _resolve_device("auto") in {"cpu", "cuda"}


def test_carbon_state_encoder_moves_model_to_cuda() -> None:
    model = _DeviceFakeModel()
    encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=model,
        tokenizer=FakeTokenizer(),
        pool_radius=0,
        device="cuda",
    )
    assert encoder.device == "cuda"
    assert model.moved_to == "cuda"
    # Encoding still works; tokenizer outputs without .to() pass through.
    assert encoder.encode("ACGTAC", edit_locus=0) == (1.0, 0.0)


def test_carbon_state_encoder_cpu_does_not_move_model() -> None:
    model = _DeviceFakeModel()
    encoder = CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=model,
        tokenizer=FakeTokenizer(),
        pool_radius=0,
        device="cpu",
    )
    assert encoder.device == "cpu"
    assert model.moved_to is None
