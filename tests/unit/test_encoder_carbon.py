"""Unit tests for the lazy Carbon state encoder wrapper."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

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
