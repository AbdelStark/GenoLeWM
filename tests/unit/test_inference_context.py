# SPDX-License-Identifier: Apache-2.0
"""Tests for optional torch inference-mode helpers."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

import geno_lewm._inference as inference_mod
from geno_lewm._inference import torch_inference_context


class _RecordingContext:
    def __init__(self, events: list[str], name: str) -> None:
        self._events = events
        self._name = name

    def __enter__(self) -> None:
        self._events.append(f"enter:{self._name}")

    def __exit__(self, *exc: object) -> bool:
        self._events.append(f"exit:{self._name}")
        return False


def test_torch_inference_context_prefers_inference_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    fake_torch = SimpleNamespace(
        inference_mode=lambda: _RecordingContext(events, "inference_mode"),
        no_grad=lambda: _RecordingContext(events, "no_grad"),
    )

    monkeypatch.setattr(inference_mod.importlib, "import_module", lambda _name: fake_torch)

    with torch_inference_context():
        events.append("inside")

    assert events == ["enter:inference_mode", "inside", "exit:inference_mode"]


def test_torch_inference_context_falls_back_to_no_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    fake_torch = SimpleNamespace(no_grad=lambda: _RecordingContext(events, "no_grad"))

    monkeypatch.setattr(inference_mod.importlib, "import_module", lambda _name: fake_torch)

    with torch_inference_context():
        events.append("inside")

    assert events == ["enter:no_grad", "inside", "exit:no_grad"]


def test_torch_inference_context_keeps_torch_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_torch(name: str) -> Any:
        if name == "torch":
            raise ImportError("no torch")
        return importlib.import_module(name)

    monkeypatch.setattr(inference_mod.importlib, "import_module", missing_torch)

    with torch_inference_context():
        pass


def test_torch_inference_context_disables_grad_when_torch_is_installed() -> None:
    torch = pytest.importorskip("torch")
    was_enabled = bool(torch.is_grad_enabled())

    try:
        torch.set_grad_enabled(True)
        with torch_inference_context():
            assert bool(torch.is_grad_enabled()) is False
    finally:
        torch.set_grad_enabled(was_enabled)
