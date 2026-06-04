# SPDX-License-Identifier: Apache-2.0
"""Optional torch inference-mode helpers."""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterator


@contextlib.contextmanager
def torch_inference_context() -> Iterator[None]:
    """Enter ``torch.inference_mode`` when torch is installed.

    GenoLeWM keeps torch optional for core imports, so eval/inference paths use
    this helper instead of importing torch at module import time.
    """
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        yield
        return

    inference_mode = getattr(torch, "inference_mode", None)
    if callable(inference_mode):
        with inference_mode():
            yield
        return

    no_grad = getattr(torch, "no_grad", None)
    if callable(no_grad):
        with no_grad():
            yield
        return

    yield
