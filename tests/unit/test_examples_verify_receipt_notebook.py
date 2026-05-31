# SPDX-License-Identifier: Apache-2.0
"""Tests for the receipt-verification tutorial notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "examples" / "07_verify_receipt.ipynb"


def _code_cells() -> list[dict[str, Any]]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_verify_receipt_notebook_is_committed_executed() -> None:
    code_cells = _code_cells()

    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert any(
        output.get("output_type") == "stream" and "ok\n" in output.get("text", [])
        for cell in code_cells
        for output in cell["outputs"]
    )


def test_verify_receipt_notebook_cells_execute(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.chdir(ROOT)
    namespace: dict[str, Any] = {"__name__": "__notebook__"}

    for index, cell in enumerate(_code_cells(), start=1):
        source = "".join(cell["source"])
        exec(compile(source, f"{NOTEBOOK}:cell-{index}", "exec"), namespace)

    captured = capsys.readouterr()
    assert "manifest model_id matches receipt: True" in captured.out
    assert "input_commitment ok" in captured.out
    assert captured.out.rstrip().endswith("ok")
